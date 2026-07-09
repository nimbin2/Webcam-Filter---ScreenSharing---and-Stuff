/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * ort_rvm.c — RVM (Robust Video Matting) on ONNX Runtime.
 *
 * Built with ORT=1 this is a real backend against the ORT C API;
 * otherwise the #else half provides no-op stubs so the rest of vcam
 * links unchanged. See include/vcam_ort.h for the interface story.
 *
 * Implementation notes (learned the hard way from the OpenCV attempt):
 *   - Tensors are matched BY NAME (src, r1i.., fgr, pha, r1o..), never
 *     by position. Model I/O order is an implementation detail.
 *   - The four recurrent outputs are carried to the next frame by
 *     keeping their OrtValue* alive and feeding them back as inputs —
 *     zero copies, and exactly how RVM's own examples do it.
 *   - Every ORT call is checked; the first failure prints one line and
 *     the backend marks itself dead for the session. Callers fall back.
 */

#include "vcam_ort.h"

#include <stdio.h>

#if VCAM_HAVE_ORT

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "onnxruntime_c_api.h"

static const OrtApi        *A;
static OrtEnv              *g_env;
static OrtSessionOptions   *g_so;
static OrtSession          *g_sess;
static OrtMemoryInfo       *g_mi;      /* CPU, for wrapping caller buffers */
static OrtAllocator        *g_alloc;   /* default allocator (names, tensors) */

static OrtValue *g_rec[4];             /* recurrent state r1..r4 (owned) */
static int g_iw = 0, g_ih = 0;         /* frame size the state belongs to */
static int g_dead = 0;                 /* backend failed; stay down */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

/* fixed RVM interface names; init() verifies the model matches */
static const char *IN_NAMES[6]  = {"src","r1i","r2i","r3i","r4i","downsample_ratio"};
static const char *OUT_NAMES[6] = {"fgr","pha","r1o","r2o","r3o","r4o"};

/* Check an OrtStatus; on error print once, release it, and return 0. */
static int ok(OrtStatus *st, const char *where)
{
    if (!st) return 1;
    fprintf(stderr, "[vcam/ort] %s failed: %s\n", where, A->GetErrorMessage(st));
    A->ReleaseStatus(st);
    return 0;
}

static void release_rec(void)
{
    for (int i = 0; i < 4; i++) {
        if (g_rec[i]) { A->ReleaseValue(g_rec[i]); g_rec[i] = NULL; }
    }
}

/* Fresh [1,1,1,1] zero tensors for the recurrent inputs (the shape RVM
 * documents for the initial state; it broadcasts internally). */
static int init_rec(void)
{
    release_rec();
    const int64_t shp[4] = {1, 1, 1, 1};
    for (int i = 0; i < 4; i++) {
        if (!ok(A->CreateTensorAsOrtValue(g_alloc, shp, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_rec[i]),
                "CreateTensorAsOrtValue(rec)"))
            return 0;
        float *p = NULL;
        if (!ok(A->GetTensorMutableData(g_rec[i], (void **)&p),
                "GetTensorMutableData(rec)"))
            return 0;
        p[0] = 0.0f;
    }
    return 1;
}

/* Confirm the session's inputs/outputs are exactly the RVM interface.
 * Rejecting mismatches here means a wrong .onnx can never half-run. */
static int verify_iface(void)
{
    size_t nin = 0, nout = 0;
    if (!ok(A->SessionGetInputCount(g_sess, &nin), "SessionGetInputCount"))  return 0;
    if (!ok(A->SessionGetOutputCount(g_sess, &nout), "SessionGetOutputCount")) return 0;
    if (nin != 6 || nout != 6) {
        fprintf(stderr, "[vcam/ort] model has %zu inputs / %zu outputs; RVM needs 6/6\n",
                nin, nout);
        return 0;
    }
    int seen_in = 0, seen_out = 0;
    for (size_t i = 0; i < 6; i++) {
        char *nm = NULL;
        if (!ok(A->SessionGetInputName(g_sess, i, g_alloc, &nm), "SessionGetInputName"))
            return 0;
        for (int k = 0; k < 6; k++)
            if (strcmp(nm, IN_NAMES[k]) == 0) { seen_in |= 1 << k; break; }
        if (!ok(A->AllocatorFree(g_alloc, nm), "AllocatorFree")) return 0;
        if (!ok(A->SessionGetOutputName(g_sess, i, g_alloc, &nm), "SessionGetOutputName"))
            return 0;
        for (int k = 0; k < 6; k++)
            if (strcmp(nm, OUT_NAMES[k]) == 0) { seen_out |= 1 << k; break; }
        if (!ok(A->AllocatorFree(g_alloc, nm), "AllocatorFree")) return 0;
    }
    if (seen_in != 0x3F || seen_out != 0x3F) {
        fprintf(stderr, "[vcam/ort] model I/O names don't match the RVM interface "
                        "(src,r1i..r4i,downsample_ratio -> fgr,pha,r1o..r4o)\n");
        return 0;
    }
    return 1;
}

bool ort_available(void) { return true; }

const char *ort_version(void)
{
    return OrtGetApiBase()->GetVersionString();
}

int ort_rvm_init(const char *model_path)
{
    if (!model_path || !*model_path) return 0;
    pthread_mutex_lock(&g_mtx);
    if (g_sess) { pthread_mutex_unlock(&g_mtx); return 1; }   /* already up */
    g_dead = 0;

    if (!A) A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!A) {
        fprintf(stderr, "[vcam/ort] ORT C API version %d unavailable in this "
                        "libonnxruntime\n", ORT_API_VERSION);
        goto fail;
    }
    if (!g_env &&
        !ok(A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "vcam", &g_env), "CreateEnv"))
        goto fail;
    if (!g_so) {
        if (!ok(A->CreateSessionOptions(&g_so), "CreateSessionOptions")) goto fail;
        if (!ok(A->SetSessionGraphOptimizationLevel(g_so, ORT_ENABLE_ALL),
                "SetGraphOptimizationLevel")) goto fail;
        /* leave intra-op threads at ORT's default (all cores) */
    }
    if (!g_mi &&
        !ok(A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mi),
            "CreateCpuMemoryInfo"))
        goto fail;
    if (!g_alloc &&
        !ok(A->GetAllocatorWithDefaultOptions(&g_alloc), "GetAllocator"))
        goto fail;

    if (!ok(A->CreateSession(g_env, model_path, g_so, &g_sess), "CreateSession"))
        goto fail;
    if (!verify_iface()) {
        A->ReleaseSession(g_sess); g_sess = NULL;
        goto fail;
    }
    if (!init_rec()) {
        A->ReleaseSession(g_sess); g_sess = NULL;
        goto fail;
    }
    g_iw = g_ih = 0;
    fprintf(stderr, "[vcam/ort] RVM ready on ONNX Runtime %s\n",
            OrtGetApiBase()->GetVersionString());
    pthread_mutex_unlock(&g_mtx);
    return 1;
fail:
    g_dead = 1;
    pthread_mutex_unlock(&g_mtx);
    return 0;
}

int ort_rvm_infer(const float *src_chw, int iw, int ih, float *pha_out)
{
    if (!src_chw || !pha_out || iw < 16 || ih < 16) return 0;
    pthread_mutex_lock(&g_mtx);
    if (!g_sess || g_dead) { pthread_mutex_unlock(&g_mtx); return 0; }

    /* frame size changed -> the recurrent state's spatial dims no longer
     * match; restart it from zeros */
    if (iw != g_iw || ih != g_ih) {
        if (!init_rec()) goto dead;
        g_iw = iw; g_ih = ih;
    }

    OrtValue *in_src = NULL, *in_ds = NULL;
    OrtValue *outs[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    static float dsr = 1.0f;               /* we pre-scale frames ourselves */
    int ret = 0;

    {
        const int64_t sshp[4] = {1, 3, ih, iw};
        if (!ok(A->CreateTensorWithDataAsOrtValue(g_mi, (void *)src_chw,
                    (size_t)3 * ih * iw * sizeof(float), sshp, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_src),
                "CreateTensor(src)"))
            goto dead;
        const int64_t dshp[1] = {1};
        if (!ok(A->CreateTensorWithDataAsOrtValue(g_mi, &dsr, sizeof dsr,
                    dshp, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_ds),
                "CreateTensor(downsample_ratio)"))
            goto cleanup_dead;

        const OrtValue *ins[6] = {in_src, g_rec[0], g_rec[1], g_rec[2], g_rec[3], in_ds};
        if (!ok(A->Run(g_sess, NULL, IN_NAMES, ins, 6, OUT_NAMES, 6, outs), "Run"))
            goto cleanup_dead;

        /* pha = outs[1]; sanity-check its shape [1,1,ih,iw] */
        OrtTensorTypeAndShapeInfo *ti = NULL;
        if (!ok(A->GetTensorTypeAndShape(outs[1], &ti), "GetShape(pha)"))
            goto cleanup_dead;
        size_t nd = 0;
        int64_t dims[8] = {0};
        int shape_ok = ok(A->GetDimensionsCount(ti, &nd), "GetDimensionsCount");
        if (shape_ok && nd <= 8)
            shape_ok = ok(A->GetDimensions(ti, dims, nd), "GetDimensions");
        A->ReleaseTensorTypeAndShapeInfo(ti);
        if (!shape_ok || nd != 4 || dims[1] != 1 || dims[2] != ih || dims[3] != iw) {
            fprintf(stderr, "[vcam/ort] unexpected pha shape\n");
            goto cleanup_dead;
        }
        float *pp = NULL;
        if (!ok(A->GetTensorMutableData(outs[1], (void **)&pp), "GetData(pha)"))
            goto cleanup_dead;
        memcpy(pha_out, pp, (size_t)ih * iw * sizeof(float));

        /* cycle the recurrent state: steal r1o..r4o (outs[2..5]) as the
         * next frame's inputs; release the previous generation */
        release_rec();
        for (int i = 0; i < 4; i++) { g_rec[i] = outs[2 + i]; outs[2 + i] = NULL; }
        ret = 1;
        goto cleanup;
    }

cleanup_dead:
dead:
    g_dead = 1;
    fprintf(stderr, "[vcam/ort] backend disabled after error — vcam falls "
                    "back to its other matting paths\n");
cleanup:
    if (in_src) A->ReleaseValue(in_src);
    if (in_ds)  A->ReleaseValue(in_ds);
    for (int i = 0; i < 6; i++)
        if (outs[i]) A->ReleaseValue(outs[i]);
    pthread_mutex_unlock(&g_mtx);
    return ret;
}

void ort_rvm_reset(void)
{
    pthread_mutex_lock(&g_mtx);
    if (g_sess && !g_dead) init_rec();
    pthread_mutex_unlock(&g_mtx);
}

void ort_shutdown(void)
{
    pthread_mutex_lock(&g_mtx);
    release_rec();
    if (g_sess) { A->ReleaseSession(g_sess); g_sess = NULL; }
    if (g_so)   { A->ReleaseSessionOptions(g_so); g_so = NULL; }
    if (g_mi)   { A->ReleaseMemoryInfo(g_mi); g_mi = NULL; }
    if (g_env)  { A->ReleaseEnv(g_env); g_env = NULL; }
    g_alloc = NULL;   /* default allocator is not owned */
    g_iw = g_ih = 0;
    pthread_mutex_unlock(&g_mtx);
}

#else /* !VCAM_HAVE_ORT — stubs so everything links without ORT */

bool ort_available(void) { return false; }

const char *ort_version(void) { return "not built in"; }

int ort_rvm_init(const char *p) { (void)p; return 0; }

int ort_rvm_infer(const float *s, int w, int h, float *o)
{ (void)s; (void)w; (void)h; (void)o; return 0; }

void ort_rvm_reset(void) { }

void ort_shutdown(void) { }

#endif /* VCAM_HAVE_ORT */
