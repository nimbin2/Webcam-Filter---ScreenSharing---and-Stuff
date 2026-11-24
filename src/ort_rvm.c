/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Modified-by: Claude Opus 5.5 (one session builder for all models: NPU via
 *              Vitis AI EP with per-model CPU fallback, thread cap, timing)
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
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <sys/stat.h>

/* ── ML device config [Claude Opus 5.5] ───────────────────────────── */
static char   g_ml_req[8]     = "auto";   /* auto | npu | cpu */
static int    g_ml_threads    = 4;
static char   g_npu_cfg[1024] = "";
static bool   g_ml_async      = true;
static double g_mms[ORT_NMODELS];          /* avg inference ms */

static bool valid_dev(const char *d)
{
    return d && (!strcmp(d, "auto") || !strcmp(d, "npu") || !strcmp(d, "cpu"));
}

void ort_config(const char *device, int threads, const char *npu_config, bool async)
{
    if (valid_dev(device)) snprintf(g_ml_req, sizeof g_ml_req, "%s", device);
    if (threads > 0) g_ml_threads = threads;
    if (npu_config && *npu_config) snprintf(g_npu_cfg, sizeof g_npu_cfg, "%s", npu_config);
    g_ml_async = async;
}

bool ort_set_device(const char *device)
{
    if (!valid_dev(device)) return false;
    snprintf(g_ml_req, sizeof g_ml_req, "%s", device);
    return true;
}

const char *ort_ml_device(void)  { return g_ml_req; }
int         ort_ml_threads(void) { return g_ml_threads; }
bool        ort_ml_async(void)   { return g_ml_async; }
double      ort_model_ms(int m)  { return m >= 0 && m < ORT_NMODELS ? g_mms[m] : 0; }

/* Ryzen AI NPU present: /dev/accel/accel0 usable and driven by amdxdna. */
bool ort_npu_hw(void)
{
    if (access("/dev/accel/accel0", R_OK | W_OK) != 0) return false;
    char link[PATH_MAX];
    ssize_t n = readlink("/sys/class/accel/accel0/device/driver", link, sizeof link - 1);
    if (n <= 0) return false;
    link[n] = '\0';
    const char *base = strrchr(link, '/');
    return strcmp(base ? base + 1 : link, "amdxdna") == 0;
}

static __attribute__((unused)) void ms_note(int m, const struct timespec *t0)
{
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0->tv_sec) * 1e3 + (t1.tv_nsec - t0->tv_nsec) / 1e6;
    g_mms[m] = g_mms[m] == 0 ? ms : g_mms[m] * 0.9 + ms * 0.1;
}

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

/* ── one session builder for every model [Claude Opus 5.5] ──
 * g_so          : CPU, thread-capped (shared)
 * npu_options() : per model, + Vitis AI EP (Ryzen AI NPU); ops the NPU
 *                 can't run stay on the CPU inside ORT. If a model fails to
 *                 load on the NPU, only that model falls back to g_so. */
static int         g_npu_rt = -1;          /* Vitis AI EP in this ORT: -1 unknown */
static const char *g_mdev[ORT_NMODELS];    /* "npu" / "cpu" once loaded */
static const char *MNAME[ORT_NMODELS] = { "matting", "pose", "depth", "img2img" };

static int common_init(void)
{
    if (!A) A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!A) {
        fprintf(stderr, "[vcam/ort] ORT C API version %d unavailable in this "
                        "libonnxruntime\n", ORT_API_VERSION);
        return 0;
    }
    if (!g_env &&
        !ok(A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "vcam", &g_env), "CreateEnv"))
        return 0;
    if (!g_so) {
        if (!ok(A->CreateSessionOptions(&g_so), "CreateSessionOptions")) return 0;
        if (!ok(A->SetSessionGraphOptimizationLevel(g_so, ORT_ENABLE_ALL),
                "SetGraphOptimizationLevel")) return 0;
        /* ORT's default is one thread per core — a single model then
         * loads every core. Cap it (--ml-threads). */
        if (!ok(A->SetIntraOpNumThreads(g_so, g_ml_threads), "SetIntraOpNumThreads"))
            return 0;
    }
    if (!g_mi &&
        !ok(A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mi),
            "CreateCpuMemoryInfo"))
        return 0;
    if (!g_alloc &&
        !ok(A->GetAllocatorWithDefaultOptions(&g_alloc), "GetAllocator"))
        return 0;
    return 1;
}

bool ort_npu_runtime(void)
{
    if (g_npu_rt >= 0) return g_npu_rt == 1;
    if (!A) A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!A) return false;
    char **names = NULL;
    int n = 0;
    g_npu_rt = 0;
    if (ok(A->GetAvailableProviders(&names, &n), "GetAvailableProviders")) {
        for (int i = 0; i < n; i++)
            if (!strcmp(names[i], "VitisAIExecutionProvider")) g_npu_rt = 1;
        A->ReleaseAvailableProviders(names, n);
    }
    return g_npu_rt == 1;
}

static void mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
}

/* Session options with the Vitis AI EP for one model. Compiling for the
 * NPU takes a while on first load; cache_dir/cache_key keep the result
 * for the next start. The caller releases the options after CreateSession. */
static OrtSessionOptions *npu_options(const char *model_path)
{
    OrtSessionOptions *so = NULL;
    if (!ok(A->CreateSessionOptions(&so), "CreateSessionOptions(npu)")) return NULL;
    A->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    A->SetIntraOpNumThreads(so, g_ml_threads);   /* for ops left on the CPU */

    char cache[PATH_MAX];
    const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    if (xdg && *xdg) snprintf(cache, sizeof cache, "%s/vcam/npu", xdg);
    else             snprintf(cache, sizeof cache, "%s/.cache/vcam/npu", home ? home : "/tmp");
    mkdir_p(cache);

    char key[256];                               /* model file name, no .onnx */
    const char *base = strrchr(model_path, '/');
    snprintf(key, sizeof key, "%s", base ? base + 1 : model_path);
    char *dot = strrchr(key, '.');
    if (dot) *dot = '\0';

    const char *keys[16], *vals[16];
    size_t n = 0;
    keys[n] = "cache_dir";                   vals[n++] = cache;
    keys[n] = "cache_key";                   vals[n++] = key;
    keys[n] = "enable_cache_file_io_in_mem"; vals[n++] = "0";   /* INT8: write cache */
    if (g_npu_cfg[0]) { keys[n] = "config_file"; vals[n++] = g_npu_cfg; }
    /* Extra / override options: VCAM_NPU_OPTIONS="key=value,key=value" */
    char extra[1024];
    const char *ev = getenv("VCAM_NPU_OPTIONS");
    if (ev && *ev) {
        snprintf(extra, sizeof extra, "%s", ev);
        char *save = NULL;
        for (char *kv = strtok_r(extra, ",", &save); kv && n < 16; kv = strtok_r(NULL, ",", &save)) {
            char *eq = strchr(kv, '=');
            if (!eq) continue;
            *eq = '\0';
            keys[n] = kv; vals[n++] = eq + 1;
        }
    }
    if (!ok(A->SessionOptionsAppendExecutionProvider(so, "VitisAI", keys, vals, n),
            "AppendExecutionProvider(VitisAI)")) {
        A->ReleaseSessionOptions(so);
        return NULL;
    }
    return so;
}

/* Create a session for model m on the best allowed device. */
static OrtSession *open_session(const char *model_path, int m)
{
    OrtSession *s = NULL;
    bool want = strcmp(g_ml_req, "cpu") != 0;
    bool hw   = ort_npu_hw();
    bool rt   = ort_npu_runtime();
    if (want && hw && rt) {
        OrtSessionOptions *so = npu_options(model_path);
        bool up = so && ok(A->CreateSession(g_env, model_path, so, &s), "CreateSession(npu)");
        if (so) A->ReleaseSessionOptions(so);
        if (up) {
            g_mdev[m] = "npu";
            fprintf(stderr, "[vcam/ort] %s model on the NPU\n", MNAME[m]);
            return s;
        }
        s = NULL;
        fprintf(stderr, "[vcam/ort] %s model: NPU load failed — using the CPU\n", MNAME[m]);
    } else if (!strcmp(g_ml_req, "npu")) {
        fprintf(stderr, "[vcam/ort] NPU requested but not usable (hardware %s, "
                        "Vitis AI runtime %s) — %s model on the CPU\n",
                hw ? "ok" : "missing", rt ? "ok" : "missing", MNAME[m]);
    }
    if (!ok(A->CreateSession(g_env, model_path, g_so, &s), "CreateSession"))
        return NULL;
    g_mdev[m] = "cpu";
    fprintf(stderr, "[vcam/ort] %s model on the CPU (%d threads)\n", MNAME[m], g_ml_threads);
    return s;
}

const char *ort_model_device(int m)
{
    return m >= 0 && m < ORT_NMODELS && g_mdev[m] ? g_mdev[m] : "off";
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

    if (!common_init()) goto fail;

    if (!(g_sess = open_session(model_path, ORT_M_RVM)))
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

static int rvm_infer_impl(const float *src_chw, int iw, int ih, float ds,
                  float *pha_out)
{
    if (!src_chw || !pha_out || iw < 16 || ih < 16) return 0;
    if (ds <= 0.0f || ds > 1.0f) ds = 1.0f;
    pthread_mutex_lock(&g_mtx);
    if (!g_sess || g_dead) { pthread_mutex_unlock(&g_mtx); return 0; }

    /* frame size or downsample ratio changed -> the recurrent state's
     * spatial dims no longer match; restart it from zeros */
    static float g_ds = 0.0f;
    if (iw != g_iw || ih != g_ih || ds != g_ds) {
        if (!init_rec()) goto dead;
        g_iw = iw; g_ih = ih; g_ds = ds;
    }

    OrtValue *in_src = NULL, *in_ds = NULL;
    OrtValue *outs[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    float dsr = ds;
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

/* ── BlazePose landmarks (MediaPipe pose, OpenCV-zoo export):
 * input [1,256,256,3] RGB [0,1] NHWC; outputs selected by element
 * count: 195 floats = 39 landmarks x (x,y,z,vis,pres), 1 float =
 * person-presence confidence. ── */
static OrtSession *g_pose;
static char g_pose_in[128];
static char g_pose_outs[8][128];
static size_t g_pose_nout = 0;
static int g_pose_dead = 0;

int ort_pose_init(const char *model_path)
{
    if (!model_path || !*model_path) return 0;
    pthread_mutex_lock(&g_mtx);
    if (g_pose) { pthread_mutex_unlock(&g_mtx); return 1; }
    g_pose_dead = 0;
    if (!common_init()) goto fail;
    if (!(g_pose = open_session(model_path, ORT_M_POSE))) goto fail;
    {
        size_t nin = 0;
        if (!ok(A->SessionGetInputCount(g_pose, &nin), "pose InCount") ||
            nin != 1 ||
            !ok(A->SessionGetOutputCount(g_pose, &g_pose_nout), "pose OutCount") ||
            g_pose_nout < 2 || g_pose_nout > 8) {
            fprintf(stderr, "[vcam/ort] unexpected pose model interface\n");
            A->ReleaseSession(g_pose); g_pose = NULL; goto fail;
        }
        char *nm = NULL;
        if (!ok(A->SessionGetInputName(g_pose, 0, g_alloc, &nm), "pose InName"))
            { A->ReleaseSession(g_pose); g_pose = NULL; goto fail; }
        snprintf(g_pose_in, sizeof g_pose_in, "%s", nm);
        if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
        for (size_t i = 0; i < g_pose_nout; i++) {
            if (!ok(A->SessionGetOutputName(g_pose, i, g_alloc, &nm), "pose OutName"))
                { A->ReleaseSession(g_pose); g_pose = NULL; goto fail; }
            snprintf(g_pose_outs[i], sizeof g_pose_outs[i], "%s", nm);
            if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
        }
    }
    fprintf(stderr, "[vcam/ort] pose model ready (%zu outputs)\n", g_pose_nout);
    pthread_mutex_unlock(&g_mtx);
    return 1;
fail:
    g_pose_dead = 1;
    pthread_mutex_unlock(&g_mtx);
    return 0;
}

static int pose_infer_impl(const float *rgb_nhwc_256, float *landmarks195, float *conf1)
{
    if (!rgb_nhwc_256 || !landmarks195 || !conf1) return 0;
    pthread_mutex_lock(&g_mtx);
    if (!g_pose || g_pose_dead) { pthread_mutex_unlock(&g_mtx); return 0; }
    OrtValue *iv = NULL, *outs[8] = {0};
    const char *names[8];
    int ret = 0;
    {
        for (size_t i = 0; i < g_pose_nout; i++) names[i] = g_pose_outs[i];
        const int64_t shp[4] = {1, 256, 256, 3};
        if (!ok(A->CreateTensorWithDataAsOrtValue(g_mi, (void *)rgb_nhwc_256,
                    (size_t)3*256*256*sizeof(float), shp, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &iv),
                "pose CreateTensor")) goto done_dead;
        const OrtValue *ivs[1] = {iv};
        const char *ins[1] = {g_pose_in};
        if (!ok(A->Run(g_pose, NULL, ins, ivs, 1, names, g_pose_nout, outs),
                "pose Run")) goto done_dead;
        int got_lm = 0, got_cf = 0;
        for (size_t i = 0; i < g_pose_nout; i++) {
            OrtTensorTypeAndShapeInfo *sh = NULL;
            size_t cnt = 0;
            if (!ok(A->GetTensorTypeAndShape(outs[i], &sh), "pose Shape"))
                goto done_dead;
            int sok = ok(A->GetTensorShapeElementCount(sh, &cnt), "pose Count");
            A->ReleaseTensorTypeAndShapeInfo(sh);
            if (!sok) goto done_dead;
            float *p = NULL;
            if (cnt == 195 && !got_lm) {
                if (!ok(A->GetTensorMutableData(outs[i], (void **)&p), "pose LM"))
                    goto done_dead;
                memcpy(landmarks195, p, 195*sizeof(float));
                got_lm = 1;
            } else if (cnt == 1 && !got_cf) {
                if (!ok(A->GetTensorMutableData(outs[i], (void **)&p), "pose CF"))
                    goto done_dead;
                *conf1 = p[0];
                got_cf = 1;
            }
        }
        if (!got_lm || !got_cf) {
            fprintf(stderr, "[vcam/ort] pose outputs missing landmarks/conf\n");
            goto done_dead;
        }
        ret = 1;
        goto done;
    }
done_dead:
    g_pose_dead = 1;
    fprintf(stderr, "[vcam/ort] pose backend disabled after error\n");
done:
    if (iv) A->ReleaseValue(iv);
    for (size_t i = 0; i < g_pose_nout; i++)
        if (outs[i]) A->ReleaseValue(outs[i]);
    pthread_mutex_unlock(&g_mtx);
    return ret;
}

void ort_pose_shutdown(void)
{
    pthread_mutex_lock(&g_mtx);
    if (g_pose) { A->ReleaseSession(g_pose); g_pose = NULL; }
    g_pose_dead = 0;
    pthread_mutex_unlock(&g_mtx);
}

/* ── monocular depth (MiDaS-small interface): fixed [1,3,256,256]
 * RGB [0,1] in -> [1,256,256] (or [1,1,256,256]) inverse depth out. ── */
static OrtSession *g_dep;
static char g_dep_in[128], g_dep_out[128];
static int g_dep_dead = 0;

int ort_depth_init(const char *model_path)
{
    if (!model_path || !*model_path) return 0;
    pthread_mutex_lock(&g_mtx);
    if (g_dep) { pthread_mutex_unlock(&g_mtx); return 1; }
    g_dep_dead = 0;
    if (!common_init()) goto fail;
    if (!(g_dep = open_session(model_path, ORT_M_DEPTH))) goto fail;
    {
        size_t nin = 0, nout = 0;
        if (!ok(A->SessionGetInputCount(g_dep, &nin), "depth InCount") ||
            !ok(A->SessionGetOutputCount(g_dep, &nout), "depth OutCount") ||
            nin != 1 || nout != 1) {
            fprintf(stderr, "[vcam/ort] depth model must be 1-in/1-out\n");
            A->ReleaseSession(g_dep); g_dep = NULL; goto fail;
        }
        char *nm = NULL;
        if (!ok(A->SessionGetInputName(g_dep, 0, g_alloc, &nm), "depth InName"))
            { A->ReleaseSession(g_dep); g_dep = NULL; goto fail; }
        snprintf(g_dep_in, sizeof g_dep_in, "%s", nm);
        if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
        if (!ok(A->SessionGetOutputName(g_dep, 0, g_alloc, &nm), "depth OutName"))
            { A->ReleaseSession(g_dep); g_dep = NULL; goto fail; }
        snprintf(g_dep_out, sizeof g_dep_out, "%s", nm);
        if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
    }
    fprintf(stderr, "[vcam/ort] depth model ready ('%s'->'%s')\n",
            g_dep_in, g_dep_out);
    pthread_mutex_unlock(&g_mtx);
    return 1;
fail:
    g_dep_dead = 1;
    pthread_mutex_unlock(&g_mtx);
    return 0;
}

static int depth_infer_impl(const float *rgb_chw_256, float *depth_out_65536)
{
    if (!rgb_chw_256 || !depth_out_65536) return 0;
    pthread_mutex_lock(&g_mtx);
    if (!g_dep || g_dep_dead) { pthread_mutex_unlock(&g_mtx); return 0; }
    OrtValue *iv = NULL, *ov = NULL;
    int ret = 0;
    {
        const int64_t shp[4] = {1, 3, 256, 256};
        if (!ok(A->CreateTensorWithDataAsOrtValue(g_mi, (void *)rgb_chw_256,
                    (size_t)3*256*256*sizeof(float), shp, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &iv),
                "depth CreateTensor")) goto done_dead;
        const char *ins[1] = {g_dep_in}, *outs2[1] = {g_dep_out};
        const OrtValue *ivs[1] = {iv};
        if (!ok(A->Run(g_dep, NULL, ins, ivs, 1, outs2, 1, &ov), "depth Run"))
            goto done_dead;
        OrtTensorTypeAndShapeInfo *sh = NULL;
        size_t cnt = 0;
        if (!ok(A->GetTensorTypeAndShape(ov, &sh), "depth OutShape"))
            goto done_dead;
        int sok = ok(A->GetTensorShapeElementCount(sh, &cnt), "depth Count");
        A->ReleaseTensorTypeAndShapeInfo(sh);
        if (!sok || cnt != (size_t)256*256) {
            fprintf(stderr, "[vcam/ort] depth output must be 256x256\n");
            goto done_dead;
        }
        float *p = NULL;
        if (!ok(A->GetTensorMutableData(ov, (void **)&p), "depth OutData"))
            goto done_dead;
        memcpy(depth_out_65536, p, (size_t)256*256*sizeof(float));
        ret = 1;
        goto done;
    }
done_dead:
    g_dep_dead = 1;
    fprintf(stderr, "[vcam/ort] depth backend disabled after error\n");
done:
    if (iv) A->ReleaseValue(iv);
    if (ov) A->ReleaseValue(ov);
    pthread_mutex_unlock(&g_mtx);
    return ret;
}

void ort_depth_shutdown(void)
{
    pthread_mutex_lock(&g_mtx);
    if (g_dep) { A->ReleaseSession(g_dep); g_dep = NULL; }
    if (g_pose) { A->ReleaseSession(g_pose); g_pose = NULL; }
    g_dep_dead = 0;
    pthread_mutex_unlock(&g_mtx);
}

/* ── generic img2img (single input, single output, dynamic H/W) ── */
static OrtSession *g_i2i;
static char g_i2i_in[128], g_i2i_out[128];
static int g_i2i_nhwc = 0, g_i2i_dead = 0;

int ort_i2i_layout(void) { return g_i2i_nhwc; }

int ort_i2i_init(const char *model_path)
{
    if (!model_path || !*model_path) return 0;
    pthread_mutex_lock(&g_mtx);
    if (g_i2i) { pthread_mutex_unlock(&g_mtx); return 1; }
    g_i2i_dead = 0;
    if (!common_init()) goto fail;
    if (!(g_i2i = open_session(model_path, ORT_M_I2I))) goto fail;
    {
        size_t nin = 0, nout = 0;
        if (!ok(A->SessionGetInputCount(g_i2i, &nin), "i2i InputCount") ||
            !ok(A->SessionGetOutputCount(g_i2i, &nout), "i2i OutputCount") ||
            nin != 1 || nout != 1) {
            fprintf(stderr, "[vcam/ort] img2img model must have exactly "
                            "1 input and 1 output (has %zu/%zu)\n", nin, nout);
            A->ReleaseSession(g_i2i); g_i2i = NULL; goto fail;
        }
        char *nm = NULL;
        if (!ok(A->SessionGetInputName(g_i2i, 0, g_alloc, &nm), "i2i InName"))
            { A->ReleaseSession(g_i2i); g_i2i = NULL; goto fail; }
        snprintf(g_i2i_in, sizeof g_i2i_in, "%s", nm);
        if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
        if (!ok(A->SessionGetOutputName(g_i2i, 0, g_alloc, &nm), "i2i OutName"))
            { A->ReleaseSession(g_i2i); g_i2i = NULL; goto fail; }
        snprintf(g_i2i_out, sizeof g_i2i_out, "%s", nm);
        if (!ok(A->AllocatorFree(g_alloc, nm), "free")) { }
        /* layout: channels dim == 3 at index 1 -> NCHW, at 3 -> NHWC */
        OrtTypeInfo *ti = NULL;
        const OrtTensorTypeAndShapeInfo *sh = NULL;
        int64_t dims[8] = {0}; size_t nd = 0;
        if (ok(A->SessionGetInputTypeInfo(g_i2i, 0, &ti), "i2i TypeInfo") &&
            ok(A->CastTypeInfoToTensorInfo(ti, &sh), "i2i Cast") &&
            ok(A->GetDimensionsCount(sh, &nd), "i2i DimCount") && nd == 4 &&
            ok(A->GetDimensions(sh, dims, nd), "i2i Dims")) {
            g_i2i_nhwc = (dims[3] == 3) ? 1 : 0;
        } else {
            g_i2i_nhwc = 0;   /* dynamic everywhere: NCHW is the safer bet */
        }
        if (ti) A->ReleaseTypeInfo(ti);
    }
    fprintf(stderr, "[vcam/ort] img2img model ready (io '%s'->'%s', %s)\n",
            g_i2i_in, g_i2i_out, g_i2i_nhwc ? "NHWC" : "NCHW");
    pthread_mutex_unlock(&g_mtx);
    return 1;
fail:
    g_i2i_dead = 1;
    pthread_mutex_unlock(&g_mtx);
    return 0;
}

static int i2i_infer_impl(const float *in, int w, int h, float *out)
{
    if (!in || !out || w < 16 || h < 16) return 0;
    pthread_mutex_lock(&g_mtx);
    if (!g_i2i || g_i2i_dead) { pthread_mutex_unlock(&g_mtx); return 0; }
    OrtValue *iv = NULL, *ov = NULL;
    int ret = 0;
    {
        int64_t shp[4];
        if (g_i2i_nhwc) { shp[0]=1; shp[1]=h; shp[2]=w; shp[3]=3; }
        else            { shp[0]=1; shp[1]=3; shp[2]=h; shp[3]=w; }
        if (!ok(A->CreateTensorWithDataAsOrtValue(g_mi, (void *)in,
                    (size_t)3*w*h*sizeof(float), shp, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &iv),
                "i2i CreateTensor")) goto done_dead;
        const char *ins[1] = {g_i2i_in}, *outs2[1] = {g_i2i_out};
        const OrtValue *ivs[1] = {iv};
        if (!ok(A->Run(g_i2i, NULL, ins, ivs, 1, outs2, 1, &ov), "i2i Run"))
            goto done_dead;
        OrtTensorTypeAndShapeInfo *sh = NULL;
        size_t cnt = 0;
        if (!ok(A->GetTensorTypeAndShape(ov, &sh), "i2i OutShape"))
            goto done_dead;
        int sok = ok(A->GetTensorShapeElementCount(sh, &cnt), "i2i Count");
        A->ReleaseTensorTypeAndShapeInfo(sh);
        if (!sok || cnt != (size_t)3*w*h) {
            fprintf(stderr, "[vcam/ort] img2img output size mismatch\n");
            goto done_dead;
        }
        float *p = NULL;
        if (!ok(A->GetTensorMutableData(ov, (void **)&p), "i2i OutData"))
            goto done_dead;
        memcpy(out, p, (size_t)3*w*h*sizeof(float));
        ret = 1;
        goto done;
    }
done_dead:
    g_i2i_dead = 1;
    fprintf(stderr, "[vcam/ort] img2img backend disabled after error\n");
done:
    if (iv) A->ReleaseValue(iv);
    if (ov) A->ReleaseValue(ov);
    pthread_mutex_unlock(&g_mtx);
    return ret;
}

void ort_i2i_shutdown(void)
{
    pthread_mutex_lock(&g_mtx);
    if (g_i2i) { A->ReleaseSession(g_i2i); g_i2i = NULL; }
    if (g_dep) { A->ReleaseSession(g_dep); g_dep = NULL; }
    g_i2i_dead = 0;
    pthread_mutex_unlock(&g_mtx);
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
    if (g_i2i) { A->ReleaseSession(g_i2i); g_i2i = NULL; }
    if (g_sess) { A->ReleaseSession(g_sess); g_sess = NULL; }
    if (g_so)   { A->ReleaseSessionOptions(g_so); g_so = NULL; }
    for (int i = 0; i < ORT_NMODELS; i++) g_mdev[i] = NULL;
    if (g_mi)   { A->ReleaseMemoryInfo(g_mi); g_mi = NULL; }
    if (g_env)  { A->ReleaseEnv(g_env); g_env = NULL; }
    g_alloc = NULL;   /* default allocator is not owned */
    g_iw = g_ih = 0;
    pthread_mutex_unlock(&g_mtx);
}

/* timed public entry points (time shows up in 'vcam-ctl stats') */
int ort_rvm_infer(const float *src_chw, int iw, int ih, float ds, float *pha_out)
{
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = rvm_infer_impl(src_chw, iw, ih, ds, pha_out);
    if (r) ms_note(ORT_M_RVM, &t0);
    return r;
}
int ort_pose_infer(const float *rgb_nhwc_256, float *landmarks195, float *conf1)
{
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = pose_infer_impl(rgb_nhwc_256, landmarks195, conf1);
    if (r) ms_note(ORT_M_POSE, &t0);
    return r;
}
int ort_depth_infer(const float *rgb_chw_256, float *depth_out_65536)
{
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = depth_infer_impl(rgb_chw_256, depth_out_65536);
    if (r) ms_note(ORT_M_DEPTH, &t0);
    return r;
}
int ort_i2i_infer(const float *in, int w, int h, float *out)
{
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = i2i_infer_impl(in, w, h, out);
    if (r) ms_note(ORT_M_I2I, &t0);
    return r;
}

#else /* !VCAM_HAVE_ORT — stubs so everything links without ORT */

bool ort_npu_runtime(void) { return false; }
const char *ort_model_device(int m) { (void)m; return "off"; }


bool ort_available(void) { return false; }

const char *ort_version(void) { return "not built in"; }

int ort_rvm_init(const char *p) { (void)p; return 0; }

int ort_rvm_infer(const float *s, int w, int h, float d, float *o)
{ (void)s; (void)w; (void)h; (void)d; (void)o; return 0; }

int ort_i2i_init(const char *p) { (void)p; return 0; }

int ort_i2i_layout(void) { return 0; }

int ort_i2i_infer(const float *i, int w, int h, float *o)
{ (void)i; (void)w; (void)h; (void)o; return 0; }

void ort_i2i_shutdown(void) { }

int ort_depth_init(const char *p) { (void)p; return 0; }

int ort_depth_infer(const float *i, float *o) { (void)i; (void)o; return 0; }

void ort_depth_shutdown(void) { }

int ort_pose_init(const char *p) { (void)p; return 0; }

int ort_pose_infer(const float *i, float *l, float *c)
{ (void)i; (void)l; (void)c; return 0; }

void ort_pose_shutdown(void) { }

void ort_rvm_reset(void) { }

void ort_shutdown(void) { }

#endif /* VCAM_HAVE_ORT */
