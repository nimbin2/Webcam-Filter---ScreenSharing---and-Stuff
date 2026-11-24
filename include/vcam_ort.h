/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * vcam_ort.h — C ABI for the optional ONNX Runtime backend.
 *
 * Why this exists: OpenCV 5's new DNN graph engine LOADS the RVM video
 * matting model but miscomputes it (empty mattes — verified on a real
 * OpenCV 5.1 install). ONNX Runtime is the reference runtime RVM was
 * built and tested on; through it the model computes correctly.
 *
 * The backend is opt-in at build time:
 *
 *     vcam-setup --ort                # one-time: fetch onnxruntime
 *     make OPENCV=1 ORT=1            # build with the backend
 *
 * When built without ORT=1 (the default), src/ort_rvm.c compiles its
 * stub half: ort_available() returns false and everything else no-ops,
 * so callers can invoke these unconditionally. The OpenCV shim tries
 * this backend first for RVM and falls back to its own paths when it
 * isn't present or fails.
 *
 * Threading: calls are serialized by the caller (the segmentation lock
 * in the OpenCV shim); an internal mutex additionally guards state so
 * misuse degrades safely rather than corrupting memory.
 */

#ifndef VCAM_ORT_H
#define VCAM_ORT_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when vcam was built with ORT=1 and libonnxruntime loaded. */
bool ort_available(void);

/* ONNX Runtime version string ("1.27.0"), or "not built in". Never NULL. */
const char *ort_version(void);

/* Load the RVM model from the given .onnx path and prepare a session.
 * Verifies the model exposes the RVM interface (src, r1i..r4i,
 * downsample_ratio -> fgr, pha, r1o..r4o); anything else is rejected.
 * Returns 1 on success, 0 on failure (a one-time reason is printed).
 * Safe to call again after failure or with a different path. */
int ort_rvm_init(const char *model_path);

/* Run one frame of RVM.
 *   src_chw : float32 RGB, values 0..1, planar CHW (3 * ih * iw floats)
 *   iw, ih  : the frame's width/height (any size; /16-aligned is best)
 *   ds      : RVM's downsample_ratio (0..1]. The encoder runs at
 *             iw*ds x ih*ds while refinement stays at full input res —
 *             THE speed lever. ~256/iw is a good value; <=0 -> 1.0.
 *   pha_out : caller buffer for the alpha matte, ih*iw floats, 0..1
 * The recurrent state is carried between calls internally (that's what
 * makes RVM temporally stable) and re-initialized automatically if the
 * frame size or ds changes. Returns 1 on success, 0 on failure. */
int ort_rvm_infer(const float *src_chw, int iw, int ih, float ds,
                  float *pha_out);

/* ── generic img2img session (AnimeGAN / CartoonGAN / any 1-in 1-out
 * image transformer). Values are [-1,1] floats, the GAN convention.
 * Layout is auto-detected from the model at init. */
int  ort_i2i_init(const char *model_path);   /* 1 ok, 0 fail */
int  ort_i2i_layout(void);                   /* 0 = NCHW, 1 = NHWC */
/* in/out are w*h*3 floats in the model's own layout; same buffer
 * layout for input and output. 1 ok, 0 fail. */
int  ort_i2i_infer(const float *in, int w, int h, float *out);
void ort_i2i_shutdown(void);

/* ── BlazePose landmarks (MediaPipe pose): input RGB [0,1] NHWC at
 * 256x256; landmarks195 = 39 x (x,y,z,vis,pres) in 256-crop space;
 * conf1 = person-presence confidence. */
int  ort_pose_init(const char *model_path);
int  ort_pose_infer(const float *rgb_nhwc_256, float *landmarks195,
                    float *conf1);
void ort_pose_shutdown(void);

/* ── monocular depth (MiDaS-small interface): input RGB [0,1] planar
 * CHW at exactly 256x256; output 256*256 floats of relative inverse
 * depth (bigger = closer). */
int  ort_depth_init(const char *model_path);
int  ort_depth_infer(const float *rgb_chw_256, float *depth_out_65536);
void ort_depth_shutdown(void);

/* Zero the recurrent state (e.g. on a scene cut). Cheap; optional. */
void ort_rvm_reset(void);

/* Release the session and all state. Safe in every build. */
void ort_shutdown(void);

/* ── ML device selection [Claude Opus 5.5] ──
 * Every model is loaded through one builder: on the Ryzen AI NPU (Vitis AI
 * execution provider) when allowed and available, else on the CPU with a
 * thread cap. A model that fails on the NPU falls back alone. */
enum { ORT_M_RVM, ORT_M_POSE, ORT_M_DEPTH, ORT_M_I2I, ORT_NMODELS };
void        ort_config(const char *device, int threads, const char *npu_config, bool async);
bool        ort_set_device(const char *device);  /* auto|npu|cpu, for models loaded later */
const char *ort_ml_device(void);     /* requested device */
int         ort_ml_threads(void);
bool        ort_ml_async(void);      /* matting runs beside the video loop */
bool        ort_npu_hw(void);        /* /dev/accel/accel0 driven by amdxdna */
bool        ort_npu_runtime(void);   /* this ONNX Runtime has the Vitis AI EP */
const char *ort_model_device(int m); /* "npu" | "cpu" | "off" */
double      ort_model_ms(int m);     /* average inference time, 0 = never run */

#ifdef __cplusplus
}
#endif

#endif /* VCAM_ORT_H */
