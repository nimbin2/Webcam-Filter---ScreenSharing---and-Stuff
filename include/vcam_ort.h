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
 *   pha_out : caller buffer for the alpha matte, ih*iw floats, 0..1
 * The recurrent state is carried between calls internally (that's what
 * makes RVM temporally stable) and re-initialized automatically if the
 * frame size changes. Returns 1 on success, 0 on failure. */
int ort_rvm_infer(const float *src_chw, int iw, int ih, float *pha_out);

/* Zero the recurrent state (e.g. on a scene cut). Cheap; optional. */
void ort_rvm_reset(void);

/* Release the session and all state. Safe in every build. */
void ort_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* VCAM_ORT_H */
