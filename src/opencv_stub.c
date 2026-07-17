/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * opencv_stub.c — no-op implementations of the vcam_cv.h C ABI, built
 * when vcam is compiled WITHOUT OpenCV (the default). Every effect
 * leaves the frame untouched; cv_available() returns false so callers
 * (and the IPC layer) can tell the user OpenCV isn't compiled in.
 *
 * This lets filter.c call the cv_fx_* functions unconditionally — the
 * linker resolves them either to the real OpenCV shim or to these.
 */

#include "vcam_cv.h"

bool cv_available(void) { return false; }

const char *cv_version(void) { return "none"; }

const char *cv_engine_info(void) { return "opencv not built in (rebuild with 'make OPENCV=1')"; }

const char *cv_model_dir(void) { return "(opencv not built in)"; }

/* All effects are no-ops: the BGR24 frame passes through unchanged. */
void cv_fx_cartoon(unsigned char *d, int w, int h, int s, float x)
{ (void)d; (void)w; (void)h; (void)s; (void)x; }

void cv_fx_pencil(unsigned char *d, int w, int h, int s, int c)
{ (void)d; (void)w; (void)h; (void)s; (void)c; }

void cv_fx_stylize(unsigned char *d, int w, int h, int s, float x)
{ (void)d; (void)w; (void)h; (void)s; (void)x; }

void cv_fx_flow(unsigned char *d, int w, int h, int s, float m)
{ (void)d; (void)w; (void)h; (void)s; (void)m; }

void cv_fx_bgsub(unsigned char *d, int w, int h, int s, float x)
{ (void)d; (void)w; (void)h; (void)s; (void)x; }

void cv_fx_face(unsigned char *d, int w, int h, int s, int m, float a, float sp)
{ (void)d; (void)w; (void)h; (void)s; (void)m; (void)a; (void)sp; }

void cv_fx_bgblur(unsigned char *d, int w, int h, int s, int m, float e, float sm, float sz, float fx)
{ (void)d; (void)w; (void)h; (void)s; (void)m; (void)e; (void)sm; (void)sz; (void)fx; }

void cv_fx_style(unsigned char *d, int w, int h, int s, int st, float b, float sz)
{ (void)d; (void)w; (void)h; (void)s; (void)st; (void)b; (void)sz; }

void cv_fx_mood(unsigned char *d, int w, int h, int s, int m, float g, float e)
{ (void)d; (void)w; (void)h; (void)s; (void)m; (void)g; (void)e; }

void cv_fx_paintflow(unsigned char *d, int w, int h, int s, float st, float dc)
{ (void)d; (void)w; (void)h; (void)s; (void)st; (void)dc; }

void cv_fx_speedlines(unsigned char *d, int w, int h, int s, float a, float b)
{ (void)d; (void)w; (void)h; (void)s; (void)a; (void)b; }

int  cv_room_set_region(int x, int y, int w, int h)
{ (void)x; (void)y; (void)w; (void)h; return 0; }

void cv_room_off(void) { }

void cv_fx_pose(unsigned char *d, int w, int h, int s, int m, float fx)
{ (void)d; (void)w; (void)h; (void)s; (void)m; (void)fx; }

int  cv_set_pose_model(const char *p) { (void)p; return 0; }

void cv_fx_depth(unsigned char *d, int w, int h, int s, int m, float fx)
{ (void)d; (void)w; (void)h; (void)s; (void)m; (void)fx; }

int  cv_set_depth_model(const char *p) { (void)p; return 0; }

void cv_fx_genai(unsigned char *d, int w, int h, int s, float b, float sz, float r)
{ (void)d; (void)w; (void)h; (void)s; (void)b; (void)sz; (void)r; }

void cv_fx_enhance(unsigned char *d, int w, int h, int s, float a, float b)
{ (void)d; (void)w; (void)h; (void)s; (void)a; (void)b; }

int  cv_set_gen_model(const char *p) { (void)p; return 0; }

void cv_clear_gen_model(void) { }

int  cv_set_face_image(const char *p) { (void)p; return 0; }

void cv_clear_face_image(void) { }

int  cv_set_bg_image(const char *p) { (void)p; return 0; }

void cv_clear_bg_image(void) { }

void cv_shutdown(void) { }
