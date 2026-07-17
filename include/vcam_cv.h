/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * vcam_cv.h — C ABI for the optional OpenCV-backed effects.
 *
 * vcam's core is C11; OpenCV 5 is a C++17 library. To keep the core
 * pure C and OpenCV strictly optional, all OpenCV code lives behind
 * this flat C interface, implemented either by:
 *
 *   src/opencv_fx.cpp   — the real thing (built when OPENCV=1)
 *   src/opencv_stub.c   — no-op fallbacks (built otherwise)
 *
 * Every function operates on a BGR24 buffer in place (data/width/
 * height/stride matching struct Frame). All of them are safe to call
 * even when OpenCV is unavailable — the stub versions simply leave the
 * frame untouched, and cv_available() returns 0.
 *
 * Functions never throw across this boundary: the C++ side wraps every
 * body in a try/catch and degrades to a no-op on error, so a bad model
 * file or an unexpected frame can never crash the pipeline.
 */

#ifndef VCAM_CV_H
#define VCAM_CV_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when vcam was built with OpenCV support (OPENCV=1). The stub
 * build returns false. Cheap; may be called every frame. */
bool cv_available(void);

/* Human-readable OpenCV version string ("5.0.0"), or "none" in the
 * stub build. Never NULL. */
const char *cv_version(void);

/* One-line description of the OpenCV generation and DNN engine in this
 * build, e.g. "OpenCV 5.0.0 — new DNN graph engine (ENGINE_AUTO), FP16
 * blend" or "OpenCV 4.6.0 — classic DNN engine, FP32 blend", or
 * "opencv not built in" for the stub. Never NULL. */
const char *cv_engine_info(void);

/* Directory the DNN filters look in for their .onnx model files.
 * Defaults to, in order of preference:
 *   $VCAM_MODEL_DIR, then ~/.local/share/vcam/models,
 *   then /usr/local/share/vcam/models, then /usr/share/vcam/models.
 * Returns the first that exists, or the compiled default. Never NULL. */
const char *cv_model_dir(void);

/* ── Classic-CV effects (no model files needed) ──
 * These use only imgproc / photo / video / objdetect and run realtime
 * on a laptop CPU. Each is self-contained and manages its own state. */

/* Edge-preserving cartoon: bilateral smoothing + adaptive-threshold
 * ink outlines. strength 0..1 (0 → default). */
void cv_fx_cartoon(unsigned char *data, int w, int h, int stride,
                   float strength);

/* Non-photorealistic pencil sketch (photo::pencilSketch).
 * color != 0 keeps color; 0 gives graphite gray. */
void cv_fx_pencil(unsigned char *data, int w, int h, int stride,
                  int color);

/* photo::stylization — a vivid watercolor/poster look. strength 0..1. */
void cv_fx_stylize(unsigned char *data, int w, int h, int stride,
                   float strength);

/* Dense optical-flow visualization (Farnebäck): motion direction is
 * mapped to hue, magnitude to brightness, blended over the frame.
 * mix 0..1 = how much flow color vs. original (0 → 0.6). */
void cv_fx_flow(unsigned char *data, int w, int h, int stride,
                float mix);

/* MOG2 background subtraction: foreground (moving) pixels are kept in
 * color, the learned static background is darkened/desaturated.
 * Person walks in and "lights up" against a dim room.
 * sensitivity 0..1 (0 → default). */
void cv_fx_bgsub(unsigned char *data, int w, int h, int stride,
                 float sensitivity);

/* ── DNN effects (need an .onnx model in cv_model_dir()) ──
 * If the model file is missing these degrade gracefully: the frame is
 * left unchanged and a one-time warning is printed. Inference runs on
 * a downscaled copy and is internally rate-limited, so they stay light
 * on a laptop. */

/* YuNet face detection. Draws a stylized effect on each detected face.
 * mode selects the look:
 *   0 = privacy blur (pixelate faces)
 *   1 = glossy highlight ring + landmark dots
 *   2 = "googly eyes" drawn at the eye landmarks
 * Needs face_detection_yunet_2023mar.onnx. */
void cv_fx_face(unsigned char *data, int w, int h, int stride,
                int mode, float amount, float speed);

/* PP-HumanSeg person segmentation → background effect.
 * mode selects what happens to the background:
 *   0 = blur the background (portrait mode)
 *   1 = desaturate/darken the background (spotlight the person)
 *   2 = replace background with a moving color gradient
 * Needs human_segmentation_pphumanseg_2023mar.onnx. */
void cv_fx_bgblur(unsigned char *data, int w, int h, int stride,
                  int mode, float edge, float smooth, float size, float fx);

/* Live neural style transfer (fast-neural-style, 5 styles).
 *   style: 0=mosaic 1=candy 2=rain-princess 3=udnie 4=pointilism
 *   blend: 0..1 stylized-vs-original mix (0 -> 0.85)
 *   size:  inference resolution 96..384, the quality<->speed knob
 *          (0 -> 192). Needs the style .onnx files
 *          ('vcam-setup --models-style'). */
void cv_fx_style(unsigned char *data, int w, int h, int stride,
                 int style, float blend, float size);

/* Emotion-reactive effects: a facial-expression model (7 emotions)
 * reads your mood every few frames and the effect responds.
 *   mode: 0 = aura (colored vignette breathing with the emotion)
 *         1 = HUD (labeled box + confidence bar)
 *         2 = party (happy->confetti, sad->rain, surprised->flash,
 *                    angry->screen shake)
 *   gain: sensitivity multiplier 0..2 (0 -> 1).
 * Needs the YuNet face model AND the FER model ('vcam-setup --models'). */
void cv_fx_mood(unsigned char *data, int w, int h, int stride,
                int mode, float gain, float extra);

/* Flow-advected paint canvas: dense optical flow drags a persistent
 * canvas — anything that moves smears reality like wet paint.
 *   strength: flow multiplier 0..3 (0 -> 1)
 *   decay:    canvas persistence 0.80..0.99 (0 -> 0.93) */
void cv_fx_paintflow(unsigned char *data, int w, int h, int stride,
                     float strength, float decay);

/* Anime speed lines: when the mean optical flow exceeds a threshold,
 * white motion lines streak in from the border opposite the movement.
 *   sens 0..3 (0->1), density 0..1 (0->0.5). */
void cv_fx_speedlines(unsigned char *data, int w, int h, int stride,
                      float sens, float density);

/* virtual-room (personfx 132): select the screen region containing the
 * other person; vcam grabs it live, cuts the person out, and seats you
 * both in a rendered room. */
int  cv_room_set_region(int x, int y, int w, int h);
void cv_room_off(void);

/* posefx: BlazePose body landmarks (MediaPipe pose via ORT).
 * Model: <models>/pose_estimation_mediapipe_2023mar.onnx
 * (vcam-setup --models-pose) or 'filter posemodel <path>'.
 * modes: 0 skeleton-glow, 1 neon-bones (dark room, glowing skeleton),
 *        2 energy-hands (orbs in your palms; merge them!), 3 wrist-trails.
 * fx = size/strength 0.2..3. */
void cv_fx_pose(unsigned char *data, int w, int h, int stride,
                int mode, float fx);
int  cv_set_pose_model(const char *path);

/* depthfx: effects driven by a monocular depth model (MiDaS v2.1
 * small interface) on the ORT backend. Model at <models>/model-small.onnx
 * (vcam-setup --models-depth) or set via 'filter depthmodel <path>'.
 * modes: 0 depth-blur (true depth-of-field), 1 depth-fog,
 *        2 depth-view (colormapped), 3 parallax-wiggle.
 * fx = effect strength 0.2..3. Adaptive pacing keeps it live. */
void cv_fx_depth(unsigned char *data, int w, int h, int stride,
                 int mode, float fx);
int  cv_set_depth_model(const char *path);

/* genai: run a generic img2img ONNX model (AnimeGAN, CartoonGAN, any
 * 1-in/1-out [-1,1] transformer) on the frame via the ORT backend.
 * blend 0.2..1 (how much of the transformed image shows, def 1),
 * size 128..512 (inference resolution, def 288). Model file is set
 * with cv_set_gen_model() ('filter genmodel <path>'), defaulting to
 * <models>/img2img.onnx if present. Adaptive pacing keeps it live. */
void cv_fx_genai(unsigned char *data, int w, int h, int stride,
                 float blend, float size, float range01);

/* enhance: retinex low-light rescue (no model needed). strength
 * 0.2..2 def 1; denoise 0..1 = shadow noise smoothing. */
void cv_fx_enhance(unsigned char *data, int w, int h, int stride,
                   float strength, float denoise);
int  cv_set_gen_model(const char *path);
void cv_clear_gen_model(void);

/* Set the face-overlay image for facefx mode 24 (any PNG; alpha
 * channel respected — an animal face with transparency tracks your
 * head: scaled by face width, rotated by eye angle). Returns 1 on
 * success. cv_clear_face_image() removes it. */
int  cv_set_face_image(const char *path);
void cv_clear_face_image(void);

/* Set/replace the user background image used by cv_fx_bgblur mode 3
 * (person keeps the camera, background becomes this image, cover-
 * scaled). Returns 1 on success, 0 if the file can't be loaded.
 * cv_clear_bg_image() removes it (mode 3 then shows the gradient). */
int  cv_set_bg_image(const char *path);
void cv_clear_bg_image(void);

/* Free any lazily-created OpenCV state (detectors, nets, bg models,
 * flow buffers). Called at shutdown; safe to call in the stub build. */
void cv_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* VCAM_CV_H */
