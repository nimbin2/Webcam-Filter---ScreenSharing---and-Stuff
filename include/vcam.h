/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/types.h>

/* ──────────────────────────────────────────────
 * Frame  (internal BGR24)
 * ────────────────────────────────────────────── */
typedef struct {
    uint8_t *data;
    int      width, height, stride;
} Frame;

Frame *frame_alloc(int width, int height);
void   frame_free(Frame *f);

/* ──────────────────────────────────────────────
 * Overlay
 * ────────────────────────────────────────────── */
typedef enum {
    OVERLAY_IMAGE,   /* static PNG/JPEG             */
    OVERLAY_VIDEO,   /* looping video file          */
    OVERLAY_SCREEN,  /* live screen-region capture  */
    OVERLAY_WEBCAM,  /* live webcam capture (own V4L2 device) */
} OverlayType;

typedef struct Overlay Overlay;

Overlay *overlay_create_image (const char *path,
                                int dst_x, int dst_y, int dst_w, int dst_h);
Overlay *overlay_create_video (const char *path,
                                int dst_x, int dst_y, int dst_w, int dst_h,
                                bool loop);  /* loop=true: restart on EOF */

/*
 * screen_geometry: "WxH+X+Y"  (as output by slurp / wf-recorder -g)
 * dst_*: placement on the output frame (0 = use capture native size)
 */
Overlay *overlay_create_screen(const char *screen_geometry,
                                int dst_x, int dst_y, int dst_w, int dst_h);

/*
 * device: V4L2 path (e.g. "/dev/video1"). Will fail with EBUSY if the
 * main vcam process is already capturing from the same device — webcam
 * overlays are intended for use in --no-input mode where the main
 * pipeline doesn't own the camera.
 */
Overlay *overlay_create_webcam(const char *device,
                                int dst_x, int dst_y, int dst_w, int dst_h);

void overlay_free (Overlay *o);
void overlay_blit (Overlay *o, Frame *dst);

/* Advance one frame.
 * webcam_src: the latest captured webcam frame (after filters), used by
 * OVERLAY_WEBCAM overlays as their source. Pass NULL if no webcam is
 * being captured — webcam overlays will then show black.
 * Other overlay types ignore this argument. */
void overlay_tick (Overlay *o, const Frame *webcam_src);

/* Runtime geometry update */
void overlay_move  (Overlay *o, int x, int y);
void overlay_resize(Overlay *o, int w, int h);

/* Inspect overlay state (for IPC list command) */
OverlayType overlay_get_type(const Overlay *o);
void        overlay_get_rect(const Overlay *o, int *x, int *y, int *w, int *h);
void        overlay_get_src (const Overlay *o, int *w, int *h);

/* ──────────────────────────────────────────────
 * Filter chain — per-pixel effects applied to the
 * webcam frame before overlays are composited.
 * ────────────────────────────────────────────── */
typedef enum {
    FX_GRAYSCALE,
    FX_SEPIA,
    FX_INVERT,
    FX_MIRROR,      /* horizontal flip */
    FX_FLIP,        /* vertical flip   */
    FX_PIXELATE,    /* param = block size (px), default 8  */
    FX_BLUR,        /* param = radius (px), default 2      */
    FX_VIGNETTE,    /* param = strength 0..1, default 0.6  */
    FX_BRIGHTNESS,  /* param = multiplier, default 1.2     */
    FX_CONTRAST,    /* param = multiplier, default 1.4     */
    FX_POSTERIZE,   /* param = levels per channel, def 4   */
    FX_EDGES,       /* sobel edge detect, strength 0..2, default 1.0      */
    FX_SATURATE,    /* saturation multiplier 0..3, default 1.5            */

    /* ── fancier effects ── */
    FX_CHROMASHIFT, /* param = offset px, default 5 (VHS-style RGB split) */
    FX_GLITCH,      /* param = intensity 0..1, default 0.3 (animated)     */
    FX_KALEIDOSCOPE,/* param = segments, default 6                        */
    FX_SWIRL,       /* param = strength, default 1.0                      */
    FX_THERMAL,     /* heatmap palette mapped from luma                   */
    FX_OLDFILM,     /* sepia + grain + scratches + vignette               */
    FX_WAVE,        /* param = amplitude px, default 10 (sinusoidal)      */
    FX_CARTOON,     /* posterize + edge enhancement                       */
    FX_SOLARIZE,    /* param = threshold 0..255, default 128              */
    FX_RAINBOW,     /* animated hue rotation, param = speed (default 1.0) */
    FX_CRT,         /* CRT scanlines + slight curvature, 0..1, def 0.7    */

    /* ── soft / dreamy / painterly ── */
    FX_DREAM,       /* blur + brightness boost (soft hazy glow)           */
    FX_BLOOM,       /* highlight-bloom glow on bright areas               */
    FX_GLOW,        /* gentle gaussian glow over everything               */
    FX_PINCH,       /* radial inward pull toward center                   */
    FX_BULGE,       /* radial outward push from center (fisheye-ish)      */
    FX_PAINTERLY,   /* oil-painting: dominant color in neighborhood       */
    FX_VELVET,      /* desat + warm tint + soft blur, 0..1, def 0.6       */

    /* ── themed / styled looks ── */
    FX_MODERN,      /* clean teal-orange grade, default 0.7 strength      */
    FX_CYBERPUNK,   /* chromashift + neon contrast, default 0.7 intensity */
    FX_ROMANTIC,    /* soft warm haze, default 0.7 strength               */
    FX_VANGOGH,     /* heavy painterly + saturation boost                 */
    FX_PICASSO,     /* kaleidoscope + posterize + cartoon edges           */
    FX_NOIR,        /* high-contrast B&W with split-tone shadows, def 0.8 */

    /* ── mathematical patterns ── */
    FX_SINE,        /* sinusoidal x/y distortion, default 15px            */
    FX_XOR,         /* x^y bitwise overlay, 0..1                          */
    FX_MODULO,      /* color modulo, default 64                           */
    FX_POLAR,       /* polar-coordinate warp, 0..1                        */
    FX_VORONOI,     /* voronoi cell mosaic, default 20 cells              */
    FX_JULIA,       /* Julia fractal color overlay, 0..1, def 0.5         */

    /* ── pixel-crazy / animated ── */
    FX_BITCRUSH,    /* quantize bits per channel, 1..8, default 3         */
    FX_MELT,        /* animated drip downward, 0..1, default 0.5          */
    FX_POPART,      /* warhol-style 2x2 quadrants, strength 0..1          */
    FX_DISPLACE,    /* luma-based horizontal displacement, default 0.5    */
    FX_RIPPLE,      /* animated water ripple from center, default 0.5     */
    FX_SCANLINES,   /* black scanlines every N rows, gap 2..8, def 3      */

    /* ── glitch art / generative animated ── */
    FX_SHUFFLE,     /* roving pixel-shuffle blocks, intensity 0..1, def 0.4 */
    FX_PIXELSORT,   /* glitch-art row pixel-sort, density 0..1, def 0.3     */
    FX_RGB_DELAY,   /* chromatic lag (G+R from past frames), 0..1, def 0.7  */
    FX_DRIFT,       /* horizontal scanline drift, band height px, def 8     */
    FX_RAIN,        /* matrix-style digital rain trails, density 0..1, def 0.4 */

    /* ── temporal / time-bending (rgbdelay's siblings) ── */
    FX_ECHO,        /* motion trails: blend of past frames, 0..1, def 0.6   */
    FX_SLITSCAN,    /* time gradient top→bottom rows from the past, 0..1    */
    FX_MOSH,        /* fake datamosh: only update pixels that move, 0..1    */
    FX_STUTTER,     /* freeze each frame for N frames (temporal pixelate)   */
    FX_HALFTONE,    /* newspaper dot grid, cell size px, def 8              */
    FX_NEON,        /* glowing colored edges on darkened scene, 0..1        */

    /* ── batch 3: retro & warp ── */
    FX_EMBOSS,      /* grayscale relief, strength 0..2, def 1               */
    FX_DITHER,      /* ordered Bayer dither, levels/channel 2..8, def 2     */
    FX_TILES,       /* mirror-tile grid, tiles per row 2..8, def 3          */
    FX_ZOOMPULSE,   /* breathing zoom oscillation, 0..1, def 0.5            */
    FX_OILSLICK,    /* iridescent luma-keyed hue warp, 0..1, def 0.6        */

    /* ── batch 4: interactive & extravagant ── */
    FX_FEEDBACK,    /* video-feedback tunnel; decay 0..1 (+zoom,rot), def 0.85 */
    FX_RIPPLES,     /* INTERACTIVE: motion spawns water ripples, 0..1       */
    FX_LIGHTPAINT,  /* INTERACTIVE: motion paints glowing trails, fade 0..1 */
    FX_HOLOGRAM,    /* sci-fi hologram: cyan, scanlines, flicker, 0..1      */
    FX_LIFE,        /* Conway's Life seeded by image edges, density 0..1    */
    FX_MATHWARP,    /* expression-driven warp — see 'filter expr', amp mult */

    /* ── batch 5: demoscene & particles ── */
    FX_PLASMA,      /* classic demoscene plasma blend, 0..1 (+speed,scale)  */
    FX_CONTOUR,     /* glowing topographic iso-lines, levels (+dim)         */
    FX_FIREWORKS,   /* INTERACTIVE: motion sparks particle bursts, 0..1     */
    FX_INFINITY,    /* recursive picture-in-picture droste, scale (+rot)    */
    FX_MOIRE,       /* rotating interference rings, spacing px (+speed)     */

    /* ── batch 6: text, tone & tracking ── */
    FX_ASCII,       /* real ASCII-art rendering, cell px (+colormode)       */
    FX_DUOTONE,     /* two-color gradient map, hueA deg (+hueB, mix)        */
    FX_SKETCH,      /* pencil sketch (dodge blend), detail (+darkness)      */
    FX_SPOTLIGHT,   /* INTERACTIVE: spotlight follows your motion, radius   */

    /* ── batch 7: the AI-craze set ── */
    FX_PREDATOR,    /* motion-camouflage cloak: hold still → vanish         */
    FX_SHARDS,      /* broken-mirror voronoi shards w/ crack lines          */
    FX_PORTAL,      /* two ringed portals that swap their contents          */
    FX_HEATTRAIL,   /* motion leaves cooling thermal glow                   */
    FX_WORMHOLE,    /* classic demo tunnel textured with your own image     */
    FX_MAGNET,      /* image warps toward the brightest region              */
    FX_LIQUID,      /* whole frame becomes gently flowing liquid            */
    FX_NIGHTVISION, /* amplified green + noise + scanlines + hard vignette  */
    FX_BADTV,       /* vertical roll, tearing, static bursts, chroma wobble */
    FX_GHOSTS,      /* motion extraction: only movement is visible          */

    /* ── batch 8: AI-craze volume 2 ── */
    FX_CLONES,      /* periodic freezes of you populate the frame           */
    FX_MELTDOWN,    /* Doom-style: a snapshot melts off, revealing live     */
    FX_ROTOZOOM,    /* demoscene rotozoomer: rotating tiled plane of you    */
    FX_STARFIELD,   /* hyperspace starfield streaking over the video        */
    FX_LIGHTNING,   /* bolts strike wherever you move                       */
    FX_SCATTER,     /* motion shatters pixels outward; frame heals itself   */
    FX_VHS,         /* full VHS: chroma bleed, head-switch bar, timecode    */
    FX_OUTLINE,     /* marching-ants animated outline around moving subject */
    FX_SYMMETRY,    /* face symmetry mirrors (left/right/quad)              */
    FX_TIMEBUBBLE,  /* drifting bubble showing N seconds into the past      */

    /* ── OpenCV-backed (optional; built with OPENCV=1) ── */
    FX_CV_CARTOON,  /* edge-preserving bilateral cartoon + ink outlines     */
    FX_CV_PENCIL,   /* non-photorealistic pencil sketch                     */
    FX_CV_STYLIZE,  /* watercolor/poster stylization                        */
    FX_CV_FLOW,     /* dense optical-flow motion visualization              */
    FX_CV_BGSUB,    /* MOG2: moving foreground vivid, static bg dimmed      */
    FX_CV_FACE,     /* YuNet face detection: blur / ring / googly-eyes      */
    FX_CV_PERSON,   /* PP-HumanSeg: blur/spotlight/gradient/your-image bg   */
    FX_CV_STYLE,    /* live neural style transfer (5 styles, blend, size)   */
    FX_CV_MOOD,     /* emotion-reactive: aura / HUD / party / emoji         */
    FX_CV_PAINT,    /* optical-flow paint smear: motion drags reality       */
    FX_CV_SPEED,    /* anime speed lines when you move fast                 */

    FX_UNKNOWN,
} FilterType;

typedef struct {
    FilterType type;
    float      param;         /* primary — kept for backwards compat */
    float      param2, param3, param4;  /* optional additional knobs */
} Filter;

#define MAX_FILTERS 16
#define FILTER_MAX_PARAMS 4

typedef struct {
    Filter          filters[MAX_FILTERS];
    int             count;
    pthread_mutex_t lock;
} FilterChain;

void        filter_chain_init   (FilterChain *fc);
void        filter_chain_destroy(FilterChain *fc);
/* Add a filter with just the primary param. */
bool        filter_chain_add    (FilterChain *fc, FilterType t, float param);
/* Add a filter with up to FILTER_MAX_PARAMS params (unused → 0.0). */
bool        filter_chain_add_multi(FilterChain *fc, FilterType t,
                                   const float *params, int n_params);
bool        filter_chain_remove (FilterChain *fc, int index);
void        filter_chain_clear  (FilterChain *fc);
void        filter_chain_apply  (FilterChain *fc, Frame *f); /* takes lock */

/* mathwarp: compile displacement expressions used by FX_MATHWARP.
 * Variables: x y (pixel coords), w h (frame size), t (seconds),
 * r (distance from center), a (angle from center).
 * Functions: sin cos abs sqrt. Operators: + - * / ( ) unary -.
 * NO SPACES inside an expression. Returns false + err message on
 * parse failure. */
bool mathwarp_set_expr(const char *dx_expr, const char *dy_expr,
                       char *err, size_t errlen);

/* Recolor expressions for FX_MATHWARP. Each returns a channel value
 * (0..255) and may read the warped source color via cr/cg/cb, plus
 * all the usual vars (x y w h t r a). Once set, mathwarp recolors
 * every pixel until mathwarp_clear_color() is called. */
bool mathwarp_set_color(const char *r_expr, const char *g_expr,
                        const char *b_expr, char *err, size_t errlen);
void mathwarp_clear_color(void);

FilterType  filter_type_from_name(const char *name);
const char *filter_type_to_name  (FilterType t);
float       filter_default_param (FilterType t);

/* ──────────────────────────────────────────────
 * Overlay registry  (mutex-protected list)
 * ────────────────────────────────────────────── */
#define MAX_OVERLAYS 64

typedef struct {
    Overlay    *overlay;
    char        id[64];      /* user-supplied name */
} OverlaySlot;

typedef struct {
    OverlaySlot  slots[MAX_OVERLAYS];
    int          count;
    pthread_mutex_t lock;
} OverlayRegistry;

void     registry_init  (OverlayRegistry *r);
void     registry_destroy(OverlayRegistry *r);
bool     registry_add   (OverlayRegistry *r, const char *id, Overlay *o);
bool     registry_remove(OverlayRegistry *r, const char *id);
Overlay *registry_find  (OverlayRegistry *r, const char *id);  /* caller must hold lock */
bool     registry_raise (OverlayRegistry *r, const char *id);  /* draw last  (on top) */
bool     registry_lower (OverlayRegistry *r, const char *id);  /* draw first (at bottom) */
bool     registry_set_z (OverlayRegistry *r, const char *id, int target);
/*  target >= 0    : absolute position (0 = bottom)
 *  target <  0    : count from top (-1 = top, -2 = second-from-top, …)
 *  Out-of-range values are clamped.  Returns false only if id not found. */

/* ──────────────────────────────────────────────
 * IPC control server
 * Commands (newline-terminated text):
 *   add img   <id> <x> <y> [<w> <h>] <file>
 *   add vid   <id> <x> <y> [<w> <h>] <file>
 *   add screen <id> <x> <y> [<w> <h>] <geometry>
 *   remove <id>
 *   move   <id> <x> <y>
 *   resize <id> <w> <h>
 *   list
 *   quit
 * Responses are newline-terminated text lines.
 * ────────────────────────────────────────────── */
#define VCAM_SOCK_PATH "/tmp/vcam.sock"

typedef struct {
    OverlayRegistry *registry;
    int              server_fd;
    pthread_t        thread;
    volatile bool    running;
} IPCServer;

bool ipc_start(IPCServer *srv, OverlayRegistry *reg);
void ipc_stop (IPCServer *srv);

/* ──────────────────────────────────────────────
 * V4L2 capture
 * ────────────────────────────────────────────── */
typedef struct V4L2Capture V4L2Capture;

V4L2Capture *capture_open (const char *dev, int *w, int *h, int *fps);
bool         capture_read (V4L2Capture *cap, Frame *out);
void         capture_close(V4L2Capture *cap);

/* Enumerate and print every usable (format, resolution, fps) the
 * device supports. Returns false if the device can't be opened. */
bool         capture_list_formats(const char *dev);

/* ──────────────────────────────────────────────
 * V4L2 loopback output
 * ────────────────────────────────────────────── */
typedef struct V4L2Output V4L2Output;

V4L2Output *output_open (const char *dev, int w, int h, int fps);

/* Output frame dimensions — set by main.c after output_open succeeds.
 * Screen / webcam overlays clamp against these to avoid being clipped. */
extern int g_output_w;
extern int g_output_h;
bool        output_write(V4L2Output *out, const Frame *frame);
void        output_close(V4L2Output *out);

/* ──────────────────────────────────────────────
 * Recorder — spawns ffmpeg as a child process and feeds it our raw
 * BGR24 frames via a pipe. Bypasses /dev/video0 entirely, so it never
 * contends with browser / ffplay / OBS consumers of the v4l2loopback.
 * ────────────────────────────────────────────── */
typedef struct Recorder Recorder;

Recorder *recorder_new(void);
void      recorder_free(Recorder *r);

/* Start recording. file is the output path; mic may be NULL or an
 * ALSA device string ("default", "plughw:1,0", etc.). w/h/fps are
 * the dimensions of the BGR24 frames you'll pass to
 * recorder_write_frame. Returns false if ffmpeg failed to start.
 *
 * crf:     x264 quality (0..51, lower = better)
 * preset, vcodec: x264 preset / codec name
 * aoffset: audio delay in seconds (positive shifts audio LATER).
 *          Compensates for the video path's buffering latency
 *          (webcam driver + decode + pipe), which otherwise makes
 *          audio lead video by a constant ~0.3-0.5s. */
bool recorder_start(Recorder *r,
                    const char *file, const char *mic,
                    int w, int h, int fps,
                    int crf, const char *vcodec, const char *vpreset,
                    float aoffset);

/* Stop recording. Sends EOF to ffmpeg via closing the pipe so it
 * finalises the container, then reaps the child. Safe to call when
 * not recording (no-op). */
void recorder_stop(Recorder *r);

bool        recorder_active(const Recorder *r);
pid_t       recorder_pid   (const Recorder *r);
const char *recorder_file  (const Recorder *r);

/* Send one frame to ffmpeg's stdin. Frame must be BGR24, width*height*3
 * bytes. Non-blocking — if ffmpeg's pipe is full (typically because the
 * encoder briefly stalled) the frame is silently dropped rather than
 * blocking the main loop. Returns true if frame was sent (or recording
 * is inactive), false if ffmpeg died unexpectedly. */
bool recorder_write_frame(Recorder *r, const uint8_t *bgr24, size_t bytes);
