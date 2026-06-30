#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

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

    FX_UNKNOWN,
} FilterType;

typedef struct {
    FilterType type;
    float      param;
} Filter;

#define MAX_FILTERS 16

typedef struct {
    Filter          filters[MAX_FILTERS];
    int             count;
    pthread_mutex_t lock;
} FilterChain;

void        filter_chain_init   (FilterChain *fc);
void        filter_chain_destroy(FilterChain *fc);
bool        filter_chain_add    (FilterChain *fc, FilterType t, float param);
bool        filter_chain_remove (FilterChain *fc, int index);
void        filter_chain_clear  (FilterChain *fc);
void        filter_chain_apply  (FilterChain *fc, Frame *f); /* takes lock */

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
