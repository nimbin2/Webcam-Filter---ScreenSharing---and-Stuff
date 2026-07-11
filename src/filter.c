/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * filter.c – per-pixel effects applied to a BGR24 frame in place.
 *
 * All effects operate on the captured webcam frame BEFORE overlays are
 * composited on top. The chain is mutex-protected so the IPC thread can
 * modify it while the main loop is applying it.
 */

#include "vcam.h"
#include "vcam_cv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

/* ──────────────────────────────────────────────────────────────
 * Name <-> type
 * ────────────────────────────────────────────────────────────── */
static const struct { const char *name; FilterType t; } NAMES[] = {
    {"grayscale",   FX_GRAYSCALE},
    {"gray",        FX_GRAYSCALE},
    {"sepia",       FX_SEPIA},
    {"invert",      FX_INVERT},
    {"mirror",      FX_MIRROR},
    {"flip",        FX_FLIP},
    {"pixelate",    FX_PIXELATE},
    {"blur",        FX_BLUR},
    {"vignette",    FX_VIGNETTE},
    {"brightness",  FX_BRIGHTNESS},
    {"contrast",    FX_CONTRAST},
    {"posterize",   FX_POSTERIZE},
    {"edges",       FX_EDGES},
    /* fancy */
    {"chromashift", FX_CHROMASHIFT},
    {"glitch",      FX_GLITCH},
    {"kaleidoscope",FX_KALEIDOSCOPE},
    {"swirl",       FX_SWIRL},
    {"thermal",     FX_THERMAL},
    {"oldfilm",     FX_OLDFILM},
    {"wave",        FX_WAVE},
    {"cartoon",     FX_CARTOON},
    {"solarize",    FX_SOLARIZE},
    {"rainbow",     FX_RAINBOW},
    /* soft/painterly */
    {"dream",       FX_DREAM},
    {"bloom",       FX_BLOOM},
    {"glow",        FX_GLOW},
    {"pinch",       FX_PINCH},
    {"bulge",       FX_BULGE},
    {"painterly",   FX_PAINTERLY},
    /* themed looks */
    {"modern",      FX_MODERN},
    {"cyberpunk",   FX_CYBERPUNK},
    {"romantic",    FX_ROMANTIC},
    {"vangogh",     FX_VANGOGH},
    {"picasso",     FX_PICASSO},
    /* mathematical */
    {"sine",        FX_SINE},
    {"xor",         FX_XOR},
    {"modulo",      FX_MODULO},
    {"polar",       FX_POLAR},
    {"voronoi",     FX_VORONOI},
    {"julia",       FX_JULIA},
    /* pixel-crazy */
    {"bitcrush",    FX_BITCRUSH},
    {"melt",        FX_MELT},
    {"popart",      FX_POPART},
    {"displace",    FX_DISPLACE},
    {"ripple",      FX_RIPPLE},
    {"scanlines",   FX_SCANLINES},
    /* glitch art */
    {"shuffle",     FX_SHUFFLE},
    {"pixelsort",   FX_PIXELSORT},
    {"rgbdelay",    FX_RGB_DELAY},
    {"rgb-delay",   FX_RGB_DELAY},
    {"drift",       FX_DRIFT},
    {"rain",        FX_RAIN},
    /* temporal */
    {"echo",        FX_ECHO},
    {"trails",      FX_ECHO},
    {"slitscan",    FX_SLITSCAN},
    {"timewarp",    FX_SLITSCAN},
    {"mosh",        FX_MOSH},
    {"datamosh",    FX_MOSH},
    {"stutter",     FX_STUTTER},
    {"halftone",    FX_HALFTONE},
    {"neon",        FX_NEON},
    /* retro & warp */
    {"emboss",      FX_EMBOSS},
    {"dither",      FX_DITHER},
    {"tiles",       FX_TILES},
    {"zoompulse",   FX_ZOOMPULSE},
    {"pulse",       FX_ZOOMPULSE},
    {"oilslick",    FX_OILSLICK},
    {"iridescent",  FX_OILSLICK},
    /* interactive & extravagant */
    {"feedback",    FX_FEEDBACK},
    {"tunnel",      FX_FEEDBACK},
    {"ripples",     FX_RIPPLES},
    {"water",       FX_RIPPLES},
    {"lightpaint",  FX_LIGHTPAINT},
    {"paint",       FX_LIGHTPAINT},
    {"hologram",    FX_HOLOGRAM},
    {"holo",        FX_HOLOGRAM},
    {"life",        FX_LIFE},
    {"conway",      FX_LIFE},
    {"mathwarp",    FX_MATHWARP},
    {"expr",        FX_MATHWARP},
    /* demoscene & particles */
    {"plasma",      FX_PLASMA},
    {"contour",     FX_CONTOUR},
    {"topo",        FX_CONTOUR},
    {"fireworks",   FX_FIREWORKS},
    {"sparks",      FX_FIREWORKS},
    {"infinity",    FX_INFINITY},
    {"droste",      FX_INFINITY},
    {"moire",       FX_MOIRE},
    /* text, tone & tracking */
    {"ascii",       FX_ASCII},
    {"matrix-text", FX_ASCII},
    {"duotone",     FX_DUOTONE},
    {"sketch",      FX_SKETCH},
    {"pencil",      FX_SKETCH},
    {"spotlight",   FX_SPOTLIGHT},
    {"followme",    FX_SPOTLIGHT},
    /* the AI-craze set */
    {"predator",    FX_PREDATOR},
    {"cloak",       FX_PREDATOR},
    {"shards",      FX_SHARDS},
    {"glasswall",   FX_SHARDS},
    {"portal",      FX_PORTAL},
    {"heattrail",   FX_HEATTRAIL},
    {"heat",        FX_HEATTRAIL},
    {"wormhole",    FX_WORMHOLE},
    {"magnet",      FX_MAGNET},
    {"liquid",      FX_LIQUID},
    {"nightvision", FX_NIGHTVISION},
    {"nv",          FX_NIGHTVISION},
    {"badtv",       FX_BADTV},
    {"ghosts",      FX_GHOSTS},
    {"motionx",     FX_GHOSTS},
    /* AI-craze volume 2 */
    {"clones",      FX_CLONES},
    {"multiplicity",FX_CLONES},
    {"meltdown",    FX_MELTDOWN},
    {"doommelt",    FX_MELTDOWN},
    {"rotozoom",    FX_ROTOZOOM},
    {"starfield",   FX_STARFIELD},
    {"hyperspace",  FX_STARFIELD},
    {"lightning",   FX_LIGHTNING},
    {"tesla",       FX_LIGHTNING},
    {"scatter",     FX_SCATTER},
    {"shatterheal", FX_SCATTER},
    {"vhs",         FX_VHS},
    {"outline",     FX_OUTLINE},
    {"symmetry",    FX_SYMMETRY},
    {"timebubble",  FX_TIMEBUBBLE},
    {"pastbubble",  FX_TIMEBUBBLE},
    /* OpenCV-backed (optional) */
    {"cvcartoon",   FX_CV_CARTOON},
    {"cvpencil",    FX_CV_PENCIL},
    {"sketch2",     FX_CV_PENCIL},
    {"cvstylize",   FX_CV_STYLIZE},
    {"watercolor",  FX_CV_STYLIZE},
    {"opticalflow", FX_CV_FLOW},
    {"cvflow",      FX_CV_FLOW},
    {"bgsub",       FX_CV_BGSUB},
    {"motionmask",  FX_CV_BGSUB},
    {"facefx",      FX_CV_FACE},
    {"face",        FX_CV_FACE},
    {"personfx",    FX_CV_PERSON},
    {"portrait",    FX_CV_PERSON},
    {"styleai",     FX_CV_STYLE},
    {"neuralstyle", FX_CV_STYLE},
    {"moodfx",      FX_CV_MOOD},
    {"emotion",     FX_CV_MOOD},
    {"paintflow",   FX_CV_PAINT},
    {"smear",       FX_CV_PAINT},
    {"speedlines",  FX_CV_SPEED},
    {"anime",       FX_CV_SPEED},
    /* new effect additions */
    {"saturate",    FX_SATURATE},
    {"crt",         FX_CRT},
    {"velvet",      FX_VELVET},
    {"noir",        FX_NOIR},
};

FilterType filter_type_from_name(const char *name)
{
    if (!name) return FX_UNKNOWN;
    for (size_t i = 0; i < sizeof NAMES / sizeof NAMES[0]; i++)
        if (strcasecmp(NAMES[i].name, name) == 0)
            return NAMES[i].t;
    return FX_UNKNOWN;
}

const char *filter_type_to_name(FilterType t)
{
    for (size_t i = 0; i < sizeof NAMES / sizeof NAMES[0]; i++)
        if (NAMES[i].t == t) return NAMES[i].name;
    return "unknown";
}

float filter_default_param(FilterType t)
{
    switch (t) {
    case FX_PIXELATE:    return 8.0f;
    case FX_BLUR:        return 2.0f;
    case FX_VIGNETTE:    return 0.6f;
    case FX_BRIGHTNESS:  return 1.2f;
    case FX_CONTRAST:    return 1.4f;
    case FX_POSTERIZE:   return 4.0f;
    case FX_CHROMASHIFT: return 5.0f;
    case FX_GLITCH:      return 0.3f;
    case FX_KALEIDOSCOPE:return 6.0f;
    case FX_SWIRL:       return 1.0f;
    case FX_WAVE:        return 10.0f;
    case FX_CARTOON:     return 50.0f;
    case FX_SOLARIZE:    return 128.0f;
    case FX_RAINBOW:     return 1.0f;
    case FX_DREAM:       return 0.5f;  /* glow strength 0..1 */
    case FX_BLOOM:       return 0.6f;  /* bloom strength 0..1 */
    case FX_GLOW:        return 0.4f;  /* glow intensity 0..1 */
    case FX_PINCH:       return 0.5f;  /* pinch strength 0..1 */
    case FX_BULGE:       return 0.5f;  /* bulge strength 0..1 */
    case FX_PAINTERLY:   return 3.0f;  /* neighborhood radius */
    case FX_MODERN:      return 0.7f;
    case FX_CYBERPUNK:   return 0.7f;
    case FX_ROMANTIC:    return 0.7f;
    case FX_VANGOGH:     return 3.0f;
    case FX_PICASSO:     return 8.0f;
    case FX_SINE:        return 15.0f;
    case FX_XOR:         return 0.5f;
    case FX_MODULO:      return 64.0f;
    case FX_POLAR:       return 0.7f;
    case FX_VORONOI:     return 20.0f;
    case FX_BITCRUSH:    return 3.0f;
    case FX_MELT:        return 0.5f;
    case FX_POPART:      return 1.0f;
    case FX_DISPLACE:    return 0.5f;
    case FX_RIPPLE:      return 0.5f;
    case FX_EDGES:       return 1.0f;
    case FX_SATURATE:    return 1.5f;
    case FX_CRT:         return 0.7f;
    case FX_VELVET:      return 0.6f;
    case FX_NOIR:        return 0.8f;
    case FX_JULIA:       return 0.5f;
    case FX_SCANLINES:   return 3.0f;
    /* glitch art */
    case FX_SHUFFLE:     return 0.4f;
    case FX_PIXELSORT:   return 0.3f;
    case FX_RGB_DELAY:   return 0.7f;
    case FX_DRIFT:       return 8.0f;
    case FX_RAIN:        return 0.4f;
    /* temporal */
    case FX_ECHO:        return 0.6f;
    case FX_SLITSCAN:    return 0.7f;
    case FX_MOSH:        return 0.5f;
    case FX_STUTTER:     return 6.0f;
    case FX_HALFTONE:    return 8.0f;
    case FX_NEON:        return 0.7f;
    case FX_EMBOSS:      return 1.0f;
    case FX_DITHER:      return 2.0f;
    case FX_TILES:       return 3.0f;
    case FX_ZOOMPULSE:   return 0.5f;
    case FX_OILSLICK:    return 0.6f;
    case FX_FEEDBACK:    return 0.85f;
    case FX_RIPPLES:     return 0.6f;
    case FX_LIGHTPAINT:  return 0.94f;
    case FX_HOLOGRAM:    return 0.8f;
    case FX_LIFE:        return 0.5f;
    case FX_MATHWARP:    return 1.0f;
    case FX_PLASMA:      return 0.5f;
    case FX_CONTOUR:     return 8.0f;
    case FX_FIREWORKS:   return 0.6f;
    case FX_INFINITY:    return 0.55f;
    case FX_MOIRE:       return 12.0f;
    case FX_ASCII:       return 8.0f;
    case FX_DUOTONE:     return 220.0f;
    case FX_SKETCH:      return 2.0f;
    case FX_SPOTLIGHT:   return 180.0f;
    case FX_PREDATOR:    return 0.15f;
    case FX_SHARDS:      return 24.0f;
    case FX_PORTAL:      return 130.0f;
    case FX_HEATTRAIL:   return 0.7f;
    case FX_WORMHOLE:    return 60.0f;
    case FX_MAGNET:      return 0.5f;
    case FX_LIQUID:      return 12.0f;
    case FX_NIGHTVISION: return 2.0f;
    case FX_BADTV:       return 0.5f;
    case FX_GHOSTS:      return 6.0f;
    case FX_CLONES:      return 3.0f;
    case FX_MELTDOWN:    return 3.0f;
    case FX_ROTOZOOM:    return 1.0f;
    case FX_STARFIELD:   return 0.5f;
    case FX_LIGHTNING:   return 0.3f;
    case FX_SCATTER:     return 0.6f;
    case FX_VHS:         return 0.5f;
    case FX_OUTLINE:     return 2.0f;
    case FX_SYMMETRY:    return 0.0f;
    case FX_TIMEBUBBLE:  return 1.5f;
    /* OpenCV-backed */
    case FX_CV_CARTOON:  return 0.6f;
    case FX_CV_PENCIL:   return 1.0f;   /* 1 = keep color, 0 = graphite */
    case FX_CV_STYLIZE:  return 0.6f;
    case FX_CV_FLOW:     return 0.6f;
    case FX_CV_BGSUB:    return 0.5f;
    case FX_CV_FACE:     return 0.0f;   /* mode 0 = privacy blur */
    case FX_CV_PERSON:   return 0.0f;   /* mode 0 = background blur */
    case FX_CV_STYLE:    return 0.0f;   /* style 0 = mosaic */
    case FX_CV_MOOD:     return 0.0f;   /* mode 0 = aura */
    case FX_CV_PAINT:    return 1.0f;
    case FX_CV_SPEED:    return 1.0f;
    default:             return 0.0f;
    }
}

/* ──────────────────────────────────────────────────────────────
 * Chain management
 * ────────────────────────────────────────────────────────────── */
void filter_chain_init(FilterChain *fc)
{
    memset(fc, 0, sizeof *fc);
    pthread_mutex_init(&fc->lock, NULL);
}

void filter_chain_destroy(FilterChain *fc)
{
    pthread_mutex_destroy(&fc->lock);
}

bool filter_chain_add(FilterChain *fc, FilterType t, float param)
{
    if (t == FX_UNKNOWN) return false;
    pthread_mutex_lock(&fc->lock);
    if (fc->count >= MAX_FILTERS) {
        pthread_mutex_unlock(&fc->lock);
        return false;
    }
    fc->filters[fc->count].type   = t;
    fc->filters[fc->count].param  = param;
    fc->filters[fc->count].param2 = 0.0f;
    fc->filters[fc->count].param3 = 0.0f;
    fc->filters[fc->count].param4 = 0.0f;
    fc->count++;
    pthread_mutex_unlock(&fc->lock);
    return true;
}

bool filter_chain_add_multi(FilterChain *fc, FilterType t,
                            const float *params, int n_params)
{
    if (t == FX_UNKNOWN) return false;
    if (n_params < 1) return false;
    if (n_params > FILTER_MAX_PARAMS) n_params = FILTER_MAX_PARAMS;
    pthread_mutex_lock(&fc->lock);
    if (fc->count >= MAX_FILTERS) {
        pthread_mutex_unlock(&fc->lock);
        return false;
    }
    Filter *f = &fc->filters[fc->count];
    f->type   = t;
    f->param  = params[0];
    f->param2 = (n_params > 1) ? params[1] : 0.0f;
    f->param3 = (n_params > 2) ? params[2] : 0.0f;
    f->param4 = (n_params > 3) ? params[3] : 0.0f;
    f->param5 = (n_params > 4) ? params[4] : 0.0f;
    fc->count++;
    pthread_mutex_unlock(&fc->lock);
    return true;
}

bool filter_chain_remove(FilterChain *fc, int index)
{
    pthread_mutex_lock(&fc->lock);
    if (index < 0 || index >= fc->count) {
        pthread_mutex_unlock(&fc->lock);
        return false;
    }
    for (int i = index; i < fc->count - 1; i++)
        fc->filters[i] = fc->filters[i+1];
    fc->count--;
    pthread_mutex_unlock(&fc->lock);
    return true;
}

void filter_chain_clear(FilterChain *fc)
{
    pthread_mutex_lock(&fc->lock);
    fc->count = 0;
    pthread_mutex_unlock(&fc->lock);
}

/* ──────────────────────────────────────────────────────────────
 * Effect implementations
 * All operate on BGR24 (3 bytes/pixel, byte order B,G,R).
 * ────────────────────────────────────────────────────────────── */
static inline uint8_t clamp_u8(int v)
{
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

static void fx_grayscale(Frame *f)
{
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        /* Rec.601 luma */
        int y = (p[2] * 299 + p[1] * 587 + p[0] * 114) / 1000;
        p[0] = p[1] = p[2] = (uint8_t)y;
    }
}

static void fx_sepia(Frame *f)
{
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        int b = p[0], g = p[1], r = p[2];
        int nr = (r * 393 + g * 769 + b * 189) / 1000;
        int ng = (r * 349 + g * 686 + b * 168) / 1000;
        int nb = (r * 272 + g * 534 + b * 131) / 1000;
        p[0] = clamp_u8(nb);
        p[1] = clamp_u8(ng);
        p[2] = clamp_u8(nr);
    }
}

static void fx_invert(Frame *f)
{
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++) p[i] = 255 - p[i];
}

static void fx_mirror(Frame *f)
{
    /* horizontal flip in place */
    for (int y = 0; y < f->height; y++) {
        uint8_t *row = f->data + y * f->stride;
        for (int x = 0; x < f->width / 2; x++) {
            int x2 = f->width - 1 - x;
            uint8_t tmp[3] = { row[x*3+0], row[x*3+1], row[x*3+2] };
            row[x*3+0] = row[x2*3+0];
            row[x*3+1] = row[x2*3+1];
            row[x*3+2] = row[x2*3+2];
            row[x2*3+0] = tmp[0];
            row[x2*3+1] = tmp[1];
            row[x2*3+2] = tmp[2];
        }
    }
}

static void fx_flip(Frame *f)
{
    /* vertical flip in place */
    uint8_t *tmp = malloc(f->stride);
    if (!tmp) return;
    for (int y = 0; y < f->height / 2; y++) {
        uint8_t *a = f->data + y * f->stride;
        uint8_t *b = f->data + (f->height - 1 - y) * f->stride;
        memcpy(tmp, a, f->stride);
        memcpy(a,   b, f->stride);
        memcpy(b, tmp, f->stride);
    }
    free(tmp);
}

static void fx_pixelate(Frame *f, int block)
{
    if (block < 2) block = 2;
    if (block > f->width / 2) block = f->width / 2;
    for (int by = 0; by < f->height; by += block) {
        for (int bx = 0; bx < f->width; bx += block) {
            /* Average colour of this block */
            int bw = bx + block > f->width  ? f->width  - bx : block;
            int bh = by + block > f->height ? f->height - by : block;
            int sb = 0, sg = 0, sr = 0;
            for (int dy = 0; dy < bh; dy++) {
                uint8_t *p = f->data + (by+dy) * f->stride + bx * 3;
                for (int dx = 0; dx < bw; dx++, p += 3) {
                    sb += p[0]; sg += p[1]; sr += p[2];
                }
            }
            int cnt = bw * bh;
            uint8_t cb = sb / cnt, cg = sg / cnt, cr = sr / cnt;
            for (int dy = 0; dy < bh; dy++) {
                uint8_t *p = f->data + (by+dy) * f->stride + bx * 3;
                for (int dx = 0; dx < bw; dx++, p += 3) {
                    p[0] = cb; p[1] = cg; p[2] = cr;
                }
            }
        }
    }
}

static void fx_blur(Frame *f, int radius)
{
    /* Box blur via two 1D passes. radius=1 → 3-tap, radius=2 → 5-tap, etc. */
    if (radius < 1) radius = 1;
    if (radius > 6) radius = 6;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;

    /* Horizontal pass: f → tmp */
    for (int y = 0; y < h; y++) {
        const uint8_t *src = f->data + y * f->stride;
        uint8_t *dst = tmp + y * w * 3;
        for (int x = 0; x < w; x++) {
            int sb = 0, sg = 0, sr = 0, n = 0;
            int x0 = x - radius, x1 = x + radius;
            if (x0 < 0) x0 = 0;
            if (x1 >= w) x1 = w - 1;
            for (int xx = x0; xx <= x1; xx++) {
                sb += src[xx*3+0]; sg += src[xx*3+1]; sr += src[xx*3+2]; n++;
            }
            dst[x*3+0] = sb / n; dst[x*3+1] = sg / n; dst[x*3+2] = sr / n;
        }
    }
    /* Vertical pass: tmp → f */
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int sb = 0, sg = 0, sr = 0, n = 0;
            int y0 = y - radius, y1 = y + radius;
            if (y0 < 0) y0 = 0;
            if (y1 >= h) y1 = h - 1;
            for (int yy = y0; yy <= y1; yy++) {
                uint8_t *s = tmp + yy * w * 3 + x * 3;
                sb += s[0]; sg += s[1]; sr += s[2]; n++;
            }
            uint8_t *d = f->data + y * f->stride + x * 3;
            d[0] = sb / n; d[1] = sg / n; d[2] = sr / n;
        }
    }
    free(tmp);
}

/* ── vignette ──
 * strength   : 0..1 darkening amount at the corners
 * radius     : where darkening starts as fraction of half-diagonal (0..1);
 *              0 = start at center, 1 = start at corner. Default 0 (original).
 * warmth     : 0..1 shift the darkened areas toward warm brown (film look);
 *              0 = pure darkening. Default 0.
 */
static void fx_vignette(Frame *f, float strength, float radius, float warmth)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    if (radius   < 0) radius   = 0;
    if (radius   > 0.95f) radius = 0.95f;
    if (warmth   < 0) warmth   = 0;
    if (warmth   > 1) warmth   = 1;

    int w = f->width, h = f->height;
    float cx = w / 2.0f, cy = h / 2.0f;
    float maxd = sqrtf(cx*cx + cy*cy);
    float denom = 1.0f - radius;
    if (denom < 0.05f) denom = 0.05f;

    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            float dx = x - cx, dy = y - cy;
            float d = sqrtf(dx*dx + dy*dy) / maxd;
            float t = (d - radius) / denom;
            if (t < 0) t = 0;
            float v = 1.0f - strength * t * t;
            if (v < 0) v = 0;

            /* Blend toward warm tone: (B G R) → (30 60 90)*(1-v)*warmth */
            float warmB = 30.0f  * (1.0f - v) * warmth;
            float warmG = 60.0f  * (1.0f - v) * warmth;
            float warmR = 90.0f  * (1.0f - v) * warmth;

            int b = (int)(p[0] * v + warmB);
            int g = (int)(p[1] * v + warmG);
            int r = (int)(p[2] * v + warmR);
            p[0] = clamp_u8(b);
            p[1] = clamp_u8(g);
            p[2] = clamp_u8(r);
        }
    }
}

static void fx_brightness(Frame *f, float mult)
{
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++)
        p[i] = clamp_u8((int)(p[i] * mult));
}

static void fx_contrast(Frame *f, float mult)
{
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++)
        p[i] = clamp_u8((int)((p[i] - 128) * mult + 128));
}

static void fx_posterize(Frame *f, int levels)
{
    if (levels < 2) levels = 2;
    if (levels > 16) levels = 16;
    int step = 256 / levels;
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++)
        p[i] = (uint8_t)((p[i] / step) * step);
}

static void fx_edges(Frame *f, float strength)
{
    /* Simple Sobel on luma. strength multiplies the gradient magnitude
     * before clamping: 0..2 → 0=black, 1=normal, 2=more punchy. */
    if (strength < 0) strength = 0;
    if (strength > 4) strength = 4;
    int w = f->width, h = f->height;
    uint8_t *luma = malloc((size_t)w * h);
    if (!luma) return;
    for (int y = 0; y < h; y++) {
        const uint8_t *src = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            luma[y*w + x] = (src[x*3+2]*299 + src[x*3+1]*587 + src[x*3+0]*114) / 1000;
        }
    }
    for (int y = 1; y < h - 1; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 1; x < w - 1; x++) {
            int gx = -luma[(y-1)*w + (x-1)] - 2*luma[y*w + (x-1)] - luma[(y+1)*w + (x-1)]
                     +luma[(y-1)*w + (x+1)] + 2*luma[y*w + (x+1)] + luma[(y+1)*w + (x+1)];
            int gy = -luma[(y-1)*w + (x-1)] - 2*luma[(y-1)*w + x] - luma[(y-1)*w + (x+1)]
                     +luma[(y+1)*w + (x-1)] + 2*luma[(y+1)*w + x] + luma[(y+1)*w + (x+1)];
            int v = (int)((abs(gx) + abs(gy)) * strength);
            uint8_t e = clamp_u8(v);
            dst[x*3+0] = dst[x*3+1] = dst[x*3+2] = e;
        }
    }
    /* Zero out the 1-pixel border for cleanliness */
    memset(f->data, 0, f->stride);
    memset(f->data + (h-1) * f->stride, 0, f->stride);
    free(luma);
}

/* ══════════════════════════════════════════════════════════════
 * Fancy effects
 * ══════════════════════════════════════════════════════════════ */

/* A monotonically increasing tick — drives animated effects */
static unsigned long anim_tick(void)
{
    static unsigned long t = 0;
    return t++;
}

/* Cheap pseudo-random for animation. Not crypto-grade, just chaotic. */
static unsigned hash_u(unsigned x)
{
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

/* ── chromashift: VHS-style RGB channel separation ──────────
 * dx      : horizontal offset for R (opposite direction for B)
 * dxb     : optional independent horizontal offset for B; 0 = mirror dx
 * dy_r    : optional vertical offset for R (0 = none)
 */
static void fx_chromashift(Frame *f, int dx, int dxb, int dy_r)
{
    if (dx == 0 && dxb == 0 && dy_r == 0) return;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    if (dxb == 0) dxb = -dx;   /* default: B opposite of R */

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int yr = y + dy_r;
        if (yr < 0) yr = 0;
        if (yr >= h) yr = h - 1;
        const uint8_t *src_g = tmp + y  * w * 3;
        const uint8_t *src_r = tmp + yr * w * 3;
        for (int x = 0; x < w; x++) {
            int xr = x + dx;   if (xr < 0) xr = 0; if (xr >= w) xr = w - 1;
            int xb = x + dxb;  if (xb < 0) xb = 0; if (xb >= w) xb = w - 1;
            dst[x*3+0] = src_g[xb*3+0];   /* B */
            dst[x*3+1] = src_g[x*3+1];    /* G — reference */
            dst[x*3+2] = src_r[xr*3+2];   /* R */
        }
    }
    free(tmp);
}

/* ── glitch: random horizontal slice displacement (animated) ──
 * intensity   : how far shifts go and how often (0..1)
 * slice_h_override : forced slice height in px (0 = derive from intensity)
 * max_shift_pct    : override on max shift as %-of-width (0 = default 15%)
 */
static void fx_glitch(Frame *f, float intensity, int slice_h_override,
                       float max_shift_pct)
{
    if (intensity <= 0) return;
    if (intensity > 1) intensity = 1;
    int w = f->width, h = f->height;
    unsigned seed = (unsigned)anim_tick();

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    int slice_h = slice_h_override > 0
                ? slice_h_override
                : 4 + (int)((1.0f - intensity) * 20);
    float pct = (max_shift_pct > 0.0f) ? max_shift_pct * 0.01f : 0.15f;
    int max_shift = (int)((float)w * pct * intensity);
    if (max_shift < 1) max_shift = 1;

    for (int y = 0; y < h; y += slice_h) {
        unsigned r = hash_u(seed + (unsigned)y);
        if ((r & 7) > 5) continue;
        int shift = (int)((r >> 4) % (unsigned)(2 * max_shift + 1)) - max_shift;
        int sh = (y + slice_h > h) ? h - y : slice_h;
        for (int dy = 0; dy < sh; dy++) {
            uint8_t *dst = f->data + (y+dy) * f->stride;
            const uint8_t *src = tmp + (y+dy) * w * 3;
            for (int x = 0; x < w; x++) {
                int sx = x - shift;
                if (sx < 0) sx += w;
                if (sx >= w) sx -= w;
                dst[x*3+0] = src[sx*3+0];
                dst[x*3+1] = src[sx*3+1];
                dst[x*3+2] = src[sx*3+2];
            }
        }
    }
    free(tmp);
}

/* ── kaleidoscope v2: mirror a wedge around the center ─────────
 * segments : number of wedges
 * rot_deg  : static rotation applied to output
 * spin     : additional per-frame spin rate (deg/frame)
 * zoom     : magnification of the sampled wedge (0 → 1.0; >1 zooms in,
 *            <1 zooms out — animate it via vcam-tune for tunnel vibes)
 */
static void fx_kaleidoscope(Frame *f, int segments, float rot_deg,
                            float spin, float zoom)
{
    if (segments < 2) segments = 2;
    if (segments > 32) segments = 32;
    if (zoom <= 0.05f) zoom = 1.0f;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float wedge = (float)(3.14159265358979f * 2.0f) / segments;
    float rot   = (rot_deg + spin * (float)anim_tick()) * 3.14159265f / 180.0f;
    float inv_zoom = 1.0f / zoom;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy) * inv_zoom;
            float a = atan2f(dy, dx) + rot;
            float m = fmodf(a, wedge);
            if (m < 0) m += wedge;
            if (m > wedge / 2) m = wedge - m;
            int sx = (int)(cx + r * cosf(m));
            int sy = (int)(cy + r * sinf(m));
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ── swirl: radial rotation that intensifies with distance ── */
/* ── swirl v2 ──
 * strength : twist amount (radians at center)
 * radius   : p2 — fraction of frame the swirl covers, 0.2..1 (0 → 1)
 * spin     : p3 — animate the twist over time (rate; 0 = static)
 */
static void fx_swirl(Frame *f, float strength, float radius_frac, float spin)
{
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    if (radius_frac <= 0.05f) radius_frac = 1.0f;
    if (radius_frac > 1.0f) radius_frac = 1.0f;
    float s = strength;
    if (spin != 0.0f)
        s += sinf((float)anim_tick() / 30.0f * spin) * strength;

    float cx = w / 2.0f, cy = h / 2.0f;
    float maxr = sqrtf(cx*cx + cy*cy) * radius_frac;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy);
            if (r >= maxr) continue;   /* outside swirl radius: untouched */
            float theta = atan2f(dy, dx) + s * (1.0f - r / maxr) * 3.14159f;
            int sx = (int)(cx + r * cosf(theta));
            int sy = (int)(cy + r * sinf(theta));
            if (sx < 0 || sx >= w || sy < 0 || sy >= h) {
                dst[x*3+0] = dst[x*3+1] = dst[x*3+2] = 0;
            } else {
                const uint8_t *src = tmp + sy * w * 3 + sx * 3;
                dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
            }
        }
    }
    free(tmp);
}

/* ── thermal: map luma to a heatmap palette ─────────────────── */
static void fx_thermal(Frame *f)
{
    /* black → blue → purple → red → orange → yellow → white */
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        int y = (p[2] * 299 + p[1] * 587 + p[0] * 114) / 1000;
        int b, g, r;
        if (y < 51) {        /* 0-50: black → blue */
            float t = y / 51.0f;
            b = (int)(t * 255); g = 0; r = 0;
        } else if (y < 102) { /* 51-101: blue → purple */
            float t = (y - 51) / 51.0f;
            b = 255; g = 0; r = (int)(t * 128);
        } else if (y < 153) { /* 102-152: purple → red */
            float t = (y - 102) / 51.0f;
            b = (int)((1 - t) * 255); g = 0; r = (int)(128 + t * 127);
        } else if (y < 204) { /* 153-203: red → orange */
            float t = (y - 153) / 51.0f;
            b = 0; g = (int)(t * 128); r = 255;
        } else {              /* 204-255: orange → yellow → white */
            float t = (y - 204) / 51.0f;
            b = (int)(t * 255); g = (int)(128 + t * 127); r = 255;
        }
        p[0] = (uint8_t)b; p[1] = (uint8_t)g; p[2] = (uint8_t)r;
    }
}

/* ── oldfilm: sepia + grain + vertical scratches + vignette ── */
static void fx_oldfilm(Frame *f)
{
    int w = f->width, h = f->height;
    unsigned seed = (unsigned)anim_tick();

    /* sepia */
    fx_sepia(f);

    /* film grain */
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            int g = (int)(hash_u(seed + (unsigned)(y * w + x)) % 41) - 20;
            p[0] = clamp_u8(p[0] + g);
            p[1] = clamp_u8(p[1] + g);
            p[2] = clamp_u8(p[2] + g);
        }
    }

    /* occasional vertical scratch lines */
    int scratches = (int)(hash_u(seed) % 4);
    for (int s = 0; s < scratches; s++) {
        int x = (int)(hash_u(seed + 17 + (unsigned)s) % (unsigned)w);
        for (int y = 0; y < h; y++) {
            uint8_t *p = f->data + y * f->stride + x * 3;
            p[0] = p[1] = p[2] = 240;
        }
    }

    /* vignette */
    fx_vignette(f, 0.7f, 0.0f, 0.0f);
}

/* ── wave: sinusoidal vertical wave displacement ────────────── */
/* ── wave: sinusoidal x-shift, now with frequency and orientation ──
 * amplitude : pixel-magnitude of the displacement
 * freq_hz   : spatial frequency (waves per screen); 0 = default
 * speed     : phase advance rate (px/frame); 0 = default
 * vertical  : if nonzero, shifts columns vertically instead of rows horizontally
 */
static void fx_wave(Frame *f, float amplitude, float freq_hz, float speed,
                    int vertical)
{
    if (amplitude < 1) amplitude = 1;
    int w = f->width, h = f->height;
    unsigned long t = anim_tick();
    /* Defaults calibrated to match the original behaviour. */
    float phase = t * (speed > 0.0f ? speed : 0.1f);
    /* freq_hz: convert "waves per screen" → radians per pixel. */
    float k = (freq_hz > 0.0f)
            ? (freq_hz * 2.0f * 3.14159265f / (vertical ? (float)w : (float)h))
            : 0.05f;   /* original: sin(y*0.05) */

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    if (!vertical) {
        for (int y = 0; y < h; y++) {
            int shift = (int)(amplitude * sinf(y * k + phase));
            uint8_t *dst = f->data + y * f->stride;
            const uint8_t *src = tmp + y * w * 3;
            for (int x = 0; x < w; x++) {
                int sx = x - shift;
                if (sx < 0) sx = 0;
                if (sx >= w) sx = w - 1;
                dst[x*3+0] = src[sx*3+0];
                dst[x*3+1] = src[sx*3+1];
                dst[x*3+2] = src[sx*3+2];
            }
        }
    } else {
        /* Vertical: each column shifts up/down by sin(x * k + phase). */
        for (int x = 0; x < w; x++) {
            int shift = (int)(amplitude * sinf(x * k + phase));
            for (int y = 0; y < h; y++) {
                int sy = y - shift;
                if (sy < 0) sy = 0;
                if (sy >= h) sy = h - 1;
                const uint8_t *src = tmp + sy * w * 3 + x * 3;
                uint8_t       *dst = f->data + y * f->stride + x * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    }
    free(tmp);
}

/* ── cartoon: posterize + emphasize edges ───────────────────── */
/* cartoon v2: levels (p2) = posterize color levels, 2..12 (0 → 5) */
static void fx_cartoon(Frame *f, float edge_thresh, float levels_f)
{
    int w = f->width, h = f->height;
    int levels = (int)levels_f;
    if (levels < 2) levels = 5;
    if (levels > 12) levels = 12;

    /* Compute luma + Sobel magnitude */
    uint8_t *luma = malloc((size_t)w * h);
    if (!luma) return;
    for (int y = 0; y < h; y++) {
        const uint8_t *src = f->data + y * f->stride;
        for (int x = 0; x < w; x++)
            luma[y*w + x] = (src[x*3+2]*299 + src[x*3+1]*587 + src[x*3+0]*114) / 1000;
    }

    /* Posterize colors */
    int step = 256 / levels;
    int n = w * h * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++) p[i] = (uint8_t)((p[i] / step) * step + step/2);

    /* Black-out strong edges */
    for (int y = 1; y < h - 1; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 1; x < w - 1; x++) {
            int gx = -luma[(y-1)*w + (x-1)] - 2*luma[y*w + (x-1)] - luma[(y+1)*w + (x-1)]
                     +luma[(y-1)*w + (x+1)] + 2*luma[y*w + (x+1)] + luma[(y+1)*w + (x+1)];
            int gy = -luma[(y-1)*w + (x-1)] - 2*luma[(y-1)*w + x] - luma[(y-1)*w + (x+1)]
                     +luma[(y+1)*w + (x-1)] + 2*luma[(y+1)*w + x] + luma[(y+1)*w + (x+1)];
            if (abs(gx) + abs(gy) > edge_thresh)
                dst[x*3+0] = dst[x*3+1] = dst[x*3+2] = 0;
        }
    }
    free(luma);
}

/* ── solarize: invert pixels above threshold ────────────────── */
static void fx_solarize(Frame *f, int threshold)
{
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++)
        if (p[i] > threshold) p[i] = 255 - p[i];
}

/* ── rainbow: animated additive hue rotation ────────────────── */
static void fx_rainbow(Frame *f, float speed)
{
    unsigned long t = anim_tick();
    /* Three offset sinusoids → smooth rainbow rotation per pixel */
    float phase = t * 0.04f * speed;
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        float y = (p[2]*299 + p[1]*587 + p[0]*114) / 1000.0f;
        /* Add color back based on phase + position */
        float sx = (i % f->width) * 0.01f;
        float sy = (i / f->width) * 0.01f;
        float a = phase + sx * 0.5f + sy * 0.3f;
        int r = (int)(y * (0.5f + 0.5f * sinf(a)));
        int g = (int)(y * (0.5f + 0.5f * sinf(a + 2.094f))); /* +120° */
        int b = (int)(y * (0.5f + 0.5f * sinf(a + 4.189f))); /* +240° */
        p[0] = clamp_u8(b);
        p[1] = clamp_u8(g);
        p[2] = clamp_u8(r);
    }
}

/* ══════════════════════════════════════════════════════════════
 * Soft / dreamy / painterly effects
 * ══════════════════════════════════════════════════════════════ */

/* Box-blur helper into a separate buffer. Used by glow effects so we
 * don't lose the original image (unlike fx_blur which is in-place). */
static void box_blur_into(const uint8_t *src, uint8_t *dst,
                          int w, int h, int radius)
{
    if (radius < 1) radius = 1;
    if (radius > 8) radius = 8;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;

    /* horizontal pass: src → tmp */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sb=0, sg=0, sr=0, n=0;
            int x0 = x - radius, x1 = x + radius;
            if (x0 < 0) x0 = 0;
            if (x1 >= w) x1 = w - 1;
            for (int xx = x0; xx <= x1; xx++) {
                const uint8_t *s = src + y * w * 3 + xx * 3;
                sb += s[0]; sg += s[1]; sr += s[2]; n++;
            }
            uint8_t *t = tmp + y * w * 3 + x * 3;
            t[0] = sb/n; t[1] = sg/n; t[2] = sr/n;
        }
    }
    /* vertical pass: tmp → dst */
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int sb=0, sg=0, sr=0, n=0;
            int y0 = y - radius, y1 = y + radius;
            if (y0 < 0) y0 = 0;
            if (y1 >= h) y1 = h - 1;
            for (int yy = y0; yy <= y1; yy++) {
                const uint8_t *s = tmp + yy * w * 3 + x * 3;
                sb += s[0]; sg += s[1]; sr += s[2]; n++;
            }
            uint8_t *d = dst + y * w * 3 + x * 3;
            d[0] = sb/n; d[1] = sg/n; d[2] = sr/n;
        }
    }
    free(tmp);
}

/* ── dream: heavy blur layered over original ─────────────────── */
static void fx_dream(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    uint8_t *blur = malloc((size_t)w * h * 3);
    if (!blur) return;
    box_blur_into(f->data, blur, w, h, 6);

    /* dst = (1-s)*orig + s*blur, then bias brightness up */
    int total = w * h * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < total; i++) {
        int v = (int)((1.0f - strength) * p[i] + strength * blur[i]);
        v += (int)(strength * 25); /* gentle brightness lift */
        p[i] = clamp_u8(v);
    }
    free(blur);
}

/* ── bloom: extract bright pixels, blur, add back ────────────── */
static void fx_bloom(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;

    /* extract highlights into a buffer (everything above threshold) */
    uint8_t *hi = malloc((size_t)w * h * 3);
    if (!hi) return;
    const int thresh = 180;
    int total = w * h * 3;
    for (int i = 0; i < total; i++)
        hi[i] = f->data[i] > thresh ? (uint8_t)(f->data[i] - thresh) * 2 : 0;

    /* blur the highlights */
    uint8_t *bloom = malloc((size_t)w * h * 3);
    if (!bloom) { free(hi); return; }
    box_blur_into(hi, bloom, w, h, 6);
    free(hi);

    /* add bloom on top of the original (screen blend-ish) */
    for (int i = 0; i < total; i++) {
        int v = f->data[i] + (int)(bloom[i] * strength);
        f->data[i] = clamp_u8(v);
    }
    free(bloom);
}

/* ── glow: uniform gaussian-like haze over the whole image ──── */
static void fx_glow(Frame *f, float intensity)
{
    if (intensity < 0) intensity = 0;
    if (intensity > 1) intensity = 1;
    int w = f->width, h = f->height;
    uint8_t *blur = malloc((size_t)w * h * 3);
    if (!blur) return;
    box_blur_into(f->data, blur, w, h, 4);

    /* Screen blend: 1 - (1-a)*(1-b) — softens highlights, lifts mids */
    int total = w * h * 3;
    for (int i = 0; i < total; i++) {
        int a = f->data[i];
        int b = (int)(blur[i] * intensity);
        int v = 255 - ((255 - a) * (255 - b)) / 255;
        f->data[i] = clamp_u8(v);
    }
    free(blur);
}

/* ── pinch v2: inward radial pull ──
 * strength; cx_frac/cy_frac (p2/p3): center position as 0..1 of the
 * frame (0 → centered). Off-center pinches look like gravity wells. */
static void fx_pinch(Frame *f, float strength, float cx_frac, float cy_frac)
{
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = (cx_frac > 0.01f && cx_frac < 0.99f) ? cx_frac * w : w / 2.0f;
    float cy = (cy_frac > 0.01f && cy_frac < 0.99f) ? cy_frac * h : h / 2.0f;
    float maxr = sqrtf(cx*cx + cy*cy);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy) / maxr; /* 0..1 */
            /* Pinch curve: pull r toward 0 by exponentiating */
            float scaled = powf(r, 1.0f + strength);
            float factor = (r > 0.001f) ? (scaled / r) : 1.0f;
            int sx = (int)(cx + dx * factor);
            int sy = (int)(cy + dy * factor);
            if (sx < 0 || sx >= w || sy < 0 || sy >= h) {
                dst[x*3+0] = dst[x*3+1] = dst[x*3+2] = 0;
            } else {
                const uint8_t *src = tmp + sy * w * 3 + sx * 3;
                dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
            }
        }
    }
    free(tmp);
}

/* ── bulge v2: outward radial push (fisheye-like) ──
 * strength; cx_frac/cy_frac (p2/p3): center as 0..1 (0 → centered). */
static void fx_bulge(Frame *f, float strength, float cx_frac, float cy_frac)
{
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = (cx_frac > 0.01f && cx_frac < 0.99f) ? cx_frac * w : w / 2.0f;
    float cy = (cy_frac > 0.01f && cy_frac < 0.99f) ? cy_frac * h : h / 2.0f;
    float maxr = sqrtf(cx*cx + cy*cy);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy) / maxr;
            /* Bulge: push r away from 0 — sample from nearer */
            float scaled = powf(r, 1.0f / (1.0f + strength));
            float factor = (r > 0.001f) ? (scaled / r) : 1.0f;
            int sx = (int)(cx + dx * factor);
            int sy = (int)(cy + dy * factor);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w-1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h-1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ── painterly: oil-painting via dominant color in neighborhood
 *   For each pixel, bucket nearby pixels into a few intensity bins,
 *   take the most popular bin's average color. */
static void fx_painterly(Frame *f, int radius)
{
    if (radius < 1) radius = 1;
    if (radius > 5) radius = 5;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    enum { NBINS = 6 };
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            int cnt[NBINS] = {0};
            int sb[NBINS] = {0}, sg[NBINS] = {0}, sr[NBINS] = {0};
            int x0 = x - radius, x1 = x + radius;
            int y0 = y - radius, y1 = y + radius;
            if (x0 < 0) x0 = 0;
            if (x1 >= w) x1 = w - 1;
            if (y0 < 0) y0 = 0;
            if (y1 >= h) y1 = h - 1;
            for (int yy = y0; yy <= y1; yy++) {
                for (int xx = x0; xx <= x1; xx++) {
                    const uint8_t *s = tmp + yy * w * 3 + xx * 3;
                    int luma = (s[2]*299 + s[1]*587 + s[0]*114) / 1000;
                    int bin = (luma * NBINS) / 256;
                    if (bin >= NBINS) bin = NBINS - 1;
                    cnt[bin]++;
                    sb[bin] += s[0]; sg[bin] += s[1]; sr[bin] += s[2];
                }
            }
            int best = 0;
            for (int b = 1; b < NBINS; b++) if (cnt[b] > cnt[best]) best = b;
            int c = cnt[best];
            dst[x*3+0] = sb[best] / c;
            dst[x*3+1] = sg[best] / c;
            dst[x*3+2] = sr[best] / c;
        }
    }
    free(tmp);
}

#include <limits.h>  /* for INT_MAX */

/* ══════════════════════════════════════════════════════════════
 * Themed / aesthetic looks
 * ══════════════════════════════════════════════════════════════ */

/* ── modern: teal-orange cinematic grade ────────────────────── */
static void fx_modern(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int n = f->width * f->height;
    uint8_t *p = f->data;
    int push = (int)(20 * strength);
    for (int i = 0; i < n; i++, p += 3) {
        int b = p[0], g = p[1], r = p[2];
        int luma = (r*299 + g*587 + b*114) / 1000;
        if (luma < 128) {
            /* shadows → teal */
            p[0] = clamp_u8(b + push);
            p[1] = clamp_u8(g + push/2);
            p[2] = clamp_u8(r - push/2);
        } else {
            /* highlights → orange */
            p[2] = clamp_u8(r + push);
            p[1] = clamp_u8(g + push/4);
            p[0] = clamp_u8(b - push);
        }
    }
    fx_contrast(f, 1.0f + 0.15f * strength);
    fx_vignette(f, 0.3f * strength, 0.0f, 0.0f);
}

/* ── cyberpunk: neon contrast + chromashift ─────────────────── */
static void fx_cyberpunk(Frame *f, float intensity)
{
    if (intensity < 0) intensity = 0;
    if (intensity > 1) intensity = 1;

    /* Lift blues and magentas, drop yellows */
    int n = f->width * f->height;
    uint8_t *p = f->data;
    int boost = (int)(25 * intensity);
    for (int i = 0; i < n; i++, p += 3) {
        p[0] = clamp_u8(p[0] + boost);            /* B up */
        if (p[2] > p[1]) p[2] = clamp_u8(p[2] + boost); /* magenta zones */
        if (p[1] > p[2]) p[1] = clamp_u8(p[1] - boost/2); /* drop yellow-green */
    }
    fx_contrast(f, 1.0f + 0.4f * intensity);
    fx_chromashift(f, (int)(intensity * 10), 0, 0);
}

/* ── romantic: soft warm haze ──────────────────────────────── */
static void fx_romantic(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;

    /* Warm tint */
    int n = f->width * f->height;
    uint8_t *p = f->data;
    int warm = (int)(strength * 25);
    for (int i = 0; i < n; i++, p += 3) {
        p[0] = clamp_u8(p[0] - warm);     /* less B */
        p[1] = clamp_u8(p[1] + warm/4);   /* tiny G */
        p[2] = clamp_u8(p[2] + warm);     /* more R */
    }
    /* Soft blur, brightness lift, vignette */
    fx_blur(f, 2);
    fx_brightness(f, 1.0f + strength * 0.15f);
    fx_vignette(f, strength * 0.55f, 0.0f, 0.0f);
}

/* ── vangogh: thick painterly with saturated brush strokes ── */
static void fx_vangogh(Frame *f, float radius)
{
    int r = (int)radius;
    if (r < 1) r = 1;
    if (r > 5) r = 5;
    fx_painterly(f, r);

    /* Saturation boost — push every channel away from luma */
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        int b = p[0], g = p[1], rr = p[2];
        int luma = (rr*299 + g*587 + b*114) / 1000;
        p[0] = clamp_u8(luma + (int)((b  - luma) * 1.6f));
        p[1] = clamp_u8(luma + (int)((g  - luma) * 1.6f));
        p[2] = clamp_u8(luma + (int)((rr - luma) * 1.6f));
    }
    fx_contrast(f, 1.15f);
}

/* ── picasso: cubist abstraction ───────────────────────────── */
static void fx_picasso(Frame *f, float segments)
{
    int s = (int)segments;
    if (s < 4) s = 4;
    if (s > 16) s = 16;
    fx_kaleidoscope(f, s, 0.0f, 0.0f, 0.0f);
    fx_posterize(f, 4);
    fx_cartoon(f, 35, 0);
}

/* ══════════════════════════════════════════════════════════════
 * Mathematical patterns
 * ══════════════════════════════════════════════════════════════ */

/* ── sine: animated sinusoidal x and y displacement ────────── */
static void fx_sine(Frame *f, float amp)
{
    if (amp < 1) amp = 1;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    unsigned long t = anim_tick();
    float phase = t * 0.06f;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            int sx = x + (int)(amp * sinf(y * 0.04f + phase));
            int sy = y + (int)(amp * sinf(x * 0.04f + phase * 0.7f));
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ── xor: pixel-position XOR overlay ───────────────────────── */
static void fx_xor(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            int v = (x ^ y) & 0xff;
            int delta = (int)(v * strength);
            p[0] ^= (uint8_t)delta;
            p[1] ^= (uint8_t)delta;
            p[2] ^= (uint8_t)delta;
        }
    }
}

/* ── modulo: channel values mod N, scaled back to 0-255 ────── */
static void fx_modulo(Frame *f, float div)
{
    int d = (int)div;
    if (d < 2) d = 2;
    if (d > 128) d = 128;
    int scale = 255 / d;
    int n = f->width * f->height * 3;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++) p[i] = (uint8_t)((p[i] % d) * scale);
}

/* ── polar: blend cartesian sampling with polar coordinates ── */
static void fx_polar(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float maxr = sqrtf(cx*cx + cy*cy);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy) / maxr;
            float theta = atan2f(dy, dx) / 3.14159265f;  /* -1..1 */

            int polar_x = (int)((theta * 0.5f + 0.5f) * w);
            int polar_y = (int)(r * h);

            int sx = (int)((1 - strength) * x + strength * polar_x);
            int sy = (int)((1 - strength) * y + strength * polar_y);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ── voronoi: random seed cells flat-shaded with seed's color ── */
static void fx_voronoi(Frame *f, float cell_count)
{
    int ncells = (int)cell_count;
    if (ncells < 4) ncells = 4;
    if (ncells > 64) ncells = 64;
    int w = f->width, h = f->height;

    /* Animate slowly — new seed positions every ~30 frames */
    unsigned long t = anim_tick();
    unsigned base = (unsigned)(t / 30);

    int sx[64], sy[64];
    uint8_t sb[64], sg[64], sr[64];
    for (int i = 0; i < ncells; i++) {
        sx[i] = (int)(hash_u(base + (unsigned)i * 13) % (unsigned)w);
        sy[i] = (int)(hash_u(base + (unsigned)i * 41) % (unsigned)h);
        const uint8_t *p = f->data + sy[i] * f->stride + sx[i] * 3;
        sb[i] = p[0]; sg[i] = p[1]; sr[i] = p[2];
    }

    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            int best = 0, best_d = INT_MAX;
            for (int i = 0; i < ncells; i++) {
                int dx = x - sx[i], dy = y - sy[i];
                int d = dx*dx + dy*dy;
                if (d < best_d) { best_d = d; best = i; }
            }
            p[0] = sb[best]; p[1] = sg[best]; p[2] = sr[best];
        }
    }
}

/* ══════════════════════════════════════════════════════════════
 * Pixel-crazy / animated
 * ══════════════════════════════════════════════════════════════ */

/* ── bitcrush: quantize each channel to `bits` bits ──────────── */
static void fx_bitcrush(Frame *f, float bits)
{
    int b = (int)bits;
    if (b < 1) b = 1;
    if (b > 8) b = 8;
    uint8_t mask = (uint8_t)(~((1 << (8 - b)) - 1));
    int n = f->width * f->height * 3;
    for (int i = 0; i < n; i++) f->data[i] &= mask;
}

/* ── melt: each column drifts downward by a random offset ────── */
static void fx_melt(Frame *f, float intensity)
{
    if (intensity < 0) intensity = 0;
    if (intensity > 1) intensity = 1;
    int w = f->width, h = f->height;
    unsigned long t = anim_tick();

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    /* Per-column drip — same column gets the same drift across rows,
     * which is what gives the "candle wax" look. The drift evolves
     * slowly over time so it actually melts on screen. */
    for (int x = 0; x < w; x++) {
        unsigned r = hash_u((unsigned)x * 31 + (unsigned)(t / 3));
        int drop = (int)((r % 200) * intensity / 3);
        for (int y = 0; y < h; y++) {
            int sy = y - drop;
            if (sy < 0) sy = 0;
            uint8_t *dst = f->data + y * f->stride + x * 3;
            const uint8_t *src = tmp + sy * w * 3 + x * 3;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
    }
    free(tmp);
}

/* ── popart: 2x2 quadrants each with a different Warhol tint ── */
static void fx_popart(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    int hw = w / 2, hh = h / 2;

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    /* 4 Warhol-ish palettes, (B,G,R) */
    const int tints[4][3] = {
        {200, 100, 255},  /* pink   */
        {255, 200, 100},  /* cyan   */
        {100, 255, 200},  /* green  */
        {100, 200, 255},  /* orange */
    };

    for (int y = 0; y < h; y++) {
        int qy = (y < hh) ? 0 : 1;
        for (int x = 0; x < w; x++) {
            int qx = (x < hw) ? 0 : 1;
            int q  = qy * 2 + qx;

            /* Sample from a 2x downscale so every quadrant shows the
             * full image, just smaller. */
            int sx = (x % hw) * 2;
            int sy = (y % hh) * 2;
            if (sx >= w) sx = w - 1;
            if (sy >= h) sy = h - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            int luma = (src[2]*299 + src[1]*587 + src[0]*114) / 1000;

            float lf = luma / 255.0f;
            int tb = (int)(lf * tints[q][0]);
            int tg = (int)(lf * tints[q][1]);
            int tr = (int)(lf * tints[q][2]);

            /* Blend tint with original by strength */
            uint8_t *dst = f->data + y * f->stride + x * 3;
            dst[0] = clamp_u8((int)((1.0f - strength) * src[0] + strength * tb));
            dst[1] = clamp_u8((int)((1.0f - strength) * src[1] + strength * tg));
            dst[2] = clamp_u8((int)((1.0f - strength) * src[2] + strength * tr));
        }
    }
    free(tmp);
}

/* ── displace: luma at (x,y) shifts the sample x-position ───── */
static void fx_displace(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    int amount = (int)(strength * 40);
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            const uint8_t *p = tmp + y * w * 3 + x * 3;
            int luma = (p[2]*299 + p[1]*587 + p[0]*114) / 1000;
            int dx = ((luma - 128) * amount) / 128;
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            const uint8_t *src = tmp + y * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ── ripple: animated water ripple radiating from center ────── */
static void fx_ripple(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    unsigned long t = anim_tick();
    float phase = t * 0.15f;
    float amp   = strength * 25.0f;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r  = sqrtf(dx*dx + dy*dy);
            if (r < 0.5f) {
                const uint8_t *src = tmp + y * w * 3 + x * 3;
                dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
                continue;
            }
            float wave = sinf(r * 0.05f - phase) * amp;
            float nx = dx / r, ny = dy / r;
            int sx = (int)(x + nx * wave);
            int sy = (int)(y + ny * wave);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0]; dst[x*3+1] = src[1]; dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ══════════════════════════════════════════════════════════════
 * NEW: one filter per category — saturate, crt, velvet, noir,
 * julia, scanlines.
 * ══════════════════════════════════════════════════════════════ */

/* ── saturate: chroma-distance multiplier around luma (Basic) ── */
static void fx_saturate(Frame *f, float factor)
{
    if (factor < 0) factor = 0;
    if (factor > 4) factor = 4;
    int n = f->width * f->height;
    for (int i = 0; i < n; i++) {
        uint8_t *p = f->data + i * 3;
        int luma = (p[2]*299 + p[1]*587 + p[0]*114) / 1000;
        p[0] = clamp_u8((int)(luma + (p[0] - luma) * factor));
        p[1] = clamp_u8((int)(luma + (p[1] - luma) * factor));
        p[2] = clamp_u8((int)(luma + (p[2] - luma) * factor));
    }
}

/* ── crt: TV CRT scanlines + slight barrel + R/B fringing (Fancy) ── */
static void fx_crt(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float barrel = strength * 0.15f;
    int shift   = (int)(strength * 3);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float ny = (y - cy) / cy;
        for (int x = 0; x < w; x++) {
            float nx = (x - cx) / cx;
            float r2 = nx*nx + ny*ny;
            /* Slight barrel distortion toward edges */
            float dx = nx * (1.0f + barrel * r2);
            float dy = ny * (1.0f + barrel * r2);
            int sx = (int)(dx * cx + cx);
            int sy = (int)(dy * cy + cy);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w-1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h-1;

            /* R/B channel shift for chromatic aberration */
            int rxr = sx + shift; if (rxr >= w) rxr = w-1;
            int rxb = sx - shift; if (rxb < 0)  rxb = 0;

            const uint8_t *sR = tmp + sy * w * 3 + rxr * 3;
            const uint8_t *sG = tmp + sy * w * 3 + sx  * 3;
            const uint8_t *sB = tmp + sy * w * 3 + rxb * 3;
            uint8_t r = sR[2], g = sG[1], b = sB[0];

            /* Scanline darken every other row */
            int dark = ((y & 1) == 0) ? (int)(strength * 60) : 0;
            dst[x*3+0] = clamp_u8(b - dark);
            dst[x*3+1] = clamp_u8(g - dark);
            dst[x*3+2] = clamp_u8(r - dark);
        }
    }
    free(tmp);
}

/* ── velvet: desat + warm tint + soft blur (Soft) ─────────────── */
static void fx_velvet(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;

    /* 3x3 box blur in place using a one-row scratch buffer to avoid
     * reading freshly-written pixels (cheap separable approximation). */
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        int y0 = y > 0 ? y - 1 : 0;
        int y1 = y < h - 1 ? y + 1 : h - 1;
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            int x0 = x > 0 ? x - 1 : 0;
            int x1 = x < w - 1 ? x + 1 : w - 1;
            int sb = 0, sg = 0, sr = 0;
            for (int yy = y0; yy <= y1; yy++) {
                for (int xx = x0; xx <= x1; xx++) {
                    const uint8_t *p = tmp + yy * w * 3 + xx * 3;
                    sb += p[0]; sg += p[1]; sr += p[2];
                }
            }
            int n = (y1 - y0 + 1) * (x1 - x0 + 1);
            int bb = sb / n, gg = sg / n, rr = sr / n;

            /* Desat toward luma */
            int luma = (rr*299 + gg*587 + bb*114) / 1000;
            float desat = 0.5f * strength;
            bb = (int)(bb * (1 - desat) + luma * desat);
            gg = (int)(gg * (1 - desat) + luma * desat);
            rr = (int)(rr * (1 - desat) + luma * desat);

            /* Warm tint */
            rr = clamp_u8(rr + (int)(20 * strength));
            gg = clamp_u8(gg + (int)( 8 * strength));
            bb = clamp_u8(bb - (int)(15 * strength));

            /* Blend with original by strength so '0' = no-op */
            const uint8_t *orig = tmp + y * w * 3 + x * 3;
            dst[x*3+0] = (uint8_t)(orig[0] * (1 - strength) + bb * strength);
            dst[x*3+1] = (uint8_t)(orig[1] * (1 - strength) + gg * strength);
            dst[x*3+2] = (uint8_t)(orig[2] * (1 - strength) + rr * strength);
        }
    }
    free(tmp);
}

/* ── noir: high-contrast B&W with split-tone shadows (Themed) ── */
static void fx_noir(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int n = f->width * f->height;
    for (int i = 0; i < n; i++) {
        uint8_t *p = f->data + i * 3;
        int luma = (p[2]*299 + p[1]*587 + p[0]*114) / 1000;

        /* Sigmoid-ish contrast boost around 128 */
        float l = luma / 255.0f;
        l = (l - 0.5f) * (1.0f + strength * 1.2f) + 0.5f;
        if (l < 0) l = 0;
        if (l > 1) l = 1;
        int v = (int)(l * 255);

        /* Split-tone: cool blue in shadows, warm sepia in highlights */
        int rb, gg, bb;
        if (v < 128) {
            float t = v / 128.0f;
            rb = (int)(v * (0.85f + 0.15f * t));
            gg = (int)(v * (0.95f + 0.05f * t));
            bb = (int)(v * (1.15f - 0.05f * t));
        } else {
            float t = (v - 128) / 127.0f;
            rb = (int)(v * (1.10f + 0.05f * t));
            gg = (int)(v * (1.00f));
            bb = (int)(v * (0.85f - 0.05f * t));
        }

        /* Mix with original by strength */
        p[0] = clamp_u8((int)(p[0] * (1 - strength) + bb * strength));
        p[1] = clamp_u8((int)(p[1] * (1 - strength) + gg * strength));
        p[2] = clamp_u8((int)(p[2] * (1 - strength) + rb * strength));
    }
}

/* ── julia: Julia-fractal coloured overlay blended with image (Math) ── */
static void fx_julia(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;

    /* Animate the constant c slowly over time */
    unsigned long t = anim_tick();
    float ang = t * 0.01f;
    float cr = 0.7885f * cosf(ang);
    float ci = 0.7885f * sinf(ang);

    const int max_iter = 32;
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float zy0 = (y - h / 2.0f) * 3.0f / h;
        for (int x = 0; x < w; x++) {
            float zx = (x - w / 2.0f) * 3.0f / w;
            float zy = zy0;
            int iter = 0;
            while (iter < max_iter && zx*zx + zy*zy < 4.0f) {
                float nzx = zx*zx - zy*zy + cr;
                zy = 2 * zx * zy + ci;
                zx = nzx;
                iter++;
            }
            /* Map iter to a colour gradient (HSV-ish) */
            float fi = iter / (float)max_iter;
            int rr = (int)(255 * fi);
            int gg = (int)(255 * sinf(fi * 3.14159f));
            int bb = (int)(255 * (1 - fi));

            /* Blend with the image by strength */
            dst[x*3+0] = clamp_u8((int)(dst[x*3+0] * (1 - strength) + bb * strength));
            dst[x*3+1] = clamp_u8((int)(dst[x*3+1] * (1 - strength) + gg * strength));
            dst[x*3+2] = clamp_u8((int)(dst[x*3+2] * (1 - strength) + rr * strength));
        }
    }
}

/* ── scanlines: hard black lines every N rows (Pixel-crazy) ─── */
static void fx_scanlines(Frame *f, float gap)
{
    int g = (int)gap;
    if (g < 2) g = 2;
    if (g > 16) g = 16;
    for (int y = 0; y < f->height; y++) {
        if ((y % g) == 0) {
            memset(f->data + y * f->stride, 0, f->width * 3);
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * Glitch-art filters — generative, stateful, time-varying.
 * These maintain per-filter persistent state across frames (in
 * function-local static variables) so they can animate. All state
 * survives until the process exits; toggling a filter on and off
 * resumes from where it left off, which is what users typically
 * expect for these effects.
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_shuffle ──
 * Roving pixel-shuffle blocks. Picks random regions of the frame,
 * gradually scrambles them over ~1.5 seconds, holds fully shuffled
 * briefly, then UNSHUFFLES back over another ~1.5 seconds. Multiple
 * regions active simultaneously (cap scales with intensity), each
 * on its own life cycle.
 *
 * intensity (0..1): caps how many blocks can be active at once, and
 * also sets the block size (higher intensity → larger blocks).
 *
 * The gradual shuffle uses partial-progress permutations: at progress
 * p in [0..1], we apply p × N random swaps of the block's pixels
 * (where N is enough for a full shuffle). The exact same swap
 * sequence is played back in reverse for the unshuffle phase, so
 * the block ends up pixel-perfect where it started — no smearing,
 * no loss.
 */
#define SHUFFLE_MAX_ACTIVE   16
#define SHUFFLE_BLOCK_MIN    32
#define SHUFFLE_BLOCK_MAX    96
#define SHUFFLE_PHASE_FRAMES 45   /* ~1.5s at 30fps for each of the 3 phases */
#define SHUFFLE_MAX_SWAPS   (SHUFFLE_BLOCK_MAX * SHUFFLE_BLOCK_MAX / 2)

typedef struct {
    int x, y;             /* top-left of block */
    int size;             /* actual block side in px */
    int frame;            /* frames since spawn, counts up */
    /* Fixed swap sequence generated at spawn. Each entry is a pair
     * of pixel indices to swap. Playing the whole sequence forward
     * fully shuffles the block; playing it in reverse un-shuffles it.
     * We store pairs as uint16_t indices packed 2 per uint32_t. */
    uint32_t swaps[SHUFFLE_MAX_SWAPS];
    int      n_swaps;
} ShuffleBlock;

/* Apply the first k swaps of a block's swap sequence to its pixels
 * (in place). Passing k=0 leaves the block untouched. Passing
 * k=n_swaps fully shuffles it. Passing the same k twice = no-op
 * because each swap is its own inverse — so playing 0..k forward
 * followed by 0..k reverse restores original. */
static void shuffle_apply_swaps(uint8_t *tmp, int size,
                                const uint32_t *swaps, int k)
{
    for (int i = 0; i < k; i++) {
        int a = (int)(swaps[i] & 0xFFFF);
        int b = (int)((swaps[i] >> 16) & 0xFFFF);
        uint8_t *pa = tmp + a * 3;
        uint8_t *pb = tmp + b * 3;
        uint8_t t0 = pa[0], t1 = pa[1], t2 = pa[2];
        pa[0] = pb[0]; pa[1] = pb[1]; pa[2] = pb[2];
        pb[0] = t0;    pb[1] = t1;    pb[2] = t2;
    }
    (void)size;
}

/* shuffle v2:
 *   size_px (p2): block size override, 16..96 (0 → scale by intensity)
 *   phase_s (p3): seconds per phase (shuffle-in / hold / unshuffle),
 *                 0.3..5 (0 → 1.5) */
static void fx_shuffle(Frame *f, float intensity, float size_px, float phase_s)
{
    static ShuffleBlock blocks[SHUFFLE_MAX_ACTIVE];
    static int n_active = 0;

    if (intensity < 0.0f) intensity = 0.0f;
    if (intensity > 1.0f) intensity = 1.0f;

    int max_active = 1 + (int)(intensity * (SHUFFLE_MAX_ACTIVE - 1));
    int block_size;
    if (size_px >= 16.0f) {
        block_size = (int)size_px;
        if (block_size > SHUFFLE_BLOCK_MAX) block_size = SHUFFLE_BLOCK_MAX;
    } else {
        block_size = SHUFFLE_BLOCK_MIN
                   + (int)(intensity * (SHUFFLE_BLOCK_MAX - SHUFFLE_BLOCK_MIN));
    }
    block_size = (block_size / 4) * 4;

    int phase_frames = SHUFFLE_PHASE_FRAMES;
    if (phase_s >= 0.3f && phase_s <= 5.0f)
        phase_frames = (int)(phase_s * 30.0f);
    const int total_life = 3 * phase_frames;

    /* Age & evict */
    int kept = 0;
    for (int i = 0; i < n_active; i++) {
        blocks[i].frame++;
        if (blocks[i].frame < total_life) {
            if (kept != i) blocks[kept] = blocks[i];
            kept++;
        }
    }
    n_active = kept;

    /* Spawn new blocks: max 1 per frame, low probability so it doesn't
     * fill up in a burst. Roughly every 2 seconds one new block on
     * average at intensity 0.5. */
    if (n_active < max_active && (rand() % 60) < 1 + (int)(intensity * 4)) {
        int cols = (f->width  - block_size) / block_size;
        int rows = (f->height - block_size) / block_size;
        if (cols >= 1 && rows >= 1) {
            ShuffleBlock *b = &blocks[n_active++];
            b->x = (rand() % cols) * block_size;
            b->y = (rand() % rows) * block_size;
            b->size = block_size;
            b->frame = 0;

            /* Generate the swap sequence. Roughly N/2 random-pair
             * swaps is enough to fully shuffle N pixels (Fisher–Yates
             * uses N swaps, but they're not independent — for a
             * pairwise-swap "reversible shuffle" N/2 is enough to
             * make the result look uniformly random). */
            int npx = block_size * block_size;
            int n_swaps = npx / 2;
            if (n_swaps > SHUFFLE_MAX_SWAPS) n_swaps = SHUFFLE_MAX_SWAPS;
            b->n_swaps = n_swaps;
            for (int j = 0; j < n_swaps; j++) {
                int a = rand() % npx;
                int c = rand() % npx;
                b->swaps[j] = ((uint32_t)c << 16) | (uint32_t)a;
            }
        }
    }

    /* Scratch: big enough for the biggest possible block. */
    static uint8_t tmp[SHUFFLE_BLOCK_MAX * SHUFFLE_BLOCK_MAX * 3];

    for (int i = 0; i < n_active; i++) {
        ShuffleBlock *b = &blocks[i];
        int size = b->size;
        if (b->x + size > f->width || b->y + size > f->height) continue;

        /* Determine how many swaps to apply based on which phase we're in.
         *   Phase 0 (frame 0             .. PHASE_FRAMES-1):   shuffle IN, 0 → n_swaps
         *   Phase 1 (PHASE_FRAMES        .. 2*PHASE_FRAMES-1): hold fully shuffled
         *   Phase 2 (2*PHASE_FRAMES      .. 3*PHASE_FRAMES-1): unshuffle,  n_swaps → 0
         */
        int k;
        if (b->frame < phase_frames) {
            k = (b->frame * b->n_swaps) / phase_frames;
        } else if (b->frame < 2 * phase_frames) {
            k = b->n_swaps;
        } else {
            int rev = b->frame - 2 * phase_frames;
            k = b->n_swaps - (rev * b->n_swaps) / phase_frames;
        }
        if (k <= 0) continue;  /* no-op frame (shouldn't change pixels) */

        /* Copy the current block into tmp, apply k swaps, copy back. */
        for (int by = 0; by < size; by++) {
            uint8_t *src = f->data + (b->y + by) * f->stride + b->x * 3;
            memcpy(tmp + by * size * 3, src, (size_t)size * 3);
        }
        shuffle_apply_swaps(tmp, size, b->swaps, k);
        for (int by = 0; by < size; by++) {
            uint8_t *dst = f->data + (b->y + by) * f->stride + b->x * 3;
            memcpy(dst, tmp + by * size * 3, (size_t)size * 3);
        }
    }
}

/* ─── fx_pixelsort ──
 * The classic pixel-sort glitch effect. Picks random horizontal spans
 * across random rows and sorts the pixels by luminance, creating
 * the unmistakable "vertical bleed" look used in glitch-art.
 *
 * density (0..1): how many active sort lines to maintain at once.
 */
#define PIXELSORT_MAX_ACTIVE 24

typedef struct {
    int row, x0, x1;
    int descending;       /* 0 = light→dark, 1 = dark→light */
    int frames_left;
} SortLine;

static int sort_cmp_asc(const void *a, const void *b)
{
    const uint8_t *pa = a, *pb = b;
    int la = pa[0] + 2 * pa[1] + pa[2];  /* B + 2G + R, cheap luma */
    int lb = pb[0] + 2 * pb[1] + pb[2];
    return la - lb;
}
static int sort_cmp_desc(const void *a, const void *b)
{
    return sort_cmp_asc(b, a);
}

static void fx_pixelsort(Frame *f, float density)
{
    static SortLine lines[PIXELSORT_MAX_ACTIVE];
    static int n_active = 0;

    if (density < 0.0f) density = 0.0f;
    if (density > 1.0f) density = 1.0f;
    int max_active = (int)(density * PIXELSORT_MAX_ACTIVE);
    if (max_active < 1) max_active = 1;

    int kept = 0;
    for (int i = 0; i < n_active; i++) {
        lines[i].frames_left--;
        if (lines[i].frames_left > 0) {
            if (kept != i) lines[kept] = lines[i];
            kept++;
        }
    }
    n_active = kept;

    int spawn_budget = 2;
    while (n_active < max_active && spawn_budget-- > 0 && (rand() % 100) < 40) {
        SortLine *l = &lines[n_active++];
        l->row = rand() % f->height;
        int len = 80 + rand() % (f->width / 2 + 1);
        l->x0 = rand() % (f->width > len ? f->width - len : 1);
        l->x1 = l->x0 + len;
        if (l->x1 > f->width) l->x1 = f->width;
        l->descending  = rand() & 1;
        l->frames_left = 3 + (rand() % 10);
    }

    for (int i = 0; i < n_active; i++) {
        SortLine *l = &lines[i];
        uint8_t *row = f->data + l->row * f->stride + l->x0 * 3;
        int n = l->x1 - l->x0;
        if (n < 2) continue;
        qsort(row, (size_t)n, 3, l->descending ? sort_cmp_desc : sort_cmp_asc);
    }
}

/* ─── fx_rgb_delay ──
 * Chromatic lag. Each color channel comes from a different point in
 * recent history: B is "live", G is 1 frame behind, R is up to N
 * frames behind (where N grows with strength). When you move, you
 * leave a green and red "ghost" trail. Subtle at strength=0.3,
 * trippy at strength=1.0.
 *
 * Stores past frames in a 4-deep ring buffer that grows/shrinks
 * with the frame size. Memory: 4 × stride × height ≈ 11 MB for 720p.
 */
typedef struct {
    uint8_t *frames[4];
    int      frame_w, frame_h, stride;
    int      write_idx;
    int      filled;       /* how many slots have been written into */
} RGBDelayState;

static RGBDelayState g_rgb_delay = {{0}, 0, 0, 0, 0, 0};

static void fx_rgb_delay(Frame *f, float strength)
{
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    int max_delay = 1 + (int)(strength * 2.5f);   /* 1..3 frames */
    if (max_delay > 3) max_delay = 3;

    int size = f->stride * f->height;

    /* (Re)allocate if first call or size changed. */
    if (g_rgb_delay.frame_w != f->width ||
        g_rgb_delay.frame_h != f->height ||
        g_rgb_delay.stride  != f->stride) {
        for (int i = 0; i < 4; i++) {
            free(g_rgb_delay.frames[i]);
            g_rgb_delay.frames[i] = malloc((size_t)size);
            if (!g_rgb_delay.frames[i]) {
                /* OOM — leave everything cleared, skip filter this frame */
                for (int j = 0; j < 4; j++) {
                    free(g_rgb_delay.frames[j]);
                    g_rgb_delay.frames[j] = NULL;
                }
                g_rgb_delay.frame_w = g_rgb_delay.frame_h = g_rgb_delay.stride = 0;
                return;
            }
            memcpy(g_rgb_delay.frames[i], f->data, (size_t)size);
        }
        g_rgb_delay.frame_w   = f->width;
        g_rgb_delay.frame_h   = f->height;
        g_rgb_delay.stride    = f->stride;
        g_rgb_delay.write_idx = 0;
        g_rgb_delay.filled    = 4;
    }

    /* Store the CURRENT (unmodified) frame in the ring first, so the
     * G/R reads further down pull from a clean history rather than
     * compounding their own modifications across frames. */
    int w = g_rgb_delay.write_idx;
    memcpy(g_rgb_delay.frames[w], f->data, (size_t)size);

    int idx_g = (w - 1 + 4) % 4;
    int idx_r = (w - max_delay + 4) % 4;

    uint8_t *cur   = f->data;
    uint8_t *gpast = g_rgb_delay.frames[idx_g];
    uint8_t *rpast = g_rgb_delay.frames[idx_r];

    int n = f->width * f->height;
    for (int i = 0; i < n; i++) {
        int o = i * 3;
        /* cur[o+0] = B (live, untouched) */
        cur[o + 1] = gpast[o + 1];   /* G from 1 frame ago     */
        cur[o + 2] = rpast[o + 2];   /* R from max_delay back  */
    }

    g_rgb_delay.write_idx = (w + 1) % 4;
}

/* ─── fx_drift ──
 * Horizontal scanline drift. The image is split into bands of
 * configurable height, and each band gets its own slowly-drifting
 * horizontal offset. Looks like a misaligned VHS or a corrupted
 * data stream. Each band's target offset is randomised every few
 * frames; current offsets ease toward target so the motion is
 * smooth rather than jittery.
 *
 * param: band height in pixels (clamped 2..32, default 8).
 */
typedef struct {
    int    band_h;
    int    n_bands;
    float *offset;        /* current x-offset, smoothed */
    float *target;        /* target x-offset, updated periodically */
    int    frame_w, frame_h;
} DriftState;

static DriftState g_drift = {0, 0, NULL, NULL, 0, 0};

/* drift v2: max_shift (p2) px 5..120 (0 → 20 with occasional 60 jolts) */
static void fx_drift(Frame *f, float param, float max_shift_f)
{
    int band_h = (int)param;
    if (band_h < 2)  band_h = 2;
    if (band_h > 32) band_h = 32;

    if (g_drift.frame_w  != f->width ||
        g_drift.frame_h  != f->height ||
        g_drift.band_h   != band_h) {
        free(g_drift.offset);
        free(g_drift.target);
        g_drift.n_bands = (f->height + band_h - 1) / band_h;
        g_drift.offset = calloc((size_t)g_drift.n_bands, sizeof(float));
        g_drift.target = calloc((size_t)g_drift.n_bands, sizeof(float));
        g_drift.band_h  = band_h;
        g_drift.frame_w = f->width;
        g_drift.frame_h = f->height;
        if (!g_drift.offset || !g_drift.target) {
            free(g_drift.offset); free(g_drift.target);
            g_drift.offset = g_drift.target = NULL;
            g_drift.frame_w = g_drift.frame_h = 0;
            return;
        }
    }
    if (!g_drift.offset || !g_drift.target) return;

    /* Every ~8 frames, randomly pick new targets for a fraction of
     * the bands. Small magnitudes most of the time, occasional jolts. */
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 8 == 0) {
        for (int i = 0; i < g_drift.n_bands; i++) {
            if (rand() % 5 == 0) {
                int base = 20, big = 60;
                if (max_shift_f >= 5.0f && max_shift_f <= 120.0f) {
                    big  = (int)max_shift_f;
                    base = big / 3 > 2 ? big / 3 : 2;
                }
                int jolt = (rand() % 7 == 0) ? big : base;
                g_drift.target[i] = (float)((rand() % (2 * jolt + 1)) - jolt);
            }
        }
    }

    /* Ease current → target each frame */
    for (int i = 0; i < g_drift.n_bands; i++)
        g_drift.offset[i] = g_drift.offset[i] * 0.85f + g_drift.target[i] * 0.15f;

    /* Apply: for each row, shift by its band's offset. Use a scratch
     * row buffer so we don't read-after-write. */
    uint8_t *tmp = malloc((size_t)f->stride);
    if (!tmp) return;

    for (int y = 0; y < f->height; y++) {
        int band = y / band_h;
        if (band >= g_drift.n_bands) band = g_drift.n_bands - 1;
        int offset = (int)g_drift.offset[band];
        if (offset == 0) continue;

        uint8_t *row = f->data + y * f->stride;
        memcpy(tmp, row, (size_t)f->stride);
        for (int x = 0; x < f->width; x++) {
            int sx = x - offset;
            if (sx < 0) sx = 0;
            if (sx >= f->width) sx = f->width - 1;
            row[x * 3 + 0] = tmp[sx * 3 + 0];
            row[x * 3 + 1] = tmp[sx * 3 + 1];
            row[x * 3 + 2] = tmp[sx * 3 + 2];
        }
    }
    free(tmp);
}

/* ─── fx_rain ──
 * Digital "Matrix"-style rain. Random columns of the frame get a
 * bright-green head moving downward, with a fading trail behind it.
 * The webcam content shows through (it's tinted, not replaced), so
 * it overlays cleanly on top of the rest of the pipeline.
 *
 * density (0..1): how many simultaneous rain columns to spawn.
 */
typedef struct {
    int *head;       /* y-position of leading edge, -1 = inactive */
    int *speed;      /* pixels per frame for active columns */
    int frame_w, frame_h;
} RainState;

static RainState g_rain = {NULL, NULL, 0, 0};

/* rain v2: speed_mult (p2) 0.3..3 (0 → 1); trail (p3) px 8..80 (0 → 28) */
static void fx_rain(Frame *f, float density, float speed_mult, float trail_f)
{
    if (density < 0.0f) density = 0.0f;
    if (density > 1.0f) density = 1.0f;

    if (g_rain.frame_w != f->width || g_rain.frame_h != f->height) {
        free(g_rain.head);
        free(g_rain.speed);
        g_rain.head  = malloc((size_t)f->width * sizeof(int));
        g_rain.speed = malloc((size_t)f->width * sizeof(int));
        g_rain.frame_w = f->width;
        g_rain.frame_h = f->height;
        if (!g_rain.head || !g_rain.speed) {
            free(g_rain.head); free(g_rain.speed);
            g_rain.head = g_rain.speed = NULL;
            g_rain.frame_w = g_rain.frame_h = 0;
            return;
        }
        for (int x = 0; x < f->width; x++) {
            g_rain.head[x] = -1;
            g_rain.speed[x] = 0;
        }
    }
    if (!g_rain.head) return;

    /* Spawn up to target_active simultaneous trails. Density 1.0
     * roughly means 1 trail per 20 columns active. */
    int target_active = (int)((float)f->width * density / 20.0f);
    int active = 0;
    for (int x = 0; x < f->width; x++) if (g_rain.head[x] >= 0) active++;

    int spawn_attempts = (target_active - active) * 2;
    if (spawn_attempts > 32) spawn_attempts = 32;
    for (int i = 0; i < spawn_attempts; i++) {
        int x = rand() % f->width;
        if (g_rain.head[x] < 0) {
            g_rain.head[x]  = 0;
            float sm = (speed_mult >= 0.3f && speed_mult <= 3.0f) ? speed_mult : 1.0f;
            g_rain.speed[x] = 1 + (int)((2 + rand() % 7) * sm);
        }
    }

    /* Apply each trail as a luma-tinting overlay */
    int trail_len = 28;
    if (trail_f >= 8.0f && trail_f <= 80.0f) trail_len = (int)trail_f;
    for (int x = 0; x < f->width; x++) {
        if (g_rain.head[x] < 0) continue;
        int head_y = g_rain.head[x];

        for (int t = 0; t < trail_len; t++) {
            int y = head_y - t;
            if (y < 0 || y >= f->height) continue;
            uint8_t *p = f->data + y * f->stride + x * 3;

            /* Fade is 1.0 at head, 0 at tail. Head is white-green;
             * trail becomes pure green, fading to nothing. */
            float fade = 1.0f - (float)t / (float)trail_len;
            float g_boost = fade * 220.0f;
            float dim     = 1.0f - fade * 0.7f;

            int g = (int)((float)p[1] * dim + g_boost);
            if (g > 255) g = 255;
            p[0] = (uint8_t)((float)p[0] * dim);
            p[1] = (uint8_t)g;
            p[2] = (uint8_t)((float)p[2] * dim);

            /* Head pixel: pure white for a single-frame "spark" */
            if (t == 0) { p[0] = 240; p[1] = 255; p[2] = 240; }
        }

        g_rain.head[x] += g_rain.speed[x];
        if (g_rain.head[x] > f->height + trail_len) {
            g_rain.head[x] = -1;
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * Temporal filters — rgbdelay's siblings. All maintain persistent
 * frame history and re-allocate transparently on resolution change.
 * ────────────────────────────────────────────────────────────── */

/* Shared helper: (re)allocate a persistent BGR frame buffer to match
 * the current frame, seeding it with the current content on (re)alloc.
 * Returns NULL on OOM. */
static uint8_t *temporal_buf(uint8_t **buf, int *bw, int *bh, int *bs,
                             const Frame *f)
{
    int size = f->stride * f->height;
    if (*bw != f->width || *bh != f->height || *bs != f->stride) {
        free(*buf);
        *buf = malloc((size_t)size);
        if (!*buf) { *bw = *bh = *bs = 0; return NULL; }
        memcpy(*buf, f->data, (size_t)size);
        *bw = f->width; *bh = f->height; *bs = f->stride;
    }
    return *buf;
}

/* ─── fx_echo ──
 * Motion trails. Keeps an exponential moving average of past frames
 * and blends the current frame into it. Anything that moves leaves a
 * fading ghost behind. strength (0..1): trail persistence — 0.3 is a
 * subtle smear, 0.9 is full "light painting".
 */
static void fx_echo(Frame *f, float strength)
{
    static uint8_t *acc = NULL;
    static int aw = 0, ah = 0, as = 0;

    if (strength < 0.0f) strength = 0.0f;
    if (strength > 0.97f) strength = 0.97f;

    if (!temporal_buf(&acc, &aw, &ah, &as, f)) return;

    int n = f->stride * f->height;
    float keep = strength, mix = 1.0f - strength;
    for (int i = 0; i < n; i++) {
        int v = (int)(acc[i] * keep + f->data[i] * mix);
        acc[i] = (uint8_t)v;
        f->data[i] = acc[i];
    }
}

/* ─── fx_slitscan ──
 * Time gradient: the top row of the output is "now", and each row
 * further down samples a progressively OLDER frame from a ring
 * buffer. Wave your hand and it bends through time like taffy.
 * depth (0..1): how far back the bottom row reaches (up to the ring
 * size). One of the all-time great video-art effects.
 */
#define SLITSCAN_RING 16

/* slitscan v2: dir (p2): 0 = top is 'now' (default), 1 = bottom is
 * 'now' (time flows upward) */
static void fx_slitscan(Frame *f, float depth, float dir)
{
    static uint8_t *ring[SLITSCAN_RING] = {0};
    static int rw = 0, rh = 0, rs = 0, widx = 0;

    if (depth < 0.0f) depth = 0.0f;
    if (depth > 1.0f) depth = 1.0f;

    int size = f->stride * f->height;

    if (rw != f->width || rh != f->height || rs != f->stride) {
        for (int i = 0; i < SLITSCAN_RING; i++) {
            free(ring[i]);
            ring[i] = malloc((size_t)size);
            if (!ring[i]) {
                for (int j = 0; j <= i; j++) { free(ring[j]); ring[j] = NULL; }
                rw = rh = rs = 0;
                return;
            }
            memcpy(ring[i], f->data, (size_t)size);
        }
        rw = f->width; rh = f->height; rs = f->stride;
        widx = 0;
    }

    /* Store current frame */
    memcpy(ring[widx], f->data, (size_t)size);

    int max_back = (int)(depth * (SLITSCAN_RING - 1));
    if (max_back < 1) max_back = 1;

    int flip = dir >= 0.5f;
    for (int y = 0; y < f->height; y++) {
        /* Row y reads from (y/h)*max_back frames ago */
        int ry = flip ? (f->height - 1 - y) : y;
        int back = (ry * max_back) / (f->height > 1 ? f->height - 1 : 1);
        int idx = (widx - back + SLITSCAN_RING) % SLITSCAN_RING;
        memcpy(f->data + y * f->stride,
               ring[idx] + y * f->stride,
               (size_t)f->width * 3);
    }

    widx = (widx + 1) % SLITSCAN_RING;
}

/* ─── fx_mosh ──
 * Fake datamosh. Keeps a "held" frame and only lets pixels through
 * when they've changed enough since the hold — so static background
 * freezes and smears while motion punches through, like a corrupted
 * P-frame stream. Every ~90 frames the hold refreshes (a fake
 * "I-frame") so it doesn't diverge forever.
 * sensitivity (0..1): lower = more smear (higher change threshold).
 */
/* mosh v2: keyint (p2) frames between auto-resync keyframes,
 * 30..600 (0 → 90). Longer = deeper smear before reality returns. */
static void fx_mosh(Frame *f, float sensitivity, float keyint_f)
{
    static uint8_t *held = NULL;
    static int hw = 0, hh = 0, hs = 0;
    static int frames_since_key = 0;

    if (sensitivity < 0.0f) sensitivity = 0.0f;
    if (sensitivity > 1.0f) sensitivity = 1.0f;
    int thresh = 12 + (int)((1.0f - sensitivity) * 80.0f);  /* 12..92 */

    if (!temporal_buf(&held, &hw, &hh, &hs, f)) return;

    int keyint = 90;
    if (keyint_f >= 30.0f && keyint_f <= 600.0f) keyint = (int)keyint_f;

    /* Periodic keyframe: resync so the mosh doesn't decay into mud */
    if (++frames_since_key >= keyint) {
        memcpy(held, f->data, (size_t)(f->stride * f->height));
        frames_since_key = 0;
        return;
    }

    for (int y = 0; y < f->height; y++) {
        uint8_t *cur = f->data + y * f->stride;
        uint8_t *hld = held    + y * f->stride;
        for (int x = 0; x < f->width; x++) {
            int o = x * 3;
            int d = abs(cur[o] - hld[o]) + abs(cur[o+1] - hld[o+1])
                  + abs(cur[o+2] - hld[o+2]);
            if (d > thresh) {
                /* Motion: let it through and update the hold */
                hld[o]   = cur[o];
                hld[o+1] = cur[o+1];
                hld[o+2] = cur[o+2];
            } else {
                /* Static: keep showing the held (possibly stale) pixel */
                cur[o]   = hld[o];
                cur[o+1] = hld[o+1];
                cur[o+2] = hld[o+2];
            }
        }
    }
}

/* ─── fx_stutter ──
 * Temporal pixelation: capture a frame, freeze it for N frames, then
 * capture the next. The classic music-video stop-motion stutter.
 * period: frames to hold each capture (2..30, default 6).
 */
static void fx_stutter(Frame *f, float period)
{
    static uint8_t *held = NULL;
    static int hw = 0, hh = 0, hs = 0;
    static int countdown = 0;

    int p = (int)period;
    if (p < 2)  p = 2;
    if (p > 30) p = 30;

    int size = f->stride * f->height;
    if (hw != f->width || hh != f->height || hs != f->stride) {
        free(held);
        held = malloc((size_t)size);
        if (!held) { hw = hh = hs = 0; return; }
        memcpy(held, f->data, (size_t)size);
        hw = f->width; hh = f->height; hs = f->stride;
        countdown = p;
    }

    if (--countdown <= 0) {
        memcpy(held, f->data, (size_t)size);  /* grab a fresh frame */
        countdown = p;
    } else {
        memcpy(f->data, held, (size_t)size);  /* keep showing the hold */
    }
}

/* ─── fx_halftone ──
 * Newspaper print: divide into cells, average each cell's color, and
 * draw a filled disc whose radius scales with brightness on a dark
 * background. CMYK-ish pop look, works great over motion.
 * cell: cell size in px (4..32, default 8).
 */
static void fx_halftone(Frame *f, float cell_f)
{
    int cell = (int)cell_f;
    if (cell < 4)  cell = 4;
    if (cell > 32) cell = 32;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int cy = 0; cy < h; cy += cell) {
        for (int cx = 0; cx < w; cx += cell) {
            int cw = (cx + cell > w) ? w - cx : cell;
            int ch = (cy + cell > h) ? h - cy : cell;

            /* average color + luma of the cell */
            long sb = 0, sg = 0, sr = 0;
            for (int y = 0; y < ch; y++) {
                const uint8_t *p = tmp + (cy+y) * w * 3 + cx * 3;
                for (int x = 0; x < cw; x++, p += 3) {
                    sb += p[0]; sg += p[1]; sr += p[2];
                }
            }
            int npx = cw * ch;
            int ab = (int)(sb / npx), ag = (int)(sg / npx), ar = (int)(sr / npx);
            int luma = (ab * 114 + ag * 587 + ar * 299) / 1000;

            /* disc radius proportional to brightness */
            float max_r = cell * 0.5f;
            float r = max_r * (float)luma / 255.0f;
            float r2 = r * r;
            float ccx = cx + cw / 2.0f, ccy = cy + ch / 2.0f;

            for (int y = 0; y < ch; y++) {
                uint8_t *p = f->data + (cy+y) * f->stride + cx * 3;
                for (int x = 0; x < cw; x++, p += 3) {
                    float dx = (cx + x) - ccx, dy = (cy + y) - ccy;
                    if (dx*dx + dy*dy <= r2) {
                        p[0] = (uint8_t)ab; p[1] = (uint8_t)ag; p[2] = (uint8_t)ar;
                    } else {
                        p[0] = 12; p[1] = 10; p[2] = 14;   /* near-black paper */
                    }
                }
            }
        }
    }
    free(tmp);
}

/* ─── fx_neon v2 ──
 * Glowing colored edges over a darkened scene, now with real bloom.
 *   strength (p1): glow intensity + background darkening, 0..1
 *   thresh   (p2): edge sensitivity 10..120 (lower = more lines; 0 → 40)
 *   huespeed (p3): rainbow drift rate (0 → 1.0)
 *   glow     (p4): bloom spread 0..1 (0 → 0.5) — edges bleed light
 * Edge hue still comes from gradient direction, so contours sweep
 * through the spectrum as they curve.
 */
static void fx_neon(Frame *f, float strength, float thresh_f,
                    float huespeed, float glow)
{
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    int thresh = (int)thresh_f;
    if (thresh <= 0) thresh = 40;
    if (thresh < 10) thresh = 10;
    if (thresh > 120) thresh = 120;
    if (huespeed <= 0.0f) huespeed = 1.0f;
    if (glow < 0.0f) glow = 0.0f;
    if (glow == 0.0f) glow = 0.5f;
    if (glow > 1.0f) glow = 1.0f;

    int w = f->width, h = f->height;
    uint8_t *luma = malloc((size_t)w * h);
    uint8_t *ebuf = calloc((size_t)w * h * 3, 1);   /* edge-light layer */
    if (!luma || !ebuf) { free(luma); free(ebuf); return; }

    for (int y = 0; y < h; y++) {
        const uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3)
            luma[y*w + x] = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
    }

    unsigned long t = anim_tick();
    float hue_drift = (float)(t % 6283) * 0.001f * huespeed;

    /* Pass 1: edge light into ebuf */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = -luma[(y-1)*w + (x-1)] + luma[(y-1)*w + (x+1)]
                     -2*luma[y*w + (x-1)]   + 2*luma[y*w + (x+1)]
                     -luma[(y+1)*w + (x-1)] + luma[(y+1)*w + (x+1)];
            int gy = -luma[(y-1)*w + (x-1)] - 2*luma[(y-1)*w + x] - luma[(y-1)*w + (x+1)]
                     +luma[(y+1)*w + (x-1)] + 2*luma[(y+1)*w + x] + luma[(y+1)*w + (x+1)];
            int mag = abs(gx) + abs(gy);
            if (mag <= thresh) continue;
            float ang = atan2f((float)gy, (float)gx) + hue_drift;
            float e = (float)(mag > 255 ? 255 : mag) / 255.0f * strength;
            uint8_t *ep = ebuf + (y*w + x) * 3;
            ep[0] = clamp_u8((int)((sinf(ang)          * 0.5f + 0.5f) * 255.0f * e));
            ep[1] = clamp_u8((int)((sinf(ang + 2.094f) * 0.5f + 0.5f) * 255.0f * e));
            ep[2] = clamp_u8((int)((sinf(ang + 4.189f) * 0.5f + 0.5f) * 255.0f * e));
        }
    }

    /* Pass 2: composite — darken base, add edge light + bloom taps */
    float dim = 1.0f - strength * 0.85f;
    int gpix = (int)(glow * 96.0f);   /* bloom contribution scale */
    for (int y = 2; y < h - 2; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 2; x < w - 2; x++) {
            int o = x * 3;
            const uint8_t *e0 = ebuf + (y*w + x) * 3;
            /* bloom: sample 4 neighbours at distance 2 */
            const uint8_t *e1 = ebuf + (y*w + x - 2) * 3;
            const uint8_t *e2 = ebuf + (y*w + x + 2) * 3;
            const uint8_t *e3 = ebuf + ((y-2)*w + x) * 3;
            const uint8_t *e4 = ebuf + ((y+2)*w + x) * 3;
            for (int c = 0; c < 3; c++) {
                int bloom = (e1[c] + e2[c] + e3[c] + e4[c]) * gpix / 255;
                dst[o+c] = clamp_u8((int)(dst[o+c] * dim) + e0[c] + bloom);
            }
        }
    }
    free(luma);
    free(ebuf);
}

/* ──────────────────────────────────────────────────────────────
 * Batch 3: retro & warp
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_emboss ──
 * Classic relief: diagonal difference + mid-gray. Face becomes a
 * pencil-carved bas-relief. strength 0..2.
 */
static void fx_emboss(Frame *f, float strength)
{
    if (strength < 0.1f) strength = 0.1f;
    if (strength > 2.0f) strength = 2.0f;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 1; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        const uint8_t *cur  = tmp + y * w * 3;
        const uint8_t *diag = tmp + (y-1) * w * 3;
        for (int x = 1; x < w; x++) {
            int o = x * 3, od = (x-1) * 3;
            /* luma difference against the up-left neighbour */
            int lc = (cur[o]  *114 + cur[o+1] *587 + cur[o+2] *299) / 1000;
            int ld = (diag[od]*114 + diag[od+1]*587 + diag[od+2]*299) / 1000;
            int v = 128 + (int)((lc - ld) * strength * 2.0f);
            uint8_t g = clamp_u8(v);
            dst[o] = dst[o+1] = dst[o+2] = g;
        }
    }
    free(tmp);
}

/* ─── fx_dither ──
 * Ordered Bayer 4×4 dithering to N levels per channel. levels=2 gives
 * the beloved 8-color retro-computer look with the characteristic
 * crosshatch texture; higher levels are subtler.
 */
static void fx_dither(Frame *f, float levels_f)
{
    int levels = (int)levels_f;
    if (levels < 2) levels = 2;
    if (levels > 8) levels = 8;

    /* 4x4 Bayer matrix, values 0..15 */
    static const int bayer[4][4] = {
        { 0,  8,  2, 10},
        {12,  4, 14,  6},
        { 3, 11,  1,  9},
        {15,  7, 13,  5},
    };

    int w = f->width, h = f->height;
    float step = 255.0f / (float)(levels - 1);

    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            /* threshold offset in [-0.5, +0.5) scaled to one level step */
            float t = ((float)bayer[y & 3][x & 3] / 16.0f - 0.5f) * step;
            for (int c = 0; c < 3; c++) {
                float v = (float)p[c] + t;
                int q = (int)((v / step) + 0.5f);
                if (q < 0) q = 0;
                if (q > levels - 1) q = levels - 1;
                p[c] = clamp_u8((int)(q * step));
            }
        }
    }
}

/* ─── fx_tiles ──
 * Mirror-tile grid: the frame becomes an n×n grid of downscaled
 * copies, alternately mirrored horizontally/vertically so edges meet
 * seamlessly — a wallpaper of yourself. n = tiles per row (2..8).
 */
static void fx_tiles(Frame *f, float n_f)
{
    int n = (int)n_f;
    if (n < 2) n = 2;
    if (n > 8) n = 8;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int ty = y * n;               /* position within a virtual n× tall image */
        int tile_y = ty / h;          /* which tile row */
        int sy = ty % h;              /* source y within the tile */
        if (tile_y & 1) sy = h - 1 - sy;  /* mirror odd rows */
        for (int x = 0; x < w; x++) {
            int tx = x * n;
            int tile_x = tx / w;
            int sx = tx % w;
            if (tile_x & 1) sx = w - 1 - sx;  /* mirror odd cols */
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0];
            dst[x*3+1] = src[1];
            dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ─── fx_zoompulse ──
 * The image breathes: zoom factor oscillates sinusoidally around 1.
 * strength (0..1) controls the amplitude (up to ±25% zoom).
 */
/* zoompulse v2: rate (p2) oscillation speed (0 → 1) */
static void fx_zoompulse(Frame *f, float strength, float rate)
{
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;

    unsigned long t = anim_tick();
    if (rate <= 0.0f) rate = 1.0f;
    float zoom = 1.0f + strength * 0.25f * sinf((float)t * 0.12f * rate);

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float inv = 1.0f / zoom;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float sy_f = cy + (y - cy) * inv;
        int sy = (int)sy_f;
        if (sy < 0) sy = 0;
        if (sy >= h) sy = h - 1;
        for (int x = 0; x < w; x++) {
            float sx_f = cx + (x - cx) * inv;
            int sx = (int)sx_f;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            const uint8_t *src = tmp + sy * w * 3 + sx * 3;
            dst[x*3+0] = src[0];
            dst[x*3+1] = src[1];
            dst[x*3+2] = src[2];
        }
    }
    free(tmp);
}

/* ─── fx_oilslick ──
 * Iridescence: each pixel's hue rotates by an angle keyed to its own
 * brightness plus a slow time drift — like light on an oil film or
 * soap bubble. Bright and dark areas swirl through different colors.
 * strength (0..1): how far hues rotate.
 */
/* oilslick v2: speed (p2) hue drift rate (0 → 1) */
static void fx_oilslick(Frame *f, float strength, float spd)
{
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;

    unsigned long t = anim_tick();
    if (spd <= 0.0f) spd = 1.0f;
    float drift = (float)(t % 62830) * 0.001f * spd;

    int w = f->width, h = f->height;
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            int B = p[0], G = p[1], R = p[2];
            int luma = (B*114 + G*587 + R*299) / 1000;

            /* rotation angle: luma sweep + time drift */
            float a = strength * ((float)luma * 0.045f + drift * 3.0f);
            float s = sinf(a), c = cosf(a);

            /* rotate the color vector in an isoluminant-ish plane:
             * cheap approximation using two channel-difference axes */
            float u = (float)(R - G) * 0.7071f;
            float v = (float)(B - (R + G) / 2) * 0.8165f;
            float u2 = u * c - v * s;
            float v2 = u * s + v * c;

            int nr = luma + (int)(u2 * 0.7071f * 1.0f) + (int)(v2 * -0.4082f);
            int ng = luma + (int)(u2 * -0.7071f)       + (int)(v2 * -0.4082f);
            int nb = luma + (int)(v2 * 0.8165f);

            p[0] = clamp_u8(nb);
            p[1] = clamp_u8(ng);
            p[2] = clamp_u8(nr);
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * Batch 4: interactive & extravagant
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_feedback ──
 * Video feedback — like pointing a camera at its own monitor. The
 * previous OUTPUT frame is zoomed and rotated around the center,
 * then the current camera frame is blended on top. Anything bright
 * spirals off into an infinite recursive tunnel behind you.
 *   decay (0..1): how much of the past survives each frame (0.85)
 *   zoom  (p2):   scale per frame, e.g. 1.03 pushes outward,
 *                 0.97 pulls inward; 0 → default 1.04
 *   rot   (p3):   degrees rotated per frame (default 1.5)
 */
static void fx_feedback(Frame *f, float decay, float zoom, float rot_deg)
{
    static uint8_t *prev = NULL;
    static int pw = 0, ph = 0, ps = 0;

    if (decay < 0.0f) decay = 0.0f;
    if (decay > 0.98f) decay = 0.98f;
    if (zoom <= 0.01f) zoom = 1.04f;
    if (rot_deg == 0.0f) rot_deg = 1.5f;

    int w = f->width, h = f->height;
    int size = f->stride * h;

    if (pw != w || ph != h || ps != f->stride) {
        free(prev);
        prev = malloc((size_t)size);
        if (!prev) { pw = ph = ps = 0; return; }
        memcpy(prev, f->data, (size_t)size);
        pw = w; ph = h; ps = f->stride;
    }

    float cx = w / 2.0f, cy = h / 2.0f;
    float ang = rot_deg * 3.14159265f / 180.0f;
    /* Inverse mapping: to fill dst pixel, sample prev at the
     * inverse transform (rotate -ang, scale 1/zoom). */
    float ca = cosf(-ang) / zoom, sa = sinf(-ang) / zoom;

    float keep = decay, mix = 1.0f - decay;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy0 = y - cy;
        for (int x = 0; x < w; x++) {
            float dx0 = x - cx;
            int sx = (int)(cx + dx0 * ca - dy0 * sa);
            int sy = (int)(cy + dx0 * sa + dy0 * ca);
            int o = x * 3;
            if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                const uint8_t *pp = prev + sy * f->stride + sx * 3;
                dst[o]   = (uint8_t)(dst[o]   * mix + pp[0] * keep);
                dst[o+1] = (uint8_t)(dst[o+1] * mix + pp[1] * keep);
                dst[o+2] = (uint8_t)(dst[o+2] * mix + pp[2] * keep);
            }
            /* outside prev: current frame stays as-is */
        }
    }
    memcpy(prev, f->data, (size_t)size);
}

/* ─── fx_ripples v2 — INTERACTIVE ──
 * Water-wave simulation driven by YOUR MOTION, now with weather.
 *   sensitivity (p1): how easily motion excites the water, 0..1
 *   rain        (p2): ambient droplets per second, 0 = off (try 3-8)
 *   refract     (p3): refraction strength (0 → 2.0; up to ~6 = glassy)
 * Simulation runs at 1/4 resolution.
 */
static void fx_ripples(Frame *f, float sensitivity, float rain,
                       float refract)
{
    static float  *u0 = NULL, *u1 = NULL;
    static uint8_t *prev_luma = NULL;
    static int gw = 0, gh = 0, fw = 0, fh = 0;

    if (sensitivity < 0.05f) sensitivity = 0.05f;
    if (sensitivity > 1.0f)  sensitivity = 1.0f;
    if (rain < 0.0f) rain = 0.0f;
    if (rain > 30.0f) rain = 30.0f;
    if (refract <= 0.0f) refract = 2.0f;
    if (refract > 8.0f) refract = 8.0f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    if (GW < 8 || GH < 8) return;

    if (fw != w || fh != h) {
        free(u0); free(u1); free(prev_luma);
        u0 = calloc((size_t)GW * GH, sizeof(float));
        u1 = calloc((size_t)GW * GH, sizeof(float));
        prev_luma = calloc((size_t)GW * GH, 1);
        if (!u0 || !u1 || !prev_luma) {
            free(u0); free(u1); free(prev_luma);
            u0 = u1 = NULL; prev_luma = NULL;
            fw = fh = 0;
            return;
        }
        gw = GW; gh = GH; fw = w; fh = h;
    }

    /* 1. Downsampled luma + motion injection */
    float inject = 12.0f * sensitivity;
    int thresh = 14;
    for (int gy = 1; gy < gh - 1; gy++) {
        for (int gx = 1; gx < gw - 1; gx++) {
            const uint8_t *p = f->data + (gy*4) * f->stride + (gx*4) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw + gx;
            int d = abs((int)l - (int)prev_luma[gi]);
            prev_luma[gi] = l;
            if (d > thresh)
                u1[gi] -= inject * ((float)d / 255.0f);
        }
    }

    /* 1b. Ambient rain: random droplets at `rain` per second */
    if (rain > 0.0f) {
        int chance = (int)(1000.0f * rain / 30.0f);   /* per-mille per frame */
        while (chance > 0) {
            if ((rand() % 1000) < (chance > 1000 ? 1000 : chance)) {
                int gx = 2 + rand() % (gw - 4);
                int gy = 2 + rand() % (gh - 4);
                float amp = 6.0f + (float)(rand() % 100) / 12.0f;
                u1[gy * gw + gx] -= amp;
                /* small splash: neighbours too */
                u1[gy * gw + gx + 1]  -= amp * 0.4f;
                u1[gy * gw + gx - 1]  -= amp * 0.4f;
                u1[(gy+1) * gw + gx]  -= amp * 0.4f;
                u1[(gy-1) * gw + gx]  -= amp * 0.4f;
            }
            chance -= 1000;
        }
    }

    /* 2. Wave equation step */
    for (int gy = 1; gy < gh - 1; gy++) {
        for (int gx = 1; gx < gw - 1; gx++) {
            int gi = gy * gw + gx;
            float nsum = u1[gi-1] + u1[gi+1] + u1[gi-gw] + u1[gi+gw];
            u0[gi] = (nsum * 0.5f - u0[gi]) * 0.985f;
        }
    }
    { float *tmpp = u0; u0 = u1; u1 = tmpp; }

    /* 3. Refract + shimmer */
    uint8_t *src = malloc((size_t)w * h * 3);
    if (!src) return;
    memcpy(src, f->data, (size_t)w * h * 3);

    for (int y = 4; y < h - 4; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        if (gy < 1) gy = 1;
        if (gy > gh - 2) gy = gh - 2;
        for (int x = 4; x < w - 4; x++) {
            int gx = x / 4;
            if (gx < 1) gx = 1;
            if (gx > gw - 2) gx = gw - 2;
            int gi = gy * gw + gx;
            float ddx = u1[gi+1]  - u1[gi-1];
            float ddy = u1[gi+gw] - u1[gi-gw];
            int sx = x + (int)(ddx * refract);
            int sy = y + (int)(ddy * refract);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = src + sy * w * 3 + sx * 3;
            int spec = (int)((ddx + ddy) * 6.0f * refract);
            int o = x * 3;
            dst[o]   = clamp_u8(sp[0] + spec);
            dst[o+1] = clamp_u8(sp[1] + spec);
            dst[o+2] = clamp_u8(sp[2] + spec);
        }
    }
    free(src);
}

/* ─── fx_lightpaint — INTERACTIVE ──
 * Motion leaves glowing trails: wherever the image changes, glowing
 * color (hue cycling over time) is deposited into a persistent
 * canvas that slowly fades. Wave your arms and paint light.
 *   fade (0..1): trail persistence per frame (0.94 ≈ a few seconds).
 */
/* lightpaint v2: thresh (p2) 10..120 motion sensitivity (0 → 40, lower
 * = easier to paint); huespd (p3) brush color cycle rate (0 → 1) */
static void fx_lightpaint(Frame *f, float fade, float thresh_f, float huespd)
{
    static uint8_t *canvas = NULL;   /* additive glow layer */
    static uint8_t *prevf  = NULL;   /* previous frame for motion */
    static int cw = 0, chh = 0, cs = 0;

    if (fade < 0.5f) fade = 0.5f;
    if (fade > 0.995f) fade = 0.995f;

    int w = f->width, h = f->height;
    int size = f->stride * h;

    if (cw != w || chh != h || cs != f->stride) {
        free(canvas); free(prevf);
        canvas = calloc(1, (size_t)size);
        prevf  = malloc((size_t)size);
        if (!canvas || !prevf) {
            free(canvas); free(prevf);
            canvas = prevf = NULL;
            cw = chh = cs = 0;
            return;
        }
        memcpy(prevf, f->data, (size_t)size);
        cw = w; chh = h; cs = f->stride;
    }

    int mthresh = 40;
    if (thresh_f >= 10.0f && thresh_f <= 120.0f) mthresh = (int)thresh_f;
    if (huespd <= 0.0f) huespd = 1.0f;

    /* Brush color cycles through the spectrum */
    unsigned long t = anim_tick();
    float hue = (float)(t % 3000) / 300.0f * 6.28318f * huespd;
    int br = (int)((sinf(hue)            * 0.5f + 0.5f) * 255.0f);
    int bg = (int)((sinf(hue + 2.094f)   * 0.5f + 0.5f) * 255.0f);
    int bb = (int)((sinf(hue + 4.189f)   * 0.5f + 0.5f) * 255.0f);

    for (int y = 0; y < h; y++) {
        uint8_t *cur = f->data + y * f->stride;
        uint8_t *pre = prevf   + y * f->stride;
        uint8_t *can = canvas  + y * f->stride;
        for (int x = 0; x < w; x++) {
            int o = x * 3;
            int d = abs(cur[o] - pre[o]) + abs(cur[o+1] - pre[o+1])
                  + abs(cur[o+2] - pre[o+2]);
            /* fade the canvas */
            can[o]   = (uint8_t)(can[o]   * fade);
            can[o+1] = (uint8_t)(can[o+1] * fade);
            can[o+2] = (uint8_t)(can[o+2] * fade);
            if (d > mthresh) {
                /* deposit brush proportional to motion energy */
                int e = d > 255 ? 255 : d;
                int nb = can[o]   + (bb * e) / 255;
                int ng = can[o+1] + (bg * e) / 255;
                int nr = can[o+2] + (br * e) / 255;
                can[o]   = clamp_u8(nb);
                can[o+1] = clamp_u8(ng);
                can[o+2] = clamp_u8(nr);
            }
            /* remember current frame, then composite additively */
            pre[o] = cur[o]; pre[o+1] = cur[o+1]; pre[o+2] = cur[o+2];
            cur[o]   = clamp_u8(cur[o]   + can[o]);
            cur[o+1] = clamp_u8(cur[o+1] + can[o+1]);
            cur[o+2] = clamp_u8(cur[o+2] + can[o+2]);
        }
    }
}

/* ─── fx_hologram ──
 * Sci-fi hologram projection: cyan-blue tint, horizontal scanlines,
 * brightness flicker, a faint vertical ghost, and occasional
 * horizontal band displacement. "Help me, Obi-Wan."
 */
/* hologram v2: gap (p2) scanline spacing 2..8 (0 → 3);
 * glitchrate (p3) 0..1 band-glitch frequency (0 → 0.3) */
static void fx_hologram(Frame *f, float strength, float gap_f, float glitchrate)
{
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;

    int w = f->width, h = f->height;
    unsigned long t = anim_tick();

    /* Flicker: slow sine + per-frame jitter */
    float flick = 0.92f
                + 0.06f * sinf((float)t * 0.9f)
                + 0.02f * (float)(hash_u((unsigned)t) & 0xFF) / 255.0f;

    int gap = 3;
    if (gap_f >= 2.0f && gap_f <= 8.0f) gap = (int)gap_f;
    if (glitchrate <= 0.0f) glitchrate = 0.3f;
    if (glitchrate > 1.0f) glitchrate = 1.0f;
    unsigned glitch_mod = (unsigned)(4 + (1.0f - glitchrate) * 90.0f);

    /* Occasional glitch band */
    int band_y0 = -1, band_h = 0, band_shift = 0;
    if ((hash_u((unsigned)t * 7u) % glitch_mod) == 0) {
        band_y0    = (int)(hash_u((unsigned)t * 13u) % (unsigned)h);
        band_h     = 8 + (int)(hash_u((unsigned)t * 17u) % 24u);
        band_shift = (int)(hash_u((unsigned)t * 23u) % 41u) - 20;
    }

    uint8_t *src = malloc((size_t)w * h * 3);
    if (!src) return;
    memcpy(src, f->data, (size_t)w * h * 3);

    const int ghost_dy = 5;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int sy = y;
        if (band_y0 >= 0 && y >= band_y0 && y < band_y0 + band_h) sy = y; /* row kept, x shifted below */
        float scan = (y % gap == 0) ? 0.55f : 1.0f;   /* scanline darkening */
        float lum = flick * scan;
        int gy = y + ghost_dy < h ? y + ghost_dy : y;

        for (int x = 0; x < w; x++) {
            int sx = x;
            if (band_y0 >= 0 && y >= band_y0 && y < band_y0 + band_h) {
                sx = x + band_shift;
                if (sx < 0) sx = 0;
                if (sx >= w) sx = w - 1;
            }
            const uint8_t *sp = src + sy * w * 3 + sx * 3;
            const uint8_t *gp = src + gy * w * 3 + sx * 3;

            /* luma of source + faint ghost */
            int l  = (sp[0]*114 + sp[1]*587 + sp[2]*299) / 1000;
            int lg = (gp[0]*114 + gp[1]*587 + gp[2]*299) / 1000;
            float v = (l + lg * 0.25f) * lum;

            /* map to hologram cyan */
            int hb = (int)(v * 1.15f);
            int hg = (int)(v * 0.95f);
            int hr = (int)(v * 0.30f);

            int o = x * 3;
            /* blend with original by strength */
            dst[o]   = clamp_u8((int)(sp[0] * (1-strength) + hb * strength));
            dst[o+1] = clamp_u8((int)(sp[1] * (1-strength) + hg * strength));
            dst[o+2] = clamp_u8((int)(sp[2] * (1-strength) + hr * strength));
        }
    }
    free(src);
}

/* ─── fx_life v2 ──
 * Conway's Game of Life living on your image, now tracking CELL AGE.
 *   density (p1): seeding rate from edges, 0..1
 *   color   (p2): 0 = classic green; 1 = age rainbow (newborns cyan,
 *                 elders migrate through the spectrum)
 *   speed   (p3): frames between generations, 1..6 (0 → 2)
 * Cells store their age; colors evolve as colonies persist.
 */
static void fx_life(Frame *f, float density, float colormode, float speed)
{
    static uint8_t *cells = NULL, *next = NULL;   /* 0=dead, else age */
    static int gw = 0, gh = 0, fw = 0, fh = 0;

    if (density < 0.05f) density = 0.05f;
    if (density > 1.0f)  density = 1.0f;
    int step_iv = (int)speed;
    if (step_iv <= 0) step_iv = 2;
    if (step_iv > 6)  step_iv = 6;
    int rainbow = colormode >= 0.5f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    if (GW < 8 || GH < 8) return;

    if (fw != w || fh != h) {
        free(cells); free(next);
        cells = calloc((size_t)GW * GH, 1);
        next  = calloc((size_t)GW * GH, 1);
        if (!cells || !next) {
            free(cells); free(next);
            cells = next = NULL;
            fw = fh = 0;
            return;
        }
        gw = GW; gh = GH; fw = w; fh = h;
    }

    /* 1. Seed from edges */
    int seed_chance = 4 + (int)((1.0f - density) * 28.0f);
    for (int gy = 1; gy < gh - 1; gy++) {
        const uint8_t *row = f->data + (gy*4) * f->stride;
        for (int gx = 1; gx < gw - 1; gx++) {
            const uint8_t *p0 = row + (gx*4 - 4) * 3;
            const uint8_t *p1 = row + (gx*4 + 4) * 3;
            int l0 = p0[0] + 2*p0[1] + p0[2];
            int l1 = p1[0] + 2*p1[1] + p1[2];
            if (abs(l1 - l0) > 160 && (rand() % seed_chance) == 0)
                if (!cells[gy * gw + gx]) cells[gy * gw + gx] = 1;
        }
    }

    /* 2. Life step every step_iv frames; survivors age */
    static int tick = 0;
    if ((tick++ % step_iv) == 0) {
        for (int gy = 1; gy < gh - 1; gy++) {
            for (int gx = 1; gx < gw - 1; gx++) {
                int gi = gy * gw + gx;
                int n = (cells[gi-1]      ? 1:0) + (cells[gi+1]      ? 1:0)
                      + (cells[gi-gw]     ? 1:0) + (cells[gi+gw]     ? 1:0)
                      + (cells[gi-gw-1]   ? 1:0) + (cells[gi-gw+1]   ? 1:0)
                      + (cells[gi+gw-1]   ? 1:0) + (cells[gi+gw+1]   ? 1:0);
                if (cells[gi]) {
                    if (n == 2 || n == 3) {
                        int age = cells[gi] + 1;
                        next[gi] = (uint8_t)(age > 250 ? 250 : age);
                    } else next[gi] = 0;
                } else {
                    next[gi] = (n == 3) ? 1 : 0;
                }
            }
        }
        uint8_t *tmpp = cells; cells = next; next = tmpp;
    }

    /* 3. Render */
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        if (gy >= gh) gy = gh - 1;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            if (gx >= gw) gx = gw - 1;
            int o = x * 3;
            uint8_t age = cells[gy * gw + gx];
            if (age) {
                int cb, cg, cr;
                if (rainbow) {
                    float ha = (float)age * 0.10f;   /* hue by age */
                    cb = (int)((sinf(ha + 4.189f) * 0.5f + 0.5f) * 235.0f) + 20;
                    cg = (int)((sinf(ha + 2.094f) * 0.5f + 0.5f) * 235.0f) + 20;
                    cr = (int)((sinf(ha)          * 0.5f + 0.5f) * 235.0f) + 20;
                } else {
                    cb = 40; cg = 200; cr = 60;
                }
                dst[o]   = clamp_u8(dst[o]   / 2 + cb);
                dst[o+1] = clamp_u8(dst[o+1] / 2 + cg);
                dst[o+2] = clamp_u8(dst[o+2] / 2 + cr);
            } else {
                dst[o]   = (uint8_t)(dst[o]   * 3 / 4);
                dst[o+1] = (uint8_t)(dst[o+1] * 3 / 4);
                dst[o+2] = (uint8_t)(dst[o+2] * 3 / 4);
            }
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * mathwarp — user-supplied displacement expressions, compiled at
 * runtime by a small recursive-descent parser into an AST that's
 * evaluated on a coarse grid (every 4th pixel) and bilinearly
 * upsampled — fast enough for 720p30 with plenty of headroom.
 *
 * Grammar:  expr   := term (('+'|'-') term)*
 *           term   := factor (('*'|'/') factor)*
 *           factor := NUM | var | fn '(' expr ')' | '(' expr ')'
 *                     | '-' factor
 *           var    := x | y | w | h | t | r | a
 *           fn     := sin | cos | abs | sqrt
 * ────────────────────────────────────────────────────────────── */

enum {
    EN_CONST, EN_X, EN_Y, EN_W, EN_H, EN_T, EN_R, EN_A,
    EN_CR, EN_CG, EN_CB,   /* input pixel color channels 0..255 */
    EN_ADD, EN_SUB, EN_MUL, EN_DIV, EN_MOD, EN_NEG,
    EN_SIN, EN_COS, EN_ABS, EN_SQRT, EN_FLOOR,
    EN_GT, EN_LT, EN_GE, EN_LE, EN_EQ, EN_NE,
    EN_TERN,          /* a = condition, b = then, c = else */
};

#define EXPR_MAX_NODES 96

typedef struct {
    uint8_t op;
    float   val;      /* EN_CONST */
    int16_t a, b, c;  /* child node indices */
} ExprNode;

typedef struct {
    ExprNode nodes[EXPR_MAX_NODES];
    int      n;
    int      root;
    bool     valid;
} ExprProg;

typedef struct {
    const char *s;
    ExprProg   *prog;
    char       *err;
    size_t      errlen;
    bool        failed;
} ExprParser;

static void expr_fail(ExprParser *p, const char *msg)
{
    if (!p->failed && p->err)
        snprintf(p->err, p->errlen, "%s near '%.12s'", msg, *p->s ? p->s : "end");
    p->failed = true;
}

static int expr_node(ExprParser *p, int op, float val, int a, int b)
{
    if (p->failed) return -1;
    if (p->prog->n >= EXPR_MAX_NODES) { expr_fail(p, "expression too long"); return -1; }
    ExprNode *nd = &p->prog->nodes[p->prog->n];
    nd->op = (uint8_t)op; nd->val = val;
    nd->a = (int16_t)a; nd->b = (int16_t)b; nd->c = -1;
    return p->prog->n++;
}

static int expr_node3(ExprParser *p, int op, int a, int b, int c)
{
    int idx = expr_node(p, op, 0, a, b);
    if (idx >= 0) p->prog->nodes[idx].c = (int16_t)c;
    return idx;
}

static int expr_parse_ternary(ExprParser *p);

static int expr_parse_factor(ExprParser *p)
{
    if (p->failed) return -1;
    const char *s = p->s;

    if (*s == '-') { p->s = s + 1; int c = expr_parse_factor(p); return expr_node(p, EN_NEG, 0, c, -1); }
    if (*s == '(') {
        p->s = s + 1;
        int c = expr_parse_ternary(p);
        if (*p->s != ')') { expr_fail(p, "expected ')'"); return -1; }
        p->s++;
        return c;
    }
    if ((*s >= '0' && *s <= '9') || *s == '.') {
        char *end;
        float v = strtof(s, &end);
        p->s = end;
        return expr_node(p, EN_CONST, v, -1, -1);
    }
    /* functions */
    if (strncmp(s, "sin", 3) == 0 || strncmp(s, "cos", 3) == 0 ||
        strncmp(s, "abs", 3) == 0 || strncmp(s, "sqrt", 4) == 0 ||
        strncmp(s, "floor", 5) == 0) {
        int op; int len;
        if      (strncmp(s, "floor", 5) == 0) { op = EN_FLOOR; len = 5; }
        else if (strncmp(s, "sqrt",  4) == 0) { op = EN_SQRT;  len = 4; }
        else if (strncmp(s, "sin",   3) == 0) { op = EN_SIN;   len = 3; }
        else if (strncmp(s, "cos",   3) == 0) { op = EN_COS;   len = 3; }
        else                                   { op = EN_ABS;   len = 3; }
        s += len;
        if (*s != '(') { p->s = s; expr_fail(p, "expected '(' after function"); return -1; }
        p->s = s + 1;
        int c = expr_parse_ternary(p);
        if (*p->s != ')') { expr_fail(p, "expected ')'"); return -1; }
        p->s++;
        return expr_node(p, op, 0, c, -1);
    }
    /* two-char color-input variables: cr cg cb (must test before the
     * single-char switch, since 'c' alone is not a variable) */
    if (s[0] == 'c') {
        if (s[1] == 'r') { p->s = s + 2; return expr_node(p, EN_CR, 0, -1, -1); }
        if (s[1] == 'g') { p->s = s + 2; return expr_node(p, EN_CG, 0, -1, -1); }
        if (s[1] == 'b') { p->s = s + 2; return expr_node(p, EN_CB, 0, -1, -1); }
    }
    switch (*s) {
    case 'x': p->s = s + 1; return expr_node(p, EN_X, 0, -1, -1);
    case 'y': p->s = s + 1; return expr_node(p, EN_Y, 0, -1, -1);
    case 'w': p->s = s + 1; return expr_node(p, EN_W, 0, -1, -1);
    case 'h': p->s = s + 1; return expr_node(p, EN_H, 0, -1, -1);
    case 't': p->s = s + 1; return expr_node(p, EN_T, 0, -1, -1);
    case 'r': p->s = s + 1; return expr_node(p, EN_R, 0, -1, -1);
    case 'a': p->s = s + 1; return expr_node(p, EN_A, 0, -1, -1);
    }
    expr_fail(p, "unexpected character");
    return -1;
}

static int expr_parse_term(ExprParser *p)
{
    int lhs = expr_parse_factor(p);
    while (!p->failed && (*p->s == '*' || *p->s == '/' || *p->s == '%')) {
        char op = *p->s;
        p->s++;
        int rhs = expr_parse_factor(p);
        int en = (op == '*') ? EN_MUL : (op == '/') ? EN_DIV : EN_MOD;
        lhs = expr_node(p, en, 0, lhs, rhs);
    }
    return lhs;
}

static int expr_parse_sum(ExprParser *p)
{
    int lhs = expr_parse_term(p);
    while (!p->failed && (*p->s == '+' || *p->s == '-')) {
        char op = *p->s;
        p->s++;
        int rhs = expr_parse_term(p);
        lhs = expr_node(p, op == '+' ? EN_ADD : EN_SUB, 0, lhs, rhs);
    }
    return lhs;
}

/* comparisons: return 1.0 or 0.0. One comparison per level (no
 * chaining like a<b<c — parenthesize if you need to combine). */
static int expr_parse_cmp(ExprParser *p)
{
    int lhs = expr_parse_sum(p);
    if (p->failed) return lhs;
    int op = -1, adv = 0;
    if      (p->s[0] == '>' && p->s[1] == '=') { op = EN_GE; adv = 2; }
    else if (p->s[0] == '<' && p->s[1] == '=') { op = EN_LE; adv = 2; }
    else if (p->s[0] == '=' && p->s[1] == '=') { op = EN_EQ; adv = 2; }
    else if (p->s[0] == '!' && p->s[1] == '=') { op = EN_NE; adv = 2; }
    else if (p->s[0] == '>')                   { op = EN_GT; adv = 1; }
    else if (p->s[0] == '<')                   { op = EN_LT; adv = 1; }
    if (op < 0) return lhs;
    p->s += adv;
    int rhs = expr_parse_sum(p);
    return expr_node(p, op, 0, lhs, rhs);
}

/* ternary: cond ? then : else  (right-associative, lowest precedence) */
static int expr_parse_ternary(ExprParser *p)
{
    int cond = expr_parse_cmp(p);
    if (p->failed || *p->s != '?') return cond;
    p->s++;
    int then_e = expr_parse_ternary(p);
    if (*p->s != ':') { expr_fail(p, "expected ':' in ?:"); return -1; }
    p->s++;
    int else_e = expr_parse_ternary(p);
    return expr_node3(p, EN_TERN, cond, then_e, else_e);
}

typedef struct { float x, y, w, h, t, r, a, cr, cg, cb; } ExprVars;

static float expr_eval(const ExprProg *pr, int node, const ExprVars *v)
{
    const ExprNode *n = &pr->nodes[node];
    switch (n->op) {
    case EN_CONST: return n->val;
    case EN_X: return v->x;
    case EN_Y: return v->y;
    case EN_W: return v->w;
    case EN_H: return v->h;
    case EN_T: return v->t;
    case EN_R: return v->r;
    case EN_A: return v->a;
    case EN_CR: return v->cr;
    case EN_CG: return v->cg;
    case EN_CB: return v->cb;
    case EN_ADD: return expr_eval(pr, n->a, v) + expr_eval(pr, n->b, v);
    case EN_SUB: return expr_eval(pr, n->a, v) - expr_eval(pr, n->b, v);
    case EN_MUL: return expr_eval(pr, n->a, v) * expr_eval(pr, n->b, v);
    case EN_DIV: {
        float d = expr_eval(pr, n->b, v);
        return d != 0.0f ? expr_eval(pr, n->a, v) / d : 0.0f;
    }
    case EN_MOD: {
        float d = expr_eval(pr, n->b, v);
        return d != 0.0f ? fmodf(expr_eval(pr, n->a, v), d) : 0.0f;
    }
    case EN_NEG:   return -expr_eval(pr, n->a, v);
    case EN_SIN:   return sinf(expr_eval(pr, n->a, v));
    case EN_COS:   return cosf(expr_eval(pr, n->a, v));
    case EN_ABS:   return fabsf(expr_eval(pr, n->a, v));
    case EN_FLOOR: return floorf(expr_eval(pr, n->a, v));
    case EN_SQRT: {
        float s = expr_eval(pr, n->a, v);
        return s > 0.0f ? sqrtf(s) : 0.0f;
    }
    case EN_GT: return expr_eval(pr, n->a, v) >  expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_LT: return expr_eval(pr, n->a, v) <  expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_GE: return expr_eval(pr, n->a, v) >= expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_LE: return expr_eval(pr, n->a, v) <= expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_EQ: return expr_eval(pr, n->a, v) == expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_NE: return expr_eval(pr, n->a, v) != expr_eval(pr, n->b, v) ? 1.0f : 0.0f;
    case EN_TERN:
        return expr_eval(pr, n->a, v) != 0.0f
             ? expr_eval(pr, n->b, v)
             : expr_eval(pr, n->c, v);
    }
    return 0.0f;
}

static bool expr_compile(ExprProg *prog, const char *src,
                         char *err, size_t errlen)
{
    ExprParser p = { src, prog, err, errlen, false };
    prog->n = 0;
    prog->valid = false;
    prog->root = expr_parse_ternary(&p);
    if (!p.failed && *p.s != '\0')
        expr_fail(&p, "trailing characters");
    if (p.failed || prog->root < 0) return false;
    prog->valid = true;
    return true;
}

/* Global mathwarp programs (one instance each; protected by one lock
 * because IPC compiles while the render thread evaluates).
 *   dx, dy       — pixel displacement (warp)
 *   cr, cg, cb   — output color per channel (recolor); when unset the
 *                  warped source color passes through unchanged. */
static ExprProg g_mw_dx, g_mw_dy;
static ExprProg g_mw_cr, g_mw_cg, g_mw_cb;
static bool     g_mw_color_on = false;
static pthread_mutex_t g_mw_lock = PTHREAD_MUTEX_INITIALIZER;

bool mathwarp_set_expr(const char *dx_expr, const char *dy_expr,
                       char *err, size_t errlen)
{
    ExprProg dx, dy;
    if (!expr_compile(&dx, dx_expr, err, errlen)) return false;
    if (!expr_compile(&dy, dy_expr, err, errlen)) return false;
    pthread_mutex_lock(&g_mw_lock);
    g_mw_dx = dx;
    g_mw_dy = dy;
    pthread_mutex_unlock(&g_mw_lock);
    return true;
}

bool mathwarp_set_color(const char *r_expr, const char *g_expr,
                        const char *b_expr, char *err, size_t errlen)
{
    ExprProg cr, cg, cb;
    if (!expr_compile(&cr, r_expr, err, errlen)) return false;
    if (!expr_compile(&cg, g_expr, err, errlen)) return false;
    if (!expr_compile(&cb, b_expr, err, errlen)) return false;
    pthread_mutex_lock(&g_mw_lock);
    g_mw_cr = cr;
    g_mw_cg = cg;
    g_mw_cb = cb;
    g_mw_color_on = true;
    pthread_mutex_unlock(&g_mw_lock);
    return true;
}

void mathwarp_clear_color(void)
{
    pthread_mutex_lock(&g_mw_lock);
    g_mw_color_on = false;
    pthread_mutex_unlock(&g_mw_lock);
}

/* ─── fx_mathwarp ──
 * Displaces each pixel by (dx, dy) from the user's warp expressions
 * (evaluated on a 1/4 grid + bilinearly upsampled), then optionally
 * recolors each pixel with the color expressions (evaluated per-pixel
 * so they can read that pixel's own cr/cg/cb).
 *   amp (param): global multiplier on the displacement.
 * Set expressions live:
 *   vcam-ctl filter expr  <dx> <dy>        (warp)
 *   vcam-ctl filter color <r> <g> <b>      (recolor; 0..255 each)
 */
static void fx_mathwarp(Frame *f, float amp)
{
    pthread_mutex_lock(&g_mw_lock);
    if (!g_mw_dx.valid) {
        /* Sensible default so 'filter add mathwarp' shows something */
        char e[64];
        expr_compile(&g_mw_dx, "sin(y/20+t*2)*12", e, sizeof e);
        expr_compile(&g_mw_dy, "cos(x/20+t*2)*12", e, sizeof e);
    }
    ExprProg dx_prog = g_mw_dx;
    ExprProg dy_prog = g_mw_dy;
    bool color_on = g_mw_color_on;
    ExprProg cr_prog = g_mw_cr, cg_prog = g_mw_cg, cb_prog = g_mw_cb;
    pthread_mutex_unlock(&g_mw_lock);
    if (!dx_prog.valid || !dy_prog.valid) return;
    if (color_on && (!cr_prog.valid || !cg_prog.valid || !cb_prog.valid))
        color_on = false;

    int w = f->width, h = f->height;
    int GW = w / 4 + 2, GH = h / 4 + 2;

    float *grid = malloc((size_t)GW * GH * 2 * sizeof(float));
    uint8_t *src = malloc((size_t)w * h * 3);
    if (!grid || !src) { free(grid); free(src); return; }
    memcpy(src, f->data, (size_t)w * h * 3);

    ExprVars v;
    v.w = (float)w;
    v.h = (float)h;
    v.t = (float)anim_tick() / 30.0f;
    v.cr = v.cg = v.cb = 0.0f;   /* unused during warp-grid eval */
    float cx = w / 2.0f, cy = h / 2.0f;

    for (int gy = 0; gy < GH; gy++) {
        v.y = (float)(gy * 4);
        for (int gx = 0; gx < GW; gx++) {
            v.x = (float)(gx * 4);
            float ddx = v.x - cx, ddy = v.y - cy;
            v.r = sqrtf(ddx*ddx + ddy*ddy);
            v.a = atan2f(ddy, ddx);
            int gi = (gy * GW + gx) * 2;
            grid[gi]     = expr_eval(&dx_prog, dx_prog.root, &v) * amp;
            grid[gi + 1] = expr_eval(&dy_prog, dy_prog.root, &v) * amp;
        }
    }

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        float fy = (float)(y & 3) / 4.0f;
        v.y = (float)y;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            float fx = (float)(x & 3) / 4.0f;
            int g00 = (gy * GW + gx) * 2;
            int g10 = g00 + 2;
            int g01 = g00 + GW * 2;
            int g11 = g01 + 2;
            float ddx = grid[g00]   * (1-fx)*(1-fy) + grid[g10]   * fx*(1-fy)
                      + grid[g01]   * (1-fx)*fy     + grid[g11]   * fx*fy;
            float ddy = grid[g00+1] * (1-fx)*(1-fy) + grid[g10+1] * fx*(1-fy)
                      + grid[g01+1] * (1-fx)*fy     + grid[g11+1] * fx*fy;
            int sx = x + (int)ddx;
            int sy = y + (int)ddy;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = src + sy * w * 3 + sx * 3;
            int o = x * 3;

            if (!color_on) {
                dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
            } else {
                /* per-pixel recolor: expose the warped source color as
                 * cr/cg/cb (and this pixel's x/y/r/a) to the exprs */
                v.x = (float)x;
                float pdx = x - cx, pdy = y - cy;
                v.r = sqrtf(pdx*pdx + pdy*pdy);
                v.a = atan2f(pdy, pdx);
                v.cb = (float)sp[0];   /* BGR order in buffer */
                v.cg = (float)sp[1];
                v.cr = (float)sp[2];
                int rr = (int)expr_eval(&cr_prog, cr_prog.root, &v);
                int gg = (int)expr_eval(&cg_prog, cg_prog.root, &v);
                int bb = (int)expr_eval(&cb_prog, cb_prog.root, &v);
                dst[o]   = clamp_u8(bb);
                dst[o+1] = clamp_u8(gg);
                dst[o+2] = clamp_u8(rr);
            }
        }
    }
    free(grid);
    free(src);
}

/* ──────────────────────────────────────────────────────────────
 * Batch 5: demoscene & particles
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_plasma ──
 * The classic demoscene plasma — layered animated sines mapped
 * through a sin-RGB palette, blended over the video.
 *   blend (p1): mix amount 0..1
 *   speed (p2): animation rate (0 → 1)
 *   scale (p3): pattern size in px (0 → 60)
 * Computed on a 1/4 grid + upsampled, so it's nearly free.
 */
static void fx_plasma(Frame *f, float blend, float speed, float scale)
{
    if (blend < 0.0f) blend = 0.0f;
    if (blend > 0.9f) blend = 0.9f;
    if (speed <= 0.0f) speed = 1.0f;
    if (scale <= 1.0f) scale = 60.0f;

    int w = f->width, h = f->height;
    int GW = w / 4 + 2, GH = h / 4 + 2;
    uint8_t *grid = malloc((size_t)GW * GH * 3);
    if (!grid) return;

    float t = (float)anim_tick() / 30.0f * speed;
    float inv = 1.0f / scale;
    float cx = w / 2.0f, cy = h / 2.0f;

    for (int gy = 0; gy < GH; gy++) {
        float y = (float)(gy * 4);
        for (int gx = 0; gx < GW; gx++) {
            float x = (float)(gx * 4);
            float dx = x - cx, dy = y - cy;
            float v = sinf(x * inv + t)
                    + sinf(y * inv * 1.3f - t * 1.2f)
                    + sinf((x + y) * inv * 0.7f + t * 0.7f)
                    + sinf(sqrtf(dx*dx + dy*dy) * inv + t * 1.5f);
            uint8_t *g = grid + (gy * GW + gx) * 3;
            g[0] = (uint8_t)((sinf(v * 1.57f)          * 0.5f + 0.5f) * 255.0f);
            g[1] = (uint8_t)((sinf(v * 1.57f + 2.094f) * 0.5f + 0.5f) * 255.0f);
            g[2] = (uint8_t)((sinf(v * 1.57f + 4.189f) * 0.5f + 0.5f) * 255.0f);
        }
    }

    float keep = 1.0f - blend;
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        for (int x = 0; x < w; x++) {
            const uint8_t *g = grid + (gy * GW + x / 4) * 3;
            int o = x * 3;
            dst[o]   = (uint8_t)(dst[o]   * keep + g[0] * blend);
            dst[o+1] = (uint8_t)(dst[o+1] * keep + g[1] * blend);
            dst[o+2] = (uint8_t)(dst[o+2] * keep + g[2] * blend);
        }
    }
    free(grid);
}

/* ─── fx_contour ──
 * Topographic map of your face: luma is quantized into N bands, and
 * the borders between bands are drawn as glowing iso-lines whose hue
 * follows the elevation. Everything else fades toward dark.
 *   levels (p1): number of elevation bands, 3..24 (default 8)
 *   dim    (p2): how much of the original survives, 0..1 (0 → 0.25)
 */
static void fx_contour(Frame *f, float levels_f, float dim)
{
    int levels = (int)levels_f;
    if (levels < 3)  levels = 3;
    if (levels > 24) levels = 24;
    if (dim <= 0.0f) dim = 0.25f;
    if (dim > 1.0f)  dim = 1.0f;

    int w = f->width, h = f->height;
    uint8_t *q = malloc((size_t)w * h);
    if (!q) return;
    int step = 256 / levels;
    for (int y = 0; y < h; y++) {
        const uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3)
            q[y*w + x] = (uint8_t)(((p[0]*114 + p[1]*587 + p[2]*299) / 1000) / step);
    }

    unsigned long t = anim_tick();
    float hue_t = (float)(t % 6283) * 0.0005f;

    for (int y = 0; y < h - 1; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w - 1; x++) {
            int o = x * 3;
            uint8_t here = q[y*w + x];
            if (here != q[y*w + x + 1] || here != q[(y+1)*w + x]) {
                /* iso-line: hue by elevation + slow drift */
                float ha = (float)here * 0.7f + hue_t;
                dst[o]   = clamp_u8(dst[o]/3   + (int)((sinf(ha+4.189f)*0.5f+0.5f)*255));
                dst[o+1] = clamp_u8(dst[o+1]/3 + (int)((sinf(ha+2.094f)*0.5f+0.5f)*255));
                dst[o+2] = clamp_u8(dst[o+2]/3 + (int)((sinf(ha)       *0.5f+0.5f)*255));
            } else {
                dst[o]   = (uint8_t)(dst[o]   * dim);
                dst[o+1] = (uint8_t)(dst[o+1] * dim);
                dst[o+2] = (uint8_t)(dst[o+2] * dim);
            }
        }
    }
    free(q);
}

/* ─── fx_fireworks — INTERACTIVE ──
 * Your motion strikes sparks: movement spawns bursts of glowing
 * particles that fly out, arc under gravity, and fade. Snap your
 * fingers in front of the camera and watch it detonate.
 *   sensitivity (p1): 0..1 — how easily motion sparks
 *   burst       (p2): particles per burst multiplier (0 → 1)
 */
#define FW_MAX_PARTICLES 640

typedef struct {
    float x, y, vx, vy;
    int   life;           /* frames remaining; 0 = dead */
    uint8_t cb, cg, cr;
} FwParticle;

static void fx_fireworks(Frame *f, float sensitivity, float burst_mult)
{
    static FwParticle ps[FW_MAX_PARTICLES];
    static uint8_t *prev_luma = NULL;
    static int gw = 0, gh = 0, fw2 = 0, fh2 = 0;

    if (sensitivity < 0.05f) sensitivity = 0.05f;
    if (sensitivity > 1.0f)  sensitivity = 1.0f;
    if (burst_mult <= 0.0f) burst_mult = 1.0f;
    if (burst_mult > 3.0f)  burst_mult = 3.0f;

    int w = f->width, h = f->height;
    int GW = w / 8, GH = h / 8;
    if (GW < 8 || GH < 8) return;

    if (fw2 != w || fh2 != h) {
        free(prev_luma);
        prev_luma = calloc((size_t)GW * GH, 1);
        if (!prev_luma) { fw2 = fh2 = 0; return; }
        gw = GW; gh = GH; fw2 = w; fh2 = h;
        memset(ps, 0, sizeof ps);
    }

    /* 1. Motion detection on an 8px grid; strong changes spawn bursts */
    int thresh = 30 + (int)((1.0f - sensitivity) * 60.0f);
    int spawn_budget = 3;   /* max bursts per frame — keeps it tasteful */
    for (int gy = 1; gy < gh - 1 && spawn_budget > 0; gy++) {
        for (int gx = 1; gx < gw - 1 && spawn_budget > 0; gx++) {
            const uint8_t *p = f->data + (gy*8) * f->stride + (gx*8) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw + gx;
            int d = abs((int)l - (int)prev_luma[gi]);
            prev_luma[gi] = l;
            if (d > thresh && (rand() % 6) == 0) {
                spawn_budget--;
                /* burst! */
                int n = (int)((10 + rand() % 10) * burst_mult);
                float hue = (float)(rand() % 628) * 0.01f;
                uint8_t cb = (uint8_t)((sinf(hue+4.189f)*0.5f+0.5f)*255);
                uint8_t cg = (uint8_t)((sinf(hue+2.094f)*0.5f+0.5f)*255);
                uint8_t cr = (uint8_t)((sinf(hue)       *0.5f+0.5f)*255);
                for (int k = 0; k < n; k++) {
                    /* find a dead slot */
                    for (int s = 0; s < FW_MAX_PARTICLES; s++) {
                        if (ps[s].life > 0) continue;
                        float ang = (float)(rand() % 628) * 0.01f;
                        float spd = 1.5f + (float)(rand() % 100) * 0.05f;
                        ps[s].x = (float)(gx * 8);
                        ps[s].y = (float)(gy * 8);
                        ps[s].vx = cosf(ang) * spd;
                        ps[s].vy = sinf(ang) * spd - 1.0f;  /* slight up-kick */
                        ps[s].life = 20 + rand() % 25;
                        ps[s].cb = cb; ps[s].cg = cg; ps[s].cr = cr;
                        break;
                    }
                }
            }
        }
    }

    /* 2. Update + render particles (additive 2×2 with fading tail) */
    for (int s = 0; s < FW_MAX_PARTICLES; s++) {
        if (ps[s].life <= 0) continue;
        ps[s].life--;
        ps[s].x += ps[s].vx;
        ps[s].y += ps[s].vy;
        ps[s].vy += 0.18f;              /* gravity */
        ps[s].vx *= 0.985f;             /* drag */

        int px = (int)ps[s].x, py = (int)ps[s].y;
        if (px < 1 || px >= w - 2 || py < 1 || py >= h - 2) {
            ps[s].life = 0;
            continue;
        }
        float fade = (float)ps[s].life / 45.0f;
        if (fade > 1.0f) fade = 1.0f;
        int ab = (int)(ps[s].cb * fade);
        int ag = (int)(ps[s].cg * fade);
        int ar = (int)(ps[s].cr * fade);
        for (int dy = 0; dy < 2; dy++) {
            uint8_t *dst = f->data + (py + dy) * f->stride + px * 3;
            for (int dx = 0; dx < 2; dx++) {
                dst[0] = clamp_u8(dst[0] + ab);
                dst[1] = clamp_u8(dst[1] + ag);
                dst[2] = clamp_u8(dst[2] + ar);
                dst += 3;
            }
        }
    }
}

/* ─── fx_infinity ──
 * Recursive picture-in-picture (droste-style): a scaled copy of the
 * PREVIOUS output frame is drawn into the center — which itself
 * contained a smaller copy, which contained a smaller copy… turtles
 * all the way down, one cheap blit per frame.
 *   scale (p1): size of the inner copy, 0.2..0.85 (default 0.55)
 *   rot   (p2): degrees the inner world rotates per level (default 0)
 */
static void fx_infinity(Frame *f, float scale, float rot_deg)
{
    static uint8_t *prev = NULL;
    static int pw = 0, ph = 0, ps = 0;

    if (scale < 0.2f)  scale = 0.55f;
    if (scale > 0.85f) scale = 0.85f;

    int w = f->width, h = f->height;
    int size = f->stride * h;

    if (pw != w || ph != h || ps != f->stride) {
        free(prev);
        prev = malloc((size_t)size);
        if (!prev) { pw = ph = ps = 0; return; }
        memcpy(prev, f->data, (size_t)size);
        pw = w; ph = h; ps = f->stride;
    }

    /* Inner rect, centered */
    int iw = (int)(w * scale), ih = (int)(h * scale);
    int x0 = (w - iw) / 2,     y0 = (h - ih) / 2;

    float ang = rot_deg * 3.14159265f / 180.0f;
    float ca = cosf(-ang), sa = sinf(-ang);
    float cx = w / 2.0f, cy = h / 2.0f;
    float inv = 1.0f / scale;

    for (int y = 0; y < ih; y++) {
        uint8_t *dst = f->data + (y0 + y) * f->stride + x0 * 3;
        for (int x = 0; x < iw; x++) {
            /* map inner-rect pixel back to full prev frame,
             * with optional rotation about the center */
            float fx0 = (float)(x0 + x) - cx;
            float fy0 = (float)(y0 + y) - cy;
            float rx = fx0 * ca - fy0 * sa;
            float ry = fx0 * sa + fy0 * ca;
            int sx = (int)(cx + rx * inv);
            int sy = (int)(cy + ry * inv);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = prev + sy * f->stride + sx * 3;
            dst[0] = sp[0]; dst[1] = sp[1]; dst[2] = sp[2];
            dst += 3;
        }
    }
    /* Thin frame border around the inner world for legibility */
    for (int x = x0; x < x0 + iw; x++) {
        uint8_t *a = f->data + y0 * f->stride + x * 3;
        uint8_t *b = f->data + (y0 + ih - 1) * f->stride + x * 3;
        a[0] = a[1] = a[2] = 230;
        b[0] = b[1] = b[2] = 230;
    }
    for (int y = y0; y < y0 + ih; y++) {
        uint8_t *a = f->data + y * f->stride + x0 * 3;
        uint8_t *b = f->data + y * f->stride + (x0 + iw - 1) * 3;
        a[0] = a[1] = a[2] = 230;
        b[0] = b[1] = b[2] = 230;
    }

    memcpy(prev, f->data, (size_t)size);
}

/* ─── fx_moire ──
 * Two ring systems — one centered, one orbiting — multiply into
 * living interference patterns over the image. Deeply hypnotic.
 *   spacing (p1): ring spacing in px, 4..40 (default 12)
 *   speed   (p2): orbit + phase animation rate (0 → 1)
 */
static void fx_moire(Frame *f, float spacing, float speed)
{
    if (spacing < 4.0f)  spacing = 12.0f;
    if (spacing > 40.0f) spacing = 40.0f;
    if (speed <= 0.0f) speed = 1.0f;

    int w = f->width, h = f->height;
    float t = (float)anim_tick() / 30.0f * speed;
    float k = 6.28318f / spacing;

    float cx1 = w / 2.0f, cy1 = h / 2.0f;
    /* second center orbits the first */
    float orb = (float)(w < h ? w : h) * 0.18f;
    float cx2 = cx1 + cosf(t * 0.7f) * orb;
    float cy2 = cy1 + sinf(t * 0.9f) * orb;

    /* Coarse grid + upsample (two sqrts per grid point only) */
    int GW = w / 2 + 1, GH = h / 2 + 1;
    int8_t *grid = malloc((size_t)GW * GH);
    if (!grid) return;
    for (int gy = 0; gy < GH; gy++) {
        float y = (float)(gy * 2);
        for (int gx = 0; gx < GW; gx++) {
            float x = (float)(gx * 2);
            float d1 = sqrtf((x-cx1)*(x-cx1) + (y-cy1)*(y-cy1));
            float d2 = sqrtf((x-cx2)*(x-cx2) + (y-cy2)*(y-cy2));
            float m = sinf(d1 * k + t * 2.0f) * sinf(d2 * k);
            grid[gy * GW + gx] = (int8_t)(m * 100.0f);  /* -100..100 */
        }
    }

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        const int8_t *grow = grid + (y / 2) * GW;
        for (int x = 0; x < w; x++) {
            int m = grow[x / 2];   /* -100..100 */
            int o = x * 3;
            /* multiply brightness by 0.55..1.45; extremes tint */
            dst[o]   = clamp_u8(dst[o]   + (dst[o]   * m) / 220 + (m > 60 ? 30 : 0));
            dst[o+1] = clamp_u8(dst[o+1] + (dst[o+1] * m) / 220);
            dst[o+2] = clamp_u8(dst[o+2] + (dst[o+2] * m) / 220 + (m < -60 ? 30 : 0));
        }
    }
    free(grid);
}

/* ──────────────────────────────────────────────────────────────
 * Batch 6: text, tone & tracking
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_ascii ──
 * Real ASCII-art: the frame becomes a grid of characters chosen by
 * brightness from the classic density ramp " .:-=+*#%@", rendered
 * from an embedded 8×8 bitmap font.
 *   cell (p1): 8 or 16 px (16 = chunkier, glyphs pixel-doubled)
 *   mode (p2): 0 = green terminal, 1 = source-colored, 2 = white
 */
static const uint8_t ascii_font[10][8] = {
    /* ' ' */ {0,0,0,0,0,0,0,0},
    /* '.' */ {0,0,0,0,0,0x18,0x18,0},
    /* ':' */ {0,0x18,0x18,0,0,0x18,0x18,0},
    /* '-' */ {0,0,0,0x7E,0,0,0,0},
    /* '=' */ {0,0,0x7E,0,0x7E,0,0,0},
    /* '+' */ {0,0x18,0x18,0x7E,0x7E,0x18,0x18,0},
    /* '*' */ {0,0x66,0x3C,0xFF,0x3C,0x66,0,0},
    /* '#' */ {0x24,0x24,0x7E,0x24,0x7E,0x24,0x24,0},
    /* '%' */ {0x62,0x64,0x08,0x10,0x26,0x46,0,0},
    /* '@' */ {0x3C,0x42,0x5A,0x5A,0x5C,0x40,0x3C,0},
};

static void fx_ascii(Frame *f, float cell_f, float mode_f)
{
    int scale = ((int)cell_f >= 16) ? 2 : 1;   /* 8 or 16 px cells */
    int cell = 8 * scale;
    int mode = (int)mode_f;
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int cy = 0; cy + cell <= h; cy += cell) {
        for (int cx = 0; cx + cell <= w; cx += cell) {
            /* cell average color + luma */
            long sb = 0, sg = 0, sr = 0;
            for (int y = 0; y < cell; y += scale) {
                const uint8_t *p = tmp + (cy+y) * w * 3 + cx * 3;
                for (int x = 0; x < cell; x += scale, p += 3 * scale) {
                    sb += p[0]; sg += p[1]; sr += p[2];
                }
            }
            int npx = (cell/scale) * (cell/scale);
            int ab = (int)(sb/npx), ag = (int)(sg/npx), ar = (int)(sr/npx);
            int luma = (ab*114 + ag*587 + ar*299) / 1000;
            int gi = (luma * 10) / 256;          /* glyph index 0..9 */
            if (gi > 9) gi = 9;

            int fb, fg2, fr;                     /* foreground color */
            switch (mode) {
            case 1:  fb = ab*3/2; fg2 = ag*3/2; fr = ar*3/2; break;
            case 2:  fb = fg2 = fr = 235; break;
            default: fb = 40; fg2 = 220; fr = 60; break;
            }
            if (fb > 255) fb = 255;
            if (fg2 > 255) fg2 = 255;
            if (fr > 255) fr = 255;

            for (int y = 0; y < cell; y++) {
                uint8_t *dst = f->data + (cy+y) * f->stride + cx * 3;
                uint8_t bits = ascii_font[gi][y / scale];
                for (int x = 0; x < cell; x++) {
                    int on = bits & (0x80 >> (x / scale));
                    if (on) {
                        dst[0] = (uint8_t)fb;
                        dst[1] = (uint8_t)fg2;
                        dst[2] = (uint8_t)fr;
                    } else {
                        dst[0] = 8; dst[1] = 10; dst[2] = 8;
                    }
                    dst += 3;
                }
            }
        }
    }
    free(tmp);
}

/* ─── fx_duotone ──
 * Maps luma onto a gradient between two hues — the classic music-
 * poster look (shadows one color, highlights another).
 *   hueA (p1): shadow hue in degrees, 0..360 (default 220 = blue)
 *   hueB (p2): highlight hue in degrees (0 → 30 = orange)
 *   mix  (p3): blend with the original, 0..1 (0 → 1 = full duotone)
 */
static void fx_duotone(Frame *f, float hueA, float hueB, float mix)
{
    if (hueB <= 0.0f) hueB = 30.0f;
    if (mix <= 0.0f || mix > 1.0f) mix = 1.0f;
    float a = hueA * 3.14159265f / 180.0f;
    float b = hueB * 3.14159265f / 180.0f;

    /* Endpoint colors from hue via sin palette, darkened / lightened */
    float Ab = (sinf(a+4.189f)*0.5f+0.5f) * 90.0f;
    float Ag = (sinf(a+2.094f)*0.5f+0.5f) * 90.0f;
    float Ar = (sinf(a)       *0.5f+0.5f) * 90.0f;
    float Bb = 120.0f + (sinf(b+4.189f)*0.5f+0.5f) * 135.0f;
    float Bg = 120.0f + (sinf(b+2.094f)*0.5f+0.5f) * 135.0f;
    float Br = 120.0f + (sinf(b)       *0.5f+0.5f) * 135.0f;

    float keep = 1.0f - mix;
    int n = f->width * f->height;
    uint8_t *p = f->data;
    for (int i = 0; i < n; i++, p += 3) {
        float l = (float)(p[0]*114 + p[1]*587 + p[2]*299) / 255000.0f; /* 0..1 */
        int db = (int)((Ab + (Bb - Ab) * l) * mix + p[0] * keep);
        int dg = (int)((Ag + (Bg - Ag) * l) * mix + p[1] * keep);
        int dr = (int)((Ar + (Br - Ar) * l) * mix + p[2] * keep);
        p[0] = clamp_u8(db);
        p[1] = clamp_u8(dg);
        p[2] = clamp_u8(dr);
    }
}

/* ─── fx_sketch ──
 * Pencil sketch via the classic color-dodge trick: grayscale, invert,
 * blur the inversion, then dodge-blend — lines survive, flats go
 * paper-white.
 *   detail   (p1): blur radius 1..4 — smaller = finer pencil (def 2)
 *   darkness (p2): 0..1 line darkness boost (0 → 0.5)
 */
static void fx_sketch(Frame *f, float detail, float darkness)
{
    int radius = (int)detail;
    if (radius < 1) radius = 1;
    if (radius > 4) radius = 4;
    if (darkness <= 0.0f) darkness = 0.5f;
    if (darkness > 1.0f) darkness = 1.0f;

    int w = f->width, h = f->height;
    uint8_t *g  = malloc((size_t)w * h);    /* grayscale */
    uint8_t *bi = malloc((size_t)w * h);    /* blurred inverse */
    if (!g || !bi) { free(g); free(bi); return; }

    for (int y = 0; y < h; y++) {
        const uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3)
            g[y*w + x] = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
    }
    /* box blur of the inverse */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0, cnt = 0;
            for (int dy = -radius; dy <= radius; dy++) {
                int yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (int dx = -radius; dx <= radius; dx++) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    sum += 255 - g[yy*w + xx];
                    cnt++;
                }
            }
            bi[y*w + x] = (uint8_t)(sum / cnt);
        }
    }
    /* color dodge: result = g / (1 - bi) */
    float dk = 1.0f - darkness * 0.35f;
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            int base = g[y*w + x];
            int blur = bi[y*w + x];
            int v = (blur >= 255) ? 255 : (base * 255) / (255 - blur);
            if (v > 255) v = 255;
            v = (int)(v * dk + 255 * (1.0f - dk) * ((float)v / 255.0f));
            p[0] = p[1] = p[2] = (uint8_t)v;
        }
    }
    free(g);
    free(bi);
}

/* ─── fx_spotlight — INTERACTIVE ──
 * A theatrical spotlight that FOLLOWS YOUR MOTION. The motion
 * centroid is tracked on a coarse grid and the light glides toward
 * it; everything outside the beam falls into darkness. Walk across
 * the frame and the light chases you.
 *   radius (p1): beam radius in px (default 180)
 *   dark   (p2): outside darkness 0..1 (0 → 0.85; 1 = pitch black)
 *   speed  (p3): follow smoothing 0.02..0.5 (0 → 0.12)
 */
static void fx_spotlight(Frame *f, float radius, float dark, float speed)
{
    static uint8_t *prev_luma = NULL;
    static int gw = 0, gh = 0, fw3 = 0, fh3 = 0;
    static float sx = -1.0f, sy = -1.0f;   /* smoothed spotlight center */

    if (radius < 40.0f) radius = 180.0f;
    if (dark <= 0.0f) dark = 0.85f;
    if (dark > 1.0f) dark = 1.0f;
    if (speed <= 0.0f) speed = 0.12f;
    if (speed > 0.5f) speed = 0.5f;

    int w = f->width, h = f->height;
    int GW = w / 8, GH = h / 8;
    if (GW < 8 || GH < 8) return;

    if (fw3 != w || fh3 != h) {
        free(prev_luma);
        prev_luma = calloc((size_t)GW * GH, 1);
        if (!prev_luma) { fw3 = fh3 = 0; return; }
        gw = GW; gh = GH; fw3 = w; fh3 = h;
        sx = w / 2.0f; sy = h / 2.0f;
    }

    /* Motion centroid on the coarse grid */
    long mx = 0, my = 0, mtot = 0;
    for (int gy = 1; gy < gh - 1; gy++) {
        for (int gx = 1; gx < gw - 1; gx++) {
            const uint8_t *p = f->data + (gy*8) * f->stride + (gx*8) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw + gx;
            int d = abs((int)l - (int)prev_luma[gi]);
            prev_luma[gi] = l;
            if (d > 18) {
                mx += (long)gx * 8 * d;
                my += (long)gy * 8 * d;
                mtot += d;
            }
        }
    }
    if (mtot > 200) {   /* enough motion to retarget */
        float tx = (float)mx / (float)mtot;
        float ty = (float)my / (float)mtot;
        sx += (tx - sx) * speed;
        sy += (ty - sy) * speed;
    }

    /* Render: smooth radial falloff, slight boost inside the beam */
    float r2 = radius * radius;
    float fall = radius * 0.5f;              /* soft edge width */
    float fall2 = (radius + fall) * (radius + fall);
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = (float)y - sy;
        for (int x = 0; x < w; x++) {
            float dx = (float)x - sx;
            float d2 = dx*dx + dy*dy;
            float lum;
            if (d2 <= r2)        lum = 1.12f;                 /* in beam */
            else if (d2 >= fall2) lum = 1.0f - dark;          /* outside */
            else {
                float tt = (sqrtf(d2) - radius) / fall;       /* 0..1 */
                lum = 1.12f + tt * ((1.0f - dark) - 1.12f);
            }
            int o = x * 3;
            dst[o]   = clamp_u8((int)(dst[o]   * lum));
            dst[o+1] = clamp_u8((int)(dst[o+1] * lum));
            dst[o+2] = clamp_u8((int)(dst[o+2] * lum));
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * Batch 7: the AI-craze set
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_predator — motion camouflage cloak ──
 * Learns the background wherever nothing moves, then shows you the
 * BACKGROUND instead of the subject. Hold still → you fade out of
 * existence. Move → a shimmering refraction betrays your outline,
 * exactly like the movie cloak.
 *   visibility (p1): 0 = fully cloaked, 1 = normal, def 0.15
 *   shimmer    (p2): refraction amplitude on motion px (0 → 8)
 *   adapt      (p3): background learn rate 0.005..0.2 (0 → 0.02)
 */
static void fx_predator(Frame *f, float visibility, float shimmer,
                        float adapt)
{
    static uint8_t *bg = NULL;
    static uint8_t *motion = NULL;   /* per-quarter-cell motion energy */
    static uint8_t *pl = NULL;       /* prev luma quarter res */
    static int bw = 0, bh = 0, bs = 0, gw = 0, gh = 0;

    if (visibility < 0.0f) visibility = 0.0f;
    if (visibility > 1.0f) visibility = 1.0f;
    if (shimmer <= 0.0f) shimmer = 8.0f;
    if (shimmer > 30.0f) shimmer = 30.0f;
    if (adapt < 0.005f || adapt > 0.2f) adapt = 0.02f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    int size = f->stride * h;

    if (bw != w || bh != h || bs != f->stride) {
        free(bg); free(motion); free(pl);
        bg     = malloc((size_t)size);
        motion = calloc((size_t)GW * GH, 1);
        pl     = calloc((size_t)GW * GH, 1);
        if (!bg || !motion || !pl) {
            free(bg); free(motion); free(pl);
            bg = motion = pl = NULL;
            bw = bh = bs = 0;
            return;
        }
        memcpy(bg, f->data, (size_t)size);
        bw = w; bh = h; bs = f->stride;
        gw = GW; gh = GH;
    }

    /* Motion energy per cell (decaying), background learning where calm */
    unsigned long t = anim_tick();
    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            const uint8_t *p = f->data + (gy*4) * f->stride + (gx*4) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw + gx;
            int d = abs((int)l - (int)pl[gi]);
            pl[gi] = l;
            int m = motion[gi];
            m = (m * 7) / 8 + (d > 14 ? 90 : 0);   /* rise fast, decay slow */
            motion[gi] = (uint8_t)(m > 255 ? 255 : m);
        }
    }

    float ph = (float)t * 0.25f;
    for (int y = 0; y < h; y++) {
        uint8_t *cur = f->data + y * f->stride;
        uint8_t *bgr = bg      + y * f->stride;
        int gy = y / 4;
        if (gy >= gh) gy = gh - 1;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            if (gx >= gw) gx = gw - 1;
            int m = motion[gy * gw + gx];
            int o = x * 3;

            if (m < 20) {
                /* calm: learn the background here + show it (cloak) */
                bgr[o]   = (uint8_t)(bgr[o]   * (1-adapt) + cur[o]   * adapt);
                bgr[o+1] = (uint8_t)(bgr[o+1] * (1-adapt) + cur[o+1] * adapt);
                bgr[o+2] = (uint8_t)(bgr[o+2] * (1-adapt) + cur[o+2] * adapt);
                cur[o]   = (uint8_t)(bgr[o]   * (1-visibility) + cur[o]   * visibility);
                cur[o+1] = (uint8_t)(bgr[o+1] * (1-visibility) + cur[o+1] * visibility);
                cur[o+2] = (uint8_t)(bgr[o+2] * (1-visibility) + cur[o+2] * visibility);
            } else {
                /* motion: shimmering refracted background */
                float amp = shimmer * (float)m / 255.0f;
                int sx = x + (int)(sinf(y * 0.11f + ph) * amp);
                int sy = y + (int)(cosf(x * 0.09f - ph) * amp * 0.6f);
                if (sx < 0) sx = 0;
                if (sx >= w) sx = w - 1;
                if (sy < 0) sy = 0;
                if (sy >= h) sy = h - 1;
                const uint8_t *bp = bg + sy * f->stride + sx * 3;
                float vis = visibility + 0.10f;   /* faint self on motion */
                if (vis > 1.0f) vis = 1.0f;
                cur[o]   = (uint8_t)(bp[0] * (1-vis) + cur[o]   * vis);
                cur[o+1] = (uint8_t)(bp[1] * (1-vis) + cur[o+1] * vis);
                cur[o+2] = (uint8_t)(bp[2] * (1-vis) + cur[o+2] * vis);
            }
        }
    }
}

/* ─── fx_shards — broken mirror ──
 * The frame shatters into voronoi shards, each displaced by its own
 * fixed random offset, with bright crack lines along the fractures.
 *   count  (p1): number of shards 6..64 (def 24)
 *   offset (p2): max per-shard displacement px (0 → 12)
 * The shard map is computed once per resolution/count at 1/2 res.
 */
#define SHARDS_MAX 64
static void fx_shards(Frame *f, float count_f, float offset_f)
{
    static uint8_t *map = NULL;             /* half-res cell ids */
    static int8_t   offx[SHARDS_MAX], offy[SHARDS_MAX];
    static int mw = 0, mh = 0, mcount = 0;

    int count = (int)count_f;
    if (count < 6)  count = 6;
    if (count > SHARDS_MAX) count = SHARDS_MAX;
    int maxoff = (offset_f >= 1.0f && offset_f <= 60.0f) ? (int)offset_f : 12;

    int w = f->width, h = f->height;
    int MW = w / 2, MH = h / 2;

    if (mw != MW || mh != MH || mcount != count) {
        free(map);
        map = malloc((size_t)MW * MH);
        if (!map) { mw = mh = 0; return; }
        /* fixed seeds → stable shards until params change */
        int sx[SHARDS_MAX], sy[SHARDS_MAX];
        srand(1234567u + (unsigned)count);
        for (int i = 0; i < count; i++) {
            sx[i] = rand() % MW;
            sy[i] = rand() % MH;
            offx[i] = (int8_t)((rand() % (2*maxoff+1)) - maxoff);
            offy[i] = (int8_t)((rand() % (2*maxoff+1)) - maxoff);
        }
        for (int y = 0; y < MH; y++) {
            for (int x = 0; x < MW; x++) {
                int best = 0; long bd = LONG_MAX;
                for (int i = 0; i < count; i++) {
                    long dx = x - sx[i], dy = y - sy[i];
                    long d = dx*dx + dy*dy;
                    if (d < bd) { bd = d; best = i; }
                }
                map[y * MW + x] = (uint8_t)best;
            }
        }
        mw = MW; mh = MH; mcount = count;
    }

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int my = y / 2;
        if (my >= mh) my = mh - 1;
        for (int x = 0; x < w; x++) {
            int mx = x / 2;
            if (mx >= mw) mx = mw - 1;
            uint8_t cell = map[my * mw + mx];
            /* crack line: neighbouring map cells differ */
            uint8_t right = map[my * mw + (mx + 1 < mw ? mx + 1 : mx)];
            uint8_t down  = map[(my + 1 < mh ? my + 1 : my) * mw + mx];
            int o = x * 3;
            if (cell != right || cell != down) {
                dst[o] = dst[o+1] = dst[o+2] = 235;   /* crack */
                continue;
            }
            int sx2 = x + offx[cell];
            int sy2 = y + offy[cell];
            if (sx2 < 0) sx2 = 0;
            if (sx2 >= w) sx2 = w - 1;
            if (sy2 < 0) sy2 = 0;
            if (sy2 >= h) sy2 = h - 1;
            const uint8_t *sp = tmp + sy2 * w * 3 + sx2 * 3;
            dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
        }
    }
    free(tmp);
}

/* ─── fx_portal ──
 * Two glowing ringed portals; each circle shows what the OTHER
 * portal sees. Orange ring left, blue ring right — yes, that game.
 *   radius (p1): portal radius px (def 130)
 *   sep    (p2): separation as fraction of width 0.2..0.45 (0 → 0.28)
 */
static void fx_portal(Frame *f, float radius, float sep_frac)
{
    if (radius < 40.0f) radius = 130.0f;
    if (sep_frac < 0.2f || sep_frac > 0.45f) sep_frac = 0.28f;

    int w = f->width, h = f->height;
    float cxA = w * (0.5f - sep_frac), cxB = w * (0.5f + sep_frac);
    float cy = h * 0.5f;
    float dxAB = cxB - cxA;

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float r2 = radius * radius;
    float ring = radius * 0.06f + 3.0f;
    unsigned long t = anim_tick();
    float pulse = 0.75f + 0.25f * sinf((float)t * 0.2f);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = y - cy;
        float dy2 = dy * dy;
        for (int x = 0; x < w; x++) {
            float dA = (x - cxA) * (x - cxA) + dy2;
            float dB = (x - cxB) * (x - cxB) + dy2;
            int o = x * 3;

            if (dA < r2) {
                /* inside A: show B's view */
                int sx = x + (int)dxAB;
                if (sx >= w) sx = w - 1;
                const uint8_t *sp = tmp + y * w * 3 + sx * 3;
                dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
            } else if (dB < r2) {
                int sx = x - (int)dxAB;
                if (sx < 0) sx = 0;
                const uint8_t *sp = tmp + y * w * 3 + sx * 3;
                dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
            }
            /* rings */
            float rA = sqrtf(dA), rB = sqrtf(dB);
            if (fabsf(rA - radius) < ring) {
                float e = (1.0f - fabsf(rA - radius) / ring) * pulse;
                dst[o]   = clamp_u8(dst[o]   + (int)(30  * e));
                dst[o+1] = clamp_u8(dst[o+1] + (int)(120 * e));
                dst[o+2] = clamp_u8(dst[o+2] + (int)(255 * e));   /* orange */
            }
            if (fabsf(rB - radius) < ring) {
                float e = (1.0f - fabsf(rB - radius) / ring) * pulse;
                dst[o]   = clamp_u8(dst[o]   + (int)(255 * e));   /* blue */
                dst[o+1] = clamp_u8(dst[o+1] + (int)(140 * e));
                dst[o+2] = clamp_u8(dst[o+2] + (int)(30  * e));
            }
        }
    }
    free(tmp);
}

/* ─── fx_heattrail — motion leaves cooling thermal glow ──
 * A per-pixel heat field: motion pumps heat in, everything cools
 * exponentially, heat renders as an additive fire palette. Wave an
 * arm and watch the thermal wake fade from white → orange → red.
 *   gain (p1): heat per unit motion 0..1 (def 0.7)
 *   cool (p2): retention per frame 0.85..0.99 (0 → 0.94)
 */
static void fx_heattrail(Frame *f, float gain, float cool)
{
    static float  *heat = NULL;
    static uint8_t *pl = NULL;
    static int gw = 0, gh = 0, fw4 = 0, fh4 = 0;

    if (gain < 0.05f) gain = 0.05f;
    if (gain > 1.0f)  gain = 1.0f;
    if (cool < 0.85f || cool > 0.99f) cool = 0.94f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    if (GW < 8 || GH < 8) return;

    if (fw4 != w || fh4 != h) {
        free(heat); free(pl);
        heat = calloc((size_t)GW * GH, sizeof(float));
        pl   = calloc((size_t)GW * GH, 1);
        if (!heat || !pl) {
            free(heat); free(pl); heat = NULL; pl = NULL;
            fw4 = fh4 = 0; return;
        }
        gw = GW; gh = GH; fw4 = w; fh4 = h;
    }

    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            const uint8_t *p = f->data + (gy*4) * f->stride + (gx*4) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw + gx;
            int d = abs((int)l - (int)pl[gi]);
            pl[gi] = l;
            float hv = heat[gi] * cool;
            if (d > 12) hv += gain * (float)d / 40.0f;
            heat[gi] = hv > 3.0f ? 3.0f : hv;
        }
    }

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        if (gy >= gh) gy = gh - 1;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            if (gx >= gw) gx = gw - 1;
            float hv = heat[gy * gw + gx];
            if (hv < 0.02f) continue;
            /* fire palette: red → orange → yellow → white */
            float v = hv > 1.0f ? 1.0f : hv;
            int r = (int)(v * 255.0f);
            int g = (int)(v * v * 210.0f);
            int b = (int)(v * v * v * 160.0f);
            int o = x * 3;
            dst[o]   = clamp_u8(dst[o]   + b);
            dst[o+1] = clamp_u8(dst[o+1] + g);
            dst[o+2] = clamp_u8(dst[o+2] + r);
        }
    }
}

/* ─── fx_wormhole ──
 * The immortal demoscene tunnel — using YOUR LIVE IMAGE as the
 * tunnel texture. Screen coords map to (angle, 1/r): you fly down a
 * tube whose walls are made of you.
 *   depth (p1): tube depth scale (def 60)
 *   speed (p2): flight speed (0 → 1)
 */
static void fx_wormhole(Frame *f, float depth, float speed)
{
    if (depth < 10.0f) depth = 60.0f;
    if (speed <= 0.0f) speed = 1.0f;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float t = (float)anim_tick() / 30.0f * speed;
    float fly = t * 60.0f;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = y - cy;
        for (int x = 0; x < w; x++) {
            float dx = x - cx;
            float r = sqrtf(dx*dx + dy*dy) + 1.0f;
            float a = atan2f(dy, dx);
            int sx = (int)(((a + 3.14159265f) / 6.2831853f) * (float)w
                           + t * 30.0f) % w;
            if (sx < 0) sx += w;
            int sy = (int)(depth * 300.0f / r + fly) % h;
            if (sy < 0) sy += h;
            const uint8_t *sp = tmp + sy * w * 3 + sx * 3;
            /* darken toward the far center for depth */
            float dim = r / (r + 60.0f);
            int o = x * 3;
            dst[o]   = (uint8_t)(sp[0] * dim);
            dst[o+1] = (uint8_t)(sp[1] * dim);
            dst[o+2] = (uint8_t)(sp[2] * dim);
        }
    }
    free(tmp);
}

/* ─── fx_magnet ──
 * The image is drawn toward the brightest region — light literally
 * bends space. The attractor point smooth-follows the brightest
 * coarse cell, so moving a phone flashlight around drags the whole
 * frame after it.
 *   strength (p1): pull amount 0..1 (def 0.5)
 *   follow   (p2): attractor smoothing 0.02..0.5 (0 → 0.1)
 */
static void fx_magnet(Frame *f, float strength, float follow)
{
    static float ax = -1.0f, ay = -1.0f;
    static int fw5 = 0, fh5 = 0;

    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    if (follow < 0.02f || follow > 0.5f) follow = 0.1f;

    int w = f->width, h = f->height;
    if (fw5 != w || fh5 != h) {
        ax = w / 2.0f; ay = h / 2.0f;
        fw5 = w; fh5 = h;
    }

    /* brightest 16px cell */
    int best = -1, bx = w/2, by = h/2;
    for (int gy = 8; gy < h - 8; gy += 16) {
        const uint8_t *row = f->data + gy * f->stride;
        for (int gx = 8; gx < w - 8; gx += 16) {
            const uint8_t *p = row + gx * 3;
            int l = p[0] + 2*p[1] + p[2];
            if (l > best) { best = l; bx = gx; by = gy; }
        }
    }
    ax += (bx - ax) * follow;
    ay += (by - ay) * follow;

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float maxr = sqrtf((float)(w*w + h*h)) * 0.5f;
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = y - ay;
        for (int x = 0; x < w; x++) {
            float dx = x - ax;
            float r = sqrtf(dx*dx + dy*dy);
            /* pull falls off with distance */
            float pull = strength * 0.55f * (1.0f - r / maxr);
            if (pull < 0) pull = 0;
            int sx = (int)(x + dx * pull);
            int sy = (int)(y + dy * pull);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = tmp + sy * w * 3 + sx * 3;
            int o = x * 3;
            dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
        }
    }
    free(tmp);
}

/* ─── fx_liquid ──
 * The whole frame flows like disturbed water — layered animated
 * sine displacement evaluated on a coarse grid (cheap), bilinearly
 * upsampled (smooth).
 *   amp   (p1): displacement px (def 12)
 *   scale (p2): wave size px (0 → 90)
 *   speed (p3): flow rate (0 → 1)
 */
static void fx_liquid(Frame *f, float amp, float scale, float speed)
{
    if (amp < 1.0f) amp = 12.0f;
    if (amp > 60.0f) amp = 60.0f;
    if (scale < 10.0f) scale = 90.0f;
    if (speed <= 0.0f) speed = 1.0f;

    int w = f->width, h = f->height;
    int GW = w / 4 + 2, GH = h / 4 + 2;
    float *grid = malloc((size_t)GW * GH * 2 * sizeof(float));
    uint8_t *src = malloc((size_t)w * h * 3);
    if (!grid || !src) { free(grid); free(src); return; }
    memcpy(src, f->data, (size_t)w * h * 3);

    float t = (float)anim_tick() / 30.0f * speed;
    float k = 1.0f / scale;

    for (int gy = 0; gy < GH; gy++) {
        float y = (float)(gy * 4);
        for (int gx = 0; gx < GW; gx++) {
            float x = (float)(gx * 4);
            int gi = (gy * GW + gx) * 2;
            grid[gi]   = amp * (sinf(y * k * 6.28f + t * 1.3f)
                              + 0.5f * sinf((x + y) * k * 4.1f - t * 0.9f));
            grid[gi+1] = amp * (cosf(x * k * 5.7f - t * 1.1f)
                              + 0.5f * cosf((x - y) * k * 3.7f + t * 0.7f));
        }
    }

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        float fy = (float)(y & 3) / 4.0f;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            float fx = (float)(x & 3) / 4.0f;
            int g00 = (gy * GW + gx) * 2;
            int g10 = g00 + 2, g01 = g00 + GW * 2, g11 = g01 + 2;
            float ddx = grid[g00]*(1-fx)*(1-fy) + grid[g10]*fx*(1-fy)
                      + grid[g01]*(1-fx)*fy     + grid[g11]*fx*fy;
            float ddy = grid[g00+1]*(1-fx)*(1-fy) + grid[g10+1]*fx*(1-fy)
                      + grid[g01+1]*(1-fx)*fy     + grid[g11+1]*fx*fy;
            int sx = x + (int)(ddx * 0.5f);
            int sy = y + (int)(ddy * 0.5f);
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = src + sy * w * 3 + sx * 3;
            int o = x * 3;
            dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
        }
    }
    free(grid);
    free(src);
}

/* ─── fx_nightvision ──
 * Gen-3 image intensifier look: luma amplified into green, live
 * sensor noise, subtle scanlines, hard circular vignette.
 *   gain  (p1): amplification 1..4 (def 2)
 *   noise (p2): noise amplitude 0..60 (0 → 25)
 */
static void fx_nightvision(Frame *f, float gain, float noise_f)
{
    if (gain < 1.0f) gain = 2.0f;
    if (gain > 4.0f) gain = 4.0f;
    int noise = (noise_f >= 1.0f && noise_f <= 60.0f) ? (int)noise_f : 25;

    int w = f->width, h = f->height;
    float cx = w / 2.0f, cy = h / 2.0f;
    float vr = (float)(w < h ? w : h) * 0.52f;
    float vr2 = vr * vr;
    unsigned seed = (unsigned)anim_tick() * 2654435761u;

    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        float dy = y - cy;
        float scan = (y & 1) ? 0.92f : 1.0f;
        for (int x = 0; x < w; x++, p += 3) {
            float dx = x - cx;
            float d2 = dx*dx + dy*dy;
            int l = (p[0]*114 + p[1]*587 + p[2]*299) / 1000;
            float v = (float)l * gain;
            /* soft knee so highlights don't clip flat */
            if (v > 200.0f) v = 200.0f + (v - 200.0f) * 0.3f;
            /* sensor noise */
            seed = seed * 1664525u + 1013904223u;
            int n = (int)((seed >> 24) % (unsigned)(2*noise+1)) - noise;
            v = (v + n) * scan;
            /* hard circular vignette */
            if (d2 > vr2) {
                float fall = 1.0f - (sqrtf(d2) - vr) / (vr * 0.25f);
                if (fall < 0) fall = 0;
                v *= fall;
            }
            int g = clamp_u8((int)v);
            p[0] = (uint8_t)(g / 5);
            p[1] = (uint8_t)g;
            p[2] = (uint8_t)(g / 6);
        }
    }
}

/* ─── fx_badtv ──
 * Dying CRT: continuous vertical roll, per-row horizontal tearing,
 * chroma wobble, and random static bursts.
 *   amount (p1): overall degradation 0..1 (def 0.5)
 *   roll   (p2): vertical roll speed rows/frame (0 → amount*3)
 */
static void fx_badtv(Frame *f, float amount, float roll_f)
{
    static float roll_pos = 0.0f;

    if (amount < 0.0f) amount = 0.0f;
    if (amount > 1.0f) amount = 1.0f;
    float roll = (roll_f > 0.0f) ? roll_f : amount * 3.0f;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    unsigned long t = anim_tick();
    roll_pos += roll;
    while (roll_pos >= (float)h) roll_pos -= (float)h;
    int roll_i = (int)roll_pos;

    /* occasional violent roll jump */
    if ((hash_u((unsigned)t * 31u) & 127) < (unsigned)(amount * 8))
        roll_pos += (float)(hash_u((unsigned)t) % (unsigned)h);

    /* static burst band this frame? */
    int burst_y0 = -1, burst_h = 0;
    if ((hash_u((unsigned)t * 17u) & 63) < (unsigned)(amount * 6)) {
        burst_y0 = (int)(hash_u((unsigned)t * 5u) % (unsigned)h);
        burst_h  = 6 + (int)(hash_u((unsigned)t * 9u) % 30u);
    }

    unsigned seed = (unsigned)t * 2654435761u;
    int chroma = 1 + (int)(amount * 4.0f);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int sy = (y + roll_i) % h;

        /* static burst rows: pure noise */
        if (burst_y0 >= 0 && y >= burst_y0 && y < burst_y0 + burst_h) {
            for (int x = 0; x < w; x++) {
                seed = seed * 1664525u + 1013904223u;
                uint8_t n = (uint8_t)(seed >> 24);
                dst[x*3+0] = n; dst[x*3+1] = n; dst[x*3+2] = n;
            }
            continue;
        }

        /* per-row tear */
        unsigned rh = hash_u((unsigned)(t * 131u) + (unsigned)y);
        int tear = 0;
        if ((rh & 31) < (unsigned)(amount * 6))
            tear = (int)(rh >> 8 & 31u) - 16;

        int cshift = (int)(sinf((float)sy * 0.05f + (float)t * 0.3f) * chroma);

        const uint8_t *srow = tmp + sy * w * 3;
        for (int x = 0; x < w; x++) {
            int sx = x + tear;
            if (sx < 0) sx += w;
            if (sx >= w) sx -= w;
            int xr = sx + cshift;
            if (xr < 0) xr = 0;
            if (xr >= w) xr = w - 1;
            dst[x*3+0] = srow[sx*3+0];
            dst[x*3+1] = srow[sx*3+1];
            dst[x*3+2] = srow[xr*3+2];   /* chroma wobble on R */
        }
    }
    free(tmp);
}

/* ─── fx_ghosts — motion extraction ──
 * Shows the DIFFERENCE between now and a moment ago, centered on
 * gray: static scenery collapses to flat gray, anything that moves
 * appears as a vivid embossed spectre with a leading and trailing
 * edge. The internet calls this "motion extraction" — it makes
 * invisible motion (breathing, heartbeat-ish sway) visible.
 *   delay (p1): frames back to compare against 1..15 (def 6)
 *   gain  (p2): difference amplification 1..6 (0 → 2)
 */
#define GHOSTS_RING 16
static void fx_ghosts(Frame *f, float delay_f, float gain)
{
    static uint8_t *ring[GHOSTS_RING] = {0};
    static int rw = 0, rh = 0, rs = 0, widx = 0;

    int delay = (int)delay_f;
    if (delay < 1) delay = 6;
    if (delay > GHOSTS_RING - 1) delay = GHOSTS_RING - 1;
    if (gain < 1.0f) gain = 2.0f;
    if (gain > 6.0f) gain = 6.0f;

    int size = f->stride * f->height;
    if (rw != f->width || rh != f->height || rs != f->stride) {
        for (int i = 0; i < GHOSTS_RING; i++) {
            free(ring[i]);
            ring[i] = malloc((size_t)size);
            if (!ring[i]) {
                for (int j = 0; j <= i; j++) { free(ring[j]); ring[j] = NULL; }
                rw = rh = rs = 0;
                return;
            }
            memcpy(ring[i], f->data, (size_t)size);
        }
        rw = f->width; rh = f->height; rs = f->stride;
        widx = 0;
    }

    memcpy(ring[widx], f->data, (size_t)size);
    int idx = (widx - delay + GHOSTS_RING) % GHOSTS_RING;
    const uint8_t *past = ring[idx];

    int n = f->stride * f->height;
    uint8_t *cur = f->data;
    for (int i = 0; i < n; i++) {
        int d = (int)cur[i] - (int)past[i];
        cur[i] = clamp_u8(128 + (int)(d * gain));
    }
    widx = (widx + 1) % GHOSTS_RING;
}

/* ──────────────────────────────────────────────────────────────
 * Batch 8: AI-craze volume 2
 * ────────────────────────────────────────────────────────────── */

/* ─── fx_clones ──
 * Multiplicity: every `interval` seconds a freeze-frame of the scene
 * is captured into a rotating set; all captures composite under the
 * live image with lighten-blend. Strike a pose, wait, move, strike
 * another — the room fills with copies of you.
 *   count    (p1): number of simultaneous clones 1..4 (def 3)
 *   interval (p2): seconds between captures 0.5..10 (0 → 2)
 *   alpha    (p3): clone opacity 0.2..1 (0 → 0.55)
 */
static void fx_clones(Frame *f, float count_f, float interval_s, float alpha)
{
    static uint8_t *snap[4] = {0};
    static int have[4] = {0};
    static int sw = 0, sh = 0, ss = 0, next_slot = 0;
    static unsigned long last_cap = 0;

    int count = (int)count_f;
    if (count < 1) count = 1;
    if (count > 4) count = 4;
    if (interval_s < 0.5f || interval_s > 10.0f) interval_s = 2.0f;
    if (alpha < 0.2f || alpha > 1.0f) alpha = 0.55f;

    int size = f->stride * f->height;
    if (sw != f->width || sh != f->height || ss != f->stride) {
        for (int i = 0; i < 4; i++) {
            free(snap[i]);
            snap[i] = malloc((size_t)size);
            have[i] = 0;
            if (!snap[i]) {
                for (int j = 0; j <= i; j++) { free(snap[j]); snap[j] = NULL; }
                sw = sh = ss = 0;
                return;
            }
        }
        sw = f->width; sh = f->height; ss = f->stride;
        next_slot = 0;
        last_cap = 0;
    }

    unsigned long t = anim_tick();
    unsigned long iv = (unsigned long)(interval_s * 30.0f);
    if (t - last_cap >= iv) {
        memcpy(snap[next_slot], f->data, (size_t)size);
        have[next_slot] = 1;
        next_slot = (next_slot + 1) % count;
        last_cap = t;
    }

    /* lighten-blend each clone under the live image */
    int a256 = (int)(alpha * 256.0f);
    for (int s = 0; s < count; s++) {
        if (!have[s]) continue;
        const uint8_t *sp = snap[s];
        uint8_t *cur = f->data;
        for (int i = 0; i < size; i++) {
            int v = (sp[i] * a256) >> 8;
            if (v > cur[i]) cur[i] = (uint8_t)v;
        }
    }
}

/* ─── fx_meltdown ──
 * The Doom screen-melt, live: every few seconds the current frame is
 * frozen, then its columns slide down and off the screen at slightly
 * different speeds, revealing the LIVE feed underneath. Loops forever.
 *   speed (p1): base fall px/frame 1..12 (def 3)
 *   chaos (p2): per-column speed variance 0..1 (0 → 0.5)
 */
static void fx_meltdown(Frame *f, float speed_f, float chaos)
{
    static uint8_t *snap = NULL;
    static float   *coff = NULL;    /* per-column fall offset */
    static float   *cspd = NULL;    /* per-column speed factor */
    static int mw2 = 0, mh2 = 0, ms2 = 0;
    static int phase = 0;           /* 0 = waiting to snapshot, 1 = melting */
    static unsigned long hold_until = 0;

    float speed = speed_f;
    if (speed < 1.0f)  speed = 3.0f;
    if (speed > 12.0f) speed = 12.0f;
    if (chaos < 0.0f || chaos > 1.0f) chaos = 0.5f;

    int w = f->width, h = f->height;
    int size = f->stride * h;

    if (mw2 != w || mh2 != h || ms2 != f->stride) {
        free(snap); free(coff); free(cspd);
        snap = malloc((size_t)size);
        coff = calloc((size_t)w, sizeof(float));
        cspd = malloc((size_t)w * sizeof(float));
        if (!snap || !coff || !cspd) {
            free(snap); free(coff); free(cspd);
            snap = NULL; coff = NULL; cspd = NULL;
            mw2 = mh2 = ms2 = 0;
            return;
        }
        mw2 = w; mh2 = h; ms2 = f->stride;
        phase = 0;
        hold_until = anim_tick() + 45;   /* 1.5s before first melt */
    }

    unsigned long t = anim_tick();

    if (phase == 0) {
        if (t >= hold_until) {
            /* freeze this frame and start melting it */
            memcpy(snap, f->data, (size_t)size);
            for (int c = 0; c < w; c++) {
                coff[c] = -(float)(rand() % (int)(h * 0.15f + 1));  /* staggered start */
                cspd[c] = 1.0f + chaos * ((float)(rand() % 200) / 100.0f - 1.0f);
                if (cspd[c] < 0.3f) cspd[c] = 0.3f;
            }
            phase = 1;
        }
        return;   /* live feed passes through untouched */
    }

    /* melting: draw snapshot columns at their offsets over live feed */
    int done = 1;
    for (int c = 0; c < w; c++) {
        coff[c] += speed * cspd[c];
        if (coff[c] < (float)h) done = 0;
    }
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int c = 0; c < w; c++) {
            float off = coff[c];
            if (off < 0) off = 0;
            int sy = y - (int)off;
            if (sy < 0) continue;          /* above the snapshot: live shows */
            const uint8_t *sp = snap + sy * f->stride + c * 3;
            dst[c*3+0] = sp[0];
            dst[c*3+1] = sp[1];
            dst[c*3+2] = sp[2];
        }
    }
    if (done) {
        phase = 0;
        hold_until = t + 60 + (unsigned)(rand() % 90);   /* 2-5s of live */
    }
}

/* ─── fx_rotozoom ──
 * The demoscene rotozoomer: the frame becomes an infinite tiled plane
 * that rotates and breathes. Mirror-tiled so seams are invisible.
 *   speed   (p1): rotation rate (def 1)
 *   zoomamp (p2): zoom oscillation amount 0..1 (0 → 0.4)
 */
static void fx_rotozoom(Frame *f, float speed, float zoomamp)
{
    if (speed <= 0.0f) speed = 1.0f;
    if (zoomamp < 0.0f || zoomamp > 1.0f) zoomamp = 0.4f;

    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float t = (float)anim_tick() / 30.0f * speed;
    float ang = t * 0.5f;
    float zoom = 1.0f + zoomamp * 0.6f * sinf(t * 0.8f);
    float ca = cosf(ang) / zoom, sa = sinf(ang) / zoom;
    float cx = w / 2.0f, cy = h / 2.0f;
    int w2 = w * 2, h2 = h * 2;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = y - cy;
        for (int x = 0; x < w; x++) {
            float dx = x - cx;
            int sx = (int)(cx + dx * ca - dy * sa);
            int sy = (int)(cy + dx * sa + dy * ca);
            /* mirror-wrap for seamless infinite tiling */
            sx = ((sx % w2) + w2) % w2;
            sy = ((sy % h2) + h2) % h2;
            if (sx >= w) sx = w2 - 1 - sx;
            if (sy >= h) sy = h2 - 1 - sy;
            const uint8_t *sp = tmp + sy * w * 3 + sx * 3;
            int o = x * 3;
            dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
        }
    }
    free(tmp);
}

/* ─── fx_starfield ──
 * Hyperspace: stars fly out of the center with speed streaks,
 * additively over the (slightly dimmed) video.
 *   density (p1): star count fraction 0..1 (def 0.5 → ~200 stars)
 *   speed   (p2): flight speed (0 → 1)
 */
#define SF_MAX 400
static void fx_starfield(Frame *f, float density, float speed)
{
    static float sx[SF_MAX], sy[SF_MAX], sz[SF_MAX];
    static int inited = 0;

    if (density < 0.05f) density = 0.05f;
    if (density > 1.0f)  density = 1.0f;
    if (speed <= 0.0f) speed = 1.0f;
    int nstars = (int)(density * SF_MAX);

    int w = f->width, h = f->height;
    float cx = w / 2.0f, cy = h / 2.0f;

    if (!inited) {
        for (int i = 0; i < SF_MAX; i++) {
            sx[i] = ((float)(rand() % 2000) / 1000.0f - 1.0f);
            sy[i] = ((float)(rand() % 2000) / 1000.0f - 1.0f);
            sz[i] = 0.1f + (float)(rand() % 900) / 1000.0f;
        }
        inited = 1;
    }

    /* slight dim so stars pop */
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w * 3; x++)
            p[x] = (uint8_t)((p[x] * 3) / 4);
    }

    float zstep = 0.012f * speed;
    float fl = (float)(w < h ? w : h) * 0.9f;

    for (int i = 0; i < nstars; i++) {
        float z0 = sz[i], z1 = sz[i] - zstep;
        if (z1 <= 0.02f) {
            sx[i] = ((float)(rand() % 2000) / 1000.0f - 1.0f);
            sy[i] = ((float)(rand() % 2000) / 1000.0f - 1.0f);
            sz[i] = 1.0f;
            continue;
        }
        sz[i] = z1;
        int px0 = (int)(cx + sx[i] / z0 * fl * 0.5f);
        int py0 = (int)(cy + sy[i] / z0 * fl * 0.5f);
        int px1 = (int)(cx + sx[i] / z1 * fl * 0.5f);
        int py1 = (int)(cy + sy[i] / z1 * fl * 0.5f);
        if (px1 < 1 || px1 >= w - 1 || py1 < 1 || py1 >= h - 1) {
            sz[i] = 0.01f;   /* respawn next frame */
            continue;
        }
        int bright = (int)((1.0f - z1) * 255.0f);
        /* streak: a few steps between old and new position */
        for (int s = 0; s <= 4; s++) {
            int px = px0 + (px1 - px0) * s / 4;
            int py = py0 + (py1 - py0) * s / 4;
            if (px < 0 || px >= w || py < 0 || py >= h) continue;
            uint8_t *d = f->data + py * f->stride + px * 3;
            int b2 = bright * (s + 2) / 6;
            d[0] = clamp_u8(d[0] + b2);
            d[1] = clamp_u8(d[1] + b2);
            d[2] = clamp_u8(d[2] + (b2 * 9) / 10);   /* faint blue tint */
        }
    }
}

/* ─── fx_lightning — INTERACTIVE ──
 * Bolts strike from the top of the frame to WHEREVER YOU MOVE —
 * jagged branched paths with blue-white glow and a full-frame flash
 * that fades over a few frames. Snap your hand out and get struck.
 *   rate     (p1): strike probability 0..1 (def 0.3)
 *   branches (p2): forks per bolt 0..6 (0 → 3)
 */
static void draw_bolt_seg(Frame *f, int x0, int y0, int x1, int y1)
{
    int steps = abs(y1 - y0) + abs(x1 - x0);
    if (steps < 1) steps = 1;
    for (int s = 0; s <= steps; s++) {
        int x = x0 + (x1 - x0) * s / steps;
        int y = y0 + (y1 - y0) * s / steps;
        for (int dx = -1; dx <= 1; dx++) {
            int xx = x + dx;
            if (xx < 0 || xx >= f->width || y < 0 || y >= f->height) continue;
            uint8_t *d = f->data + y * f->stride + xx * 3;
            int core = (dx == 0) ? 255 : 130;
            d[0] = clamp_u8(d[0] + core);
            d[1] = clamp_u8(d[1] + (core * 4) / 5);
            d[2] = clamp_u8(d[2] + (core * 3) / 5);
        }
    }
}

static void fx_lightning(Frame *f, float rate, float branches_f)
{
    static uint8_t *pl = NULL;
    static int gw6 = 0, gh6 = 0, fw6 = 0, fh6 = 0;
    static float flash = 0.0f;

    if (rate < 0.0f) rate = 0.0f;
    if (rate > 1.0f) rate = 1.0f;
    int branches = (int)branches_f;
    if (branches_f <= 0.0f) branches = 3;
    if (branches > 6) branches = 6;

    int w = f->width, h = f->height;
    int GW = w / 8, GH = h / 8;
    if (GW < 8 || GH < 8) return;

    if (fw6 != w || fh6 != h) {
        free(pl);
        pl = calloc((size_t)GW * GH, 1);
        if (!pl) { fw6 = fh6 = 0; return; }
        gw6 = GW; gh6 = GH; fw6 = w; fh6 = h;
    }

    /* motion centroid (strike target) */
    long mx = 0, my = 0, mtot = 0;
    for (int gy = 1; gy < gh6 - 1; gy++) {
        for (int gx = 1; gx < gw6 - 1; gx++) {
            const uint8_t *p = f->data + (gy*8) * f->stride + (gx*8) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw6 + gx;
            int d = abs((int)l - (int)pl[gi]);
            pl[gi] = l;
            if (d > 20) { mx += (long)gx*8*d; my += (long)gy*8*d; mtot += d; }
        }
    }

    /* fading flash from previous strikes */
    if (flash > 0.01f) {
        int add = (int)(flash * 70.0f);
        for (int y = 0; y < h; y++) {
            uint8_t *p = f->data + y * f->stride;
            for (int x = 0; x < w * 3; x++)
                p[x] = clamp_u8(p[x] + add);
        }
        flash *= 0.6f;
    }

    /* strike? needs motion (or small ambient chance) */
    int want = 0;
    int tx = w / 2, ty = h - h / 5;
    if (mtot > 400 && (rand() % 100) < (int)(rate * 35.0f)) {
        want = 1;
        tx = (int)(mx / mtot);
        ty = (int)(my / mtot);
    } else if ((rand() % 1000) < (int)(rate * 8.0f)) {
        want = 1;
        tx = rand() % w;
        ty = h / 3 + rand() % (h / 2);
    }
    if (!want) return;

    /* jagged main bolt from random top point to target */
    int x = rand() % w, y = 0;
    int seg_h = h / 12 + 1;
    while (y < ty) {
        int ny = y + seg_h / 2 + rand() % seg_h;
        if (ny > ty) ny = ty;
        /* drift toward target with jitter */
        int nx = x + (tx - x) * (ny - y) / (ty - y + 1)
               + (rand() % (w / 6 + 1)) - w / 12;
        if (nx < 0) nx = 0;
        if (nx >= w) nx = w - 1;
        draw_bolt_seg(f, x, y, nx, ny);
        /* occasional branch shooting off */
        if (branches > 0 && (rand() % 6) < 2) {
            int bx = nx + (rand() % (w / 4 + 1)) - w / 8;
            int by = ny + h / 10 + rand() % (h / 8 + 1);
            if (bx < 0) bx = 0;
            if (bx >= w) bx = w - 1;
            if (by >= h) by = h - 1;
            draw_bolt_seg(f, nx, ny, bx, by);
            branches--;
        }
        x = nx; y = ny;
    }
    flash = 1.0f;
}

/* ─── fx_scatter — INTERACTIVE ──
 * Motion shatters the image: cells you disturb get a random outward
 * displacement impulse, then spring back — the frame explodes where
 * you touch it and heals itself.
 *   power (p1): impulse strength 0..1 (def 0.6)
 *   heal  (p2): spring return per frame 0.7..0.98 (0 → 0.88)
 */
static void fx_scatter(Frame *f, float power, float heal)
{
    static float *offx = NULL, *offy = NULL;
    static uint8_t *pl = NULL;
    static int gw7 = 0, gh7 = 0, fw7 = 0, fh7 = 0;

    if (power < 0.05f) power = 0.05f;
    if (power > 1.0f)  power = 1.0f;
    if (heal < 0.7f || heal > 0.98f) heal = 0.88f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    if (GW < 8 || GH < 8) return;

    if (fw7 != w || fh7 != h) {
        free(offx); free(offy); free(pl);
        offx = calloc((size_t)GW * GH, sizeof(float));
        offy = calloc((size_t)GW * GH, sizeof(float));
        pl   = calloc((size_t)GW * GH, 1);
        if (!offx || !offy || !pl) {
            free(offx); free(offy); free(pl);
            offx = offy = NULL; pl = NULL;
            fw7 = fh7 = 0;
            return;
        }
        gw7 = GW; gh7 = GH; fw7 = w; fh7 = h;
    }

    for (int gy = 0; gy < gh7; gy++) {
        for (int gx = 0; gx < gw7; gx++) {
            const uint8_t *p = f->data + (gy*4) * f->stride + (gx*4) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw7 + gx;
            int d = abs((int)l - (int)pl[gi]);
            pl[gi] = l;
            if (d > 16) {
                float ang = (float)(hash_u((unsigned)(gi * 2654435761u
                              + (unsigned)anim_tick())) % 628u) * 0.01f;
                float imp = power * (float)d * 0.35f;
                if (imp > 40.0f) imp = 40.0f;
                offx[gi] += cosf(ang) * imp;
                offy[gi] += sinf(ang) * imp;
            }
            offx[gi] *= heal;
            offy[gi] *= heal;
        }
    }

    uint8_t *src = malloc((size_t)w * h * 3);
    if (!src) return;
    memcpy(src, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        int gy = y / 4;
        if (gy >= gh7) gy = gh7 - 1;
        for (int x = 0; x < w; x++) {
            int gx = x / 4;
            if (gx >= gw7) gx = gw7 - 1;
            int gi = gy * gw7 + gx;
            float ox = offx[gi], oy = offy[gi];
            if (ox > -0.5f && ox < 0.5f && oy > -0.5f && oy < 0.5f) continue;
            int sx = x + (int)ox;
            int sy = y + (int)oy;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            if (sy < 0) sy = 0;
            if (sy >= h) sy = h - 1;
            const uint8_t *sp = src + sy * w * 3 + sx * 3;
            int o = x * 3;
            dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
        }
    }
    free(src);
}

/* ─── fx_vhs ──
 * The complete VHS experience: horizontal chroma bleed, luma noise,
 * a jittering head-switching noise bar at the bottom, and a live
 * counting timecode rendered from an embedded digit font.
 *   wear     (p1): degradation 0..1 (def 0.5)
 *   timecode (p2): 1 = show counter (default), 0 = hide
 */
static const uint8_t vhs_digits[11][8] = {
    {0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0}, /* 0 */
    {0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0}, /* 1 */
    {0x3C,0x66,0x06,0x0C,0x18,0x30,0x7E,0}, /* 2 */
    {0x3C,0x66,0x06,0x1C,0x06,0x66,0x3C,0}, /* 3 */
    {0x0C,0x1C,0x3C,0x6C,0x7E,0x0C,0x0C,0}, /* 4 */
    {0x7E,0x60,0x7C,0x06,0x06,0x66,0x3C,0}, /* 5 */
    {0x1C,0x30,0x60,0x7C,0x66,0x66,0x3C,0}, /* 6 */
    {0x7E,0x06,0x0C,0x18,0x30,0x30,0x30,0}, /* 7 */
    {0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0}, /* 8 */
    {0x3C,0x66,0x66,0x3E,0x06,0x0C,0x38,0}, /* 9 */
    {0x00,0x18,0x18,0x00,0x18,0x18,0x00,0}, /* : */
};

static void vhs_draw_glyph(Frame *f, int gi, int x0, int y0)
{
    for (int y = 0; y < 16; y++) {
        int yy = y0 + y;
        if (yy < 0 || yy >= f->height) continue;
        uint8_t bits = vhs_digits[gi][y / 2];
        uint8_t *row = f->data + yy * f->stride;
        for (int x = 0; x < 16; x++) {
            if (!(bits & (0x80 >> (x / 2)))) continue;
            int xx = x0 + x;
            if (xx < 0 || xx >= f->width) continue;
            uint8_t *p = row + xx * 3;
            p[0] = 240; p[1] = 240; p[2] = 240;
        }
    }
}

static void fx_vhs(Frame *f, float wear, float timecode)
{
    if (wear < 0.0f) wear = 0.0f;
    if (wear > 1.0f) wear = 1.0f;
    int show_tc = (timecode >= 0.5f) || (timecode == 0.0f);   /* default on */

    int w = f->width, h = f->height;
    unsigned long t = anim_tick();
    unsigned seed = (unsigned)t * 2654435761u;

    /* chroma bleed: R and B smeared horizontally, G stays sharp */
    int bleed = 2 + (int)(wear * 5.0f);
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);
    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        const uint8_t *src = tmp + y * w * 3;
        for (int x = 0; x < w; x++) {
            int sb = 0, sr = 0, cnt = 0;
            for (int dx = -bleed; dx <= bleed; dx += 2) {
                int xx = x + dx;
                if (xx < 0 || xx >= w) continue;
                sb += src[xx*3+0];
                sr += src[xx*3+2];
                cnt++;
            }
            int o = x * 3;
            dst[o]   = (uint8_t)(sb / cnt);
            dst[o+2] = (uint8_t)(sr / cnt);
            /* luma noise on G-ish overall */
            seed = seed * 1664525u + 1013904223u;
            int n = (int)((seed >> 26) & 15u) - 8;
            dst[o+1] = clamp_u8(src[o+1] + (n * (int)(wear * 10.0f)) / 8);
        }
    }
    free(tmp);

    /* head-switching noise bar at the bottom */
    int bar_h = 6 + (int)(wear * 10.0f);
    int jitter = (int)((hash_u((unsigned)t * 7u) % 9u)) - 4;
    for (int y = h - bar_h; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            seed = seed * 1664525u + 1013904223u;
            uint8_t n = (uint8_t)(seed >> 24);
            int sx = x + jitter;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            uint8_t *sp2 = f->data + (h - bar_h - 2) * f->stride + sx * 3;
            dst[x*3+0] = (uint8_t)((sp2[0] + n) / 2);
            dst[x*3+1] = (uint8_t)((sp2[1] + n) / 2);
            dst[x*3+2] = (uint8_t)((sp2[2] + n) / 2);
        }
    }

    /* timecode 0:00:00 bottom-left, counting real playtime */
    if (show_tc) {
        unsigned secs = (unsigned)(t / 30);
        unsigned hh2 = secs / 3600, mm = (secs / 60) % 60, ss2 = secs % 60;
        int gl[8] = { (int)(hh2 % 10), 10,
                      (int)(mm / 10), (int)(mm % 10), 10,
                      (int)(ss2 / 10), (int)(ss2 % 10), -1 };
        int x0 = 20, y0 = h - bar_h - 30;
        for (int i = 0; gl[i] >= 0; i++) {
            vhs_draw_glyph(f, gl[i], x0, y0);
            x0 += 18;
        }
    }
}

/* ─── fx_outline — INTERACTIVE ──
 * A marching-ants animated outline around whatever is MOVING —
 * comic-book selection lasso around you. Static scenery gets no
 * outline; wave and your silhouette gets traced in cycling color
 * with animated dashes.
 *   thick  (p1): outline thickness in 4px cells, 1..4 (def 2)
 *   huespd (p2): color cycle rate (0 → 1)
 */
static void fx_outline(Frame *f, float thick_f, float huespd)
{
    static uint8_t *mask = NULL;   /* decaying motion mask */
    static uint8_t *pl = NULL;
    static int gw8 = 0, gh8 = 0, fw8 = 0, fh8 = 0;

    int thick = (int)thick_f;
    if (thick < 1) thick = 2;
    if (thick > 4) thick = 4;
    if (huespd <= 0.0f) huespd = 1.0f;

    int w = f->width, h = f->height;
    int GW = w / 4, GH = h / 4;
    if (GW < 8 || GH < 8) return;

    if (fw8 != w || fh8 != h) {
        free(mask); free(pl);
        mask = calloc((size_t)GW * GH, 1);
        pl   = calloc((size_t)GW * GH, 1);
        if (!mask || !pl) {
            free(mask); free(pl); mask = NULL; pl = NULL;
            fw8 = fh8 = 0;
            return;
        }
        gw8 = GW; gh8 = GH; fw8 = w; fh8 = h;
    }

    for (int gy = 0; gy < gh8; gy++) {
        for (int gx = 0; gx < gw8; gx++) {
            const uint8_t *p = f->data + (gy*4) * f->stride + (gx*4) * 3;
            uint8_t l = (uint8_t)((p[0]*114 + p[1]*587 + p[2]*299) / 1000);
            int gi = gy * gw8 + gx;
            int d = abs((int)l - (int)pl[gi]);
            pl[gi] = l;
            int m = (mask[gi] * 6) / 8;
            if (d > 15) m = 255;
            mask[gi] = (uint8_t)m;
        }
    }

    unsigned long t = anim_tick();
    float hue = (float)(t % 300) / 300.0f * 6.28318f * huespd;
    int cb = (int)((sinf(hue + 4.189f) * 0.5f + 0.5f) * 255.0f);
    int cg = (int)((sinf(hue + 2.094f) * 0.5f + 0.5f) * 255.0f);
    int cr = (int)((sinf(hue)          * 0.5f + 0.5f) * 255.0f);

    /* border cell: masked, with an unmasked cell within `thick` */
    for (int gy = thick; gy < gh8 - thick; gy++) {
        for (int gx = thick; gx < gw8 - thick; gx++) {
            int gi = gy * gw8 + gx;
            if (mask[gi] < 60) continue;
            int edge = 0;
            for (int dy = -thick; dy <= thick && !edge; dy++)
                for (int dx = -thick; dx <= thick && !edge; dx++)
                    if (mask[(gy+dy) * gw8 + gx+dx] < 60) edge = 1;
            if (!edge) continue;
            /* marching ants: dashed by diagonal phase */
            if (((gx + gy + (int)(t / 3)) & 3) >= 2) continue;
            for (int py = 0; py < 4; py++) {
                uint8_t *dst = f->data + (gy*4 + py) * f->stride + gx*4*3;
                for (int px = 0; px < 4; px++) {
                    dst[px*3+0] = (uint8_t)cb;
                    dst[px*3+1] = (uint8_t)cg;
                    dst[px*3+2] = (uint8_t)cr;
                }
            }
        }
    }
}

/* ─── fx_symmetry ──
 * Face symmetry mirrors — the party classic.
 *   mode (p1): 0 = left half mirrored right (def)
 *              1 = right half mirrored left
 *              2 = quad symmetry (top-left kaleido-mirrored 4 ways)
 */
static void fx_symmetry(Frame *f, float mode_f)
{
    int mode = (int)mode_f;
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;

    int w = f->width, h = f->height;

    if (mode == 0) {
        for (int y = 0; y < h; y++) {
            uint8_t *row = f->data + y * f->stride;
            for (int x = w / 2; x < w; x++) {
                const uint8_t *src = row + (w - 1 - x) * 3;
                row[x*3+0] = src[0]; row[x*3+1] = src[1]; row[x*3+2] = src[2];
            }
        }
    } else if (mode == 1) {
        for (int y = 0; y < h; y++) {
            uint8_t *row = f->data + y * f->stride;
            for (int x = 0; x < w / 2; x++) {
                const uint8_t *src = row + (w - 1 - x) * 3;
                row[x*3+0] = src[0]; row[x*3+1] = src[1]; row[x*3+2] = src[2];
            }
        }
    } else {
        for (int y = 0; y < h; y++) {
            uint8_t *row = f->data + y * f->stride;
            int sy = (y < h / 2) ? y : (h - 1 - y);
            const uint8_t *srow = f->data + sy * f->stride;
            for (int x = 0; x < w; x++) {
                int sx = (x < w / 2) ? x : (w - 1 - x);
                if (sy == y && sx == x) continue;
                const uint8_t *src = srow + sx * 3;
                row[x*3+0] = src[0]; row[x*3+1] = src[1]; row[x*3+2] = src[2];
            }
        }
    }
}

/* ─── fx_timebubble ──
 * A glowing bubble drifts around the frame — INSIDE it, you see the
 * PAST. History is kept at quarter resolution (up to ~3 seconds),
 * shown pixel-doubled inside the lens with a cyan rim.
 *   delay  (p1): seconds into the past 0.3..3 (def 1.5)
 *   radius (p2): bubble radius px (0 → 160)
 */
#define TB_RING 90
static void fx_timebubble(Frame *f, float delay_s, float radius)
{
    static uint8_t *ring[TB_RING] = {0};
    static int qw = 0, qh = 0, widx2 = 0, fw9 = 0, fh9 = 0;

    if (delay_s < 0.3f || delay_s > 3.0f) delay_s = 1.5f;
    if (radius < 50.0f) radius = 160.0f;

    int w = f->width, h = f->height;
    int QW = w / 4, QH = h / 4;
    size_t qsize = (size_t)QW * QH * 3;

    if (fw9 != w || fh9 != h) {
        for (int i = 0; i < TB_RING; i++) {
            free(ring[i]);
            ring[i] = malloc(qsize);
            if (!ring[i]) {
                for (int j = 0; j <= i; j++) { free(ring[j]); ring[j] = NULL; }
                fw9 = fh9 = 0;
                return;
            }
        }
        qw = QW; qh = QH; fw9 = w; fh9 = h;
        widx2 = 0;
    }

    /* store quarter-res copy of the current frame */
    for (int qy = 0; qy < qh; qy++) {
        uint8_t *drow = ring[widx2] + qy * qw * 3;
        const uint8_t *srow = f->data + (qy*4) * f->stride;
        for (int qx = 0; qx < qw; qx++) {
            const uint8_t *sp = srow + (qx*4) * 3;
            drow[qx*3+0] = sp[0];
            drow[qx*3+1] = sp[1];
            drow[qx*3+2] = sp[2];
        }
    }

    int back = (int)(delay_s * 30.0f);
    if (back > TB_RING - 1) back = TB_RING - 1;
    const uint8_t *past = ring[(widx2 - back + TB_RING) % TB_RING];
    widx2 = (widx2 + 1) % TB_RING;

    /* drifting bubble center */
    float t = (float)anim_tick() / 30.0f;
    float cx = w * 0.5f + sinf(t * 0.31f) * w * 0.24f;
    float cy = h * 0.5f + cosf(t * 0.23f) * h * 0.22f;
    float r2 = radius * radius;
    float rim = radius * 0.05f + 3.0f;

    int y0 = (int)(cy - radius - rim), y1 = (int)(cy + radius + rim);
    if (y0 < 0) y0 = 0;
    if (y1 > h) y1 = h;
    for (int y = y0; y < y1; y++) {
        uint8_t *dst = f->data + y * f->stride;
        float dy = y - cy;
        for (int x = (int)(cx - radius - rim > 0 ? cx - radius - rim : 0);
             x < w && x < (int)(cx + radius + rim); x++) {
            float dx = x - cx;
            float d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                int qx = x / 4, qy = y / 4;
                if (qx >= qw) qx = qw - 1;
                if (qy >= qh) qy = qh - 1;
                const uint8_t *sp = past + (qy * qw + qx) * 3;
                int o = x * 3;
                dst[o] = sp[0]; dst[o+1] = sp[1]; dst[o+2] = sp[2];
            } else if (fabsf(sqrtf(d2) - radius) < rim) {
                float e = 1.0f - fabsf(sqrtf(d2) - radius) / rim;
                int o = x * 3;
                dst[o]   = clamp_u8(dst[o]   + (int)(200 * e));
                dst[o+1] = clamp_u8(dst[o+1] + (int)(160 * e));
                dst[o+2] = clamp_u8(dst[o+2] + (int)(40  * e));
            }
        }
    }
}

void filter_chain_apply(FilterChain *fc, Frame *frame)
{
    pthread_mutex_lock(&fc->lock);
    int n = fc->count;
    Filter local[MAX_FILTERS];
    if (n > 0) memcpy(local, fc->filters, n * sizeof(Filter));
    pthread_mutex_unlock(&fc->lock);

    /* Apply outside the lock so adding/removing isn't blocked while we work */
    for (int i = 0; i < n; i++) {
        Filter *f = &local[i];
        switch (f->type) {
        case FX_GRAYSCALE:    fx_grayscale(frame); break;
        case FX_SEPIA:        fx_sepia(frame); break;
        case FX_INVERT:       fx_invert(frame); break;
        case FX_MIRROR:       fx_mirror(frame); break;
        case FX_FLIP:         fx_flip(frame); break;
        case FX_PIXELATE:     fx_pixelate(frame, (int)f->param); break;
        case FX_BLUR:         fx_blur(frame, (int)f->param); break;
        case FX_VIGNETTE:     fx_vignette(frame, f->param, f->param2, f->param3); break;
        case FX_BRIGHTNESS:   fx_brightness(frame, f->param); break;
        case FX_CONTRAST:     fx_contrast(frame, f->param); break;
        case FX_POSTERIZE:    fx_posterize(frame, (int)f->param); break;
        case FX_EDGES:        fx_edges(frame, f->param); break;
        case FX_SATURATE:     fx_saturate(frame, f->param); break;
        case FX_CHROMASHIFT:  fx_chromashift(frame, (int)f->param,
                                             (int)f->param2, (int)f->param3); break;
        case FX_GLITCH:       fx_glitch(frame, f->param,
                                        (int)f->param2, f->param3); break;
        case FX_KALEIDOSCOPE: fx_kaleidoscope(frame, (int)f->param,
                                              f->param2, f->param3,
                                              f->param4); break;
        case FX_SWIRL:        fx_swirl(frame, f->param, f->param2, f->param3); break;
        case FX_THERMAL:      fx_thermal(frame); break;
        case FX_OLDFILM:      fx_oldfilm(frame); break;
        case FX_WAVE:         fx_wave(frame, f->param, f->param2,
                                      f->param3, (int)f->param4); break;
        case FX_CARTOON:      fx_cartoon(frame, f->param, f->param2); break;
        case FX_SOLARIZE:     fx_solarize(frame, (int)f->param); break;
        case FX_RAINBOW:      fx_rainbow(frame, f->param); break;
        case FX_CRT:          fx_crt(frame, f->param); break;
        case FX_DREAM:        fx_dream(frame, f->param); break;
        case FX_BLOOM:        fx_bloom(frame, f->param); break;
        case FX_GLOW:         fx_glow(frame, f->param); break;
        case FX_PINCH:        fx_pinch(frame, f->param, f->param2, f->param3); break;
        case FX_BULGE:        fx_bulge(frame, f->param, f->param2, f->param3); break;
        case FX_PAINTERLY:    fx_painterly(frame, (int)f->param); break;
        case FX_VELVET:       fx_velvet(frame, f->param); break;
        case FX_MODERN:       fx_modern(frame, f->param); break;
        case FX_CYBERPUNK:    fx_cyberpunk(frame, f->param); break;
        case FX_ROMANTIC:     fx_romantic(frame, f->param); break;
        case FX_VANGOGH:      fx_vangogh(frame, f->param); break;
        case FX_PICASSO:      fx_picasso(frame, f->param); break;
        case FX_NOIR:         fx_noir(frame, f->param); break;
        case FX_SINE:         fx_sine(frame, f->param); break;
        case FX_XOR:          fx_xor(frame, f->param); break;
        case FX_MODULO:       fx_modulo(frame, f->param); break;
        case FX_POLAR:        fx_polar(frame, f->param); break;
        case FX_VORONOI:      fx_voronoi(frame, f->param); break;
        case FX_JULIA:        fx_julia(frame, f->param); break;
        case FX_BITCRUSH:     fx_bitcrush(frame, f->param); break;
        case FX_MELT:         fx_melt(frame, f->param); break;
        case FX_POPART:       fx_popart(frame, f->param); break;
        case FX_DISPLACE:     fx_displace(frame, f->param); break;
        case FX_RIPPLE:       fx_ripple(frame, f->param); break;
        case FX_SCANLINES:    fx_scanlines(frame, f->param); break;
        /* glitch art */
        case FX_SHUFFLE:      fx_shuffle(frame, f->param, f->param2, f->param3); break;
        case FX_PIXELSORT:    fx_pixelsort(frame, f->param); break;
        case FX_RGB_DELAY:    fx_rgb_delay(frame, f->param); break;
        case FX_DRIFT:        fx_drift(frame, f->param, f->param2); break;
        case FX_RAIN:         fx_rain(frame, f->param, f->param2, f->param3); break;
        /* temporal */
        case FX_ECHO:         fx_echo(frame, f->param); break;
        case FX_SLITSCAN:     fx_slitscan(frame, f->param, f->param2); break;
        case FX_MOSH:         fx_mosh(frame, f->param, f->param2); break;
        case FX_STUTTER:      fx_stutter(frame, f->param); break;
        case FX_HALFTONE:     fx_halftone(frame, f->param); break;
        case FX_NEON:         fx_neon(frame, f->param, f->param2,
                                      f->param3, f->param4); break;
        /* retro & warp */
        case FX_EMBOSS:       fx_emboss(frame, f->param); break;
        case FX_DITHER:       fx_dither(frame, f->param); break;
        case FX_TILES:        fx_tiles(frame, f->param); break;
        case FX_ZOOMPULSE:    fx_zoompulse(frame, f->param, f->param2); break;
        case FX_OILSLICK:     fx_oilslick(frame, f->param, f->param2); break;
        /* interactive & extravagant */
        case FX_FEEDBACK:     fx_feedback(frame, f->param, f->param2, f->param3); break;
        case FX_RIPPLES:      fx_ripples(frame, f->param, f->param2, f->param3); break;
        case FX_LIGHTPAINT:   fx_lightpaint(frame, f->param, f->param2, f->param3); break;
        case FX_HOLOGRAM:     fx_hologram(frame, f->param, f->param2, f->param3); break;
        case FX_LIFE:         fx_life(frame, f->param, f->param2, f->param3); break;
        case FX_MATHWARP:     fx_mathwarp(frame, f->param); break;
        /* demoscene & particles */
        case FX_PLASMA:       fx_plasma(frame, f->param, f->param2, f->param3); break;
        case FX_CONTOUR:      fx_contour(frame, f->param, f->param2); break;
        case FX_FIREWORKS:    fx_fireworks(frame, f->param, f->param2); break;
        case FX_INFINITY:     fx_infinity(frame, f->param, f->param2); break;
        case FX_MOIRE:        fx_moire(frame, f->param, f->param2); break;
        /* text, tone & tracking */
        case FX_ASCII:        fx_ascii(frame, f->param, f->param2); break;
        case FX_DUOTONE:      fx_duotone(frame, f->param, f->param2, f->param3); break;
        case FX_SKETCH:       fx_sketch(frame, f->param, f->param2); break;
        case FX_SPOTLIGHT:    fx_spotlight(frame, f->param, f->param2, f->param3); break;
        /* the AI-craze set */
        case FX_PREDATOR:     fx_predator(frame, f->param, f->param2, f->param3); break;
        case FX_SHARDS:       fx_shards(frame, f->param, f->param2); break;
        case FX_PORTAL:       fx_portal(frame, f->param, f->param2); break;
        case FX_HEATTRAIL:    fx_heattrail(frame, f->param, f->param2); break;
        case FX_WORMHOLE:     fx_wormhole(frame, f->param, f->param2); break;
        case FX_MAGNET:       fx_magnet(frame, f->param, f->param2); break;
        case FX_LIQUID:       fx_liquid(frame, f->param, f->param2, f->param3); break;
        case FX_NIGHTVISION:  fx_nightvision(frame, f->param, f->param2); break;
        case FX_BADTV:        fx_badtv(frame, f->param, f->param2); break;
        case FX_GHOSTS:       fx_ghosts(frame, f->param, f->param2); break;
        /* AI-craze volume 2 */
        case FX_CLONES:       fx_clones(frame, f->param, f->param2, f->param3); break;
        case FX_MELTDOWN:     fx_meltdown(frame, f->param, f->param2); break;
        case FX_ROTOZOOM:     fx_rotozoom(frame, f->param, f->param2); break;
        case FX_STARFIELD:    fx_starfield(frame, f->param, f->param2); break;
        case FX_LIGHTNING:    fx_lightning(frame, f->param, f->param2); break;
        case FX_SCATTER:      fx_scatter(frame, f->param, f->param2); break;
        case FX_VHS:          fx_vhs(frame, f->param, f->param2); break;
        case FX_OUTLINE:      fx_outline(frame, f->param, f->param2); break;
        case FX_SYMMETRY:     fx_symmetry(frame, f->param); break;
        case FX_TIMEBUBBLE:   fx_timebubble(frame, f->param, f->param2); break;
        /* OpenCV-backed effects operate on the whole BGR24 frame.
         * In the non-OpenCV build these resolve to no-op stubs. */
        case FX_CV_CARTOON:   cv_fx_cartoon(frame->data, frame->width, frame->height, frame->stride, f->param); break;
        case FX_CV_PENCIL:    cv_fx_pencil (frame->data, frame->width, frame->height, frame->stride, (int)f->param); break;
        case FX_CV_STYLIZE:   cv_fx_stylize(frame->data, frame->width, frame->height, frame->stride, f->param); break;
        case FX_CV_FLOW:      cv_fx_flow   (frame->data, frame->width, frame->height, frame->stride, f->param); break;
        case FX_CV_BGSUB:     cv_fx_bgsub  (frame->data, frame->width, frame->height, frame->stride, f->param); break;
        case FX_CV_FACE:      cv_fx_face   (frame->data, frame->width, frame->height, frame->stride, (int)f->param, f->param2, f->param3); break;
        case FX_CV_PERSON:    cv_fx_bgblur (frame->data, frame->width, frame->height, frame->stride, (int)f->param, f->param2, f->param3, f->param4, f->param5); break;
        case FX_CV_STYLE:     cv_fx_style  (frame->data, frame->width, frame->height, frame->stride, (int)f->param, f->param2, f->param3); break;
        case FX_CV_MOOD:      cv_fx_mood   (frame->data, frame->width, frame->height, frame->stride, (int)f->param, f->param2, f->param3); break;
        case FX_CV_PAINT:     cv_fx_paintflow(frame->data, frame->width, frame->height, frame->stride, f->param, f->param2); break;
        case FX_CV_SPEED:     cv_fx_speedlines(frame->data, frame->width, frame->height, frame->stride, f->param, f->param2); break;
        default: break;
        }
    }
}
