/*
 * filter.c – per-pixel effects applied to a BGR24 frame in place.
 *
 * All effects operate on the captured webcam frame BEFORE overlays are
 * composited on top. The chain is mutex-protected so the IPC thread can
 * modify it while the main loop is applying it.
 */

#include "vcam.h"

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
    /* pixel-crazy */
    {"bitcrush",    FX_BITCRUSH},
    {"melt",        FX_MELT},
    {"popart",      FX_POPART},
    {"displace",    FX_DISPLACE},
    {"ripple",      FX_RIPPLE},
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
    fc->filters[fc->count].type  = t;
    fc->filters[fc->count].param = param;
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

static void fx_vignette(Frame *f, float strength)
{
    if (strength < 0) strength = 0;
    if (strength > 1) strength = 1;
    int w = f->width, h = f->height;
    float cx = w / 2.0f, cy = h / 2.0f;
    float maxd = sqrtf(cx*cx + cy*cy);
    for (int y = 0; y < h; y++) {
        uint8_t *p = f->data + y * f->stride;
        for (int x = 0; x < w; x++, p += 3) {
            float dx = x - cx, dy = y - cy;
            float d = sqrtf(dx*dx + dy*dy) / maxd;
            float v = 1.0f - strength * d * d;
            if (v < 0) v = 0;
            p[0] = (uint8_t)(p[0] * v);
            p[1] = (uint8_t)(p[1] * v);
            p[2] = (uint8_t)(p[2] * v);
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

static void fx_edges(Frame *f)
{
    /* Simple Sobel on luma */
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
            int v = abs(gx) + abs(gy);
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

/* ── chromashift: VHS-style RGB channel separation ──────────── */
static void fx_chromashift(Frame *f, int dx)
{
    if (dx == 0) return;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        const uint8_t *src = tmp + y * w * 3;
        for (int x = 0; x < w; x++) {
            int xr = x + dx;  if (xr < 0) xr = 0;  if (xr >= w) xr = w - 1;
            int xb = x - dx;  if (xb < 0) xb = 0;  if (xb >= w) xb = w - 1;
            dst[x*3+0] = src[xb*3+0]; /* B from xb */
            dst[x*3+1] = src[x*3+1];  /* G stays   */
            dst[x*3+2] = src[xr*3+2]; /* R from xr */
        }
    }
    free(tmp);
}

/* ── glitch: random horizontal slice displacement (animated) ── */
static void fx_glitch(Frame *f, float intensity)
{
    if (intensity <= 0) return;
    if (intensity > 1) intensity = 1;
    int w = f->width, h = f->height;
    unsigned seed = (unsigned)anim_tick();

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    int slice_h = 4 + (int)((1.0f - intensity) * 20);
    int max_shift = (int)(w * 0.15f * intensity);

    for (int y = 0; y < h; y += slice_h) {
        unsigned r = hash_u(seed + (unsigned)y);
        if ((r & 7) > 5) continue; /* only ~25% of slices shift */
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

/* ── kaleidoscope: mirror a wedge around the center ─────────── */
static void fx_kaleidoscope(Frame *f, int segments)
{
    if (segments < 2) segments = 2;
    if (segments > 32) segments = 32;
    int w = f->width, h = f->height;
    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    float cx = w / 2.0f, cy = h / 2.0f;
    float wedge = (float)(3.14159265358979f * 2.0f) / segments;

    for (int y = 0; y < h; y++) {
        uint8_t *dst = f->data + y * f->stride;
        for (int x = 0; x < w; x++) {
            float dx = x - cx, dy = y - cy;
            float r = sqrtf(dx*dx + dy*dy);
            float a = atan2f(dy, dx);
            /* fold angle into one wedge then mirror */
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
static void fx_swirl(Frame *f, float strength)
{
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
            float r = sqrtf(dx*dx + dy*dy);
            float theta = atan2f(dy, dx) + strength * (1.0f - r / maxr) * 3.14159f;
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
    fx_vignette(f, 0.7f);
}

/* ── wave: sinusoidal vertical wave displacement ────────────── */
static void fx_wave(Frame *f, float amplitude)
{
    if (amplitude < 1) amplitude = 1;
    int w = f->width, h = f->height;
    unsigned long t = anim_tick();
    float phase = (t * 0.1f);

    uint8_t *tmp = malloc((size_t)w * h * 3);
    if (!tmp) return;
    memcpy(tmp, f->data, (size_t)w * h * 3);

    for (int y = 0; y < h; y++) {
        int shift = (int)(amplitude * sinf(y * 0.05f + phase));
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
    free(tmp);
}

/* ── cartoon: posterize + emphasize edges ───────────────────── */
static void fx_cartoon(Frame *f, float edge_thresh)
{
    int w = f->width, h = f->height;

    /* Compute luma + Sobel magnitude */
    uint8_t *luma = malloc((size_t)w * h);
    if (!luma) return;
    for (int y = 0; y < h; y++) {
        const uint8_t *src = f->data + y * f->stride;
        for (int x = 0; x < w; x++)
            luma[y*w + x] = (src[x*3+2]*299 + src[x*3+1]*587 + src[x*3+0]*114) / 1000;
    }

    /* Posterize colors to 5 levels */
    int step = 256 / 5;
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

/* ── pinch: inward radial pull (subtle anamorphic squeeze) ──── */
static void fx_pinch(Frame *f, float strength)
{
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

/* ── bulge: outward radial push (fisheye-like) ──────────────── */
static void fx_bulge(Frame *f, float strength)
{
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
    fx_vignette(f, 0.3f * strength);
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
    fx_chromashift(f, (int)(intensity * 10));
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
    fx_vignette(f, strength * 0.55f);
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
    fx_kaleidoscope(f, s);
    fx_posterize(f, 4);
    fx_cartoon(f, 35);
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
        case FX_VIGNETTE:     fx_vignette(frame, f->param); break;
        case FX_BRIGHTNESS:   fx_brightness(frame, f->param); break;
        case FX_CONTRAST:     fx_contrast(frame, f->param); break;
        case FX_POSTERIZE:    fx_posterize(frame, (int)f->param); break;
        case FX_EDGES:        fx_edges(frame); break;
        case FX_CHROMASHIFT:  fx_chromashift(frame, (int)f->param); break;
        case FX_GLITCH:       fx_glitch(frame, f->param); break;
        case FX_KALEIDOSCOPE: fx_kaleidoscope(frame, (int)f->param); break;
        case FX_SWIRL:        fx_swirl(frame, f->param); break;
        case FX_THERMAL:      fx_thermal(frame); break;
        case FX_OLDFILM:      fx_oldfilm(frame); break;
        case FX_WAVE:         fx_wave(frame, f->param); break;
        case FX_CARTOON:      fx_cartoon(frame, f->param); break;
        case FX_SOLARIZE:     fx_solarize(frame, (int)f->param); break;
        case FX_RAINBOW:      fx_rainbow(frame, f->param); break;
        case FX_DREAM:        fx_dream(frame, f->param); break;
        case FX_BLOOM:        fx_bloom(frame, f->param); break;
        case FX_GLOW:         fx_glow(frame, f->param); break;
        case FX_PINCH:        fx_pinch(frame, f->param); break;
        case FX_BULGE:        fx_bulge(frame, f->param); break;
        case FX_PAINTERLY:    fx_painterly(frame, (int)f->param); break;
        case FX_MODERN:       fx_modern(frame, f->param); break;
        case FX_CYBERPUNK:    fx_cyberpunk(frame, f->param); break;
        case FX_ROMANTIC:     fx_romantic(frame, f->param); break;
        case FX_VANGOGH:      fx_vangogh(frame, f->param); break;
        case FX_PICASSO:      fx_picasso(frame, f->param); break;
        case FX_SINE:         fx_sine(frame, f->param); break;
        case FX_XOR:          fx_xor(frame, f->param); break;
        case FX_MODULO:       fx_modulo(frame, f->param); break;
        case FX_POLAR:        fx_polar(frame, f->param); break;
        case FX_VORONOI:      fx_voronoi(frame, f->param); break;
        case FX_BITCRUSH:     fx_bitcrush(frame, f->param); break;
        case FX_MELT:         fx_melt(frame, f->param); break;
        case FX_POPART:       fx_popart(frame, f->param); break;
        case FX_DISPLACE:     fx_displace(frame, f->param); break;
        case FX_RIPPLE:       fx_ripple(frame, f->param); break;
        default: break;
        }
    }
}
