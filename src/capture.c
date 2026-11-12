/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Modified-by: Claude Opus 5.5 (live mode switching: enum/choose/open_mode,
 *              scale+crop into a fixed output size)
 */

#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <strings.h>
#include <linux/videodev2.h>

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#define NUM_BUFS 4

typedef struct {
    void   *start;
    size_t  length;
} Buffer;

struct V4L2Capture {
    int    fd;
    int    width, height;
    int    fps;
    __u32  pixfmt;            /* actual V4L2 pixel format we got */
    Buffer bufs[NUM_BUFS];
    int    nbuf;

    /* For MJPG: a libavcodec MJPEG decoder */
    AVCodecContext *mjpeg_ctx;
    AVPacket       *mjpeg_pkt;
    AVFrame        *mjpeg_frame;

    /* swscale: source pixel format → BGR24 */
    struct SwsContext *sws;
    int    sws_src_fmt;       /* AV_PIX_FMT_* */
    int    sws_src_w, sws_src_h; /* may differ from cap->width when MJPEG */

    /* Output geometry: frames are scaled to out_w × out_h (the fixed
     * loopback size). With crop, a different aspect ratio is scaled to
     * cover (scale_w × scale_h into tmp) and the centre is cut out. */
    int    out_w, out_h;
    bool   crop;
    int    scale_w, scale_h;
    int    crop_x, crop_y;
    Frame *tmp;               /* only when cropping is needed */
};

static bool capture_start(V4L2Capture *cap, const char *device,
                          __u32 fmt, int w, int h, int target_fps);

/* ── helpers ─────────────────────────────────── */
static int xioctl(int fd, unsigned long req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static const char *fourcc(__u32 f)
{
    static _Thread_local char s[5];   /* used from main + IPC threads */
    s[0]= f      & 0xff;
    s[1]=(f>> 8) & 0xff;
    s[2]=(f>>16) & 0xff;
    s[3]=(f>>24) & 0xff;
    s[4]= 0;
    return s;
}

/*
 * Check whether the camera supports a given pixel format at a given
 * resolution and ≥ target_fps. Returns true if supported.
 */
static bool format_supported(int fd, __u32 pixfmt, int w, int h, int target_fps)
{
    struct v4l2_frmivalenum fi = {0};
    fi.pixel_format = pixfmt;
    fi.width  = w;
    fi.height = h;
    for (fi.index = 0; xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) == 0; fi.index++) {
        if (fi.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
            int fps = fi.discrete.denominator /
                      (fi.discrete.numerator ? fi.discrete.numerator : 1);
            if (fps >= target_fps) return true;
        }
    }
    return false;
}

/*
 * Pick the best (pixfmt, w, h) combination for our purposes:
 *   - Prefer the requested w/h if both are supported at ≥ target_fps
 *   - Otherwise prefer the largest resolution that supports ≥ target_fps
 *   - Prefer MJPG over YUYV for high resolutions (cameras usually support
 *     higher fps in MJPG); prefer YUYV for low resolutions (no decode cost)
 *
 * Writes chosen format into *out_*.
 */
static bool pick_format(int fd, int req_w, int req_h, int target_fps,
                         __u32 *out_fmt, int *out_w, int *out_h)
{
    /* Candidate pixel formats in our preference order */
    const __u32 prefs[] = {
        V4L2_PIX_FMT_MJPEG,   /* high-bandwidth, decoded via libavcodec     */
        V4L2_PIX_FMT_YUYV,    /* universal, no decode                       */
        V4L2_PIX_FMT_NV12,
        V4L2_PIX_FMT_YUV420,
        V4L2_PIX_FMT_BGR24,
        V4L2_PIX_FMT_RGB24,
    };
    const size_t nprefs = sizeof prefs / sizeof prefs[0];

    /* If a specific size was requested, try to honour it first */
    if (req_w > 0 && req_h > 0) {
        for (size_t i = 0; i < nprefs; i++) {
            if (format_supported(fd, prefs[i], req_w, req_h, target_fps)) {
                *out_fmt = prefs[i]; *out_w = req_w; *out_h = req_h;
                return true;
            }
        }
    }

    /* Otherwise enumerate all (format, size) pairs and pick the largest
     * resolution that meets target_fps, preferring MJPG. */
    int best_w = 0, best_h = 0;
    __u32 best_fmt = 0;
    int best_score = -1;

    struct v4l2_fmtdesc fd_desc = {0};
    fd_desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (fd_desc.index = 0;
         xioctl(fd, VIDIOC_ENUM_FMT, &fd_desc) == 0;
         fd_desc.index++) {
        /* Skip formats we don't handle */
        size_t pref_rank = nprefs;
        for (size_t i = 0; i < nprefs; i++)
            if (prefs[i] == fd_desc.pixelformat) { pref_rank = i; break; }
        if (pref_rank == nprefs) continue;

        struct v4l2_frmsizeenum fs = {0};
        fs.pixel_format = fd_desc.pixelformat;
        for (fs.index = 0;
             xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) == 0;
             fs.index++) {
            if (fs.type != V4L2_FRMSIZE_TYPE_DISCRETE) continue;
            int w = fs.discrete.width, h = fs.discrete.height;
            if (!format_supported(fd, fd_desc.pixelformat, w, h, target_fps))
                continue;
            /* Score: pixel count, with a small bonus for MJPG so it wins ties */
            int score = w * h - (int)pref_rank;
            if (score > best_score) {
                best_score = score;
                best_w = w; best_h = h;
                best_fmt = fd_desc.pixelformat;
            }
        }
    }

    if (best_score < 0) return false;
    *out_fmt = best_fmt; *out_w = best_w; *out_h = best_h;
    return true;
}

/* ── formats vcam can consume, in preference order ── */
static const __u32 g_usable[] = {
    V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_NV12,
    V4L2_PIX_FMT_YUV420, V4L2_PIX_FMT_BGR24, V4L2_PIX_FMT_RGB24,
};
#define N_USABLE ((int)(sizeof g_usable / sizeof g_usable[0]))

static int usable_rank(__u32 f)
{
    for (int i = 0; i < N_USABLE; i++) if (g_usable[i] == f) return i;
    return -1;
}

/* Name → fourcc. Accepts the fourcc itself plus common aliases. */
static bool fourcc_from_name(const char *n, __u32 *out)
{
    static const struct { const char *name; __u32 f; } names[] = {
        { "MJPG", V4L2_PIX_FMT_MJPEG }, { "MJPEG", V4L2_PIX_FMT_MJPEG },
        { "YUYV", V4L2_PIX_FMT_YUYV },  { "NV12",  V4L2_PIX_FMT_NV12 },
        { "YU12", V4L2_PIX_FMT_YUV420 },{ "YUV420",V4L2_PIX_FMT_YUV420 },
        { "I420", V4L2_PIX_FMT_YUV420 },{ "BGR3",  V4L2_PIX_FMT_BGR24 },
        { "BGR24",V4L2_PIX_FMT_BGR24 }, { "RGB3",  V4L2_PIX_FMT_RGB24 },
        { "RGB24",V4L2_PIX_FMT_RGB24 },
    };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++)
        if (strcasecmp(n, names[i].name) == 0) { *out = names[i].f; return true; }
    return false;
}

/* Highest frame rate the device offers for (fmt, w, h); 0 if unknown. */
static int max_fps_for(int fd, __u32 pixfmt, int w, int h)
{
    int best = 0;
    struct v4l2_frmivalenum fi = {0};
    fi.pixel_format = pixfmt;
    fi.width = w; fi.height = h;
    for (fi.index = 0; xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) == 0; fi.index++) {
        struct v4l2_fract *f = fi.type == V4L2_FRMIVAL_TYPE_DISCRETE
                             ? &fi.discrete : &fi.stepwise.min;
        int fps = f->denominator / (f->numerator ? f->numerator : 1);
        if (fps > best) best = fps;
        if (fi.type != V4L2_FRMIVAL_TYPE_DISCRETE) break;
    }
    return best;
}

static int enum_modes_fd(int fd, CaptureMode *out, int max)
{
    int n = 0;
    struct v4l2_fmtdesc fd_desc = {0};
    fd_desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (fd_desc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &fd_desc) == 0; fd_desc.index++) {
        if (usable_rank(fd_desc.pixelformat) < 0) continue;
        struct v4l2_frmsizeenum fs = {0};
        fs.pixel_format = fd_desc.pixelformat;
        for (fs.index = 0; xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) == 0 && n < max; fs.index++) {
            if (fs.type != V4L2_FRMSIZE_TYPE_DISCRETE) continue;
            CaptureMode *m = &out[n++];
            snprintf(m->fmt, sizeof m->fmt, "%s", fourcc(fd_desc.pixelformat));
            m->w   = (int)fs.discrete.width;
            m->h   = (int)fs.discrete.height;
            m->fps = max_fps_for(fd, fd_desc.pixelformat, m->w, m->h);
        }
    }
    return n;
}

int capture_enum_modes(const char *device, CaptureMode *out, int max)
{
    int fd = open(device, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) return -1;
    int n = enum_modes_fd(fd, out, max);
    close(fd);
    return n;
}

/*
 * Pick a mode. fmt / w,h / fps may each be "unset" (NULL or "", 0):
 * unset fields keep the current mode's value when the camera allows it.
 * fmt "auto" = best format; w,h = -1 = largest size.
 * Explicit values are never changed silently: if they can't be met → error.
 */
bool capture_choose(const CaptureMode *modes, int n,
                    const char *fmt, int w, int h, int fps,
                    const CaptureMode *cur, int default_fps,
                    CaptureMode *out, char *err, size_t errlen)
{
    char want_fmt[8] = "";
    bool fmt_explicit = false;
    if (fmt && *fmt && strcasecmp(fmt, "auto") != 0) {
        __u32 f;
        if (!fourcc_from_name(fmt, &f)) {
            snprintf(err, errlen, "unknown format '%s' (MJPG, YUYV, NV12, YU12, BGR3, RGB3, auto)", fmt);
            return false;
        }
        snprintf(want_fmt, sizeof want_fmt, "%s", fourcc(f));
        fmt_explicit = true;
    } else if (!fmt || !*fmt) {
        if (cur && cur->fmt[0]) snprintf(want_fmt, sizeof want_fmt, "%s", cur->fmt);
    }
    bool size_explicit = w > 0 && h > 0;
    bool size_max      = w < 0;
    int  want_w = size_explicit ? w : (!size_max && cur ? cur->w : 0);
    int  want_h = size_explicit ? h : (!size_max && cur ? cur->h : 0);
    int  target = fps > 0 ? fps : (cur && cur->fps > 0 ? cur->fps : default_fps);
    if (target <= 0) target = 30;

    /* Attempts, in order: as asked → any size → any format → both.
     * A field the caller set explicitly is never dropped. */
    static const bool keep[4][2] = { {1,1}, {1,0}, {0,1}, {0,0} };   /* {fmt, size} */
    for (int a = 0; a < 4; a++) {
        if (!keep[a][0] && fmt_explicit)  continue;
        if (!keep[a][1] && size_explicit) continue;
        const char *ff = keep[a][0] ? want_fmt : "";
        int fw = keep[a][1] ? want_w : 0, fh = keep[a][1] ? want_h : 0;

        const CaptureMode *best = NULL;
        int best_eff = -1; long best_area = -1; int best_rank = 99;
        for (int i = 0; i < n; i++) {
            const CaptureMode *m = &modes[i];
            if (ff[0] && strcmp(m->fmt, ff) != 0) continue;
            if (fw > 0 && (m->w != fw || m->h != fh)) continue;
            int  eff  = m->fps <= 0 ? target : (m->fps < target ? m->fps : target);
            long area = fw > 0 ? 0 : (long)m->w * m->h;
            __u32 f = 0; fourcc_from_name(m->fmt, &f);
            int  rank = usable_rank(f);
            if (eff > best_eff ||
                (eff == best_eff && area > best_area) ||
                (eff == best_eff && area == best_area && rank < best_rank)) {
                best = m; best_eff = eff; best_area = area; best_rank = rank;
            }
        }
        if (!best) continue;
        if (fps > 0 && best->fps > 0 && best->fps < fps) {
            snprintf(err, errlen, "%s %dx%d supports at most %d fps",
                     best->fmt, best->w, best->h, best->fps);
            return false;
        }
        *out = *best;
        out->fps = best_eff;
        return true;
    }
    if (!fmt_explicit) want_fmt[0] = '\0';   /* every format was tried */
    if (size_explicit)
        snprintf(err, errlen, "camera has no mode %s%s%dx%d (see 'capture formats')",
                 want_fmt, want_fmt[0] ? " " : "", w, h);
    else
        snprintf(err, errlen, "camera has no %s mode (see 'capture formats')",
                 want_fmt[0] ? want_fmt : "usable");
    return false;
}

/* ── list supported formats ───────────────────────
 * Opens the device, enumerates every (pixel format, resolution, fps)
 * combination the driver reports, and prints a human-readable table
 * to stdout. Used by `vcam --list-formats`. Returns true on success.
 *
 * Only the pixel formats vcam can actually consume are shown, since
 * listing formats we can't decode would be misleading. For each
 * discrete frame size the maximum discrete frame rate is reported. */
bool capture_list_formats(const char *device)
{
    int fd = open(device, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        fprintf(stderr, "vcam: cannot open %s: %s\n", device, strerror(errno));
        return false;
    }

    struct v4l2_capability caps = {0};
    if (xioctl(fd, VIDIOC_QUERYCAP, &caps) == 0)
        printf("Capture device: %s (%s)\n", device, caps.card);
    else
        printf("Capture device: %s\n", device);

    /* Formats vcam can consume, in the same preference order as pick_format */
    const __u32 usable[] = {
        V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_NV12,
        V4L2_PIX_FMT_YUV420, V4L2_PIX_FMT_BGR24, V4L2_PIX_FMT_RGB24,
    };
    const size_t n_usable = sizeof usable / sizeof usable[0];

    printf("\n  %-8s %-12s %s\n", "format", "resolution", "max fps");
    printf("  %-8s %-12s %s\n", "------", "----------", "-------");

    int shown = 0;
    struct v4l2_fmtdesc fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (fmt.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0; fmt.index++) {
        bool ok = false;
        for (size_t i = 0; i < n_usable; i++)
            if (usable[i] == fmt.pixelformat) { ok = true; break; }
        if (!ok) continue;   /* skip formats we can't decode */

        struct v4l2_frmsizeenum fs = {0};
        fs.pixel_format = fmt.pixelformat;
        for (fs.index = 0;
             xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) == 0;
             fs.index++) {
            if (fs.type != V4L2_FRMSIZE_TYPE_DISCRETE) continue;
            int w = fs.discrete.width, h = fs.discrete.height;

            /* find the highest discrete fps for this size */
            int max_fps = 0;
            struct v4l2_frmivalenum fi = {0};
            fi.pixel_format = fmt.pixelformat;
            fi.width = w; fi.height = h;
            for (fi.index = 0;
                 xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) == 0;
                 fi.index++) {
                if (fi.type != V4L2_FRMIVAL_TYPE_DISCRETE) continue;
                int fps = fi.discrete.denominator /
                          (fi.discrete.numerator ? fi.discrete.numerator : 1);
                if (fps > max_fps) max_fps = fps;
            }

            char res[16];
            snprintf(res, sizeof res, "%dx%d", w, h);
            if (max_fps > 0)
                printf("  %-8s %-12s %d\n", fourcc(fmt.pixelformat), res, max_fps);
            else
                printf("  %-8s %-12s %s\n", fourcc(fmt.pixelformat), res, "?");
            shown++;
        }
    }

    if (shown == 0)
        printf("  (no usable discrete formats reported)\n");

    printf("\nPick one with:  vcam --res WxH [-f FPS]\n");
    printf("Live switch:    vcam-ctl capture set <FMT> <WxH> [FPS]   (output size stays)\n");
    close(fd);
    return true;
}

/* ── open ─────────────────────────────────────── */
V4L2Capture *capture_open(const char *device,
                           int *out_width, int *out_height, int *out_fps)
{
    V4L2Capture *cap = calloc(1, sizeof *cap);
    if (!cap) return NULL;

    cap->fd = open(device, O_RDWR | O_NONBLOCK);
    if (cap->fd < 0) {
        perror("capture_open: open");
        free(cap);
        return NULL;
    }

    /* ── pick the best format/size for what was requested ── */
    int target_fps = *out_fps > 0 ? *out_fps : 30;
    __u32 chosen_fmt = 0;
    int   chosen_w = 0, chosen_h = 0;

    if (!pick_format(cap->fd, *out_width, *out_height, target_fps,
                      &chosen_fmt, &chosen_w, &chosen_h)) {
        /* Fallback: ask driver for whatever it has */
        fprintf(stderr, "[capture] No suitable format found at ≥%d fps; "
                        "using driver default\n", target_fps);
        struct v4l2_format gf = {0};
        gf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(cap->fd, VIDIOC_G_FMT, &gf) < 0) {
            perror("VIDIOC_G_FMT"); goto err;
        }
        chosen_fmt = gf.fmt.pix.pixelformat;
        chosen_w   = gf.fmt.pix.width;
        chosen_h   = gf.fmt.pix.height;
    }

    cap->crop = true;   /* only matters once out size ≠ capture size */
    if (!capture_start(cap, device, chosen_fmt, chosen_w, chosen_h, target_fps))
        goto err;

    *out_width  = cap->width;
    *out_height = cap->height;
    *out_fps    = cap->fps;
    return cap;

err:
    capture_close(cap);
    return NULL;
}

/* Open the device in an exact mode, scaling frames to out_w × out_h. */
V4L2Capture *capture_open_mode(const char *device, const CaptureMode *mode,
                               int out_w, int out_h, bool crop)
{
    __u32 f;
    if (!fourcc_from_name(mode->fmt, &f)) return NULL;
    V4L2Capture *cap = calloc(1, sizeof *cap);
    if (!cap) return NULL;
    cap->fd = open(device, O_RDWR | O_NONBLOCK);
    if (cap->fd < 0) {
        perror("capture_open_mode: open");
        free(cap);
        return NULL;
    }
    cap->out_w = out_w;
    cap->out_h = out_h;
    cap->crop  = crop;
    if (!capture_start(cap, device, f, mode->w, mode->h, mode->fps > 0 ? mode->fps : 30)) {
        capture_close(cap);
        return NULL;
    }
    return cap;
}

/*
 * Scaling geometry for capture → output. Rebuilds the swscale context
 * (MJPEG: lazily on the next decoded frame) and the crop buffer.
 */
static bool setup_scaling(V4L2Capture *cap)
{
    if (cap->out_w <= 0 || cap->out_h <= 0) {
        cap->out_w = cap->width;
        cap->out_h = cap->height;
    }
    frame_free(cap->tmp);
    cap->tmp = NULL;
    int sw = cap->width, sh = cap->height, ow = cap->out_w, oh = cap->out_h;
    cap->scale_w = ow; cap->scale_h = oh;
    cap->crop_x  = cap->crop_y = 0;
    if (cap->crop && (long)sw * oh != (long)sh * ow) {
        if ((long)sw * oh > (long)sh * ow) {          /* source wider: cut sides */
            cap->scale_w = (int)(((long)sw * oh + sh - 1) / sh);
        } else {                                      /* source taller: cut top/bottom */
            cap->scale_h = (int)(((long)sh * ow + sw - 1) / sw);
        }
        cap->crop_x = (cap->scale_w - ow) / 2;
        cap->crop_y = (cap->scale_h - oh) / 2;
        cap->tmp = frame_alloc(cap->scale_w, cap->scale_h);
        if (!cap->tmp) return false;
    }
    if (cap->sws) { sws_freeContext(cap->sws); cap->sws = NULL; }
    if (cap->pixfmt != V4L2_PIX_FMT_MJPEG) {
        cap->sws = sws_getContext(sw, sh, cap->sws_src_fmt,
                                  cap->scale_w, cap->scale_h, AV_PIX_FMT_BGR24,
                                  SWS_BILINEAR, NULL, NULL, NULL);
        if (!cap->sws) return false;
    }
    return true;
}

bool capture_set_fit(V4L2Capture *cap, bool crop)
{
    cap->crop = crop;
    return setup_scaling(cap);
}

void capture_get(const V4L2Capture *cap, CaptureMode *mode, bool *crop)
{
    snprintf(mode->fmt, sizeof mode->fmt, "%s", fourcc(cap->pixfmt));
    mode->w   = cap->width;
    mode->h   = cap->height;
    mode->fps = cap->fps;
    if (crop) *crop = cap->crop;
}

/* Copy the centre of the oversized tmp frame into out. */
static void crop_into(const V4L2Capture *cap, Frame *out)
{
    const Frame *t = cap->tmp;
    for (int y = 0; y < cap->out_h; y++)
        memcpy(out->data + (size_t)y * out->stride,
               t->data + (size_t)(y + cap->crop_y) * t->stride + (size_t)cap->crop_x * 3,
               (size_t)cap->out_w * 3);
}

/* S_FMT, frame rate, buffers, STREAMON, decoder + scaler. */
static bool capture_start(V4L2Capture *cap, const char *device,
                          __u32 chosen_fmt, int chosen_w, int chosen_h, int target_fps)
{
    /* ── apply the chosen format ── */
    struct v4l2_format fmt = {0};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = chosen_w;
    fmt.fmt.pix.height      = chosen_h;
    fmt.fmt.pix.pixelformat = chosen_fmt;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(cap->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        return false;
    }

    cap->width  = fmt.fmt.pix.width;
    cap->height = fmt.fmt.pix.height;
    cap->pixfmt = fmt.fmt.pix.pixelformat;

    fprintf(stderr, "[capture] %s  %dx%d  fmt=%s\n",
            device, cap->width, cap->height, fourcc(cap->pixfmt));

    /* ── set frame rate ── */
    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cap->fd, VIDIOC_G_PARM, &parm) == 0 &&
        parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) {
        parm.parm.capture.timeperframe.numerator   = 1;
        parm.parm.capture.timeperframe.denominator = target_fps;
        xioctl(cap->fd, VIDIOC_S_PARM, &parm);
        xioctl(cap->fd, VIDIOC_G_PARM, &parm);
        struct v4l2_fract *tf = &parm.parm.capture.timeperframe;
        cap->fps = tf->denominator / (tf->numerator ? tf->numerator : 1);
    }
    if (cap->fps <= 0) cap->fps = target_fps;

    /* ── mmap buffers ── */
    struct v4l2_requestbuffers req = {0};
    req.count  = NUM_BUFS;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cap->fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS"); return false;
    }
    cap->nbuf = req.count;

    for (int i = 0; i < cap->nbuf; i++) {
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(cap->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF"); return false;
        }
        cap->bufs[i].length = buf.length;
        cap->bufs[i].start  = mmap(NULL, buf.length,
                                    PROT_READ | PROT_WRITE, MAP_SHARED,
                                    cap->fd, buf.m.offset);
        if (cap->bufs[i].start == MAP_FAILED) {
            perror("mmap"); return false;
        }
    }

    for (int i = 0; i < cap->nbuf; i++) {
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(cap->fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF"); return false;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cap->fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON"); return false;
    }

    /* ── set up MJPEG decoder if needed ── */
    if (cap->pixfmt == V4L2_PIX_FMT_MJPEG) {
        const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
        if (!codec) { fprintf(stderr, "[capture] no MJPEG decoder\n"); return false; }
        cap->mjpeg_ctx = avcodec_alloc_context3(codec);
        cap->mjpeg_ctx->width  = cap->width;
        cap->mjpeg_ctx->height = cap->height;
        if (avcodec_open2(cap->mjpeg_ctx, codec, NULL) < 0) {
            fprintf(stderr, "[capture] avcodec_open2 (MJPEG) failed\n");
            return false;
        }
        cap->mjpeg_pkt   = av_packet_alloc();
        cap->mjpeg_frame = av_frame_alloc();
        /* sws will be set lazily on first decoded frame since the decoder
         * tells us the exact pixfmt (typically YUVJ420P or YUVJ422P) */
        cap->sws = NULL;
    } else {
        /* Set up swscale directly from the V4L2 pixel format */
        int avfmt;
        switch (cap->pixfmt) {
        case V4L2_PIX_FMT_YUYV:   avfmt = AV_PIX_FMT_YUYV422; break;
        case V4L2_PIX_FMT_YUV420: avfmt = AV_PIX_FMT_YUV420P; break;
        case V4L2_PIX_FMT_NV12:   avfmt = AV_PIX_FMT_NV12;    break;
        case V4L2_PIX_FMT_BGR24:  avfmt = AV_PIX_FMT_BGR24;   break;
        case V4L2_PIX_FMT_RGB24:  avfmt = AV_PIX_FMT_RGB24;   break;
        default:
            fprintf(stderr, "[capture] unsupported format %s\n", fourcc(cap->pixfmt));
            return false;
        }
        cap->sws_src_fmt = avfmt;
        cap->sws_src_w   = cap->width;
        cap->sws_src_h   = cap->height;
    }
    return setup_scaling(cap);   /* builds the BGR24 scaler for out size */
}

/* ── read one frame ───────────────────────────── */
bool capture_read(V4L2Capture *cap, Frame *out)
{
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(cap->fd, &fds);
    struct timeval tv = { .tv_sec = 2 };
    if (select(cap->fd + 1, &fds, NULL, NULL, &tv) <= 0)
        return false;

    struct v4l2_buffer buf = {0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cap->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) return false;
        perror("VIDIOC_DQBUF");
        return false;
    }

    bool ok = false;

    if (cap->pixfmt == V4L2_PIX_FMT_MJPEG) {
        /* Decode JPEG → AVFrame → BGR24 via swscale */
        cap->mjpeg_pkt->data = cap->bufs[buf.index].start;
        cap->mjpeg_pkt->size = buf.bytesused;

        if (avcodec_send_packet(cap->mjpeg_ctx, cap->mjpeg_pkt) == 0 &&
            avcodec_receive_frame(cap->mjpeg_ctx, cap->mjpeg_frame) == 0) {

            /* Build/refresh swscale context if needed */
            int src_fmt = cap->mjpeg_frame->format;
            int src_w   = cap->mjpeg_frame->width;
            int src_h   = cap->mjpeg_frame->height;
            if (!cap->sws ||
                cap->sws_src_fmt != src_fmt ||
                cap->sws_src_w   != src_w ||
                cap->sws_src_h   != src_h) {
                if (cap->sws) sws_freeContext(cap->sws);
                cap->sws = sws_getContext(src_w, src_h, src_fmt,
                                           cap->scale_w, cap->scale_h, AV_PIX_FMT_BGR24,
                                           SWS_BILINEAR, NULL, NULL, NULL);
                cap->sws_src_fmt = src_fmt;
                cap->sws_src_w   = src_w;
                cap->sws_src_h   = src_h;
            }

            Frame   *target  = cap->tmp ? cap->tmp : out;
            uint8_t *dst[4]  = { target->data, NULL, NULL, NULL };
            int      dl[4]   = { target->stride, 0, 0, 0 };
            if (cap->sws) {
                sws_scale(cap->sws,
                          (const uint8_t *const *)cap->mjpeg_frame->data,
                          cap->mjpeg_frame->linesize, 0, src_h, dst, dl);
                if (cap->tmp) crop_into(cap, out);
                ok = true;
            }
        }
    } else {
        const uint8_t *src[4]   = { cap->bufs[buf.index].start, NULL, NULL, NULL };
        int            sl[4]    = { 0 };
        av_image_fill_linesizes(sl, cap->sws_src_fmt, cap->width);

        Frame         *target   = cap->tmp ? cap->tmp : out;
        uint8_t       *dst[4]   = { target->data, NULL, NULL, NULL };
        int            dl[4]    = { target->stride, 0, 0, 0 };

        sws_scale(cap->sws, src, sl, 0, cap->height, dst, dl);
        if (cap->tmp) crop_into(cap, out);
        ok = true;
    }

    if (xioctl(cap->fd, VIDIOC_QBUF, &buf) < 0)
        perror("VIDIOC_QBUF (re-enqueue)");

    return ok;
}

/* ── close ──────────────────────────────────── */
void capture_close(V4L2Capture *cap)
{
    if (!cap) return;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(cap->fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < cap->nbuf; i++)
        munmap(cap->bufs[i].start, cap->bufs[i].length);
    if (cap->sws) sws_freeContext(cap->sws);
    if (cap->mjpeg_ctx)   avcodec_free_context(&cap->mjpeg_ctx);
    if (cap->mjpeg_pkt)   av_packet_free(&cap->mjpeg_pkt);
    if (cap->mjpeg_frame) av_frame_free(&cap->mjpeg_frame);
    frame_free(cap->tmp);
    if (cap->fd >= 0) close(cap->fd);
    free(cap);
}
