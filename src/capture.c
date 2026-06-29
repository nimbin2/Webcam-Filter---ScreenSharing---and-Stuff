#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
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
};

/* ── helpers ─────────────────────────────────── */
static int xioctl(int fd, unsigned long req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static const char *fourcc(__u32 f)
{
    static char s[5];
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

    /* ── apply the chosen format ── */
    struct v4l2_format fmt = {0};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = chosen_w;
    fmt.fmt.pix.height      = chosen_h;
    fmt.fmt.pix.pixelformat = chosen_fmt;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(cap->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        goto err;
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
        perror("VIDIOC_REQBUFS"); goto err;
    }
    cap->nbuf = req.count;

    for (int i = 0; i < cap->nbuf; i++) {
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(cap->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF"); goto err;
        }
        cap->bufs[i].length = buf.length;
        cap->bufs[i].start  = mmap(NULL, buf.length,
                                    PROT_READ | PROT_WRITE, MAP_SHARED,
                                    cap->fd, buf.m.offset);
        if (cap->bufs[i].start == MAP_FAILED) {
            perror("mmap"); goto err;
        }
    }

    for (int i = 0; i < cap->nbuf; i++) {
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(cap->fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF"); goto err;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cap->fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON"); goto err;
    }

    /* ── set up MJPEG decoder if needed ── */
    if (cap->pixfmt == V4L2_PIX_FMT_MJPEG) {
        const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
        if (!codec) { fprintf(stderr, "[capture] no MJPEG decoder\n"); goto err; }
        cap->mjpeg_ctx = avcodec_alloc_context3(codec);
        cap->mjpeg_ctx->width  = cap->width;
        cap->mjpeg_ctx->height = cap->height;
        if (avcodec_open2(cap->mjpeg_ctx, codec, NULL) < 0) {
            fprintf(stderr, "[capture] avcodec_open2 (MJPEG) failed\n");
            goto err;
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
            goto err;
        }
        cap->sws_src_fmt = avfmt;
        cap->sws_src_w   = cap->width;
        cap->sws_src_h   = cap->height;
        cap->sws = sws_getContext(cap->width, cap->height, avfmt,
                                   cap->width, cap->height, AV_PIX_FMT_BGR24,
                                   SWS_BILINEAR, NULL, NULL, NULL);
        if (!cap->sws) goto err;
    }

    *out_width  = cap->width;
    *out_height = cap->height;
    *out_fps    = cap->fps;
    return cap;

err:
    if (cap->fd >= 0) close(cap->fd);
    free(cap);
    return NULL;
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
                                           cap->width, cap->height, AV_PIX_FMT_BGR24,
                                           SWS_BILINEAR, NULL, NULL, NULL);
                cap->sws_src_fmt = src_fmt;
                cap->sws_src_w   = src_w;
                cap->sws_src_h   = src_h;
            }

            uint8_t *dst[4]  = { out->data, NULL, NULL, NULL };
            int      dl[4]   = { out->stride, 0, 0, 0 };
            sws_scale(cap->sws,
                      (const uint8_t *const *)cap->mjpeg_frame->data,
                      cap->mjpeg_frame->linesize, 0, src_h, dst, dl);
            ok = true;
        }
    } else {
        const uint8_t *src[4]   = { cap->bufs[buf.index].start, NULL, NULL, NULL };
        int            sl[4]    = { 0 };
        av_image_fill_linesizes(sl, cap->sws_src_fmt, cap->width);

        uint8_t       *dst[4]   = { out->data, NULL, NULL, NULL };
        int            dl[4]    = { out->stride, 0, 0, 0 };

        sws_scale(cap->sws, src, sl, 0, cap->height, dst, dl);
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
    close(cap->fd);
    free(cap);
}
