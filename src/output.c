/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * V4L2 output to v4l2loopback using streaming I/O (MMAP buffers).
 *
 * This matches how real V4L2 capture drivers feed their output buffers,
 * making the loopback device behave the same way as a physical webcam
 * from the consumer's perspective (Firefox, Chromium, OBS, etc.).
 *
 * Flow:
 *   1. open device
 *   2. VIDIOC_S_FMT  (YUYV422, proper colorspace tags)
 *   3. VIDIOC_REQBUFS (OUTPUT type, MMAP, N buffers)
 *   4. mmap each buffer
 *   5. for each frame:
 *        - find a free (dequeued) buffer
 *        - convert BGR24 → YUYV422 directly into the buffer
 *        - VIDIOC_QBUF to publish it
 *        - VIDIOC_STREAMON on the first publish
 *        - VIDIOC_DQBUF when no free buffers remain to recycle one
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
#include <linux/videodev2.h>

#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#define NUM_OUT_BUFS 4

typedef struct {
    void  *start;
    size_t length;
    bool   queued; /* true if currently held by the kernel */
} OutBuf;

struct V4L2Output {
    int    fd;
    int    width, height;
    int    frame_size;            /* bytes per YUYV frame */
    OutBuf bufs[NUM_OUT_BUFS];
    int    nbuf;
    bool   streaming;             /* STREAMON called yet? */

    struct SwsContext *sws;       /* BGR24 → YUYV422 */
};

static int xioctl(int fd, unsigned long req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

V4L2Output *output_open(const char *device, int width, int height, int fps)
{
    V4L2Output *out = calloc(1, sizeof *out);
    if (!out) return NULL;

    out->fd = open(device, O_RDWR);
    if (out->fd < 0) {
        fprintf(stderr, "[output] Cannot open %s: %s\n"
                        "  Load v4l2loopback with exclusive_caps=1:\n"
                        "    sudo modprobe v4l2loopback video_nr=0 "
                        "card_label=\"vcam\" exclusive_caps=1\n",
                device, strerror(errno));
        free(out);
        return NULL;
    }

    /* ── verify device exposes V4L2_CAP_VIDEO_OUTPUT ── */
    struct v4l2_capability cap = {0};
    if (xioctl(out->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("[output] VIDIOC_QUERYCAP");
        close(out->fd); free(out); return NULL;
    }
    __u32 caps = cap.capabilities & V4L2_CAP_DEVICE_CAPS
                ? cap.device_caps : cap.capabilities;
    if (!(caps & V4L2_CAP_VIDEO_OUTPUT)) {
        fprintf(stderr, "[output] %s does not support VIDEO_OUTPUT\n"
                        "  Is this really a v4l2loopback device?\n", device);
        close(out->fd); free(out); return NULL;
    }

    /* ── set format: YUYV422 with proper colorspace tags ── */
    struct v4l2_format fmt = {0};
    fmt.type                       = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width              = width;
    fmt.fmt.pix.height             = height;
    fmt.fmt.pix.pixelformat        = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field              = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline       = width * 2;
    fmt.fmt.pix.sizeimage          = width * height * 2;
    /* Use BT.709 for HD, BT.601 for SD — same as real cameras */
    fmt.fmt.pix.colorspace         = (height >= 720)
                                      ? V4L2_COLORSPACE_REC709
                                      : V4L2_COLORSPACE_SMPTE170M;
    fmt.fmt.pix.ycbcr_enc          = V4L2_YCBCR_ENC_DEFAULT;
    fmt.fmt.pix.quantization       = V4L2_QUANTIZATION_DEFAULT;
    fmt.fmt.pix.xfer_func          = V4L2_XFER_FUNC_DEFAULT;

    if (xioctl(out->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("[output] VIDIOC_S_FMT");
        close(out->fd); free(out); return NULL;
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        char got[5] = {0};
        memcpy(got, &fmt.fmt.pix.pixelformat, 4);
        fprintf(stderr, "[output] WARNING: driver gave us '%s' not YUYV\n", got);
    }

    /* Use whatever the driver settled on (may have rounded dims) */
    out->width      = fmt.fmt.pix.width;
    out->height     = fmt.fmt.pix.height;
    out->frame_size = out->width * out->height * 2;

    /* ── advertise frame rate ── */
    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    parm.parm.output.timeperframe.numerator   = 1;
    parm.parm.output.timeperframe.denominator = fps > 0 ? fps : 30;
    if (xioctl(out->fd, VIDIOC_S_PARM, &parm) < 0) {
        /* not fatal — some drivers don't accept S_PARM on output */
    }

    /* ── request mmap output buffers ── */
    struct v4l2_requestbuffers req = {0};
    req.count  = NUM_OUT_BUFS;
    req.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(out->fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("[output] VIDIOC_REQBUFS");
        close(out->fd); free(out); return NULL;
    }
    out->nbuf = req.count;
    if (out->nbuf < 2) {
        fprintf(stderr, "[output] need at least 2 buffers, got %d\n", out->nbuf);
        close(out->fd); free(out); return NULL;
    }

    /* ── mmap each buffer ── */
    for (int i = 0; i < out->nbuf; i++) {
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(out->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("[output] VIDIOC_QUERYBUF");
            close(out->fd); free(out); return NULL;
        }
        out->bufs[i].length = buf.length;
        out->bufs[i].start  = mmap(NULL, buf.length,
                                    PROT_READ | PROT_WRITE, MAP_SHARED,
                                    out->fd, buf.m.offset);
        if (out->bufs[i].start == MAP_FAILED) {
            perror("[output] mmap");
            close(out->fd); free(out); return NULL;
        }
        out->bufs[i].queued = false;
    }

    /* ── BGR24 → YUYV422 converter ── */
    out->sws = sws_getContext(out->width, out->height, AV_PIX_FMT_BGR24,
                               out->width, out->height, AV_PIX_FMT_YUYV422,
                               SWS_BILINEAR, NULL, NULL, NULL);
    if (!out->sws) {
        fprintf(stderr, "[output] sws_getContext failed\n");
        close(out->fd); free(out); return NULL;
    }

    fprintf(stderr, "[output] %s  %dx%d @ %d fps  YUYV422  "
                    "colorspace=%s  buffers=%d (streaming I/O)\n",
            device, out->width, out->height, fps,
            (out->height >= 720) ? "BT.709" : "BT.601",
            out->nbuf);
    return out;
}

bool output_write(V4L2Output *out, const Frame *frame)
{
    /* Find a free (not-queued) buffer */
    int idx = -1;
    for (int i = 0; i < out->nbuf; i++) {
        if (!out->bufs[i].queued) { idx = i; break; }
    }

    if (idx < 0) {
        /* All buffers in kernel — dequeue one back */
        struct v4l2_buffer buf = {0};
        buf.type   = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(out->fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) return true;
            perror("[output] VIDIOC_DQBUF");
            return false;
        }
        idx = buf.index;
        out->bufs[idx].queued = false;
    }

    /* Convert BGR24 → YUYV422 directly into the mmap'd buffer */
    const uint8_t *src[4] = { frame->data, NULL, NULL, NULL };
    int src_stride[4]     = { frame->stride, 0, 0, 0 };
    uint8_t *dst[4]       = { out->bufs[idx].start, NULL, NULL, NULL };
    int dst_stride[4]     = { out->width * 2, 0, 0, 0 };

    sws_scale(out->sws, src, src_stride, 0, out->height, dst, dst_stride);

    /* Queue it to the kernel */
    struct v4l2_buffer qbuf = {0};
    qbuf.type      = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    qbuf.memory    = V4L2_MEMORY_MMAP;
    qbuf.index     = idx;
    qbuf.bytesused = out->frame_size;
    qbuf.field     = V4L2_FIELD_NONE;
    if (xioctl(out->fd, VIDIOC_QBUF, &qbuf) < 0) {
        perror("[output] VIDIOC_QBUF");
        return false;
    }
    out->bufs[idx].queued = true;

    /* Start streaming once the first buffer is queued */
    if (!out->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        if (xioctl(out->fd, VIDIOC_STREAMON, &type) < 0) {
            perror("[output] VIDIOC_STREAMON");
            return false;
        }
        out->streaming = true;
        fprintf(stderr, "[output] streaming started\n");
    }
    return true;
}

void output_close(V4L2Output *out)
{
    if (!out) return;
    if (out->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        xioctl(out->fd, VIDIOC_STREAMOFF, &type);
    }
    for (int i = 0; i < out->nbuf; i++)
        if (out->bufs[i].start && out->bufs[i].start != MAP_FAILED)
            munmap(out->bufs[i].start, out->bufs[i].length);
    if (out->sws) sws_freeContext(out->sws);
    close(out->fd);
    free(out);
}
