/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Modified-by: Claude Opus 5.5 (--version, g_no_input, socket path)
 */

/*
 * vcam – virtual webcam pipeline with runtime control
 *
 * Usage:  vcam [OPTIONS]
 *
 * Pipeline options:
 *   -i <device>   Input webcam device   (default: /dev/video1)
 *   -o <device>   Output v4l2loopback   (default: /dev/video0)
 *   --res WxH     Output resolution     (default: camera native)
 *   --list-formats  List supported input resolutions, then exit
 *   -f <fps>      Output frame rate     (default: camera native)
 *
 * Startup overlays (same syntax as IPC 'add' command):
 *   --img    <id> <x> <y> [<w> <h>] <file>
 *   --vid    <id> <x> <y> [<w> <h>] <file>
 *   --screen <id> <x> <y> [<w> <h>] <geometry>
 *
 * Runtime control via Unix socket /tmp/vcam.sock:
 *   echo "list"                       | nc -U /tmp/vcam.sock
 *   echo "add img logo 10 10 200 200 /path/to/logo.png" | nc -U /tmp/vcam.sock
 *   echo "remove logo"                | nc -U /tmp/vcam.sock
 *   echo "move logo 50 50"            | nc -U /tmp/vcam.sock
 *   echo "resize logo 300 300"        | nc -U /tmp/vcam.sock
 *   echo "add screen scr 0 0 640 360 1920x1080+0+0" | nc -U /tmp/vcam.sock
 *   vcam-ctl <command>                (convenience wrapper)
 */

#include "vcam.h"
#include "vcam_cv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>
#include <sys/wait.h>

#include <libavutil/log.h>

/* ── global state ─────────────────────────────── */
static volatile bool    g_running = true;
static OverlayRegistry  g_registry;
static IPCServer        g_ipc;
FilterChain             g_filters; /* extern'd by ipc.c */
int                     g_output_w = 0; /* extern'd by overlay.c */
int                     g_output_h = 0;
int                     g_output_fps = 30; /* extern'd by ipc.c (record) */
Recorder               *g_recorder = NULL; /* extern'd by ipc.c */
bool                    g_no_input = false; /* extern'd by api.c (status) */

static void on_signal(int sig)
{
    (void)sig;
    g_running = false;
}

static void on_sigchld(int sig)
{
    (void)sig;
    /* Reap any dead screen-capture child processes */
    while (waitpid(-1, NULL, WNOHANG) > 0) {}
}

/* ── arg helpers ───────────────────────────────── */
static bool is_flag(const char *s)
{
    return s && s[0] == '-';
}

/* Parse optional [w] [h] before the final path/geometry token.
 * Supports: "w h path", "w path", or just "path".
 * Returns how many extra argv slots were consumed. */
static int try_wh(char *argv[], int i, int argc, int *w, int *h)
{
    if (i >= argc || is_flag(argv[i])) return 0;
    char *endp;
    long v0 = strtol(argv[i], &endp, 10);
    if (*endp != '\0') return 0;

    if (i+1 < argc && !is_flag(argv[i+1])) {
        long v1 = strtol(argv[i+1], &endp, 10);
        if (*endp == '\0' && i+2 < argc) {
            *w = (int)v0; *h = (int)v1;
            return 2;
        }
    }
    if (i+1 < argc) { *w = (int)v0; *h = 0; return 1; }
    return 0;
}

/* Parse a "WxH" string (e.g. "1280x720") into w/h. Returns true on
 * success. Accepts 'x' or 'X' as the separator. */
static bool parse_wxh(const char *s, int *w, int *h)
{
    char *ep;
    long ww = strtol(s, &ep, 10);
    if (ep == s || (*ep != 'x' && *ep != 'X')) return false;
    const char *hs = ep + 1;
    long hh = strtol(hs, &ep, 10);
    if (ep == hs || *ep != '\0') return false;
    if (ww <= 0 || hh <= 0 || ww > 16384 || hh > 16384) return false;
    *w = (int)ww; *h = (int)hh;
    return true;
}

/* ── main ──────────────────────────────────────── */
int main(int argc, char *argv[])
{
    const char *input_dev  = "/dev/video1";
    const char *output_dev = "/dev/video0";
    int  width = 0, height = 0, fps = 0;
    bool no_input = false;     /* --no-input: skip webcam capture entirely */

    /* Pre-scan for --list-formats (and the -i that feeds it). This is a
     * query-and-exit mode, so we handle it before the main parse and
     * before touching any pipeline state. */
    {
        const char *dev = input_dev;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
                dev = argv[i+1];
        }
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-V") == 0) {
                puts(api_version_string());
                return 0;
            }
            if (strcmp(argv[i], "--list-formats") == 0 ||
                strcmp(argv[i], "--list") == 0) {
                return capture_list_formats(dev) ? 0 : 1;
            }
        }
    }

    registry_init(&g_registry);
    filter_chain_init(&g_filters);

    /* ── parse args ── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc) {
            input_dev = argv[++i];

        } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            output_dev = argv[++i];

        } else if ((strcmp(argv[i], "--res") == 0 ||
                    strcmp(argv[i], "-r") == 0) && i+1 < argc) {
            /* --res WxH : the preferred way to choose resolution.
             * See 'vcam --list-formats' for what the device supports. */
            if (!parse_wxh(argv[++i], &width, &height)) {
                fprintf(stderr, "vcam: bad --res '%s' (expected WxH, "
                        "e.g. 1280x720)\n", argv[i]);
                return 1;
            }

        } else if (strcmp(argv[i], "-W") == 0 && i+1 < argc) {
            width  = atoi(argv[++i]);   /* legacy; --res is preferred */

        } else if (strcmp(argv[i], "-H") == 0 && i+1 < argc) {
            height = atoi(argv[++i]);   /* legacy; --res is preferred */

        } else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            fps    = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--no-input") == 0) {
            no_input = true;

        } else if (strcmp(argv[i], "--filter") == 0 && i+1 < argc) {
            /* --filter <name> [<param>] */
            const char *name = argv[++i];
            FilterType t = filter_type_from_name(name);
            if (t == FX_UNKNOWN) {
                fprintf(stderr, "[main] Unknown filter '%s'\n", name);
                continue;
            }
            float param = filter_default_param(t);
            if (i+1 < argc && !is_flag(argv[i+1])) {
                char *ep;
                float v = strtof(argv[i+1], &ep);
                if (*ep == '\0') { param = v; i++; }
            }
            filter_chain_add(&g_filters, t, param);

        } else if ((strcmp(argv[i], "--img") == 0 ||
                    strcmp(argv[i], "--vid") == 0 ||
                    strcmp(argv[i], "--vid-once") == 0 ||
                    strcmp(argv[i], "--screen") == 0) && i+4 < argc) {

            const char *flag = argv[i];
            const char *kind = (strncmp(flag, "--vid", 5) == 0) ? "vid" :
                               (strcmp(flag, "--img") == 0)     ? "img" : "screen";
            bool vid_loop    = (strcmp(flag, "--vid-once") != 0); /* once = no loop */
            const char *id   = argv[++i];
            int x = atoi(argv[++i]);
            int y = atoi(argv[++i]);
            int w = 0, h = 0;
            int extra = try_wh(argv, i+1, argc, &w, &h);
            i += extra;
            if (i+1 >= argc) {
                fprintf(stderr, "[main] Missing file/geometry for %s\n", flag);
                continue;
            }
            const char *path = argv[++i];

            Overlay *o = NULL;
            if (strcmp(kind, "img") == 0)
                o = overlay_create_image(path, x, y, w, h);
            else if (strcmp(kind, "vid") == 0)
                o = overlay_create_video(path, x, y, w, h, vid_loop);
            else
                o = overlay_create_screen(path, x, y, w, h);

            if (!o) {
                fprintf(stderr, "[main] Failed to create %s overlay '%s'\n", kind, id);
                continue;
            }
            if (!registry_add(&g_registry, id, o))
                overlay_free(o);

        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            puts(
"vcam – virtual webcam pipeline\n\n"
"Usage: vcam [OPTIONS]\n\n"
"Pipeline:\n"
"  -i <dev>          Input webcam       (default /dev/video1)\n"
"  -o <dev>          Output loopback    (default /dev/video0)\n"
"  --res WxH         Output resolution  (e.g. --res 1280x720)\n"
"  -f <fps>          Frame rate\n"
"  --list-formats    List the resolutions the input device supports, then exit\n"
"  --version, -V     Print version + build hash, then exit\n"
"  --no-input        Don't open a webcam (use screen overlays as source).\n"
"                    Combine with --res and a fullscreen --screen overlay.\n\n"
"Startup overlays:\n"
"  --img    <id> <x> <y> [<w> [<h>]] <file>\n"
"  --vid    <id> <x> <y> [<w> [<h>]] <file>\n"
"  --screen <id> <x> <y> [<w> [<h>]] <WxH+X+Y>\n\n"
"Startup filters (chain order matters):\n"
"  --filter <name> [<param>]\n"
"  Available: grayscale, sepia, invert, mirror, flip,\n"
"             pixelate <block>, blur <radius>, vignette <0..1>,\n"
"             brightness <mult>, contrast <mult>, posterize <levels>, edges\n\n"
"Runtime control (vcam-ctl):\n"
"  vcam-ctl list\n"
"  vcam-ctl add img|vid|screen <id> <x> <y> [<w> [<h>]] <file>\n"
"  vcam-ctl remove|move|resize ...\n"
"  vcam-ctl filter add <name> [<param>]\n"
"  vcam-ctl filter remove <index>\n"
"  vcam-ctl filter list / clear\n"
"  vcam-ctl status [--json] | get <key>\n"
"  vcam-ctl share start|stop|toggle\n"
"  vcam-ctl cam on|off|pos|move|size\n"
            );
            registry_destroy(&g_registry);
            filter_chain_destroy(&g_filters);
            return 0;

        } else {
            fprintf(stderr, "[main] Unknown option: %s  (try --help)\n", argv[i]);
        }
    }

    g_no_input = no_input;

    /* ── signals ── */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGCHLD, on_sigchld);
    signal(SIGPIPE, SIG_IGN);

    /* Silence libav's INFO/WARNING spam — "deprecated pixel format used"
     * (JPEG full-range YUV), "EOI missing, emulating" (slightly malformed
     * MJPEG from some webcams), etc. We still see real errors. */
    av_log_set_level(AV_LOG_ERROR);

    /* ── open capture (unless --no-input) ── */
    V4L2Capture *cap = NULL;
    if (!no_input) {
        cap = capture_open(input_dev, &width, &height, &fps);
        if (!cap) { registry_destroy(&g_registry); filter_chain_destroy(&g_filters); return 1; }
    } else {
        if (width <= 0 || height <= 0) {
            fprintf(stderr, "[main] --no-input requires -W and -H\n");
            registry_destroy(&g_registry); filter_chain_destroy(&g_filters); return 1;
        }
        if (fps <= 0) fps = 30;
        fprintf(stderr, "[main] No-input mode: %dx%d @ %d fps (no webcam opened)\n",
                width, height, fps);
    }

    /* ── open output ── */
    V4L2Output *out = output_open(output_dev, width, height, fps);
    if (!out) {
        if (cap) capture_close(cap);
        registry_destroy(&g_registry); filter_chain_destroy(&g_filters);
        return 1;
    }
    g_output_w = width;
    g_output_h = height;
    g_output_fps = fps;

    /* Recorder: idle until 'record start' IPC fires it up. */
    g_recorder = recorder_new();

    /* ── start IPC server ── */
    if (!ipc_start(&g_ipc, &g_registry)) {
        fprintf(stderr, "[main] IPC server failed to start (continuing without it)\n");
    }

    /* ── allocate work frames ──
     * cam_buf holds the filtered webcam (or stays black in --no-input
     * mode); webcam overlays sample from it. frame is the output
     * frame that overlays composite onto. Keeping them separate so
     * overlay blits don't pollute the webcam source. */
    Frame *cam_buf = frame_alloc(width, height);
    Frame *frame   = frame_alloc(width, height);
    if (!cam_buf || !frame) {
        ipc_stop(&g_ipc);
        output_close(out);
        if (cap) capture_close(cap);
        registry_destroy(&g_registry); filter_chain_destroy(&g_filters);
        return 1;
    }

    fprintf(stderr, "[main] Pipeline: %s → %s  %dx%d @ %d fps\n",
            no_input ? "(no input)" : input_dev,
            output_dev, width, height, fps);
    fprintf(stderr, "[main] IPC socket: %s\n", ipc_sock_path());
    fprintf(stderr, "[main] %d startup overlay(s), %d startup filter(s).\n",
            g_registry.count, g_filters.count);

    long frame_ns = fps > 0 ? (1000000000L / fps) : 33333333L;

    /* ── main loop ── */
    while (g_running) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* 1. Capture the webcam (if open) into cam_buf */
        if (cap) {
            if (!capture_read(cap, cam_buf))
                continue;
        } else {
            memset(cam_buf->data, 0,
                   (size_t)cam_buf->stride * cam_buf->height);
        }

        /* 2. Apply filter chain to cam_buf — filters operate on the
         *    raw camera stream, so they affect both the base AND any
         *    webcam-corner overlays consistently. */
        filter_chain_apply(&g_filters, cam_buf);

        /* 3. Build the base frame. Default = filtered cam. With
         *    --no-input the base is black (screen/image overlays will
         *    paint on top). */
        if (!no_input) {
            memcpy(frame->data, cam_buf->data,
                   (size_t)cam_buf->stride * cam_buf->height);
        } else {
            memset(frame->data, 0,
                   (size_t)frame->stride * frame->height);
        }

        /* 4. Composite overlays. Webcam overlays read from cam_buf
         *    (so they always show the latest filtered cam, not the
         *    accumulating composite). */
        pthread_mutex_lock(&g_registry.lock);
        for (int i = 0; i < g_registry.count; i++) {
            Overlay *o = g_registry.slots[i].overlay;
            overlay_tick(o, cam_buf);
            overlay_blit(o, frame);
        }
        pthread_mutex_unlock(&g_registry.lock);

        output_write(out, frame);

        /* Mirror the composed frame to the recorder, if active.
         * This is BGR24 — same format we asked ffmpeg to expect
         * via -f rawvideo -pixel_format bgr24. */
        if (recorder_active(g_recorder)) {
            recorder_write_frame(g_recorder, frame->data,
                                 (size_t)frame->stride * frame->height);
        }

        /* Frame pacing */
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000000L
                     + (t1.tv_nsec - t0.tv_nsec);
        long remaining = frame_ns - elapsed;
        if (remaining > 1000000L) {
            struct timespec sl = { remaining / 1000000000L, remaining % 1000000000L };
            nanosleep(&sl, NULL);
        }
    }

    fprintf(stderr, "\n[main] Shutting down.\n");

    ipc_stop(&g_ipc);
    recorder_free(g_recorder);   /* stops + finalises any active recording */
    g_recorder = NULL;
    frame_free(frame);
    frame_free(cam_buf);
    output_close(out);
    if (cap) capture_close(cap);
    registry_destroy(&g_registry);
    filter_chain_destroy(&g_filters);
    cv_shutdown();   /* free any OpenCV state (no-op in non-OpenCV builds) */
    return 0;
}
