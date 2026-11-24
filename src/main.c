/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Modified-by: Claude Opus 5.5 (--version, g_no_input, socket path,
 *              live capture mode switching, default 1280x720, fast path,
 *              idle mode, stats, ML device options, libav log de-dup)
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
#include "vcam_ort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <poll.h>

#include <libavutil/log.h>
#include <stdarg.h>

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

/* ── live capture switching ───────────────────── [Claude Opus 5.5]
 * api.c validates a request and posts it with capture_request(); the
 * main loop applies it between two frames (close → open new mode →
 * reopen the old one if that fails) and signals the result back.
 */
static const char     *g_input_dev = "/dev/video1";
static pthread_mutex_t g_cap_lock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cap_cond  = PTHREAD_COND_INITIALIZER;
static CaptureState    g_cap_state;            /* current; guarded by g_cap_lock */
static CaptureMode     g_req_mode;
static bool            g_req_reopen;
static int             g_req_fit = -1;
static unsigned        g_req_seq, g_done_seq;
static bool            g_req_ok;
static char            g_req_err[192];

const char *capture_device_path(void) { return g_input_dev; }

void capture_state_get(CaptureState *st)
{
    pthread_mutex_lock(&g_cap_lock);
    *st = g_cap_state;
    pthread_mutex_unlock(&g_cap_lock);
}

static void capture_state_set(V4L2Capture *cap)
{
    pthread_mutex_lock(&g_cap_lock);
    g_cap_state.active = cap != NULL;
    if (cap) capture_get(cap, &g_cap_state.mode, &g_cap_state.crop);
    /* closed: keep mode + crop, so idle can resume the same mode */
    pthread_mutex_unlock(&g_cap_lock);
}

bool capture_request(const CaptureMode *mode, int fit, char *err, size_t errlen)
{
    pthread_mutex_lock(&g_cap_lock);
    g_req_reopen = mode != NULL;
    if (mode) g_req_mode = *mode;
    g_req_fit = fit;
    unsigned seq = ++g_req_seq;
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += 6;
    int rc = 0;
    while (g_done_seq < seq && rc == 0)
        rc = pthread_cond_timedwait(&g_cap_cond, &g_cap_lock, &dl);
    bool done = g_done_seq >= seq;
    bool ok   = done && g_req_ok;
    if (!ok) snprintf(err, errlen, "%s", done ? g_req_err : "timeout (camera not responding)");
    pthread_mutex_unlock(&g_cap_lock);
    return ok;
}

/* Called by the main loop once per frame; cheap when nothing is pending. */
static void capture_service(V4L2Capture **cap)
{
    pthread_mutex_lock(&g_cap_lock);
    if (g_done_seq == g_req_seq) { pthread_mutex_unlock(&g_cap_lock); return; }
    unsigned     seq    = g_req_seq;
    CaptureMode  m      = g_req_mode;
    bool         reopen = g_req_reopen;
    int          fit    = g_req_fit;
    CaptureState old    = g_cap_state;
    pthread_mutex_unlock(&g_cap_lock);

    bool crop = fit < 0 ? old.crop : fit == 1;
    bool ok = true;
    char err[192] = "";

    if (reopen || !*cap) {
        if (!reopen) m = old.mode;     /* fit change while camera is down: retry it */
        if (*cap) { capture_close(*cap); *cap = NULL; }
        *cap = capture_open_mode(g_input_dev, &m, g_output_w, g_output_h, crop);
        if (!*cap) {
            ok = false;
            snprintf(err, sizeof err, "camera refused %s %dx%d@%d", m.fmt, m.w, m.h, m.fps);
            if (old.active) {
                *cap = capture_open_mode(g_input_dev, &old.mode, g_output_w, g_output_h, old.crop);
                if (!*cap)
                    snprintf(err + strlen(err), sizeof err - strlen(err), "; old mode failed too, no picture");
            }
        }
    } else if (!capture_set_fit(*cap, crop)) {
        ok = false;
        snprintf(err, sizeof err, "out of memory changing fit");
    }
    fprintf(stderr, "[capture] switch %s → %s\n", ok ? "ok" : "FAILED", ok ? m.fmt : err);

    capture_state_set(*cap);
    pthread_mutex_lock(&g_cap_lock);
    g_req_ok = ok;
    snprintf(g_req_err, sizeof g_req_err, "%s", err);
    g_done_seq = seq;
    pthread_cond_broadcast(&g_cap_cond);
    pthread_mutex_unlock(&g_cap_lock);
}

/* ── idle + stats ─────────────────────────────── [Claude Opus 5.5]
 * Idle: when no other process has the loopback device open (see
 * consumers.c) for IDLE_GRACE_S, the camera is released and one black
 * frame per second keeps the virtual camera alive. An open() wakes us.
 * Stats: per-stage times summed over ~1 s windows, published for 'stats'.
 */
#define IDLE_GRACE_S 3.0

static volatile bool g_idle_auto = true;
static volatile bool g_idle      = false;
static volatile int  g_consumers = -1;

bool vcam_idle_get(int *consumers, bool *idle_auto)
{
    if (consumers) *consumers = g_consumers;
    if (idle_auto) *idle_auto = g_idle_auto;
    return g_idle;
}
void vcam_idle_set_auto(bool on) { g_idle_auto = on; }

static pthread_mutex_t g_stats_lock = PTHREAD_MUTEX_INITIALIZER;
static VcamStats       g_stats;            /* last published window */

void vcam_stats_get(VcamStats *st)
{
    pthread_mutex_lock(&g_stats_lock);
    *st = g_stats;
    pthread_mutex_unlock(&g_stats_lock);
}

typedef struct {
    struct timespec t0;
    double cpu0;
    long   in, out;
    long   ns_cap, ns_fil, ns_comp, ns_out;
    long   total_out;
} StatsWin;

static double ts_s(const struct timespec *t) { return t->tv_sec + t->tv_nsec / 1e9; }

static double cpu_seconds(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 +
           ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
}

static long ns_between(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1000000000L + (b->tv_nsec - a->tv_nsec);
}

/* Publish the window if ≥ 1 s old, then start a new one. */
static void stats_roll(StatsWin *w, const char *path, int fps)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = ts_s(&now) - ts_s(&w->t0);
    if (dt < 1.0) return;
    double cpu = cpu_seconds();
    VcamStats st = {0};
    long n = w->out > 0 ? w->out : 1;
    st.window_s   = dt;
    st.fps_in     = w->in  / dt;
    st.fps_out    = w->out / dt;
    st.cpu        = (cpu - w->cpu0) / dt * 100.0;
    snprintf(st.path, sizeof st.path, "%s", path);
    st.capture_ms = w->ns_cap  / 1e6 / n;
    st.filters_ms = w->ns_fil  / 1e6 / n;
    st.compose_ms = w->ns_comp / 1e6 / n;
    st.output_ms  = w->ns_out  / 1e6 / n;
    st.work_ms    = st.capture_ms + st.filters_ms + st.compose_ms + st.output_ms;
    st.budget_ms  = fps > 0 ? 1000.0 / fps : 0;
    st.frames     = w->total_out;
    pthread_mutex_lock(&g_stats_lock);
    g_stats = st;
    pthread_mutex_unlock(&g_stats_lock);
    long total = w->total_out;
    memset(w, 0, sizeof *w);
    w->t0 = now; w->cpu0 = cpu; w->total_out = total;
}

/* ── libav log filter ──────────────────────────── [Claude Opus 5.5]
 * Some webcams put metadata in every JPEG that FFmpeg can't parse, and it
 * reports that at *error* level once per frame ("unable to decode APP
 * fields"), 30 lines a second. Print each distinct message once, hide
 * repeats. VCAM_AV_LOG=all shows everything. */
#define AVLOG_SEEN_MAX 64
static pthread_mutex_t g_avlog_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned        g_avlog_seen[AVLOG_SEEN_MAX];
static int             g_avlog_n;

static void avlog_once(void *avcl, int level, const char *fmt, va_list vl)
{
    if (level > av_log_get_level()) return;
    char line[1024];
    int prefix = 1;
    va_list cp;
    va_copy(cp, vl);
    av_log_format_line2(avcl, level, fmt, cp, line, sizeof line, &prefix);
    va_end(cp);

    unsigned h = 2166136261u;                         /* FNV-1a */
    for (const char *p = line; *p; p++) h = (h ^ (unsigned char)*p) * 16777619u;
    pthread_mutex_lock(&g_avlog_lock);
    for (int i = 0; i < g_avlog_n; i++)
        if (g_avlog_seen[i] == h) { pthread_mutex_unlock(&g_avlog_lock); return; }
    bool tracked = g_avlog_n < AVLOG_SEEN_MAX;
    if (tracked) g_avlog_seen[g_avlog_n++] = h;
    pthread_mutex_unlock(&g_avlog_lock);

    size_t len = strlen(line);
    if (tracked && len && line[len - 1] == '\n') {
        line[len - 1] = '\0';
        fprintf(stderr, "%s (repeats hidden)\n", line);
    } else {
        fputs(line, stderr);
    }
}

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
    const char *ml_device  = getenv("VCAM_ML_DEVICE");   /* auto|npu|cpu */
    const char *npu_config = getenv("VCAM_NPU_CONFIG");
    int  ml_threads = 0;       /* 0 = vcam default (4) */
    bool ml_async   = true;

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

        } else if (strcmp(argv[i], "--idle") == 0 && i+1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "auto") && strcmp(v, "off")) {
                fprintf(stderr, "vcam: --idle takes auto|off\n");
                return 1;
            }
            g_idle_auto = strcmp(v, "auto") == 0;

        } else if (strcmp(argv[i], "--ml-device") == 0 && i+1 < argc) {
            ml_device = argv[++i];
            if (strcmp(ml_device, "auto") && strcmp(ml_device, "npu") && strcmp(ml_device, "cpu")) {
                fprintf(stderr, "vcam: --ml-device takes auto|npu|cpu\n");
                return 1;
            }

        } else if (strcmp(argv[i], "--ml-threads") == 0 && i+1 < argc) {
            ml_threads = atoi(argv[++i]);

        } else if (strcmp(argv[i], "--ml-async") == 0 && i+1 < argc) {
            ml_async = strcmp(argv[++i], "off") != 0;

        } else if (strcmp(argv[i], "--npu-config") == 0 && i+1 < argc) {
            npu_config = argv[++i];

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
"  --res WxH         Output resolution  (default 1280x720, or the camera mode\n"
"                    closest to it; see --list-formats)\n"
"  -f <fps>          Frame rate\n"
"  --list-formats    List the resolutions the input device supports, then exit\n"
"  --version, -V     Print version + build hash, then exit\n"
"  --no-input        Don't open a webcam (use screen overlays as source).\n"
"                    Combine with --res and a fullscreen --screen overlay.\n"
"  --idle auto|off   auto (default): release the camera and idle while no app\n"
"                    reads the virtual camera\n\n"
"Machine learning (portrait/background, posefx, depthfx, genai):\n"
"  --ml-device auto|npu|cpu   auto (default): NPU if usable, else CPU\n"
"  --ml-threads N    CPU threads per model (default 4)\n"
"  --ml-async on|off on (default): models run beside the video, never stall it\n"
"  --npu-config FILE Vitis AI config file for the NPU (env VCAM_NPU_CONFIG)\n\n"
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
"  vcam-ctl stats [--json] | idle [auto|off] | ml\n"
            );
            registry_destroy(&g_registry);
            filter_chain_destroy(&g_filters);
            return 0;

        } else {
            fprintf(stderr, "[main] Unknown option: %s  (try --help)\n", argv[i]);
        }
    }

    g_no_input = no_input;
    ort_config(ml_device, ml_threads, npu_config, ml_async);

    /* ── signals ── */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGCHLD, on_sigchld);
    signal(SIGPIPE, SIG_IGN);

    /* Silence libav's INFO/WARNING spam — "deprecated pixel format used"
     * (JPEG full-range YUV), "EOI missing, emulating" (slightly malformed
     * MJPEG from some webcams), etc. We still see real errors. */
    av_log_set_level(AV_LOG_ERROR);
    const char *avl = getenv("VCAM_AV_LOG");
    if (!avl || strcmp(avl, "all") != 0)
        av_log_set_callback(avlog_once);

    /* ── open capture (unless --no-input) ── */
    V4L2Capture *cap = NULL;
    if (!no_input) {
        /* No --res: 1280x720 (or the closest mode) instead of the camera's
         * maximum — a 5 MP output costs ~5x the CPU and apps scale it down. */
        if ((width <= 0 || height <= 0) &&
            capture_default_size(input_dev, 1280, 720, fps, &width, &height))
            fprintf(stderr, "[main] no --res given: using %dx%d\n", width, height);
        cap = capture_open(input_dev, &width, &height, &fps);
        if (!cap) { registry_destroy(&g_registry); filter_chain_destroy(&g_filters); return 1; }
        g_input_dev = input_dev;
        capture_state_set(cap);
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

    /* ── idle detection ── */
    Consumers cons;
    bool cons_ok = consumers_init(&cons, output_dev);
    if (cons_ok)
        fprintf(stderr, "[main] idle: %s (%d reader(s) now)\n",
                g_idle_auto ? "auto" : "off", cons.count);
    else
        fprintf(stderr, "[main] idle: unavailable (cannot watch %s) — always running\n",
                output_dev);
    g_consumers = cons_ok ? cons.count : -1;

    struct timespec last_reader, last_black = {0}, last_reopen = {0};
    clock_gettime(CLOCK_MONOTONIC, &last_reader);
    StatsWin win = {0};
    clock_gettime(CLOCK_MONOTONIC, &win.t0);
    win.cpu0 = cpu_seconds();
    const char *path = "full";

    /* ── main loop ── */
    while (g_running) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* 0. Apply a pending live capture switch (from the API) */
        if (!no_input) capture_service(&cap);

        /* 0b. Idle? Nobody reading for IDLE_GRACE_S, not recording. */
        if (cons_ok) g_consumers = consumers_update(&cons);
        bool want_idle = cons_ok && g_idle_auto && g_consumers == 0 &&
                         !recorder_active(g_recorder);
        if (!want_idle) last_reader = t0;
        bool idle_now = want_idle && ns_between(&last_reader, &t0) >= IDLE_GRACE_S * 1e9;
        if (idle_now != g_idle) {
            g_idle = idle_now;
            fprintf(stderr, "[main] %s\n", idle_now
                    ? "idle: no app reads the virtual camera — camera released"
                    : g_consumers > 0 ? "awake: an app opened the virtual camera"
                    : !g_idle_auto     ? "awake: idle switched off"
                    :                    "awake: recording");
            if (!idle_now) last_reopen.tv_sec = 0;   /* reopen right away */
        }
        if (idle_now) {
            if (cap) { capture_close(cap); cap = NULL; capture_state_set(NULL); }
            if (ns_between(&last_black, &t0) >= 1000000000L) {
                output_write_black(out);
                last_black = t0;
            }
            path = "idle";
            stats_roll(&win, path, fps);
            struct pollfd pfd = { cons.ifd, POLLIN, 0 };
            poll(&pfd, 1, 100);           /* wakes at once on open() */
            continue;
        }

        /* 0c. Camera closed (after idle, or lost): reopen, retry every 2 s */
        if (!no_input && !cap && ns_between(&last_reopen, &t0) >= 2000000000L) {
            CaptureState cs;
            capture_state_get(&cs);
            last_reopen = t0;
            if (cs.mode.w > 0) {
                cap = capture_open_mode(g_input_dev, &cs.mode, width, height, cs.crop);
                capture_state_set(cap);
                fprintf(stderr, "[main] camera %s (%s %dx%d@%d)\n",
                        cap ? "reopened" : "reopen failed, retrying",
                        cs.mode.fmt, cs.mode.w, cs.mode.h, cs.mode.fps);
            }
        }

        /* Fast path: nothing needs BGR → camera YUV straight into the
         * loopback buffer (no YUV→BGR→YUV, no copies). */
        pthread_mutex_lock(&g_filters.lock);
        int nfilters = g_filters.count;
        pthread_mutex_unlock(&g_filters.lock);
        bool fast = cap && nfilters == 0 && g_registry.count == 0 &&
                    !recorder_active(g_recorder);
        struct timespec ta, tb, tc, td;

        if (fast) {
            int ostride;
            uint8_t *obuf = output_begin(out, &ostride);
            if (!obuf) continue;
            if (!capture_read_yuyv(cap, obuf, ostride))
                continue;                 /* buffer stays free for next time */
            win.in++;
            clock_gettime(CLOCK_MONOTONIC, &tc);
            output_commit(out);
            clock_gettime(CLOCK_MONOTONIC, &td);
            win.ns_cap += capture_last_work_ns(cap);
            win.ns_out += ns_between(&tc, &td);
            path = "fast";
        } else {
            /* 1. Capture the webcam (if open) into cam_buf */
            if (cap) {
                if (!capture_read(cap, cam_buf))
                    continue;
                win.in++;
                win.ns_cap += capture_last_work_ns(cap);
                path = "full";
            } else {
                memset(cam_buf->data, 0,
                       (size_t)cam_buf->stride * cam_buf->height);
                path = "nocam";
            }
            clock_gettime(CLOCK_MONOTONIC, &ta);

            /* 2. Apply filter chain to cam_buf — filters operate on the
             *    raw camera stream, so they affect both the base AND any
             *    webcam-corner overlays consistently. */
            filter_chain_apply(&g_filters, cam_buf);
            clock_gettime(CLOCK_MONOTONIC, &tb);

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
            clock_gettime(CLOCK_MONOTONIC, &tc);

            output_write(out, frame);

            /* Mirror the composed frame to the recorder, if active.
             * This is BGR24 — same format we asked ffmpeg to expect
             * via -f rawvideo -pixel_format bgr24. */
            if (recorder_active(g_recorder)) {
                recorder_write_frame(g_recorder, frame->data,
                                     (size_t)frame->stride * frame->height);
            }
            clock_gettime(CLOCK_MONOTONIC, &td);
            win.ns_fil  += ns_between(&ta, &tb);
            win.ns_comp += ns_between(&tb, &tc);
            win.ns_out  += ns_between(&tc, &td);
        }
        win.out++;
        win.total_out++;
        stats_roll(&win, path, fps);

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

    consumers_close(&cons);
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
