#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <poll.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <png.h>

/* ──────────────────────────────────────────────────────────────
 * Overlay struct
 * ────────────────────────────────────────────────────────────── */
struct Overlay {
    OverlayType type;

    int dst_x, dst_y, dst_w, dst_h;

    uint8_t *pixels;   /* BGR24 or BGRA pixel buffer */
    int      bpp;      /* 3 = BGR24, 4 = BGRA        */

    /* ── image-overlay-only: keep the decoded source at its native
     * resolution so we can re-scale on resize without re-reading
     * the file. NULL for non-image overlays. ── */
    uint8_t *src_pixels;
    int      src_bpp_img;

    /* ── video fields ── */
    AVFormatContext  *fmt_ctx;
    AVCodecContext   *codec_ctx;
    AVFrame          *av_frame;
    AVPacket         *av_pkt;
    struct SwsContext *sws;
    int               stream_idx;
    bool              vid_loop;        /* if true, restart on EOF */
    bool              vid_eof_sent;    /* flush sentinel already sent */
    bool              vid_finished;    /* play-once + EOF reached → keep last frame */

    int src_w, src_h;

    /* ── screen capture fields ── */
    pid_t    child_pid;
    int      pipe_fd;        /* read end of pipe/fifo from capture child   */
    char     fifo_path[128]; /* path if using a FIFO (Wayland), else ""    */
    int      src_bpp;        /* bytes per pixel from child: 3=BGR24 4=BGR0 */
    size_t   raw_bytes;      /* bytes per frame from child                 */

    /* Dedicated reader thread: pulls frames as fast as possible into
     * latest_frame, so the main loop always uses the freshest data
     * (minimizing capture-to-output latency). */
    pthread_t       reader_thread;
    bool            reader_running;
    bool            reader_shutdown;
    pthread_mutex_t latest_lock;
    uint8_t        *latest_frame;  /* protected by latest_lock */
    bool            has_frame;     /* protected by latest_lock */

    /* ── webcam-overlay fields ──
     * Webcam overlays don't open a V4L2 device themselves (V4L2 only
     * permits one streamer per cam). Instead they reference the main
     * pipeline's shared capture, which the main loop passes into
     * overlay_tick() each frame. */
    int webcam_src_w, webcam_src_h;  /* cached src dims for sws caching */
};

/* ──────────────────────────────────────────────────────────────
 * Helpers
 * ────────────────────────────────────────────────────────────── */
static void rebuild_sws(Overlay *o, int sw, int sh, int sfmt)
{
    if (o->sws) sws_freeContext(o->sws);
    o->sws = sws_getContext(sw, sh, sfmt,
                             o->dst_w, o->dst_h, AV_PIX_FMT_BGR24,
                             SWS_BILINEAR, NULL, NULL, NULL);
}

/* ──────────────────────────────────────────────────────────────
 * PNG loader → BGRA
 * ────────────────────────────────────────────────────────────── */
static uint8_t *load_png(const char *path, int *w, int *h, bool *has_alpha)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info  = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) { fclose(fp); return NULL; }

    png_init_io(png, fp);
    png_read_info(png, info);

    *w = (int)png_get_image_width(png, info);
    *h = (int)png_get_image_height(png, info);
    png_byte ct = png_get_color_type(png, info);
    png_byte bd = png_get_bit_depth(png, info);

    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_RGB || ct == PNG_COLOR_TYPE_GRAY ||
        ct == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_read_update_info(png, info);

    *has_alpha = !!(png_get_color_type(png, info) & PNG_COLOR_MASK_ALPHA);
    size_t rb = png_get_rowbytes(png, info);
    uint8_t *rgba = malloc(rb * *h);
    png_bytep *rows = malloc(*h * sizeof *rows);
    for (int i = 0; i < *h; i++) rows[i] = rgba + i * rb;
    png_read_image(png, rows);
    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    uint8_t *bgra = malloc((size_t)*w * *h * 4);
    for (int i = 0; i < *w * *h; i++) {
        bgra[i*4+0] = rgba[i*4+2];
        bgra[i*4+1] = rgba[i*4+1];
        bgra[i*4+2] = rgba[i*4+0];
        bgra[i*4+3] = rgba[i*4+3];
    }
    free(rgba);
    return bgra;
}

/* ──────────────────────────────────────────────────────────────
 * JPEG / generic image loader → BGR24 (via FFmpeg)
 * ────────────────────────────────────────────────────────────── */
static uint8_t *load_jpeg_bgr(const char *path, int *w, int *h)
{
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return NULL;
    avformat_find_stream_info(fmt, NULL);

    int idx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++)
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            { idx = (int)i; break; }
    if (idx < 0) { avformat_close_input(&fmt); return NULL; }

    const AVCodec *codec = avcodec_find_decoder(fmt->streams[idx]->codecpar->codec_id);
    AVCodecContext *cc   = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cc, fmt->streams[idx]->codecpar);
    avcodec_open2(cc, codec, NULL);

    AVFrame  *frame = av_frame_alloc();
    AVPacket *pkt   = av_packet_alloc();
    uint8_t  *out   = NULL;

    while (av_read_frame(fmt, pkt) >= 0 && !out) {
        if (pkt->stream_index == idx) {
            avcodec_send_packet(cc, pkt);
            if (avcodec_receive_frame(cc, frame) == 0) {
                *w = frame->width; *h = frame->height;
                out = malloc((size_t)*w * *h * 3);
                struct SwsContext *sws = sws_getContext(
                    *w, *h, frame->format, *w, *h, AV_PIX_FMT_BGR24,
                    SWS_BILINEAR, NULL, NULL, NULL);
                uint8_t *d[4] = { out }; int dl[4] = { *w*3 };
                sws_scale(sws, (const uint8_t *const *)frame->data,
                          frame->linesize, 0, *h, d, dl);
                sws_freeContext(sws);
            }
        }
        av_packet_unref(pkt);
    }
    av_frame_free(&frame); av_packet_free(&pkt);
    avcodec_free_context(&cc); avformat_close_input(&fmt);
    return out;
}

/* ──────────────────────────────────────────────────────────────
 * Compute dst dimensions: support specifying only w or only h,
 * deriving the other from the source aspect ratio.
 * dst_w / dst_h of 0 means "not specified".
 * ────────────────────────────────────────────────────────────── */
static void resolve_dims(int src_w, int src_h,
                          int req_w, int req_h,
                          int *out_w, int *out_h)
{
    if (req_w > 0 && req_h > 0) {
        *out_w = req_w; *out_h = req_h;
    } else if (req_w > 0) {
        *out_w = req_w;
        *out_h = src_h > 0 ? (req_w * src_h / src_w) : req_w;
    } else if (req_h > 0) {
        *out_h = req_h;
        *out_w = src_w > 0 ? (req_h * src_w / src_h) : req_h;
    } else {
        *out_w = src_w; *out_h = src_h;
    }
    /* Ensure even dimensions (some codecs require it) */
    if (*out_w & 1) (*out_w)++;
    if (*out_h & 1) (*out_h)++;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_create_image
 * ────────────────────────────────────────────────────────────── */
Overlay *overlay_create_image(const char *path,
                               int dst_x, int dst_y, int dst_w, int dst_h)
{
    Overlay *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->type    = OVERLAY_IMAGE;
    o->dst_x   = dst_x; o->dst_y = dst_y;
    o->pipe_fd = -1;

    const char *ext = strrchr(path, '.');
    bool is_png = ext && strcasecmp(ext, ".png") == 0;

    int sw = 0, sh = 0;
    bool has_alpha = false;
    uint8_t *src = NULL;

    if (is_png) {
        src = load_png(path, &sw, &sh, &has_alpha);
        if (!src) { free(o); return NULL; }
        o->bpp = 4;
    } else {
        src = load_jpeg_bgr(path, &sw, &sh);
        if (!src) { free(o); return NULL; }
        o->bpp = 3;
    }

    /* Keep source pixels for future resizes */
    o->src_pixels   = src;
    o->src_bpp_img  = o->bpp;
    o->src_w = sw; o->src_h = sh;

    resolve_dims(sw, sh, dst_w, dst_h, &o->dst_w, &o->dst_h);

    /* Allocate dst buffer and scale into it */
    o->pixels = calloc(1, (size_t)o->dst_w * o->dst_h * o->bpp);
    if (!o->pixels) { overlay_free(o); return NULL; }

    enum AVPixelFormat fmt = (o->bpp == 4) ? AV_PIX_FMT_BGRA : AV_PIX_FMT_BGR24;
    if (sw == o->dst_w && sh == o->dst_h) {
        memcpy(o->pixels, src, (size_t)sw * sh * o->bpp);
    } else {
        struct SwsContext *sws = sws_getContext(
            sw, sh, fmt,
            o->dst_w, o->dst_h, fmt,
            SWS_BILINEAR, NULL, NULL, NULL);
        const uint8_t *s[4] = { src }; int sl[4] = { sw * o->bpp };
        uint8_t       *d[4] = { o->pixels }; int dl[4] = { o->dst_w * o->bpp };
        sws_scale(sws, s, sl, 0, sh, d, dl);
        sws_freeContext(sws);
    }

    fprintf(stderr, "[overlay] image '%s' %dx%d → %dx%d at (%d,%d)\n",
            path, sw, sh, o->dst_w, o->dst_h, dst_x, dst_y);
    return o;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_create_video
 * ────────────────────────────────────────────────────────────── */
Overlay *overlay_create_video(const char *path,
                               int dst_x, int dst_y, int dst_w, int dst_h,
                               bool loop)
{
    Overlay *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->type      = OVERLAY_VIDEO;
    o->dst_x     = dst_x; o->dst_y = dst_y;
    o->bpp       = 3;
    o->pipe_fd   = -1;
    o->vid_loop  = loop;

    if (avformat_open_input(&o->fmt_ctx, path, NULL, NULL) < 0) {
        fprintf(stderr, "[overlay] Cannot open %s\n", path);
        free(o); return NULL;
    }
    avformat_find_stream_info(o->fmt_ctx, NULL);

    o->stream_idx = -1;
    for (unsigned i = 0; i < o->fmt_ctx->nb_streams; i++)
        if (o->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            { o->stream_idx = (int)i; break; }
    if (o->stream_idx < 0) { overlay_free(o); return NULL; }

    AVCodecParameters *cp = o->fmt_ctx->streams[o->stream_idx]->codecpar;
    const AVCodec *codec  = avcodec_find_decoder(cp->codec_id);
    o->codec_ctx          = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(o->codec_ctx, cp);
    if (avcodec_open2(o->codec_ctx, codec, NULL) < 0) { overlay_free(o); return NULL; }

    o->src_w = cp->width; o->src_h = cp->height;
    resolve_dims(o->src_w, o->src_h, dst_w, dst_h, &o->dst_w, &o->dst_h);

    fprintf(stderr, "[overlay] video '%s' %dx%d → %dx%d at (%d,%d) loop=%d\n",
            path, o->src_w, o->src_h, o->dst_w, o->dst_h, dst_x, dst_y, loop);

    o->av_frame = av_frame_alloc();
    o->av_pkt   = av_packet_alloc();
    /* DON'T build sws here — codec_ctx->pix_fmt may be NONE until the
     * first frame is decoded. We build it lazily in overlay_tick using
     * av_frame->format, which is always correct. */
    o->sws = NULL;

    o->pixels = calloc(1, (size_t)o->dst_w * o->dst_h * 3); /* black, not garbage */
    if (!o->pixels) { overlay_free(o); return NULL; }
    return o;
}

/* ──────────────────────────────────────────────────────────────
 * Screen reader thread
 *
 * Runs per-screen-overlay. Drains the FIFO/pipe as fast as the kernel
 * delivers data, keeping `latest_frame` always pointing at the most
 * recent fully-decoded frame. The main compositor thread just copies
 * from latest_frame under the mutex.
 *
 * Uses poll() with a 100ms timeout so it can check `reader_shutdown`
 * and exit promptly when the overlay is destroyed.
 * ────────────────────────────────────────────────────────────── */
static void *screen_reader_thread(void *arg)
{
    Overlay *o = arg;
    uint8_t *acc = malloc(o->raw_bytes);
    if (!acc) return NULL;
    size_t partial = 0;

    /*
     * NOTE: on Linux, a FIFO opened read-only reports POLLHUP from poll()
     * whenever no writer is currently connected — including BEFORE the
     * writer (wf-recorder) has had a chance to open its end. So we must
     * NOT treat POLLHUP as a signal to exit. The only authoritative EOF
     * is `read() == 0` (which only happens after a writer connected and
     * then closed). Until then we just keep polling.
     */
    while (!o->reader_shutdown) {
        struct pollfd pfd = { .fd = o->pipe_fd, .events = POLLIN };
        int pr = poll(&pfd, 1, 100);  /* 100 ms wakeup to check shutdown */
        if (pr <= 0) continue;        /* timeout or interrupt */
        if (!(pfd.revents & POLLIN)) continue; /* HUP w/o data — keep waiting */

        ssize_t n = read(o->pipe_fd, acc + partial, o->raw_bytes - partial);
        if (n > 0) {
            partial += (size_t)n;
            while (partial >= o->raw_bytes) {
                pthread_mutex_lock(&o->latest_lock);
                memcpy(o->latest_frame, acc, o->raw_bytes);
                o->has_frame = true;
                pthread_mutex_unlock(&o->latest_lock);

                size_t leftover = partial - o->raw_bytes;
                if (leftover > 0)
                    memmove(acc, acc + o->raw_bytes, leftover);
                partial = leftover;
            }
        } else if (n == 0) {
            /* Real EOF — a writer connected and then closed. We can exit. */
            break;
        } else if (errno != EINTR && errno != EAGAIN) {
            break;
        }
    }

    free(acc);
    return NULL;
}

/* Snap dimension DOWN to a multiple of `align`, but keep at least `align`. */
static int snap_down(int v, int align)
{
    int r = (v / align) * align;
    return r >= align ? r : align;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_create_screen
 *
 * geometry: "WxH+X+Y"
 *
 * IMPORTANT: width and height are snapped DOWN to multiples of 16 to
 * match the alignment that wf-recorder / wlr-screencopy actually
 * deliver. Mismatched stride causes the captured image to "scroll"
 * because each frame is offset from the previous by (real - requested).
 * ────────────────────────────────────────────────────────────── */
Overlay *overlay_create_screen(const char *geometry,
                                int dst_x, int dst_y, int dst_w, int dst_h)
{
    Overlay *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->type      = OVERLAY_SCREEN;
    o->dst_x     = dst_x; o->dst_y = dst_y;
    o->bpp       = 3;
    o->pipe_fd   = -1;
    o->fifo_path[0] = '\0';

    /* Parse "WxH+X+Y" */
    int gw = 0, gh = 0, gx = 0, gy = 0;
    if (sscanf(geometry, "%dx%d+%d+%d", &gw, &gh, &gx, &gy) != 4) {
        fprintf(stderr, "[screen] Bad geometry '%s' (expected WxH+X+Y)\n", geometry);
        free(o); return NULL;
    }

    /* Snap to 16-pixel boundaries to match compositor stride */
    int gw_snapped = snap_down(gw, 16);
    int gh_snapped = snap_down(gh, 2);
    if (gw_snapped != gw || gh_snapped != gh) {
        fprintf(stderr, "[screen] Snapping %dx%d → %dx%d for stride alignment\n",
                gw, gh, gw_snapped, gh_snapped);
    }
    gw = gw_snapped; gh = gh_snapped;

    o->src_w = gw; o->src_h = gh;
    resolve_dims(gw, gh, dst_w, dst_h, &o->dst_w, &o->dst_h);

    bool wayland = getenv("WAYLAND_DISPLAY") != NULL;

    if (wayland) {
        /* Unique FIFO path: pid + monotonic counter so two overlays never collide */
        static int fifo_counter = 0;
        int n = __atomic_add_fetch(&fifo_counter, 1, __ATOMIC_RELAXED);
        snprintf(o->fifo_path, sizeof o->fifo_path,
                 "/tmp/vcam_screen_%d_%d.fifo", (int)getpid(), n);
        unlink(o->fifo_path);
        if (mkfifo(o->fifo_path, 0600) < 0) {
            perror("[screen] mkfifo"); free(o); return NULL;
        }
        fprintf(stderr, "[screen] FIFO: %s\n", o->fifo_path);

        /* Open the FIFO read end O_NONBLOCK to avoid blocking before the
         * writer connects. We'll flip it back to blocking afterwards so
         * the reader thread can use poll() + blocking read efficiently. */
        o->pipe_fd = open(o->fifo_path, O_RDONLY | O_NONBLOCK);
        if (o->pipe_fd < 0) {
            perror("[screen] open fifo");
            unlink(o->fifo_path); free(o); return NULL;
        }

        char wfr_geo[64];
        snprintf(wfr_geo, sizeof wfr_geo, "%d,%d %dx%d", gx, gy, gw, gh);

        /* wf-recorder asks "Overwrite? Y/n:" when the output file already
         * exists (our FIFO does). With no terminal it hangs forever waiting
         * for input — that's the silent black-screen bug. Fix: feed "y\n"
         * via a stdin pipe so the prompt is auto-answered. */
        int answer_pipe[2];
        if (pipe(answer_pipe) < 0) {
            perror("[screen] stdin pipe");
            close(o->pipe_fd); unlink(o->fifo_path); free(o); return NULL;
        }
        /* Pre-load several "y\n"s — harmless if not consumed */
        const char *yes = "y\ny\ny\n";
        ssize_t wn = write(answer_pipe[1], yes, strlen(yes));
        (void)wn;

        pid_t pid = fork();
        if (pid < 0) {
            perror("[screen] fork");
            close(answer_pipe[0]); close(answer_pipe[1]);
            close(o->pipe_fd); unlink(o->fifo_path); free(o); return NULL;
        }
        if (pid == 0) {
            /* Replace stdin with the answer pipe so wf-recorder reads "y\n" */
            dup2(answer_pipe[0], STDIN_FILENO);
            close(answer_pipe[0]);
            close(answer_pipe[1]);
            execlp("wf-recorder", "wf-recorder",
                   "-g", wfr_geo,
                   "-c", "rawvideo",
                   "-x", "bgr0",
                   "--muxer", "rawvideo",
                   "-f", o->fifo_path,
                   (char *)NULL);
            perror("wf-recorder");
            _exit(1);
        }
        /* parent: close both pipe fds — child already has the read end via dup2 */
        close(answer_pipe[0]);
        close(answer_pipe[1]);

        o->child_pid = pid;
        o->src_bpp   = 4;
        o->raw_bytes = (size_t)gw * gh * 4;
        rebuild_sws(o, gw, gh, AV_PIX_FMT_BGR0);

        fprintf(stderr, "[screen] Wayland: wf-recorder -g '%s' (target overlay %dx%d)\n",
                wfr_geo, o->dst_w, o->dst_h);

        /* Brief sanity check: did the child immediately die? */
        struct timespec ts200 = { .tv_nsec = 200000000L };
        nanosleep(&ts200, NULL);
        int status; pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            fprintf(stderr, "[screen] ERROR: wf-recorder exited immediately "
                            "(status=%d). Is it installed and does your "
                            "compositor support wlr-screencopy?\n",
                    WIFEXITED(status) ? WEXITSTATUS(status) : -1);
            close(o->pipe_fd); unlink(o->fifo_path); free(o); return NULL;
        }

    } else {
        int pipefd[2];
        if (pipe(pipefd) < 0) { perror("[screen] pipe"); free(o); return NULL; }

        pid_t pid = fork();
        if (pid < 0) {
            perror("[screen] fork");
            close(pipefd[0]); close(pipefd[1]); free(o); return NULL;
        }
        if (pid == 0) {
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[0]); close(pipefd[1]);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }

            char input[64], size[32];
            snprintf(input, sizeof input, ":0.0+%d,%d", gx, gy);
            snprintf(size,  sizeof size,  "%dx%d", gw, gh);
            execlp("ffmpeg", "ffmpeg",
                   "-loglevel", "quiet",
                   "-f", "x11grab",
                   "-video_size", size,
                   "-i", input,
                   "-f", "rawvideo",
                   "-pix_fmt", "bgr24",
                   "pipe:1",
                   (char *)NULL);
            perror("[screen] ffmpeg x11grab");
            _exit(1);
        }
        close(pipefd[1]);
        o->child_pid = pid;
        o->pipe_fd   = pipefd[0];
        o->src_bpp   = 3;
        o->raw_bytes = (size_t)gw * gh * 3;
        rebuild_sws(o, gw, gh, AV_PIX_FMT_BGR24);

        fprintf(stderr, "[screen] X11: ffmpeg x11grab → pipe\n");
    }

    /* Flip the fd back to BLOCKING — poll() in the reader thread is
     * cleaner than spinning on EAGAIN. */
    int flags = fcntl(o->pipe_fd, F_GETFL);
    fcntl(o->pipe_fd, F_SETFL, flags & ~O_NONBLOCK);

    /* Allocate the latest_frame buffer + pixels buffer (calloc so initial
     * frames before the reader produces anything are black, not garbage). */
    o->latest_frame = calloc(1, o->raw_bytes);
    o->pixels       = calloc(1, (size_t)o->dst_w * o->dst_h * 3);
    if (!o->latest_frame || !o->pixels) { overlay_free(o); return NULL; }

    pthread_mutex_init(&o->latest_lock, NULL);
    o->reader_shutdown = false;
    if (pthread_create(&o->reader_thread, NULL, screen_reader_thread, o) != 0) {
        perror("[screen] pthread_create");
        overlay_free(o); return NULL;
    }
    o->reader_running = true;

    fprintf(stderr, "[screen] Capturing %dx%d at (%d,%d) → overlay %dx%d at (%d,%d)\n",
            gw, gh, gx, gy, o->dst_w, o->dst_h, dst_x, dst_y);
    return o;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_create_webcam
 *
 * A webcam overlay is just a movable/resizable window onto the main
 * pipeline's captured webcam frame. It does NOT open a V4L2 device of
 * its own — V4L2 only allows one streamer per camera, so we share the
 * main capture and let any number of webcam overlays exist
 * simultaneously (each at its own position/size).
 *
 * Source dimensions are unknown at creation time (we'd need to query
 * the main loop). For aspect-ratio computation we assume 16:9 by
 * default — the first tick with a real source frame corrects it.
 * The user can also pass explicit <w> <h> at creation.
 * ────────────────────────────────────────────────────────────── */
Overlay *overlay_create_webcam(const char *device,
                                int dst_x, int dst_y, int dst_w, int dst_h)
{
    (void)device; /* legacy arg — we no longer open per-overlay devices */

    Overlay *o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->type    = OVERLAY_WEBCAM;
    o->dst_x   = dst_x; o->dst_y = dst_y;
    o->bpp     = 3;
    o->pipe_fd = -1;

    /* Default assumption: 16:9. Corrected on first tick. */
    o->src_w = 640; o->src_h = 360;
    o->webcam_src_w = 0; o->webcam_src_h = 0; /* "not yet seen" */

    resolve_dims(o->src_w, o->src_h, dst_w, dst_h, &o->dst_w, &o->dst_h);

    o->pixels = calloc(1, (size_t)o->dst_w * o->dst_h * 3);
    if (!o->pixels) { overlay_free(o); return NULL; }

    /* sws is built lazily on first tick (when we know the real src dims) */
    o->sws = NULL;

    fprintf(stderr, "[overlay] webcam → %dx%d at (%d,%d) "
                    "(shares main pipeline's capture)\n",
            o->dst_w, o->dst_h, dst_x, dst_y);
    return o;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_tick
 * ────────────────────────────────────────────────────────────── */
void overlay_tick(Overlay *o, const Frame *webcam_src)
{
    if (!o) return;

    if (o->type == OVERLAY_SCREEN) {
        /* Reader thread fills latest_frame asynchronously. We just
         * snapshot it under the lock and convert to dst pixels. */
        pthread_mutex_lock(&o->latest_lock);
        if (!o->has_frame) {
            pthread_mutex_unlock(&o->latest_lock);
            return;
        }
        /* Copy out to a local-scope buffer pointer (the underlying memory
         * stays stable for the rest of the overlay's lifetime). */
        uint8_t *src = o->latest_frame;
        int src_stride = o->src_w * o->src_bpp;

        if (o->src_w == o->dst_w && o->src_h == o->dst_h && o->src_bpp == 3) {
            memcpy(o->pixels, src, o->raw_bytes);
        } else {
            const uint8_t *s[4] = { src };       int sl[4] = { src_stride };
            uint8_t       *d[4] = { o->pixels }; int dl[4] = { o->dst_w * 3 };
            sws_scale(o->sws, s, sl, 0, o->src_h, d, dl);
        }
        pthread_mutex_unlock(&o->latest_lock);
        return;
    }

    if (o->type == OVERLAY_WEBCAM) {
        /* Sample from the shared webcam source the main loop provides.
         * If no source (i.e. --no-input mode without a captured cam),
         * leave the previous pixels alone — they'll be calloc'd black
         * the first time round. */
        if (!webcam_src || !webcam_src->data) return;

        int sw = webcam_src->width, sh = webcam_src->height;

        /* (Re)build sws if this is the first tick or src dims changed.
         * Also handle the case where the user passed only -w at create
         * time and we estimated h from 16:9 — once we know the actual
         * src dims, recompute aspect and resize the buffer. */
        if (sw != o->webcam_src_w || sh != o->webcam_src_h) {
            /* First time we see real dims: if the user gave us width only
             * (or width = src_w by default), recompute height to match. */
            if (o->webcam_src_w == 0) {
                /* Was our placeholder src 16:9 estimate close enough?
                 * If the dst aspect matches our placeholder src aspect,
                 * the user gave just width — recompute dst from real src. */
                int new_dst_w = o->dst_w;
                int new_dst_h = (int)((long)new_dst_w * sh / sw);
                if (new_dst_h != o->dst_h) {
                    free(o->pixels);
                    o->pixels = calloc(1, (size_t)new_dst_w * new_dst_h * 3);
                    o->dst_h  = new_dst_h;
                }
            }
            o->src_w = sw; o->src_h = sh;
            o->webcam_src_w = sw; o->webcam_src_h = sh;

            if (o->sws) sws_freeContext(o->sws);
            o->sws = sws_getContext(sw, sh, AV_PIX_FMT_BGR24,
                                     o->dst_w, o->dst_h, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, NULL, NULL, NULL);
        }

        /* Fast path: src == dst */
        if (sw == o->dst_w && sh == o->dst_h) {
            memcpy(o->pixels, webcam_src->data,
                   (size_t)o->dst_w * o->dst_h * 3);
        } else if (o->sws) {
            const uint8_t *s[4] = { webcam_src->data };
            int sl[4] = { webcam_src->stride };
            uint8_t *d[4] = { o->pixels };
            int dl[4] = { o->dst_w * 3 };
            sws_scale(o->sws, s, sl, 0, sh, d, dl);
        }
        return;
    }

    if (o->type != OVERLAY_VIDEO) return;

    /* ── video tick ──
     * One overlay_tick = at most one decoded frame displayed.
     * State machine:
     *   receive_frame → 0       : display, return
     *   receive_frame → EAGAIN  : feed more packets, retry
     *   receive_frame → EOF     : either loop (seek+flush) or freeze
     *
     * vid_eof_sent prevents sending the NULL flush sentinel twice.
     * vid_finished marks play-once playback as completed (we keep the
     * last frame visible without further decoder activity).
     */
    if (o->vid_finished) return; /* play-once: keep last frame, don't touch decoder */

    int safety = 512; /* generous — covers long GOPs and large packet runs */
    while (safety-- > 0) {
        int ret = avcodec_receive_frame(o->codec_ctx, o->av_frame);

        if (ret == 0) {
            /* Got a frame — build/refresh sws based on the actual format
             * (codec_ctx->pix_fmt may be NONE until the first frame). */
            int src_fmt = o->av_frame->format;
            int src_w   = o->av_frame->width;
            int src_h   = o->av_frame->height;
            if (!o->sws || src_fmt != o->codec_ctx->pix_fmt) {
                if (o->sws) sws_freeContext(o->sws);
                o->sws = sws_getContext(src_w, src_h, src_fmt,
                                         o->dst_w, o->dst_h, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, NULL, NULL, NULL);
                o->codec_ctx->pix_fmt = src_fmt;
                o->src_w = src_w; o->src_h = src_h;
            }
            if (o->sws) {
                uint8_t *d[4] = { o->pixels }; int dl[4] = { o->dst_w*3 };
                sws_scale(o->sws, (const uint8_t *const *)o->av_frame->data,
                          o->av_frame->linesize, 0, src_h, d, dl);
            }
            return;
        }

        if (ret == AVERROR_EOF) {
            /* Decoder fully drained. */
            if (!o->vid_loop) {
                o->vid_finished = true;
                return;
            }
            /* Loop: rewind demuxer to start, reset decoder & EOF state */
            av_seek_frame(o->fmt_ctx, o->stream_idx, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(o->codec_ctx);
            o->vid_eof_sent = false;
            continue;
        }

        if (ret != AVERROR(EAGAIN)) return; /* hard error */

        /* Decoder is hungry — feed one packet from the video stream */
        bool sent = false;
        while (av_read_frame(o->fmt_ctx, o->av_pkt) >= 0) {
            if (o->av_pkt->stream_index == o->stream_idx) {
                avcodec_send_packet(o->codec_ctx, o->av_pkt);
                av_packet_unref(o->av_pkt);
                sent = true;
                break;
            }
            av_packet_unref(o->av_pkt);
        }

        if (!sent && !o->vid_eof_sent) {
            /* True end of input stream — send flush sentinel ONCE.
             * Don't send it again or send_packet returns an error. */
            avcodec_send_packet(o->codec_ctx, NULL);
            o->vid_eof_sent = true;
        }
        /* loop and try receive again */
    }
}

/* ──────────────────────────────────────────────────────────────
 * overlay_blit
 * ────────────────────────────────────────────────────────────── */
void overlay_blit(Overlay *o, Frame *dst)
{
    if (!o || !o->pixels) return;

    int bpp = o->bpp;
    int ox = o->dst_x, oy = o->dst_y, ow = o->dst_w, oh = o->dst_h;

    for (int row = 0; row < oh; row++) {
        int dy = oy + row;
        if (dy < 0 || dy >= dst->height) continue;
        uint8_t       *dst_row = dst->data + dy * dst->stride + ox * 3;
        const uint8_t *src_row = o->pixels + row * ow * bpp;

        for (int col = 0; col < ow; col++) {
            int dx = ox + col;
            if (dx < 0 || dx >= dst->width) continue;
            const uint8_t *s = src_row + col * bpp;
            uint8_t       *d = dst_row + col * 3;

            if (bpp == 4) {
                uint8_t a = s[3];
                if (a == 0) continue;
                if (a == 255) { d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; }
                else {
                    d[0] = (uint8_t)((s[0]*a + d[0]*(255-a)) / 255);
                    d[1] = (uint8_t)((s[1]*a + d[1]*(255-a)) / 255);
                    d[2] = (uint8_t)((s[2]*a + d[2]*(255-a)) / 255);
                }
            } else {
                d[0]=s[0]; d[1]=s[1]; d[2]=s[2];
            }
        }
    }
}

/* ──────────────────────────────────────────────────────────────
 * overlay_move / overlay_resize
 * ────────────────────────────────────────────────────────────── */
void overlay_move(Overlay *o, int x, int y)
{
    if (!o) return;
    o->dst_x = x; o->dst_y = y;
}

void overlay_resize(Overlay *o, int w, int h)
{
    if (!o || w <= 0 || h <= 0) return;
    if (w == o->dst_w && h == o->dst_h) return;  /* no-op */

    /* Allocate a NEW pixel buffer (zeroed) and free the old one. We
     * never resize in place because intermediate writes could be read
     * by overlay_blit on the main thread mid-resize. The registry lock
     * held by the IPC caller prevents that, but a fresh buffer is also
     * simpler reasoning. */
    uint8_t *newpx = calloc(1, (size_t)w * h * o->bpp);
    if (!newpx) return;
    free(o->pixels);
    o->pixels = newpx;
    o->dst_w  = w;
    o->dst_h  = h;

    switch (o->type) {
    case OVERLAY_IMAGE: {
        /* Rescale from the stored source pixels into the new buffer */
        if (!o->src_pixels || o->src_w <= 0 || o->src_h <= 0) break;
        enum AVPixelFormat fmt = (o->bpp == 4) ? AV_PIX_FMT_BGRA : AV_PIX_FMT_BGR24;
        if (o->src_w == w && o->src_h == h) {
            memcpy(o->pixels, o->src_pixels, (size_t)w * h * o->bpp);
        } else {
            struct SwsContext *sws = sws_getContext(
                o->src_w, o->src_h, fmt,
                w, h, fmt,
                SWS_BILINEAR, NULL, NULL, NULL);
            const uint8_t *s[4] = { o->src_pixels }; int sl[4] = { o->src_w * o->bpp };
            uint8_t       *d[4] = { o->pixels };     int dl[4] = { w * o->bpp };
            sws_scale(sws, s, sl, 0, o->src_h, d, dl);
            sws_freeContext(sws);
        }
        break;
    }

    case OVERLAY_VIDEO:
        /* sws is rebuilt lazily in overlay_tick — but it caches the LAST
         * dst dims via the codec_ctx->pix_fmt sentinel trick. Free it
         * now so the next tick definitely rebuilds with the new dims. */
        if (o->sws) { sws_freeContext(o->sws); o->sws = NULL; }
        /* Reset the cached pix_fmt marker so the lazy rebuild fires */
        if (o->codec_ctx) o->codec_ctx->pix_fmt = AV_PIX_FMT_NONE;
        break;

    case OVERLAY_SCREEN: {
        enum AVPixelFormat src_fmt = (o->src_bpp == 4)
            ? AV_PIX_FMT_BGR0 : AV_PIX_FMT_BGR24;
        rebuild_sws(o, o->src_w, o->src_h, src_fmt);
        break;
    }

    case OVERLAY_WEBCAM:
        if (o->sws) { sws_freeContext(o->sws); o->sws = NULL; }
        if (o->src_w != w || o->src_h != h) {
            o->sws = sws_getContext(o->src_w, o->src_h, AV_PIX_FMT_BGR24,
                                     w, h, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, NULL, NULL, NULL);
        }
        break;
    }
}

/* ──────────────────────────────────────────────────────────────
 * Accessors (for IPC list command)
 * ────────────────────────────────────────────────────────────── */
OverlayType overlay_get_type(const Overlay *o)
{
    return o ? o->type : OVERLAY_IMAGE;
}

void overlay_get_rect(const Overlay *o, int *x, int *y, int *w, int *h)
{
    if (!o) { *x=*y=*w=*h=0; return; }
    *x = o->dst_x; *y = o->dst_y; *w = o->dst_w; *h = o->dst_h;
}

void overlay_get_src(const Overlay *o, int *w, int *h)
{
    if (!o) { *w = *h = 0; return; }
    *w = o->src_w; *h = o->src_h;
}

/* ──────────────────────────────────────────────────────────────
 * overlay_free
 * ────────────────────────────────────────────────────────────── */
void overlay_free(Overlay *o)
{
    if (!o) return;
    free(o->pixels);
    free(o->src_pixels);
    if (o->type == OVERLAY_SCREEN) {
        /* Signal the reader thread to exit on its next poll() wakeup */
        o->reader_shutdown = true;

        /* Stop the capture child — this also causes EOF on the pipe */
        if (o->child_pid > 0) {
            kill(o->child_pid, SIGTERM);
            struct timespec ts = { .tv_nsec = 500000000L };
            nanosleep(&ts, NULL);
            kill(o->child_pid, SIGKILL);
            waitpid(o->child_pid, NULL, 0);
        }

        /* Reader sees shutdown flag OR EOF on pipe — joins quickly */
        if (o->reader_running) pthread_join(o->reader_thread, NULL);
        pthread_mutex_destroy(&o->latest_lock);

        if (o->pipe_fd >= 0) close(o->pipe_fd);
        if (o->fifo_path[0]) unlink(o->fifo_path);
        free(o->latest_frame);
        if (o->sws) sws_freeContext(o->sws);
    } else if (o->type == OVERLAY_VIDEO) {
        sws_freeContext(o->sws);
        av_frame_free(&o->av_frame);
        av_packet_free(&o->av_pkt);
        avcodec_free_context(&o->codec_ctx);
        avformat_close_input(&o->fmt_ctx);
    } else if (o->type == OVERLAY_WEBCAM) {
        if (o->sws) sws_freeContext(o->sws);
    }
    free(o);
}
