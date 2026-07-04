/*
 * src/record.c — vcam's recording subsystem.
 *
 * Spawns ffmpeg as a child process and feeds it BGR24 frames via a
 * pipe. ffmpeg encodes (x264 + AAC) to the output file the user
 * specified. Avoids /dev/video0 contention entirely.
 *
 * Threading model:
 *   - The main loop calls recorder_write_frame() every iteration.
 *   - The IPC thread calls recorder_start() and recorder_stop()
 *     in response to "record" commands.
 *   - SIGCHLD handler reaps the ffmpeg child after it exits.
 *
 * Synchronisation: an internal mutex protects state transitions
 * (start/stop). The mutex is held only briefly — never during
 * write() or waitpid() — so it doesn't block the main loop or
 * delay IPC responses.
 *
 * Stop is non-blocking: we close ffmpeg's stdin (it sees EOF and
 * cleanly finalises the file) but do NOT waitpid. SIGCHLD reaps
 * the child in the background. This is critical: ffmpeg can take
 * several seconds to write the MP4 trailer, and blocking the main
 * loop on that would freeze the webcam stream for the duration.
 */

#define _GNU_SOURCE  /* for F_SETPIPE_SZ */
#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <strings.h>      /* strcasecmp */
#include <stdint.h>       /* intptr_t */
#include <pthread.h>
#include <sys/stat.h>     /* mkfifo */

/* Per-recording context owned by the paced writer thread. The writer
 * duplicates/skips frames to feed ffmpeg at EXACTLY the target fps,
 * which makes plain -framerate timestamps true and reduces A/V sync
 * to a small constant (camera latency). */
typedef struct RecCtx {
    struct Recorder *r;
    uint64_t gen;          /* recording generation this ctx belongs to */
    int      fd;           /* video pipe write end (blocking) — owned  */
    int      fps;
    size_t   size;         /* one frame in bytes */
    uint8_t *latest;       /* most recent submitted frame (r->lock)    */
    uint8_t *scratch;      /* writer's private copy for the write()    */
    int      have_frame;   /* has anything been submitted yet?         */
    int      missed;       /* ticks where ffmpeg couldn't keep up      */
} RecCtx;

struct Recorder {
    pthread_mutex_t lock;
    uint64_t gen;       /* bumped every start; stale threads detect it */
    pid_t pid;          /* ffmpeg */
    pid_t arec_pid;     /* arecord audio-capture child, or -1 */
    RecCtx *ctx;        /* live writer context, or NULL */
    int   fifo_fd;      /* our O_RDWR handle on the audio FIFO, or -1 */
    char  fifo_path[64];
    bool  active;
    char  file[512];
    int   frame_w, frame_h;
};

Recorder *recorder_new(void)
{
    Recorder *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    pthread_mutex_init(&r->lock, NULL);
    r->pid = -1;
    r->arec_pid = -1;
    r->ctx = NULL;
    r->fifo_fd = -1;
    return r;
}

void recorder_free(Recorder *r)
{
    if (!r) return;
    recorder_stop(r);
    pthread_mutex_destroy(&r->lock);
    free(r);
}

bool recorder_active(const Recorder *r) { return r && r->active; }
pid_t recorder_pid(const Recorder *r)   { return r ? r->pid : -1; }
const char *recorder_file(const Recorder *r) { return r ? r->file : ""; }

/* Build the argv array for ffmpeg.
 * audio_fifo: path to a WAV FIFO being fed by arecord (preferred for
 *             ALSA mics — see recorder_start); NULL if not used.
 * mic:        original mic string; used directly only for "pulse..."
 *             sources or as a direct-ALSA fallback when the FIFO
 *             couldn't be set up. */
static char **build_argv(const char *file, const char *mic,
                         const char *audio_fifo, float aoffset,
                         int w, int h, int fps,
                         int crf, const char *vcodec, const char *vpreset)
{
    /* Slot budget: base(5) + video input(14) + audio input(≤10)
     * + encode(8) + ts flags(4) + cfr(4) + audio encode(8)
     * + movflags(2) + interleave(2) + file+NULL(2) = 59 used today.
     * 96 leaves generous headroom — a silent overflow here previously
     * CORRUPTED THE HEAP and randomly produced an empty output
     * filename ("Unable to choose an output format for ''"). */
    char **argv = calloc(96, sizeof *argv);
    if (!argv) return NULL;

    int i = 0;
    char buf[64];

    argv[i++] = strdup("ffmpeg");
    argv[i++] = strdup("-y");
    argv[i++] = strdup("-hide_banner");
    argv[i++] = strdup("-loglevel"); argv[i++] = strdup("warning");

    /* ── video input: raw BGR24 from our pipe ──
     * SHARED-CLOCK SYNC DESIGN: both this input and the audio input
     * are stamped with -use_wallclock_as_timestamps — the REAL moment
     * each packet arrives, from one common clock. Combined with
     * -copyts -start_at_zero on the output (see below), ffmpeg
     * preserves the true relative timing between audio and video no
     * matter when each stream started or how either momentarily
     * stalls. This is what "record the sound that plays during this
     * exact frame" means mechanically.
     *
     * The paced writer thread delivers frames at real 30fps, so the
     * wallclock stamps are smooth; -fps_mode cfr regularizes any
     * residual jitter by dup/drop WITHOUT moving content in time. */
    argv[i++] = strdup("-thread_queue_size"); argv[i++] = strdup("1024");
    argv[i++] = strdup("-use_wallclock_as_timestamps"); argv[i++] = strdup("1");
    argv[i++] = strdup("-f");          argv[i++] = strdup("rawvideo");
    argv[i++] = strdup("-pixel_format"); argv[i++] = strdup("bgr24");
    snprintf(buf, sizeof buf, "%dx%d", w, h);
    argv[i++] = strdup("-video_size"); argv[i++] = strdup(buf);
    snprintf(buf, sizeof buf, "%d", fps);
    argv[i++] = strdup("-framerate");  argv[i++] = strdup(buf);
    argv[i++] = strdup("-i");          argv[i++] = strdup("pipe:0");

    /* ── audio input ──
     * Three paths, best first:
     *
     * 1. WAV FIFO fed by arecord (audio_fifo set). ffmpeg reads a
     *    plain pipe — fully interruptible, nothing can wedge in the
     *    kernel. arecord's large-period blocking capture is exactly
     *    what flaky drivers (AMD acp DMIC!) tolerate, while ffmpeg's
     *    own small-period ALSA reads can hang the driver in
     *    uninterruptible sleep — observed as: SIGINT/SIGTERM ignored,
     *    SIGKILL required, torn MP4 fragments.
     *    WAV timestamps come from the sample count — a monotonic
     *    audio clock. No wallclock stamping here (pipe jitter would
     *    make wallclock stamps WORSE than the sample clock).
     *
     * 2. PulseAudio/PipeWire ("pulse" / "pulse:<source>") for systems
     *    that have it.
     *
     * 3. Direct ALSA — only as a fallback if the FIFO setup failed.
     */
    /* Audio inputs also get wallclock stamps — SAME clock as video.
     * (For the WAV FIFO this overrides the sample-count timestamps
     * with arrival times; the aresample async filter then absorbs
     * the small jitter between the ALSA crystal and the wall clock.)
     * -itsoffset shifts audio slightly later to compensate for the
     * video path's capture latency (webcam driver + decode) which
     * happens BEFORE our wallclock stamp. */
    char offbuf[32];
    snprintf(offbuf, sizeof offbuf, "%.3f", aoffset);

    if (audio_fifo && *audio_fifo) {
        argv[i++] = strdup("-thread_queue_size"); argv[i++] = strdup("1024");
        argv[i++] = strdup("-use_wallclock_as_timestamps"); argv[i++] = strdup("1");
        if (aoffset != 0.0f) {
            argv[i++] = strdup("-itsoffset"); argv[i++] = strdup(offbuf);
        }
        argv[i++] = strdup("-f");  argv[i++] = strdup("wav");
        argv[i++] = strdup("-i");  argv[i++] = strdup(audio_fifo);
    } else if (mic && strncmp(mic, "pulse", 5) == 0) {
        argv[i++] = strdup("-thread_queue_size"); argv[i++] = strdup("1024");
        argv[i++] = strdup("-use_wallclock_as_timestamps"); argv[i++] = strdup("1");
        if (aoffset != 0.0f) {
            argv[i++] = strdup("-itsoffset"); argv[i++] = strdup(offbuf);
        }
        argv[i++] = strdup("-f");  argv[i++] = strdup("pulse");
        argv[i++] = strdup("-i");
        argv[i++] = strdup(mic[5] == ':' && mic[6] ? mic + 6 : "default");
    } else if (mic && *mic) {
        argv[i++] = strdup("-thread_queue_size"); argv[i++] = strdup("1024");
        argv[i++] = strdup("-use_wallclock_as_timestamps"); argv[i++] = strdup("1");
        if (aoffset != 0.0f) {
            argv[i++] = strdup("-itsoffset"); argv[i++] = strdup(offbuf);
        }
        argv[i++] = strdup("-f");  argv[i++] = strdup("alsa");
        argv[i++] = strdup("-i");  argv[i++] = strdup(mic);
    }

    bool have_audio = (audio_fifo && *audio_fifo) || (mic && *mic);

    /* ── encoding ── */
    argv[i++] = strdup("-threads");  argv[i++] = strdup("0");  /* all cores */
    argv[i++] = strdup("-c:v");      argv[i++] = strdup(vcodec ? vcodec : "libx264");
    argv[i++] = strdup("-preset");   argv[i++] = strdup(vpreset ? vpreset : "ultrafast");
    snprintf(buf, sizeof buf, "%d", crf > 0 ? crf : 23);
    argv[i++] = strdup("-crf");      argv[i++] = strdup(buf);
    argv[i++] = strdup("-pix_fmt");  argv[i++] = strdup("yuv420p");

    /* ── PRESERVE cross-input alignment ──
     * Without -copyts, ffmpeg re-zeroes each input INDEPENDENTLY
     * (subtracting each input's own first timestamp) — which throws
     * away exactly the relative offset we carefully captured with the
     * shared wallclock stamps. -copyts keeps timestamps as-is;
     * -start_at_zero then shifts ALL streams together by one common
     * amount so the file starts at 0. -avoid_negative_ts make_zero is
     * a mux-level belt-and-braces that also shifts uniformly. */
    argv[i++] = strdup("-copyts");
    argv[i++] = strdup("-start_at_zero");
    argv[i++] = strdup("-avoid_negative_ts"); argv[i++] = strdup("make_zero");

    /* ── enforce CFR at the requested fps on the OUTPUT ──
     * Converts the wallclock arrival stamps into a clean constant
     * frame rate by duplicating/dropping frames — timing stays true
     * even if the writer or encoder momentarily stalls (a stall
     * becomes a briefly frozen picture, never a desync). */
    snprintf(buf, sizeof buf, "%d", fps);
    argv[i++] = strdup("-r");        argv[i++] = strdup(buf);
    argv[i++] = strdup("-fps_mode"); argv[i++] = strdup("cfr");

    if (have_audio) {
        /* Audio timestamp strategy — third iteration, and the one
         * that handles BOTH failure modes:
         *
         *   - FIFO burst jitter (±0.1-0.2s packet wobble): must NOT
         *     be chased with hard sample trims — that chops audible
         *     chunks out (aresample's default hard threshold is 0.1s,
         *     SMALLER than the jitter → constant trimming).
         *   - Real capture gaps (ALSA overrun when ffmpeg stalls and
         *     the FIFO backs up): MUST be corrected, or the timeline
         *     glues shut across the hole and everything afterwards
         *     plays early — a permanent A/V offset after a crackle.
         *
         * aresample=async=1000:min_hard_comp=0.5 does exactly this:
         * deviations under 0.5s get only gentle soft resampling
         * (≤ ~2% rate, inaudible), while a genuine ≥0.5s gap is
         * silence-padded ONCE, restoring sync. */
        argv[i++] = strdup("-af");
        argv[i++] = strdup("aresample=async=1000:min_hard_comp=0.5");
        argv[i++] = strdup("-c:a");  argv[i++] = strdup("aac");
        argv[i++] = strdup("-b:a");  argv[i++] = strdup("160k");
        argv[i++] = strdup("-ac");   argv[i++] = strdup("2");
    }

    /* ── container-specific flags ──
     * MP4 needs fragmented mode for live recording; otherwise the
     * moov atom is buffered in memory and only written on clean
     * close, and an interrupted recording is unplayable. */
    const char *ext = strrchr(file, '.');
    if (ext && strcasecmp(ext, ".mp4") == 0) {
        argv[i++] = strdup("-movflags");
        argv[i++] = strdup("+frag_keyframe+empty_moov+default_base_moof");
    }

    /* ── never let audio hold video hostage ──
     * The muxer interleaves streams by timestamp and BUFFERS video
     * while waiting for matching audio packets. Default
     * max_interleave_delta is 10 SECONDS — so if the mic stalls
     * (AMD acp DMICs are notorious for wedging in the kernel driver;
     * observed as "ALSA buffer xrun" then silence), every video frame
     * after the stall sits in RAM and never reaches the disk. A
     * 7-second recording with audio that died at 1s plays back as a
     * 1-second file. With 1s max delta, video flushes to disk at most
     * 1 second behind real time no matter what audio does. */
    argv[i++] = strdup("-max_interleave_delta");
    argv[i++] = strdup("1000000");   /* 1s in microseconds */

    argv[i++] = strdup(file);
    argv[i]   = NULL;
    return argv;
}

/* ── paced writer thread ──
 * Feeds ffmpeg's stdin one frame every 1/fps on a monotonic clock.
 * If the camera runs slower than fps, the latest frame is duplicated;
 * if faster, extra frames are skipped (only the newest survives).
 * Result: the rawvideo stream is EXACTLY constant-rate, which makes
 * -framerate timestamps true, playback duration equal to wall time,
 * and A/V sync independent of camera behaviour.
 *
 * The thread OWNS ctx and the pipe fd: it closes the fd and frees the
 * buffers when it exits. recorder_stop never touches them — it only
 * flips r->active, which the writer notices within one tick. This
 * removes every close-vs-write race by construction. */
static void *writer_thread(void *arg)
{
    RecCtx *c = arg;
    Recorder *r = c->r;

    /* Signals are the main thread's job. */
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGCHLD);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, NULL);

    const long period_ns = 1000000000L / (c->fps > 0 ? c->fps : 30);
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    for (;;) {
        /* Sleep until the next tick (EINTR-safe, absolute deadline
         * so error never accumulates). */
        next.tv_nsec += period_ns;
        while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec  += 1;
        }
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME,
                               &next, NULL) == EINTR)
            ;

        /* Still the live recording? Grab the newest frame. */
        pthread_mutex_lock(&r->lock);
        bool live = r->active && r->ctx == c && r->gen == c->gen;
        int have = c->have_frame;
        if (live && have)
            memcpy(c->scratch, c->latest, c->size);
        pthread_mutex_unlock(&r->lock);

        if (!live) break;
        if (!have) continue;   /* nothing submitted yet — wait a tick */

        /* Blocking write of the whole frame. If ffmpeg briefly
         * stalls we block here — that's fine, it's our own thread —
         * and any ticks we miss just mean the newest frame gets
         * duplicated less / content skips slightly. */
        size_t written = 0;
        bool dead = false;
        while (written < c->size) {
            ssize_t n = write(c->fd, c->scratch + written, c->size - written);
            if (n > 0) { written += (size_t)n; continue; }
            if (n < 0 && errno == EINTR) continue;
            dead = true;   /* EPIPE etc. — ffmpeg is gone */
            break;
        }
        if (dead) {
            /* ffmpeg exited unexpectedly. Mark the recording dead and
             * clean up the audio side so arecord doesn't block forever
             * writing into a FIFO nobody drains. */
            pthread_mutex_lock(&r->lock);
            if (r->active && r->ctx == c && r->gen == c->gen) {
                r->active = false;
                fprintf(stderr, "[record] ffmpeg exited mid-recording "
                                "(file=%s)\n", r->file);
                if (r->arec_pid > 0) { kill(r->arec_pid, SIGTERM); r->arec_pid = -1; }
                if (r->fifo_fd >= 0) { close(r->fifo_fd); r->fifo_fd = -1; }
                if (r->fifo_path[0]) { unlink(r->fifo_path); r->fifo_path[0] = '\0'; }
            }
            pthread_mutex_unlock(&r->lock);
            break;
        }

        /* If the deadline slipped past a whole period (long stall),
         * rebase rather than bursting to catch up. */
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long behind = (now.tv_sec - next.tv_sec) * 1000000000L
                    + (now.tv_nsec - next.tv_nsec);
        if (behind > period_ns) {
            c->missed += (int)(behind / period_ns);
            next = now;
        }
    }

    /* Thread owns the teardown of its resources. Detach ctx from the
     * recorder first (under lock) so no submitter can touch it after
     * we free. */
    pthread_mutex_lock(&r->lock);
    if (r->ctx == c) r->ctx = NULL;
    pthread_mutex_unlock(&r->lock);

    close(c->fd);          /* EOF → ffmpeg's video input ends */
    free(c->latest);
    free(c->scratch);
    if (c->missed > 0)
        fprintf(stderr, "[record] writer: %d duplicated/missed tick(s)\n",
                c->missed);
    free(c);
    return NULL;
}

bool recorder_start(Recorder *r,
                    const char *file, const char *mic,
                    int w, int h, int fps,
                    int crf, const char *vcodec, const char *vpreset,
                    float aoffset)
{
    if (!r || !file) return false;

    pthread_mutex_lock(&r->lock);
    if (r->active) {
        pthread_mutex_unlock(&r->lock);
        fprintf(stderr, "[record] already active — stop first\n");
        return false;
    }
    pthread_mutex_unlock(&r->lock);

    /* ── audio setup: arecord → WAV FIFO for ALSA mics ──
     * Keeps ALSA out of ffmpeg entirely. arecord's large-period
     * blocking capture works reliably on hardware where ffmpeg's own
     * ALSA reads wedge the kernel driver (AMD acp DMICs). ffmpeg then
     * only reads pipes, so SIGINT always works and stops are clean.
     *
     * We also keep our own O_RDWR handle on the FIFO. This means
     * ffmpeg's open() of the FIFO never blocks waiting for a writer —
     * even if arecord dies instantly, ffmpeg starts up, records
     * video, and simply gets no audio (the muxer won't stall thanks
     * to -max_interleave_delta). */
    char fifo_path[64] = "";
    int   fifo_fd = -1;
    pid_t arec    = -1;
    bool  use_pulse = mic && strncmp(mic, "pulse", 5) == 0;

    if (mic && *mic && !use_pulse) {
        snprintf(fifo_path, sizeof fifo_path,
                 "/tmp/vcam-audio-%d.wav", (int)getpid());
        unlink(fifo_path);
        if (mkfifo(fifo_path, 0600) < 0) {
            perror("[record] mkfifo (falling back to direct ALSA)");
            fifo_path[0] = '\0';
        } else {
            fifo_fd = open(fifo_path, O_RDWR);   /* never blocks */
            /* Grow the FIFO to 1MB ≈ 10 SECONDS of 48kHz mono S16.
             * The default 64KB is only 0.68s — any ffmpeg stall longer
             * than that blocked arecord, overran the ALSA ring, and
             * lost audio at capture (heard as missing chunks; visible
             * as a low average audio bitrate in the file). With a 10s
             * cushion, encoder pressure can no longer reach the mic. */
            if (fifo_fd >= 0)
                fcntl(fifo_fd, F_SETPIPE_SZ, 1 << 20);
            arec = fork();
            if (arec == 0) {
                /* arecord child: own log, no stdin, closed fds */
                int log = open("/tmp/vcam-record-audio.log",
                               O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (log >= 0) { dup2(log, 2); close(log); }
                int devnull = open("/dev/null", O_RDWR);
                if (devnull >= 0) {
                    dup2(devnull, 0);
                    dup2(devnull, 1);
                    close(devnull);
                }
                long mx = sysconf(_SC_OPEN_MAX);
                if (mx < 0 || mx > 4096) mx = 4096;
                for (int fd = 3; fd < (int)mx; fd++) close(fd);

                /* Mono 48kHz S16 — the 'plug' layer converts from
                 * whatever the device natively does. Output side
                 * upmixes to stereo AAC. */
                /* -B 1500000: 1.5s ALSA ring buffer — the driver
                 * side can ride out stalls without overrunning. */
                execlp("arecord", "arecord", "-q",
                       "-B", "1500000",
                       "-D", mic,
                       "-f", "S16_LE", "-c", "1", "-r", "48000",
                       "-t", "wav", fifo_path, (char *)NULL);
                _exit(127);
            }
            if (arec < 0) {
                perror("[record] fork arecord (falling back to direct ALSA)");
                close(fifo_fd); fifo_fd = -1;
                unlink(fifo_path);
                fifo_path[0] = '\0';
            }
        }
    }

    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("[record] pipe");
        if (arec > 0) kill(arec, SIGTERM);
        if (fifo_fd >= 0) close(fifo_fd);
        if (fifo_path[0]) unlink(fifo_path);
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("[record] fork");
        close(pipefd[0]); close(pipefd[1]);
        if (arec > 0) kill(arec, SIGTERM);
        if (fifo_fd >= 0) close(fifo_fd);
        if (fifo_path[0]) unlink(fifo_path);
        return false;
    }

    if (pid == 0) {
        /* Child: pipe → stdin, log → stderr, /dev/null → stdout, exec */
        if (dup2(pipefd[0], 0) < 0) _exit(127);
        close(pipefd[0]); close(pipefd[1]);

        int log = open("/tmp/vcam-record.log",
                       O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log >= 0) { dup2(log, 2); close(log); }

        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 1); close(devnull); }

        /* CRITICAL: close every other inherited fd. vcam has the IPC
         * client socket, the V4L2 devices, overlay pipes etc. open at
         * fork time. If ffmpeg inherits the IPC client socket, the
         * client (vcam-ctl / ncat) never sees the connection close
         * and hangs until ffmpeg exits — which is exactly the
         * "record start doesn't return" symptom. */
        long max_fd = sysconf(_SC_OPEN_MAX);
        if (max_fd < 0 || max_fd > 4096) max_fd = 4096;
        for (int fd = 3; fd < (int)max_fd; fd++)
            close(fd);

        char **argv = build_argv(file, mic,
                                  fifo_path[0] ? fifo_path : NULL,
                                  aoffset,
                                  w, h, fps,
                                  crf, vcodec, vpreset);
        if (!argv) _exit(127);
        execvp("ffmpeg", argv);
        perror("[record] execvp ffmpeg");
        _exit(127);
    }

    /* Parent */
    close(pipefd[0]);
    /* Pipe stays BLOCKING — the paced writer thread owns it, and
     * blocking there is harmless (it's not the main loop). */

    /* Build the writer context. The writer thread owns it (and the
     * pipe fd) from the moment it starts. */
    size_t frame_size = (size_t)w * h * 3;
    RecCtx *ctx = calloc(1, sizeof *ctx);
    uint8_t *latest  = ctx ? malloc(frame_size) : NULL;
    uint8_t *scratch = ctx ? malloc(frame_size) : NULL;
    if (!ctx || !latest || !scratch) {
        fprintf(stderr, "[record] out of memory for writer buffers\n");
        free(ctx); free(latest); free(scratch);
        close(pipefd[1]);
        kill(pid, SIGKILL);
        if (arec > 0) kill(arec, SIGTERM);
        if (fifo_fd >= 0) close(fifo_fd);
        if (fifo_path[0]) unlink(fifo_path);
        return false;
    }

    /* Publish state atomically under the lock, then start the writer.
     * No blocking anywhere — vcam-ctl record start returns in
     * milliseconds; if ffmpeg failed to exec, the writer's first
     * write hits EPIPE and tears the recording down cleanly. */
    pthread_mutex_lock(&r->lock);
    r->gen++;
    ctx->r    = r;
    ctx->gen  = r->gen;
    ctx->fd   = pipefd[1];
    ctx->fps  = fps > 0 ? fps : 30;
    ctx->size = frame_size;
    ctx->latest  = latest;
    ctx->scratch = scratch;
    ctx->have_frame = 0;
    ctx->missed = 0;
    r->ctx = ctx;
    r->pid = pid;
    r->arec_pid = arec;
    r->fifo_fd = fifo_fd;
    strncpy(r->fifo_path, fifo_path, sizeof r->fifo_path - 1);
    r->fifo_path[sizeof r->fifo_path - 1] = '\0';
    r->active = true;
    r->frame_w = w;
    r->frame_h = h;
    strncpy(r->file, file, sizeof r->file - 1);
    r->file[sizeof r->file - 1] = '\0';
    pthread_mutex_unlock(&r->lock);

    pthread_t wt;
    if (pthread_create(&wt, NULL, writer_thread, ctx) != 0) {
        perror("[record] writer thread");
        pthread_mutex_lock(&r->lock);
        r->ctx = NULL;
        r->active = false;
        r->pid = -1;
        r->arec_pid = -1;
        r->fifo_fd = -1;
        r->fifo_path[0] = '\0';
        pthread_mutex_unlock(&r->lock);
        close(pipefd[1]);
        free(latest); free(scratch); free(ctx);
        kill(pid, SIGKILL);
        if (arec > 0) kill(arec, SIGTERM);
        if (fifo_fd >= 0) close(fifo_fd);
        if (fifo_path[0]) unlink(fifo_path);
        return false;
    }
    pthread_detach(wt);

    fprintf(stderr, "[record] started — ffmpeg=%d%s file=%s (paced %d fps)\n",
            (int)pid,
            arec > 0 ? " (+arecord)" : "",
            file, ctx->fps);
    return true;
}

/* ── stop watchdog ──
 * SIGINT asks ffmpeg to shut down gracefully (flush + trailer), but
 * a graceful shutdown can wedge — e.g. when the ALSA input thread is
 * stuck in xrun recovery. If the old ffmpeg survives, it keeps the
 * microphone open and the next 'record start' fails with
 * "cannot open audio device (Device or resource busy)".
 *
 * So after signalling we verify the process actually dies, escalating
 * politely: SIGINT (sent by recorder_stop) → 3s grace → SIGTERM →
 * 2s grace → SIGKILL. Runs in a detached thread; recorder_stop still
 * returns instantly. kill(pid, 0) is used for liveness — it works
 * whether or not the SIGCHLD handler already reaped the child
 * (ESRCH once fully gone). */
static void *stop_watchdog(void *arg)
{
    pid_t pid = (pid_t)(intptr_t)arg;

    /* Phase 1: up to 3s for graceful SIGINT shutdown */
    for (int i = 0; i < 15; i++) {
        if (kill(pid, 0) != 0) return NULL;   /* gone — clean exit */
        usleep(200000);
    }

    fprintf(stderr, "[record] child %d ignored SIGINT — sending SIGTERM\n",
            (int)pid);
    kill(pid, SIGTERM);

    /* Phase 2: up to 2s for SIGTERM */
    for (int i = 0; i < 10; i++) {
        if (kill(pid, 0) != 0) return NULL;
        usleep(200000);
    }

    fprintf(stderr, "[record] child %d ignored SIGTERM — SIGKILL\n",
            (int)pid);
    kill(pid, SIGKILL);
    return NULL;
}

void recorder_stop(Recorder *r)
{
    if (!r) return;

    pthread_mutex_lock(&r->lock);
    if (!r->active) {
        pthread_mutex_unlock(&r->lock);
        return;
    }
    /* Mark inactive FIRST — the paced writer thread sees this within
     * one frame tick, closes the video pipe itself (EOF), frees its
     * buffers, and exits. We never touch its fd here, so there is no
     * close-vs-write race by construction. */
    r->active = false;
    int   fifo_fd = r->fifo_fd;
    pid_t pid     = r->pid;
    pid_t arec    = r->arec_pid;
    char  file[512], fifo_path[64];
    strncpy(file, r->file, sizeof file);
    file[sizeof file - 1] = '\0';
    strncpy(fifo_path, r->fifo_path, sizeof fifo_path);
    fifo_path[sizeof fifo_path - 1] = '\0';
    r->fifo_fd = -1;
    r->pid = -1;
    r->arec_pid = -1;
    r->file[0] = '\0';
    r->fifo_path[0] = '\0';
    pthread_mutex_unlock(&r->lock);

    /* Teardown:
     *   1. Stop arecord — audio EOFs.
     *   2. Close our O_RDWR FIFO handle — no writers remain.
     *   3. SIGINT ffmpeg — graceful flush + trailer. Both its inputs
     *      are plain pipes, so nothing can wedge; the watchdog's
     *      SIGTERM/SIGKILL escalation is a never-expected safety net.
     * (The video pipe closes moments later when the writer thread
     * exits.) No waitpid anywhere — SIGCHLD reaps, stop returns
     * instantly. */
    if (arec > 0) {
        kill(arec, SIGTERM);
        pthread_t at;
        if (pthread_create(&at, NULL, stop_watchdog,
                           (void *)(intptr_t)arec) == 0)
            pthread_detach(at);
    }
    if (fifo_fd >= 0) close(fifo_fd);
    if (pid > 0) {
        kill(pid, SIGINT);
        pthread_t tid;
        if (pthread_create(&tid, NULL, stop_watchdog,
                           (void *)(intptr_t)pid) == 0)
            pthread_detach(tid);
    }
    if (fifo_path[0]) unlink(fifo_path);

    fprintf(stderr, "[record] stop initiated — file=%s ffmpeg=%d%s "
                    "(finalises in background)\n",
            file, (int)pid, arec > 0 ? " +arecord" : "");
}

bool recorder_write_frame(Recorder *r, const uint8_t *bgr24, size_t bytes)
{
    if (!r || !bgr24) return true;

    /* Just publish the newest frame; the paced writer thread ships it
     * to ffmpeg on its own clock. A single memcpy (~2.7MB at 720p,
     * ~0.3ms) under the lock — the main loop never blocks on ffmpeg. */
    pthread_mutex_lock(&r->lock);
    RecCtx *c = r->ctx;
    if (r->active && c && bytes >= c->size) {
        memcpy(c->latest, bgr24, c->size);
        c->have_frame = 1;
    }
    pthread_mutex_unlock(&r->lock);
    return true;
}
