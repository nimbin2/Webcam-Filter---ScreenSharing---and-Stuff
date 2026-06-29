/*
 * test_screen – heavy-diagnostic screen capture test
 *
 * Usage:
 *   ./test_screen <WxH+X+Y> [output.raw]
 *
 * The goal of this program is to figure out WHY screen capture fails
 * when it does. It prints every step verbosely.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int snap_down(int v, int align)
{
    int r = (v / align) * align;
    return r >= align ? r : align;
}

/* Read up to `want` bytes within timeout_ms. Returns bytes obtained.
 * Treats n==0 on a FIFO as "no writer yet" (waits up to deadline). */
static size_t read_exact(int fd, uint8_t *buf, size_t want, double timeout_ms)
{
    size_t got = 0;
    double deadline = now_ms() + timeout_ms;
    while (got < want) {
        ssize_t n = read(fd, buf + got, want - got);
        if (n > 0) { got += (size_t)n; continue; }
        if (now_ms() >= deadline) break;
        if (n == 0 || (n < 0 && errno == EAGAIN)) { usleep(10000); continue; }
        if (n < 0 && errno == EINTR) continue;
        perror("read"); break;
    }
    return got;
}

static bool child_alive(pid_t pid)
{
    int status;
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == 0) return true;
    if (r == pid) {
        if (WIFEXITED(status))
            fprintf(stderr, "[diag] child exited with code %d\n", WEXITSTATUS(status));
        else if (WIFSIGNALED(status))
            fprintf(stderr, "[diag] child killed by signal %d\n", WTERMSIG(status));
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <WxH+X+Y> [output.raw]\n"
            "Example: %s 1280x720+0+0 frame.raw\n", argv[0], argv[0]);
        return 1;
    }
    const char *geometry = argv[1];
    const char *outfile  = argc >= 3 ? argv[2] : NULL;

    int gw=0, gh=0, gx=0, gy=0;
    if (sscanf(geometry, "%dx%d+%d+%d", &gw, &gh, &gx, &gy) != 4) {
        fprintf(stderr, "Bad geometry '%s' — expected WxH+X+Y\n", geometry);
        return 1;
    }

    /* Snap to alignment (same as vcam does) */
    int gw_orig = gw, gh_orig = gh;
    gw = snap_down(gw, 16);
    gh = snap_down(gh, 2);

    bool wayland = getenv("WAYLAND_DISPLAY") != NULL;
    bool x11     = getenv("DISPLAY") != NULL;

    printf("─── test_screen ───────────────────────────────────────\n");
    printf("Requested  : %dx%d at (%d,%d)\n", gw_orig, gh_orig, gx, gy);
    if (gw != gw_orig || gh != gh_orig)
        printf("Snapped to : %dx%d (16/2-aligned for compositor stride)\n", gw, gh);
    printf("WAYLAND_DISPLAY = %s\n", getenv("WAYLAND_DISPLAY") ?: "(unset)");
    printf("DISPLAY         = %s\n", getenv("DISPLAY") ?: "(unset)");
    printf("XDG_RUNTIME_DIR = %s\n", getenv("XDG_RUNTIME_DIR") ?: "(unset)");
    printf("Backend         : %s\n", wayland ? "Wayland (wf-recorder)" :
                                    (x11 ? "X11 (ffmpeg x11grab)" : "NONE"));
    printf("───────────────────────────────────────────────────────\n");

    int fd = -1;
    pid_t pid = -1;
    int   src_bpp = 3;
    char  fifo_path[128] = "";

    if (wayland) {
        snprintf(fifo_path, sizeof fifo_path, "/tmp/test_screen_%d.fifo", (int)getpid());
        unlink(fifo_path);
        if (mkfifo(fifo_path, 0600) < 0) { perror("mkfifo"); return 1; }
        printf("[step] created FIFO  : %s\n", fifo_path);

        fd = open(fifo_path, O_RDONLY | O_NONBLOCK);
        if (fd < 0) { perror("open fifo"); unlink(fifo_path); return 1; }
        printf("[step] opened read end (O_NONBLOCK)\n");

        char wfr_geo[64];
        snprintf(wfr_geo, sizeof wfr_geo, "%d,%d %dx%d", gx, gy, gw, gh);
        printf("[step] forking child to exec:\n");
        printf("           wf-recorder -g '%s' -c rawvideo -x bgr0 --muxer rawvideo -f %s\n",
               wfr_geo, fifo_path);
        printf("[step] (stdin pre-loaded with 'y\\n' to auto-answer Overwrite prompt)\n");

        /* Pre-load "y\n" on stdin so wf-recorder's "Overwrite?" prompt is
         * answered automatically. Without this it hangs forever. */
        int answer_pipe[2];
        if (pipe(answer_pipe) < 0) { perror("answer pipe"); return 1; }
        const char *yes = "y\ny\ny\n";
        ssize_t wn = write(answer_pipe[1], yes, strlen(yes));
        (void)wn;

        pid = fork();
        if (pid == 0) {
            dup2(answer_pipe[0], STDIN_FILENO);
            close(answer_pipe[0]); close(answer_pipe[1]);
            execlp("wf-recorder", "wf-recorder",
                   "-g", wfr_geo,
                   "-c", "rawvideo",
                   "-x", "bgr0",
                   "--muxer", "rawvideo",
                   "-f", fifo_path,
                   (char *)NULL);
            perror("execlp wf-recorder");
            _exit(127);
        }
        close(answer_pipe[0]); close(answer_pipe[1]);
        src_bpp = 4;

    } else if (x11) {
        int pipefd[2];
        if (pipe(pipefd) < 0) { perror("pipe"); return 1; }
        printf("[step] created stdout pipe\n");

        char input[64], size[32];
        snprintf(input, sizeof input, ":0.0+%d,%d", gx, gy);
        snprintf(size,  sizeof size,  "%dx%d", gw, gh);
        printf("[step] forking child to exec:\n");
        printf("           ffmpeg -f x11grab -video_size %s -i %s -f rawvideo -pix_fmt bgr24 pipe:1\n",
               size, input);

        pid = fork();
        if (pid == 0) {
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[0]); close(pipefd[1]);
            execlp("ffmpeg", "ffmpeg",
                   "-f", "x11grab", "-video_size", size, "-i", input,
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
                   (char *)NULL);
            perror("execlp ffmpeg");
            _exit(127);
        }
        close(pipefd[1]);
        fd = pipefd[0];
        src_bpp = 3;
    } else {
        fprintf(stderr, "[fail] Neither WAYLAND_DISPLAY nor DISPLAY is set.\n");
        return 1;
    }

    printf("[step] child pid: %d\n", (int)pid);

    /* Verify child is alive after a brief moment */
    usleep(200000);
    if (!child_alive(pid)) {
        fprintf(stderr, "[fail] child exited immediately. See its error output above.\n");
        if (fifo_path[0]) unlink(fifo_path);
        return 1;
    }
    printf("[step] child still alive after 200ms — capture starting\n");

    size_t frame_bytes = (size_t)gw * gh * src_bpp;
    printf("Frame size : %zu bytes (%.2f MB)\n", frame_bytes, frame_bytes / 1e6);
    printf("\n");

    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    uint8_t *buf = malloc(frame_bytes);
    if (!buf) { perror("malloc"); kill(pid, SIGTERM); return 1; }

    /* Wait up to 15s for first frame, polling child health */
    printf("Waiting for first frame (up to 15s)...\n");
    double t0 = now_ms();
    double deadline = t0 + 15000;
    size_t got = 0;

    while (got < frame_bytes && now_ms() < deadline) {
        if (!child_alive(pid)) {
            fprintf(stderr, "[fail] capture process died before delivering a frame.\n");
            free(buf);
            if (fifo_path[0]) unlink(fifo_path);
            return 1;
        }
        size_t step = read_exact(fd, buf + got, frame_bytes - got, 200);
        got += step;
        if (step > 0) {
            printf("  +%zu bytes (total %zu / %zu, %.0f%%)\n",
                   step, got, frame_bytes, 100.0 * got / frame_bytes);
        }
    }

    if (got < frame_bytes) {
        printf("\n[fail] only %zu / %zu bytes after 15s. Capture is producing nothing.\n",
               got, frame_bytes);
        printf("\nThings to try manually:\n");
        if (wayland) {
            printf("  wf-recorder -g '%d,%d %dx%d' -c rawvideo -x bgr0 --muxer rawvideo -f /tmp/raw.test\n",
                   gx, gy, gw, gh);
            printf("  sleep 2; pkill wf-recorder\n");
            printf("  ls -la /tmp/raw.test    # should be many MB\n");
            printf("\nIf wf-recorder writes nothing on its own either:\n");
            printf("  - Your compositor may not expose wlr-screencopy (KDE/GNOME don't).\n");
            printf("    Wayland compositors that DO: Sway, Hyprland, river, niri, wayfire.\n");
            printf("  - Try a screenshot tool to check: grim /tmp/test.png\n");
        } else {
            printf("  ffmpeg -f x11grab -video_size %dx%d -i :0.0+%d,%d -frames:v 5 /tmp/test.mp4\n",
                   gw, gh, gx, gy);
        }
        kill(pid, SIGTERM); waitpid(pid, NULL, 0);
        free(buf);
        if (fifo_path[0]) unlink(fifo_path);
        return 1;
    }

    printf("\nFirst full frame in %.0f ms\n", now_ms() - t0);

    /* Read 9 more frames to measure throughput */
    printf("Reading 9 more frames for throughput measurement...\n");
    double t1 = now_ms();
    int frames = 0;
    for (; frames < 9; frames++) {
        size_t n = read_exact(fd, buf, frame_bytes, 5000.0);
        if (n < frame_bytes) { printf("  pipe closed after %d frames\n", frames); break; }
    }
    double elapsed = now_ms() - t1;
    if (frames > 0)
        printf("Throughput : %.1f fps\n", frames * 1000.0 / elapsed);

    /* Check for content */
    uint64_t sum = 0, nonzero = 0;
    for (size_t i = 0; i < frame_bytes; i++) {
        sum += buf[i];
        if (buf[i] != 0) nonzero++;
    }
    printf("Brightness : %llu (avg byte %.1f)\n",
           (unsigned long long)sum, sum / (double)frame_bytes);
    printf("Non-zero   : %llu / %zu bytes (%.1f%%)\n",
           (unsigned long long)nonzero, frame_bytes,
           100.0 * nonzero / frame_bytes);
    if (nonzero < 100)
        printf("WARNING    : frame is essentially all zero — capture works but content is black!\n"
               "             Try a region that has actual content on screen.\n");

    if (outfile) {
        FILE *f = fopen(outfile, "wb");
        if (f) {
            fwrite(buf, 1, frame_bytes, f);
            fclose(f);
            printf("\nSaved to   : %s\n", outfile);
            printf("View with  : ffmpeg -f rawvideo -pix_fmt %s -video_size %dx%d -i %s -frames:v 1 -y %s.png && xdg-open %s.png\n",
                   src_bpp == 4 ? "bgr0" : "bgr24", gw, gh, outfile, outfile, outfile);
        }
    }

    /* Robust shutdown: SIGTERM, wait up to 500ms, then SIGKILL */
    fprintf(stderr, "\n[cleanup] stopping capture child...\n");
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        if (waitpid(pid, NULL, WNOHANG) == pid) goto reaped;
        usleep(10000); /* 10ms */
    }
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
reaped:
    free(buf);
    if (fifo_path[0]) unlink(fifo_path);
    fprintf(stderr, "[cleanup] done.\n");
    return 0;
}
