/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Created-by: Claude Opus 5.5 (idle detection)
 */

/* realpath() is an X/Open extension: the build only asks for POSIX
 * 2008, which newer glibc (2.43) takes strictly. Must precede all includes. */
#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE

/*
 * Who reads the virtual camera? Counts the other processes that have the
 * loopback device open, so vcam can go idle when nobody watches.
 *
 * - inotify on the device node reports every open() and close(), with
 *   no polling. Events are only a trigger: inotify merges identical
 *   back-to-back events (two quick opens arrive as one), so a tally
 *   would drift.
 * - The count itself comes from a /proc scan for fds on the device: on
 *   every event, again 0.3 s later (an opener's fd can appear in /proc a
 *   moment after the event), and every 10 s as a safety net.
 * Limit: the scan sees processes vcam may inspect — same user, or all
 * when vcam runs as root. Use --idle off for readers under other users.
 */

#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <time.h>

/* Count other processes holding at least one fd on the device. */
int consumers_scan(const Consumers *c)
{
    DIR *proc = opendir("/proc");
    if (!proc) return 0;
    pid_t self = getpid();
    int n = 0;
    struct dirent *pe;
    while ((pe = readdir(proc))) {
        char *ep;
        long pid = strtol(pe->d_name, &ep, 10);
        if (*ep || pid <= 0 || pid == self) continue;
        char fdpath[64];
        snprintf(fdpath, sizeof fdpath, "/proc/%ld/fd", pid);
        DIR *fds = opendir(fdpath);
        if (!fds) continue;                  /* gone, or not ours to inspect */
        struct dirent *fe;
        while ((fe = readdir(fds))) {
            if (fe->d_name[0] == '.') continue;
            char lpath[384], target[PATH_MAX];
            snprintf(lpath, sizeof lpath, "%s/%s", fdpath, fe->d_name);
            ssize_t len = readlink(lpath, target, sizeof target - 1);
            if (len <= 0) continue;
            target[len] = '\0';
            if (strcmp(target, c->path) != 0) continue;
            n++;                             /* count each process once */
            break;
        }
        closedir(fds);
    }
    closedir(proc);
    return n;
}

static long now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000L + t.tv_nsec / 1000000L;
}

bool consumers_init(Consumers *c, const char *dev)
{
    memset(c, 0, sizeof *c);
    c->ifd = -1;
    if (!realpath(dev, c->path)) return false;
    c->count = consumers_scan(c);
    c->next_scan_ms = now_ms() + 10000;
    c->ifd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (c->ifd < 0) return false;
    if (inotify_add_watch(c->ifd, c->path,
                          IN_OPEN | IN_CLOSE_WRITE | IN_CLOSE_NOWRITE) < 0) {
        close(c->ifd);
        c->ifd = -1;
        return false;
    }
    return true;
}

/* Drain pending events (non-blocking) and return the current count. */
int consumers_update(Consumers *c)
{
    if (c->ifd < 0) return -1;
    char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
    bool event = false;
    for (;;) {
        ssize_t len = read(c->ifd, buf, sizeof buf);
        if (len <= 0) break;                 /* EAGAIN: nothing pending */
        event = true;                        /* open, close or overflow */
    }
    long now = now_ms();
    if (event) {
        c->count = consumers_scan(c);
        c->recheck_ms = now + 300;           /* catch an fd not yet installed */
    } else if ((c->recheck_ms && now >= c->recheck_ms) || now >= c->next_scan_ms) {
        c->count = consumers_scan(c);
        c->recheck_ms = 0;
        c->next_scan_ms = now + 10000;
    }
    return c->count;
}

void consumers_close(Consumers *c)
{
    if (c->ifd >= 0) close(c->ifd);
    c->ifd = -1;
}
