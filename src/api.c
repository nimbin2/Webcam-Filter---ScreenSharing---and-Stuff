/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 *
 * Created-by: Claude Opus 5.5 (control API, live capture switching)
 */

/*
 * High-level control API, served on the same socket as ipc.c.
 * Built for other apps: short commands, one-line machine-readable replies.
 *
 *   status [json]              whole state: "OK k=v k=v ..." or one JSON line
 *   get <key>                  one value from status, e.g. "get cam_pos" → "OK br"
 *   version                    "OK vcam 1.1.0 (build abc1234)"  — full reference: API.md
 *   monitors                   one line per monitor: "OK <n> <name> WxH+X+Y"
 *
 *   share start [<n>|<name>|<WxH+X+Y>] [fit=stretch|contain]
 *               [cam=<pos>|off] [camw=<px>]
 *   share stop | toggle | status
 *
 *   cam on [<pos>] [<width>]   pos: tl tr bl br center full
 *   cam off | toggle | raise | get
 *   cam pos <pos>              keep size, snap to a position
 *   cam move <x> <y>           free position (pos becomes "custom")
 *   cam size <w> [<h>]         keeps its anchor; h derived from aspect
 *   cam margin <px>            gap between cam and the output edge
 *
 *   capture [get]              current camera mode
 *   capture formats [json]     every mode the camera offers
 *   capture set [<fmt>] [<WxH>|max] [<fps>] [fit=crop|stretch]
 *                              switch the camera live (output size stays)
 *   capture fit crop|stretch   how a different aspect ratio is fitted
 *
 * The share overlay uses id "share", the cam overlay id "cam" (same id
 * as 'vcam-ctl webcam'). Status also recognises screen/webcam overlays
 * added by other commands.
 */

#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>

#ifndef VCAM_BUILD
#define VCAM_BUILD "dev"
#endif

#define SHARE_ID "share"
#define CAM_ID   "cam"

extern FilterChain g_filters;
extern Recorder   *g_recorder;
extern int         g_output_w, g_output_h, g_output_fps;
extern bool        g_no_input;

/* State owned by the API. Only touched from command handlers, which
 * ipc.c serialises with a mutex, so no extra locking is needed. */
static char g_share_geo[64] = "";
static char g_share_mon[64] = "";
static int  g_cam_margin    = 0;

const char *api_version_string(void)
{
    return "vcam " VCAM_VERSION " (build " VCAM_BUILD ")";
}

/* ── reply helper (bigger buffer than ipc.c's send_line) ───── */
static void reply(int fd, const char *fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof buf - 2) n = (int)sizeof buf - 2;
    buf[n++] = '\n';
    for (int off = 0; off < n; ) {
        ssize_t w = write(fd, buf + off, (size_t)(n - off));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        off += (int)w;
    }
}

/* ══════════════════════════════════════════════════════════════
 * Monitor detection (sway, Hyprland, other wlroots, X11, DRM)
 * Coordinates are logical (what wf-recorder -g / x11grab expect).
 * ══════════════════════════════════════════════════════════════ */
typedef struct { char name[32]; int w, h, x, y; } Monitor;
#define MAX_MON 16

typedef struct {
    Monitor m;
    double  scale;
    bool    swap;    /* rotated 90/270 → swap w/h */
    bool    valid;
} Pending;

static void pend_reset(Pending *p)
{
    memset(p, 0, sizeof *p);
    p->scale = 1.0;
}

static void pend_commit(Pending *p, Monitor *out, int *n, int max)
{
    if (p->valid && p->m.w > 0 && p->m.h > 0 && *n < max) {
        Monitor m = p->m;
        if (p->swap) { int t = m.w; m.w = m.h; m.h = t; }
        if (p->scale > 0.0 && p->scale != 1.0) {
            m.w = (int)(m.w / p->scale + 0.5);
            m.h = (int)(m.h / p->scale + 0.5);
        }
        out[(*n)++] = m;
    }
    pend_reset(p);
}

static char *skip_ws(char *s) { while (*s == ' ' || *s == '\t') s++; return s; }

static bool is_rot90(const char *v)
{
    return strstr(v, "90") || strstr(v, "270");
}

/* swaymsg -p -t get_outputs */
static int parse_sway(FILE *f, Monitor *out, int max)
{
    char line[512]; int n = 0; Pending p; pend_reset(&p);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "Output ", 7) == 0) {
            pend_commit(&p, out, &n, max);
            if (sscanf(line + 7, "%31s", p.m.name) == 1) p.valid = true;
            continue;
        }
        char *s = skip_ws(line);
        if      (sscanf(s, "Current mode: %dx%d", &p.m.w, &p.m.h) == 2) {}
        else if (sscanf(s, "Position: %d,%d", &p.m.x, &p.m.y) == 2) {}
        else if (sscanf(s, "Scale factor: %lf", &p.scale) == 1) {}
        else if (strncmp(s, "Transform:", 10) == 0) p.swap = is_rot90(s + 10);
    }
    pend_commit(&p, out, &n, max);
    return n;
}

/* hyprctl monitors */
static int parse_hypr(FILE *f, Monitor *out, int max)
{
    char line[512]; int n = 0; Pending p; pend_reset(&p);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "Monitor ", 8) == 0) {
            pend_commit(&p, out, &n, max);
            if (sscanf(line + 8, "%31s", p.m.name) == 1) p.valid = true;
            continue;
        }
        char *s = skip_ws(line);
        int t;
        if      (sscanf(s, "%dx%d@%*f at %dx%d", &p.m.w, &p.m.h, &p.m.x, &p.m.y) == 4) {}
        else if (sscanf(s, "scale: %lf", &p.scale) == 1) {}
        else if (sscanf(s, "transform: %d", &t) == 1) p.swap = (t % 2) == 1;
    }
    pend_commit(&p, out, &n, max);
    return n;
}

/* wlr-randr (generic wlroots compositors) */
static int parse_wlr(FILE *f, Monitor *out, int max)
{
    char line[512]; int n = 0; Pending p; pend_reset(&p);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != ' ' && line[0] != '\t' && line[0] != '\n') {
            pend_commit(&p, out, &n, max);
            if (sscanf(line, "%31s", p.m.name) == 1) p.valid = true;
            continue;
        }
        char *s = skip_ws(line);
        if (strncmp(s, "Enabled: no", 11) == 0) p.valid = false;
        else if (strstr(s, " px") && strstr(s, "current"))
            sscanf(s, "%dx%d", &p.m.w, &p.m.h);
        else if (sscanf(s, "Position: %d,%d", &p.m.x, &p.m.y) == 2) {}
        else if (sscanf(s, "Scale: %lf", &p.scale) == 1) {}
        else if (strncmp(s, "Transform:", 10) == 0) p.swap = is_rot90(s + 10);
    }
    pend_commit(&p, out, &n, max);
    return n;
}

/* xrandr --query:  "DP-1 connected primary 1920x1080+0+0 (normal ..." */
static int parse_xrandr(FILE *f, Monitor *out, int max)
{
    char line[512]; int n = 0;
    while (fgets(line, sizeof line, f) && n < max) {
        if (!strstr(line, " connected")) continue;
        Monitor m = {0};
        if (sscanf(line, "%31s", m.name) != 1) continue;
        char *save = NULL;
        for (char *t = strtok_r(line, " \t", &save); t; t = strtok_r(NULL, " \t", &save)) {
            if (sscanf(t, "%dx%d+%d+%d", &m.w, &m.h, &m.x, &m.y) == 4) {
                out[n++] = m;
                break;
            }
        }
    }
    return n;
}

/* Last resort: /sys/class/drm gives resolution only (position 0,0). */
static int detect_drm(Monitor *out, int max)
{
    DIR *d = opendir("/sys/class/drm");
    if (!d) return 0;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < max) {
        const char *dash = strchr(e->d_name, '-');
        if (strncmp(e->d_name, "card", 4) != 0 || !dash) continue;
        char path[512], buf[64] = "";
        snprintf(path, sizeof path, "/sys/class/drm/%s/status", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        bool connected = fgets(buf, sizeof buf, f) && strncmp(buf, "connected", 9) == 0;
        fclose(f);
        if (!connected) continue;
        snprintf(path, sizeof path, "/sys/class/drm/%s/modes", e->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        Monitor m = {0};
        if (fgets(buf, sizeof buf, f) && sscanf(buf, "%dx%d", &m.w, &m.h) == 2) {
            snprintf(m.name, sizeof m.name, "%s", dash + 1);
            out[n++] = m;
        }
        fclose(f);
    }
    closedir(d);
    return n;
}

static int run_parse(const char *cmd, int (*parse)(FILE *, Monitor *, int),
                     Monitor *out, int max)
{
    FILE *f = popen(cmd, "r");
    if (!f) return 0;
    int n = parse(f, out, max);
    pclose(f);   /* may report ECHILD if main's SIGCHLD reaped it — harmless */
    return n;
}

static int detect_monitors(Monitor *out, int max)
{
    int n = 0;
    if (getenv("WAYLAND_DISPLAY")) {
        if (getenv("SWAYSOCK"))
            n = run_parse("swaymsg -p -t get_outputs 2>/dev/null", parse_sway, out, max);
        if (!n && getenv("HYPRLAND_INSTANCE_SIGNATURE"))
            n = run_parse("hyprctl monitors 2>/dev/null", parse_hypr, out, max);
        if (!n)
            n = run_parse("wlr-randr 2>/dev/null", parse_wlr, out, max);
    } else if (getenv("DISPLAY")) {
        n = run_parse("xrandr --query 2>/dev/null", parse_xrandr, out, max);
    }
    if (!n) n = detect_drm(out, max);
    return n;
}

static bool parse_geo(const char *s, int *w, int *h, int *x, int *y)
{
    char tail;
    return sscanf(s, "%dx%d+%d+%d%c", w, h, x, y, &tail) == 4 && *w > 0 && *h > 0;
}

/* Resolve a selector (empty, 1-based index, name, or WxH+X+Y) to a
 * geometry string. Returns false with *err set on failure. */
static bool resolve_share_target(const char *sel, char *geo, size_t geolen,
                                 char *mon, size_t monlen, const char **err)
{
    int w, h, x, y;
    if (sel && parse_geo(sel, &w, &h, &x, &y)) {
        snprintf(geo, geolen, "%dx%d+%d+%d", w, h, x, y);
        snprintf(mon, monlen, "-");
        return true;
    }

    Monitor m[MAX_MON];
    int n = detect_monitors(m, MAX_MON);
    if (n == 0) {
        *err = "no monitor detected — pass a geometry: share start WxH+X+Y";
        return false;
    }

    int pick = 0;   /* default: first monitor */
    if (sel && *sel) {
        char *ep;
        long idx = strtol(sel, &ep, 10);
        pick = -1;
        if (*ep == '\0') {
            if (idx >= 1 && idx <= n) pick = (int)idx - 1;
        } else {
            for (int i = 0; i < n; i++)
                if (strcmp(m[i].name, sel) == 0) { pick = i; break; }
        }
        if (pick < 0) { *err = "unknown monitor (see 'monitors')"; return false; }
    }
    snprintf(geo, geolen, "%dx%d+%d+%d", m[pick].w, m[pick].h, m[pick].x, m[pick].y);
    snprintf(mon, monlen, "%.31s", m[pick].name);
    return true;
}

/* ══════════════════════════════════════════════════════════════
 * State snapshot
 * ══════════════════════════════════════════════════════════════ */
typedef struct {
    bool share;   char share_id[64]; int sx, sy, sw, sh, s_z;
    bool cam;     char cam_id[64];   int cx, cy, cw, ch, c_z;
    bool cam_over_screen;
    int  overlays;
} State;

static bool rects_intersect(int ax, int ay, int aw, int ah,
                            int bx, int by, int bw, int bh)
{
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

static void collect(OverlayRegistry *reg, State *st)
{
    memset(st, 0, sizeof *st);
    pthread_mutex_lock(&reg->lock);
    st->overlays = reg->count;
    for (int i = 0; i < reg->count; i++) {
        OverlaySlot *s = &reg->slots[i];
        OverlayType t = overlay_get_type(s->overlay);
        int x, y, w, h;
        overlay_get_rect(s->overlay, &x, &y, &w, &h);

        /* Prefer our own ids ("share"/"cam") once seen; otherwise the
         * topmost overlay of that type wins. */
        if (t == OVERLAY_SCREEN &&
            (!st->share || strcmp(st->share_id, SHARE_ID) != 0)) {
            st->share = true;
            snprintf(st->share_id, sizeof st->share_id, "%s", s->id);
            st->sx = x; st->sy = y; st->sw = w; st->sh = h; st->s_z = i;
        }
        if (t == OVERLAY_WEBCAM &&
            (!st->cam || strcmp(st->cam_id, CAM_ID) != 0)) {
            st->cam = true;
            snprintf(st->cam_id, sizeof st->cam_id, "%s", s->id);
            st->cx = x; st->cy = y; st->cw = w; st->ch = h; st->c_z = i;
        }

        /* cam_over_screen: a webcam overlay drawn above a screen
         * overlay, overlapping it, and visible inside the output. */
        if (t == OVERLAY_WEBCAM &&
            rects_intersect(x, y, w, h, 0, 0, g_output_w, g_output_h)) {
            for (int j = 0; j < i; j++) {
                if (overlay_get_type(reg->slots[j].overlay) != OVERLAY_SCREEN) continue;
                int bx, by, bw, bh;
                overlay_get_rect(reg->slots[j].overlay, &bx, &by, &bw, &bh);
                if (rects_intersect(x, y, w, h, bx, by, bw, bh))
                    st->cam_over_screen = true;
            }
        }
    }
    pthread_mutex_unlock(&reg->lock);
}

static const char *anchor_name(int x, int y, int w, int h)
{
    int W = g_output_w, H = g_output_h, m = g_cam_margin;
    if (x <= 0 && y <= 0 && w >= W && h >= H) return "full";
    int L = m, R = W - w - m, T = m, B = H - h - m;
    if (x == L && y == T) return "tl";
    if (x == R && y == T) return "tr";
    if (x == L && y == B) return "bl";
    if (x == R && y == B) return "br";
    if (x == (W - w) / 2 && y == (H - h) / 2) return "center";
    return "custom";
}

/* Compute x/y for a named anchor. "full" also overrides w/h. */
static bool anchor_xy(const char *pos, int *x, int *y, int *w, int *h)
{
    int W = g_output_w, H = g_output_h, m = g_cam_margin;
    if      (!strcmp(pos, "tl"))     { *x = m;            *y = m; }
    else if (!strcmp(pos, "tr"))     { *x = W - *w - m;   *y = m; }
    else if (!strcmp(pos, "bl"))     { *x = m;            *y = H - *h - m; }
    else if (!strcmp(pos, "br"))     { *x = W - *w - m;   *y = H - *h - m; }
    else if (!strcmp(pos, "center")) { *x = (W - *w) / 2; *y = (H - *h) / 2; }
    else if (!strcmp(pos, "full"))   { *x = 0; *y = 0; *w = W; *h = H; }
    else return false;
    return true;
}

/* Cam height from width, using the capture aspect (cam_buf == output size).
 * Must match overlay_tick's aspect fix-up + resolve_dims' even rounding
 * exactly, or the first frame would shift the height and break the anchor. */
static int cam_h_for(int w)
{
    int h = g_output_w > 0 ? (int)((long)w * g_output_h / g_output_w) : w * 9 / 16;
    return (h & 1) ? h + 1 : h;
}

/* ══════════════════════════════════════════════════════════════
 * Status: key/value list → plain, single key, or JSON
 * ══════════════════════════════════════════════════════════════ */
typedef struct { const char *k; char v[80]; char t; } KV;   /* t: b=bool i=int s=str */
#define MAX_KV 48

static int kv_add(KV *kv, int n, const char *k, char t, const char *fmt, ...)
{
    if (n >= MAX_KV) return n;
    kv[n].k = k; kv[n].t = t;
    va_list ap; va_start(ap, fmt);
    vsnprintf(kv[n].v, sizeof kv[n].v, fmt, ap);
    va_end(ap);
    /* values must be single tokens in plain mode */
    for (char *p = kv[n].v; *p; p++) if (*p == ' ' || *p == '"' || *p == '\\') *p = '_';
    return n + 1;
}

static int build_status(OverlayRegistry *reg, KV *kv)
{
    State st; collect(reg, &st);
    int n = 0, nf;
    pthread_mutex_lock(&g_filters.lock);
    nf = g_filters.count;
    pthread_mutex_unlock(&g_filters.lock);
    bool share_ours = st.share && !strcmp(st.share_id, SHARE_ID);
    bool full = st.share && st.sx <= 0 && st.sy <= 0 &&
                st.sx + st.sw >= g_output_w && st.sy + st.sh >= g_output_h;

    n = kv_add(kv, n, "version",          's', "%s", VCAM_VERSION);
    n = kv_add(kv, n, "build",            's', "%s", VCAM_BUILD);
    n = kv_add(kv, n, "input",            's', "%s", g_no_input ? "none" : "webcam");
    n = kv_add(kv, n, "output_w",         'i', "%d", g_output_w);
    n = kv_add(kv, n, "output_h",         'i', "%d", g_output_h);
    n = kv_add(kv, n, "fps",              'i', "%d", g_output_fps);
    CaptureState cs; memset(&cs, 0, sizeof cs);
    if (!g_no_input) capture_state_get(&cs);
    n = kv_add(kv, n, "capture",          'b', "%d", cs.active);
    n = kv_add(kv, n, "capture_device",   's', "%s", g_no_input ? "-" : capture_device_path());
    n = kv_add(kv, n, "capture_fmt",      's', "%s", cs.active ? cs.mode.fmt : "-");
    n = kv_add(kv, n, "capture_w",        'i', "%d", cs.active ? cs.mode.w : 0);
    n = kv_add(kv, n, "capture_h",        'i', "%d", cs.active ? cs.mode.h : 0);
    n = kv_add(kv, n, "capture_fps",      'i', "%d", cs.active ? cs.mode.fps : 0);
    n = kv_add(kv, n, "capture_fit",      's', "%s", cs.active ? (cs.crop ? "crop" : "stretch") : "-");
    n = kv_add(kv, n, "share",            'b', "%d", st.share);
    n = kv_add(kv, n, "share_id",         's', "%s", st.share ? st.share_id : "-");
    n = kv_add(kv, n, "share_fullscreen", 'b', "%d", full);
    n = kv_add(kv, n, "share_monitor",    's', "%s", share_ours && g_share_mon[0] ? g_share_mon : "-");
    n = kv_add(kv, n, "share_geometry",   's', "%s", share_ours && g_share_geo[0] ? g_share_geo : "-");
    n = kv_add(kv, n, "share_x",          'i', "%d", st.sx);
    n = kv_add(kv, n, "share_y",          'i', "%d", st.sy);
    n = kv_add(kv, n, "share_w",          'i', "%d", st.sw);
    n = kv_add(kv, n, "share_h",          'i', "%d", st.sh);
    n = kv_add(kv, n, "cam",              'b', "%d", st.cam);
    n = kv_add(kv, n, "cam_id",           's', "%s", st.cam ? st.cam_id : "-");
    n = kv_add(kv, n, "cam_pos",          's', "%s", st.cam ? anchor_name(st.cx, st.cy, st.cw, st.ch) : "-");
    n = kv_add(kv, n, "cam_x",            'i', "%d", st.cx);
    n = kv_add(kv, n, "cam_y",            'i', "%d", st.cy);
    n = kv_add(kv, n, "cam_w",            'i', "%d", st.cw);
    n = kv_add(kv, n, "cam_h",            'i', "%d", st.ch);
    n = kv_add(kv, n, "cam_z",            'i', "%d", st.cam ? st.c_z : -1);
    n = kv_add(kv, n, "cam_margin",       'i', "%d", g_cam_margin);
    n = kv_add(kv, n, "cam_over_screen",  'b', "%d", st.cam_over_screen);
    n = kv_add(kv, n, "overlays",         'i', "%d", st.overlays);
    n = kv_add(kv, n, "filters",          'i', "%d", nf);
    n = kv_add(kv, n, "recording",        'b', "%d", recorder_active(g_recorder));
    return n;
}

static const char *kv_plain(const KV *e)
{
    if (e->t == 'b') return strcmp(e->v, "0") ? "yes" : "no";
    return e->v;
}

static void cmd_status(int fd, OverlayRegistry *reg, bool json)
{
    KV kv[MAX_KV];
    int n = build_status(reg, kv);
    char buf[3072];
    size_t off = 0;
    off += (size_t)snprintf(buf + off, sizeof buf - off, json ? "{\"ok\":true" : "OK");
    for (int i = 0; i < n && off < sizeof buf; i++) {
        if (!json)
            off += (size_t)snprintf(buf + off, sizeof buf - off, " %s=%s", kv[i].k, kv_plain(&kv[i]));
        else if (kv[i].t == 'b')
            off += (size_t)snprintf(buf + off, sizeof buf - off, ",\"%s\":%s", kv[i].k,
                                    strcmp(kv[i].v, "0") ? "true" : "false");
        else if (kv[i].t == 'i')
            off += (size_t)snprintf(buf + off, sizeof buf - off, ",\"%s\":%s", kv[i].k, kv[i].v);
        else
            off += (size_t)snprintf(buf + off, sizeof buf - off, ",\"%s\":\"%s\"", kv[i].k, kv[i].v);
    }
    if (json && off < sizeof buf) snprintf(buf + off, sizeof buf - off, "}");
    reply(fd, "%s", buf);
}

static void cmd_get(int fd, OverlayRegistry *reg, const char *key)
{
    if (!key) { reply(fd, "ERR usage: get <key>   (keys: see 'status')"); return; }
    KV kv[MAX_KV];
    int n = build_status(reg, kv);
    for (int i = 0; i < n; i++)
        if (!strcmp(kv[i].k, key)) { reply(fd, "OK %s", kv_plain(&kv[i])); return; }
    reply(fd, "ERR unknown key '%s'", key);
}

/* ══════════════════════════════════════════════════════════════
 * cam
 * ══════════════════════════════════════════════════════════════ */
static void cam_reply(int fd, OverlayRegistry *reg)
{
    State st; collect(reg, &st);
    if (!st.cam) { reply(fd, "OK cam=off"); return; }
    reply(fd, "OK cam=on id=%s pos=%s x=%d y=%d w=%d h=%d z=%d over_screen=%s",
          st.cam_id, anchor_name(st.cx, st.cy, st.cw, st.ch),
          st.cx, st.cy, st.cw, st.ch, st.c_z, st.cam_over_screen ? "yes" : "no");
}

/* Place (and create if needed) the cam overlay. pos may be NULL to keep
 * the current anchor; w <= 0 keeps the current width (or 320). */
static bool cam_place(OverlayRegistry *reg, const char *pos, int w, int h,
                      const char **err)
{
    State st; collect(reg, &st);
    const char *id = st.cam ? st.cam_id : CAM_ID;
    const char *anchor = pos ? pos
                       : st.cam ? anchor_name(st.cx, st.cy, st.cw, st.ch) : "tr";
    if (w <= 0) w = st.cam ? st.cw : 320;
    if (h <= 0) h = cam_h_for(w);
    if (w < 16 || h < 16) { *err = "cam size too small"; return false; }

    int x = st.cx, y = st.cy;
    if (strcmp(anchor, "custom") != 0 && !anchor_xy(anchor, &x, &y, &w, &h)) {
        *err = "unknown position (use tl|tr|bl|br|center|full)";
        return false;
    }

    if (!st.cam) {
        Overlay *o = overlay_create_webcam("-", x, y, w, h);
        if (!o) { *err = "failed to create webcam overlay"; return false; }
        if (!registry_add(reg, id, o)) {
            overlay_free(o);
            *err = "could not add cam overlay (registry full?)";
            return false;
        }
    } else {
        pthread_mutex_lock(&reg->lock);
        Overlay *o = registry_find(reg, id);
        if (o) { overlay_resize(o, w, h); overlay_move(o, x, y); }
        pthread_mutex_unlock(&reg->lock);
    }
    registry_raise(reg, id);   /* cam always on top of the share */
    return true;
}

static bool cam_handle(int fd, char **save, OverlayRegistry *reg)
{
    char *sub = strtok_r(NULL, " \t", save);
    const char *err = NULL;
    State st;

    if (!sub || !strcmp(sub, "get") || !strcmp(sub, "status")) {
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "on") || !strcmp(sub, "pos")) {
        char *pos = strtok_r(NULL, " \t", save);
        char *ws  = strtok_r(NULL, " \t", save);
        if (!strcmp(sub, "pos") && !pos) { reply(fd, "ERR usage: cam pos <tl|tr|bl|br|center|full>"); return true; }
        if (!cam_place(reg, pos, ws ? atoi(ws) : 0, 0, &err)) { reply(fd, "ERR %s", err); return true; }
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "off")) {
        collect(reg, &st);
        if (st.cam) registry_remove(reg, st.cam_id);
        reply(fd, "OK cam=off"); return true;
    }
    if (!strcmp(sub, "toggle")) {
        collect(reg, &st);
        if (st.cam) { registry_remove(reg, st.cam_id); reply(fd, "OK cam=off"); return true; }
        if (!cam_place(reg, NULL, 0, 0, &err)) { reply(fd, "ERR %s", err); return true; }
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "move")) {
        char *xs = strtok_r(NULL, " \t", save), *ys = strtok_r(NULL, " \t", save);
        if (!xs || !ys) { reply(fd, "ERR usage: cam move <x> <y>"); return true; }
        collect(reg, &st);
        if (!st.cam) { reply(fd, "ERR cam is off (use 'cam on')"); return true; }
        pthread_mutex_lock(&reg->lock);
        Overlay *o = registry_find(reg, st.cam_id);
        if (o) overlay_move(o, atoi(xs), atoi(ys));
        pthread_mutex_unlock(&reg->lock);
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "size")) {
        char *ws = strtok_r(NULL, " \t", save), *hs = strtok_r(NULL, " \t", save);
        if (!ws) { reply(fd, "ERR usage: cam size <w> [<h>]"); return true; }
        collect(reg, &st);
        if (!st.cam) { reply(fd, "ERR cam is off (use 'cam on')"); return true; }
        const char *a = anchor_name(st.cx, st.cy, st.cw, st.ch);
        if (!strcmp(a, "full")) a = "tr";    /* resizing leaves fullscreen */
        if (!cam_place(reg, a, atoi(ws), hs ? atoi(hs) : 0, &err)) { reply(fd, "ERR %s", err); return true; }
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "margin")) {
        char *ms = strtok_r(NULL, " \t", save);
        if (!ms) { reply(fd, "OK margin=%d", g_cam_margin); return true; }
        collect(reg, &st);
        const char *a = st.cam ? anchor_name(st.cx, st.cy, st.cw, st.ch) : NULL;
        g_cam_margin = atoi(ms) < 0 ? 0 : atoi(ms);
        if (a && strcmp(a, "custom") != 0 && strcmp(a, "full") != 0)
            cam_place(reg, a, st.cw, st.ch, &err);
        cam_reply(fd, reg); return true;
    }
    if (!strcmp(sub, "raise")) {
        collect(reg, &st);
        if (!st.cam) { reply(fd, "ERR cam is off"); return true; }
        registry_raise(reg, st.cam_id);
        cam_reply(fd, reg); return true;
    }
    reply(fd, "ERR cam: unknown subcommand (on|off|toggle|pos|move|size|margin|raise|get)");
    return true;
}

/* ══════════════════════════════════════════════════════════════
 * share
 * ══════════════════════════════════════════════════════════════ */
static void share_reply(int fd, OverlayRegistry *reg)
{
    KV kv[MAX_KV];
    int n = build_status(reg, kv);
    char buf[512]; size_t off = 0;
    off += (size_t)snprintf(buf, sizeof buf, "OK");
    for (int i = 0; i < n && off < sizeof buf; i++)
        if (!strncmp(kv[i].k, "share", 5) || !strcmp(kv[i].k, "cam") ||
            !strcmp(kv[i].k, "cam_pos") || !strcmp(kv[i].k, "cam_over_screen"))
            off += (size_t)snprintf(buf + off, sizeof buf - off, " %s=%s", kv[i].k, kv_plain(&kv[i]));
    reply(fd, "%s", buf);
}

static void share_stop(OverlayRegistry *reg)
{
    State st; collect(reg, &st);
    if (st.share) registry_remove(reg, st.share_id);
    g_share_geo[0] = g_share_mon[0] = '\0';
}

static void share_start(int fd, char **save, OverlayRegistry *reg)
{
    const char *sel = NULL, *campos = NULL, *err = NULL;
    bool contain = false;
    int camw = 0;
    char *t;
    while ((t = strtok_r(NULL, " \t", save))) {
        if      (!strncmp(t, "fit=", 4))     contain = !strcmp(t + 4, "contain");
        else if (!strncmp(t, "cam=", 4))     campos = t + 4;
        else if (!strncmp(t, "camw=", 5))    camw = atoi(t + 5);
        else if (!strncmp(t, "monitor=", 8)) sel = t + 8;
        else if (!strncmp(t, "geo=", 4))     sel = t + 4;
        else                                 sel = t;
    }
    if (g_output_w <= 0 || g_output_h <= 0) { reply(fd, "ERR output size unknown"); return; }

    char geo[64], mon[64];
    if (!resolve_share_target(sel, geo, sizeof geo, mon, sizeof mon, &err)) {
        reply(fd, "ERR %s", err); return;
    }
    int gw, gh, gx, gy;
    parse_geo(geo, &gw, &gh, &gx, &gy);

    int dx = 0, dy = 0, dw = g_output_w, dh = g_output_h;
    if (contain) {
        double sx = (double)g_output_w / gw, sy = (double)g_output_h / gh;
        double s  = sx < sy ? sx : sy;
        dw = (int)(gw * s) & ~1; dh = (int)(gh * s) & ~1;
        dx = (g_output_w - dw) / 2; dy = (g_output_h - dh) / 2;
    }

    share_stop(reg);   /* restart cleanly if already sharing */
    Overlay *o = overlay_create_screen(geo, dx, dy, dw, dh);
    if (!o) {
        reply(fd, "ERR screen capture failed (wf-recorder on Wayland / ffmpeg x11grab on X11 — see vcam stderr)");
        return;
    }
    if (!registry_add(reg, SHARE_ID, o)) {
        overlay_free(o);
        reply(fd, "ERR could not add share overlay (registry full?)");
        return;
    }
    registry_lower(reg, SHARE_ID);   /* screen is the background layer */
    snprintf(g_share_geo, sizeof g_share_geo, "%s", geo);
    snprintf(g_share_mon, sizeof g_share_mon, "%s", mon);

    if (campos && !strcmp(campos, "off")) {
        State st; collect(reg, &st);
        if (st.cam) registry_remove(reg, st.cam_id);
    } else if (campos || camw > 0) {
        if (!cam_place(reg, campos, camw, 0, &err)) { reply(fd, "ERR share on, but cam: %s", err); return; }
    } else {
        State st; collect(reg, &st);
        if (st.cam) registry_raise(reg, st.cam_id);
    }
    share_reply(fd, reg);
}

static bool share_handle(int fd, char **save, OverlayRegistry *reg)
{
    char *sub = strtok_r(NULL, " \t", save);
    if (!sub || !strcmp(sub, "status") || !strcmp(sub, "get")) { share_reply(fd, reg); return true; }
    if (!strcmp(sub, "start") || !strcmp(sub, "on")) { share_start(fd, save, reg); return true; }
    if (!strcmp(sub, "stop")  || !strcmp(sub, "off")) { share_stop(reg); share_reply(fd, reg); return true; }
    if (!strcmp(sub, "toggle")) {
        State st; collect(reg, &st);
        if (st.share) { share_stop(reg); share_reply(fd, reg); }
        else share_start(fd, save, reg);
        return true;
    }
    reply(fd, "ERR share: unknown subcommand (start|stop|toggle|status)");
    return true;
}

/* ══════════════════════════════════════════════════════════════
 * capture — live camera mode (format / resolution / fps)
 * ══════════════════════════════════════════════════════════════ */
#define MAX_MODES 128

static void capture_reply(int fd)
{
    CaptureState cs;
    capture_state_get(&cs);
    if (!cs.active) {
        reply(fd, "OK capture=off device=%s", capture_device_path());
        return;
    }
    reply(fd, "OK capture=on device=%s fmt=%s w=%d h=%d fps=%d fit=%s output=%dx%d",
          capture_device_path(), cs.mode.fmt, cs.mode.w, cs.mode.h, cs.mode.fps,
          cs.crop ? "crop" : "stretch", g_output_w, g_output_h);
}

static int parse_fit(const char *v)
{
    if (!strcmp(v, "crop"))    return 1;
    if (!strcmp(v, "stretch")) return 0;
    return -2;
}

static bool capture_handle(int fd, char **save)
{
    char *sub = strtok_r(NULL, " \t", save);
    char err[192] = "";

    if (g_no_input && sub && strcmp(sub, "get") != 0) {
        reply(fd, "ERR vcam runs with --no-input (no camera)");
        return true;
    }
    if (!sub || !strcmp(sub, "get") || !strcmp(sub, "status")) {
        if (g_no_input) reply(fd, "OK capture=off device=-");
        else            capture_reply(fd);
        return true;
    }

    if (!strcmp(sub, "formats") || !strcmp(sub, "modes")) {
        char *a = strtok_r(NULL, " \t", save);
        bool json = a && (!strcmp(a, "json") || !strcmp(a, "--json"));
        CaptureMode m[MAX_MODES];
        int n = capture_enum_modes(capture_device_path(), m, MAX_MODES);
        if (n < 0) { reply(fd, "ERR cannot open %s", capture_device_path()); return true; }
        if (!json) {
            if (n == 0) { reply(fd, "ERR camera reports no usable modes"); return true; }
            for (int i = 0; i < n; i++)
                reply(fd, "OK %s %dx%d %d", m[i].fmt, m[i].w, m[i].h, m[i].fps);
            return true;
        }
        char buf[3584]; size_t off = 0;
        off += (size_t)snprintf(buf, sizeof buf, "{\"ok\":true,\"formats\":[");
        for (int i = 0; i < n && off < sizeof buf; i++)
            off += (size_t)snprintf(buf + off, sizeof buf - off,
                                    "%s{\"fmt\":\"%s\",\"w\":%d,\"h\":%d,\"fps\":%d}",
                                    i ? "," : "", m[i].fmt, m[i].w, m[i].h, m[i].fps);
        if (off < sizeof buf) snprintf(buf + off, sizeof buf - off, "]}");
        reply(fd, "%s", buf);
        return true;
    }

    if (!strcmp(sub, "fit")) {
        char *v = strtok_r(NULL, " \t", save);
        int fit = v ? parse_fit(v) : -2;
        if (fit < 0) { reply(fd, "ERR usage: capture fit crop|stretch"); return true; }
        if (!capture_request(NULL, fit, err, sizeof err)) { reply(fd, "ERR %s", err); return true; }
        capture_reply(fd);
        return true;
    }

    if (!strcmp(sub, "set")) {
        const char *fmt = NULL;
        int w = 0, h = 0, fps = 0, fit = -1;
        char *t;
        while ((t = strtok_r(NULL, " \t", save))) {
            const char *v = t;
            if      (!strncmp(t, "fmt=", 4)) { fmt = t + 4; continue; }
            else if (!strncmp(t, "fps=", 4)) { fps = atoi(t + 4); if (fps <= 0) goto bad; continue; }
            else if (!strncmp(t, "fit=", 4)) { fit = parse_fit(t + 4); if (fit < 0) goto bad; continue; }
            else if (!strncmp(t, "res=", 4)) v = t + 4;
            char tail;
            if (!strcmp(v, "max"))                                   { w = h = -1; continue; }
            if (sscanf(v, "%dx%d%c", &w, &h, &tail) == 2 && w > 0 && h > 0) continue;
            if (v == t && strspn(t, "0123456789") == strlen(t))      { fps = atoi(t); if (fps <= 0) goto bad; continue; }
            if (v == t)                                              { fmt = t; continue; }
        bad:
            reply(fd, "ERR bad argument '%s' (capture set [<fmt>] [<WxH>|max] [<fps>] [fit=crop|stretch])", t);
            return true;
        }
        if (!fmt && !w && !fps && fit < 0) {
            reply(fd, "ERR usage: capture set [<fmt>] [<WxH>|max] [<fps>] [fit=crop|stretch]");
            return true;
        }

        CaptureState cs;
        capture_state_get(&cs);
        CaptureMode chosen;
        if (fmt || w || fps) {
            CaptureMode m[MAX_MODES];
            int n = capture_enum_modes(capture_device_path(), m, MAX_MODES);
            if (n <= 0) { reply(fd, "ERR cannot read camera modes from %s", capture_device_path()); return true; }
            if (!capture_choose(m, n, fmt, w, h, fps, cs.active ? &cs.mode : NULL,
                                g_output_fps, &chosen, err, sizeof err)) {
                reply(fd, "ERR %s", err);
                return true;
            }
        }
        bool reopen = fmt || w || fps;
        if (!capture_request(reopen ? &chosen : NULL, fit, err, sizeof err)) {
            reply(fd, "ERR %s", err);
            return true;
        }
        capture_reply(fd);
        return true;
    }

    reply(fd, "ERR capture: unknown subcommand (get|formats|set|fit)");
    return true;
}

/* ══════════════════════════════════════════════════════════════
 * entry point — returns true if the command was handled here
 * ══════════════════════════════════════════════════════════════ */
bool api_handle(int fd, const char *cmd, char **save, OverlayRegistry *reg)
{
    if (!strcmp(cmd, "status")) {
        char *a = strtok_r(NULL, " \t", save);
        cmd_status(fd, reg, a && (!strcmp(a, "json") || !strcmp(a, "--json")));
        return true;
    }
    if (!strcmp(cmd, "get"))     { cmd_get(fd, reg, strtok_r(NULL, " \t", save)); return true; }
    if (!strcmp(cmd, "version")) { reply(fd, "OK %s", api_version_string()); return true; }
    if (!strcmp(cmd, "share"))   return share_handle(fd, save, reg);
    if (!strcmp(cmd, "cam"))     return cam_handle(fd, save, reg);
    if (!strcmp(cmd, "capture")) return capture_handle(fd, save);
    if (!strcmp(cmd, "monitors")) {
        Monitor m[MAX_MON];
        int n = detect_monitors(m, MAX_MON);
        if (n == 0) { reply(fd, "ERR no monitor detected"); return true; }
        for (int i = 0; i < n; i++)
            reply(fd, "OK %d %s %dx%d+%d+%d", i + 1, m[i].name, m[i].w, m[i].h, m[i].x, m[i].y);
        return true;
    }
    return false;
}
