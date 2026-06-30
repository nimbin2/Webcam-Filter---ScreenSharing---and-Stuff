/*
 * IPC server – Unix domain socket at /tmp/vcam.sock
 *
 * Protocol: newline-terminated UTF-8 text commands.
 *
 * Commands:
 *   add img    <id> <x> <y> [<w> <h>] <file>
 *   add vid    <id> <x> <y> [<w> <h>] <file>
 *   add screen <id> <x> <y> [<w> <h>] <geometry>
 *   remove <id>
 *   move   <id> <x> <y>
 *   resize <id> <w> <h>
 *   list
 *   quit
 *
 * Every command gets a one-line response: "OK ..." or "ERR ...".
 */

#include "vcam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <fcntl.h>

/* ── helpers ─────────────────────────────────── */
static void send_line(int fd, const char *msg)
{
    size_t len = strlen(msg);
    char buf[512];
    size_t n = len < sizeof buf - 2 ? len : sizeof buf - 2;
    memcpy(buf, msg, n);
    buf[n]   = '\n';
    buf[n+1] = '\0';
    ssize_t r = write(fd, buf, n + 1);
    if (r < 0) { /* client disconnected – ignore */ }
}

/* Read a newline-terminated line from fd into buf (max len).
 * Returns number of bytes (excluding '\n'), or -1 on error/EOF. */
static int read_line(int fd, char *buf, int maxlen)
{
    int i = 0;
    while (i < maxlen - 1) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) return -1;
        if (c == '\n') break;
        if (c != '\r') buf[i++] = c;
    }
    buf[i] = '\0';
    return i;
}

/*
 * Parse optional size tokens before the final path/geometry argument.
 * Accepts:
 *   <w> <h> <path>   → both dimensions set
 *   <w>   <path>     → only w set (h=0, resolved from aspect ratio later)
 *   <path>           → neither set
 *
 * We decide by checking whether the token before the last is a number.
 * The last token is always the path/geometry.
 * Remaining tokens are collected into an array first.
 */
static void parse_size_and_path(char **save_ptr,
                                 int *out_w, int *out_h, char **out_path)
{
    char *toks[3] = { NULL, NULL, NULL };
    int n = 0;
    char *t;
    while (n < 3 && (t = strtok_r(NULL, " \t", save_ptr)) != NULL)
        toks[n++] = t;

    *out_w = 0; *out_h = 0; *out_path = NULL;

    if (n == 0) return;          /* no path */
    if (n == 1) {                /* just path */
        *out_path = toks[0]; return;
    }

    /* Check if toks[0] is numeric */
    char *ep;
    long v0 = strtol(toks[0], &ep, 10);
    bool t0_num = (*ep == '\0');

    if (n == 2) {
        if (t0_num) {            /* <w> <path> */
            *out_w = (int)v0;
            *out_path = toks[1];
        } else {                 /* <path> (extra token lost – shouldn't happen) */
            *out_path = toks[0];
        }
        return;
    }

    /* n == 3 */
    long v1 = strtol(toks[1], &ep, 10);
    bool t1_num = (*ep == '\0');

    if (t0_num && t1_num) {      /* <w> <h> <path> */
        *out_w = (int)v0; *out_h = (int)v1;
        *out_path = toks[2];
    } else if (t0_num) {         /* <w> <path> (toks[2] lost) */
        *out_w = (int)v0;
        *out_path = toks[1];
    } else {                     /* <path> (extra tokens lost) */
        *out_path = toks[0];
    }
}

/* ── command handler ─────────────────────────── */
extern FilterChain g_filters;  /* defined in main.c */

static void handle_command(int client_fd, char *line, OverlayRegistry *reg)
{
    char *save;
    char *cmd = strtok_r(line, " \t", &save);
    if (!cmd || cmd[0] == '#' || cmd[0] == '\0') {
        send_line(client_fd, "OK noop");
        return;
    }

    /* ── filter <subcmd> ──
     *   filter help
     *   filter add <name> [<param>]
     *   filter remove <index>
     *   filter clear
     *   filter list
     */
    if (strcmp(cmd, "filter") == 0) {
        char *sub = strtok_r(NULL, " \t", &save);
        if (!sub || strcmp(sub, "help") == 0) {
            send_line(client_fd, "OK filter commands:");
            send_line(client_fd, "OK   filter list                    show active filters");
            send_line(client_fd, "OK   filter add <name> [<param>]    append a filter");
            send_line(client_fd, "OK   filter set <index> <param>     update a filter's parameter");
            send_line(client_fd, "OK   filter remove <index>          remove filter at index");
            send_line(client_fd, "OK   filter clear                   remove all filters");
            send_line(client_fd, "OK   filter help                    this message");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK basic filters:");
            send_line(client_fd, "OK   grayscale                      Rec.601 luma");
            send_line(client_fd, "OK   sepia                          warm brown tone");
            send_line(client_fd, "OK   invert                         negative colors");
            send_line(client_fd, "OK   mirror                         horizontal flip");
            send_line(client_fd, "OK   flip                           vertical flip");
            send_line(client_fd, "OK   pixelate <block>               mosaic, default block=8");
            send_line(client_fd, "OK   blur <radius>                  box blur, radius 1..6, default 2");
            send_line(client_fd, "OK   vignette <strength>            darken edges, 0..1, default 0.6");
            send_line(client_fd, "OK   brightness <mult>              multiply brightness, default 1.2");
            send_line(client_fd, "OK   contrast <mult>                contrast around 128, default 1.4");
            send_line(client_fd, "OK   posterize <levels>             reduce color levels, default 4");
            send_line(client_fd, "OK   edges <strength>               Sobel edge detection, 0..2, default 1.0");
            send_line(client_fd, "OK   saturate <factor>              chroma boost, 0=gray, 1=normal, 0..3, default 1.5");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK fancy filters:");
            send_line(client_fd, "OK   chromashift <offset>           VHS-style RGB channel split, default 5");
            send_line(client_fd, "OK   glitch <intensity>             animated random horizontal slices, 0..1, default 0.3");
            send_line(client_fd, "OK   kaleidoscope <segments>        radial mirror, default 6");
            send_line(client_fd, "OK   swirl <strength>               radial twist, default 1.0");
            send_line(client_fd, "OK   thermal                        heatmap palette (luma → fire colors)");
            send_line(client_fd, "OK   oldfilm                        sepia + grain + scratches + vignette");
            send_line(client_fd, "OK   wave <amplitude>               animated sinusoidal x-shift, default 10");
            send_line(client_fd, "OK   cartoon <edge_thresh>          posterize + dark edges, default 50");
            send_line(client_fd, "OK   solarize <threshold>           invert pixels above value, default 128");
            send_line(client_fd, "OK   rainbow <speed>                animated hue rotation, default 1.0");
            send_line(client_fd, "OK   crt <strength>                 CRT scanlines + barrel + RGB fringe, 0..1, default 0.7");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK soft & dreamy filters:");
            send_line(client_fd, "OK   dream <strength>               hazy soft glow, 0..1, default 0.5");
            send_line(client_fd, "OK   bloom <strength>               highlight glow, 0..1, default 0.6");
            send_line(client_fd, "OK   glow <intensity>               gentle gaussian glow, 0..1, default 0.4");
            send_line(client_fd, "OK   pinch <strength>               radial inward pull, 0..1, default 0.5");
            send_line(client_fd, "OK   bulge <strength>               fisheye-like outward push, 0..1, default 0.5");
            send_line(client_fd, "OK   painterly <radius>             oil-painting effect, 1..5, default 3");
            send_line(client_fd, "OK   velvet <strength>              soft desat + warm tint + blur, 0..1, default 0.6");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK themed looks:");
            send_line(client_fd, "OK   modern <strength>              teal-orange cinematic grade, default 0.7");
            send_line(client_fd, "OK   cyberpunk <intensity>          neon contrast + chromashift, default 0.7");
            send_line(client_fd, "OK   romantic <strength>            soft warm haze, default 0.7");
            send_line(client_fd, "OK   vangogh <radius>               heavy painterly + saturation, 1..5, default 3");
            send_line(client_fd, "OK   picasso <segments>             cubist: kaleidoscope+poster+edges, 4..16, default 8");
            send_line(client_fd, "OK   noir <strength>                high-contrast B&W + split-tone, 0..1, default 0.8");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK mathematical patterns:");
            send_line(client_fd, "OK   sine <amplitude>               sin-wave x/y distortion, default 15");
            send_line(client_fd, "OK   xor <strength>                 (x^y) bitwise overlay, 0..1, default 0.5");
            send_line(client_fd, "OK   modulo <divisor>               color mod N, 2..128, default 64");
            send_line(client_fd, "OK   polar <strength>               polar-coordinate warp, 0..1, default 0.7");
            send_line(client_fd, "OK   voronoi <cells>                voronoi cell mosaic, 4..64, default 20");
            send_line(client_fd, "OK   julia <strength>               animated Julia fractal overlay, 0..1, default 0.5");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK pixel-crazy / animated:");
            send_line(client_fd, "OK   bitcrush <bits>                quantize bits per channel, 1..8, default 3");
            send_line(client_fd, "OK   melt <intensity>               animated downward drip, 0..1, default 0.5");
            send_line(client_fd, "OK   popart <strength>              Warhol 2x2 tinted quadrants, 0..1, default 1");
            send_line(client_fd, "OK   displace <strength>            luma-driven horizontal shift, 0..1, default 0.5");
            send_line(client_fd, "OK   ripple <strength>              animated water ripple from center, 0..1, default 0.5");
            send_line(client_fd, "OK   scanlines <gap>                black scanline every N rows, 2..16, default 3");
            return;
        }

        if (strcmp(sub, "list") == 0) {
            pthread_mutex_lock(&g_filters.lock);
            if (g_filters.count == 0) {
                send_line(client_fd, "OK (no filters)");
            } else {
                char buf[128];
                for (int i = 0; i < g_filters.count; i++) {
                    Filter *f = &g_filters.filters[i];
                    snprintf(buf, sizeof buf, "OK %d: %s param=%.2f",
                             i, filter_type_to_name(f->type), f->param);
                    send_line(client_fd, buf);
                }
            }
            pthread_mutex_unlock(&g_filters.lock);
            return;
        }

        if (strcmp(sub, "clear") == 0) {
            filter_chain_clear(&g_filters);
            send_line(client_fd, "OK all filters cleared");
            return;
        }

        if (strcmp(sub, "add") == 0) {
            char *name = strtok_r(NULL, " \t", &save);
            if (!name) { send_line(client_fd, "ERR filter add: missing name"); return; }
            FilterType t = filter_type_from_name(name);
            if (t == FX_UNKNOWN) {
                char b[128]; snprintf(b, sizeof b, "ERR unknown filter '%s'", name);
                send_line(client_fd, b); return;
            }
            float param = filter_default_param(t);
            char *pstr = strtok_r(NULL, " \t", &save);
            if (pstr) {
                char *ep; float v = strtof(pstr, &ep);
                if (*ep == '\0') param = v;
            }
            if (filter_chain_add(&g_filters, t, param)) {
                char b[128];
                snprintf(b, sizeof b, "OK added filter '%s' param=%.2f",
                         filter_type_to_name(t), param);
                send_line(client_fd, b);
            } else {
                send_line(client_fd, "ERR filter chain full");
            }
            return;
        }

        if (strcmp(sub, "set") == 0) {
            char *istr = strtok_r(NULL, " \t", &save);
            char *pstr = strtok_r(NULL, " \t", &save);
            if (!istr || !pstr) {
                send_line(client_fd, "ERR filter set: usage 'filter set <index> <param>'");
                return;
            }
            int idx = atoi(istr);
            char *ep; float v = strtof(pstr, &ep);
            if (*ep != '\0') { send_line(client_fd, "ERR filter set: bad number"); return; }
            pthread_mutex_lock(&g_filters.lock);
            if (idx < 0 || idx >= g_filters.count) {
                pthread_mutex_unlock(&g_filters.lock);
                send_line(client_fd, "ERR filter set: index out of range");
                return;
            }
            g_filters.filters[idx].param = v;
            const char *name = filter_type_to_name(g_filters.filters[idx].type);
            pthread_mutex_unlock(&g_filters.lock);
            char b[128]; snprintf(b, sizeof b, "OK %d: %s param=%.2f", idx, name, v);
            send_line(client_fd, b);
            return;
        }

        if (strcmp(sub, "remove") == 0) {
            char *arg = strtok_r(NULL, " \t", &save);
            if (!arg) {
                send_line(client_fd, "ERR filter remove: missing <name> or <index>");
                return;
            }
            int idx = -1;
            /* If the whole arg is digits, treat as index; otherwise as name. */
            bool all_digits = arg[0] != '\0';
            for (char *p = arg; *p; p++)
                if (*p < '0' || *p > '9') { all_digits = false; break; }

            if (all_digits) {
                idx = atoi(arg);
            } else {
                /* Look up filter type, then find it in the active chain. */
                FilterType t = filter_type_from_name(arg);
                if (t == FX_UNKNOWN) {
                    char b[128];
                    snprintf(b, sizeof b, "ERR filter remove: unknown filter '%s'", arg);
                    send_line(client_fd, b); return;
                }
                pthread_mutex_lock(&g_filters.lock);
                for (int i = 0; i < g_filters.count; i++)
                    if (g_filters.filters[i].type == t) { idx = i; break; }
                pthread_mutex_unlock(&g_filters.lock);
                if (idx < 0) {
                    char b[128];
                    snprintf(b, sizeof b, "ERR filter '%s' is not in the active chain", arg);
                    send_line(client_fd, b); return;
                }
            }
            if (filter_chain_remove(&g_filters, idx)) {
                char b[64];
                snprintf(b, sizeof b, "OK removed filter at index %d", idx);
                send_line(client_fd, b);
            } else {
                send_line(client_fd, "ERR no filter at that index");
            }
            return;
        }

        send_line(client_fd, "ERR filter: unknown subcommand (use add/remove/list/clear)");
        return;
    }

    /* ── help ── */
    if (strcmp(cmd, "help") == 0) {
        send_line(client_fd, "OK vcam commands:");
        send_line(client_fd, "OK   list                                 show active overlays (draw order)");
        send_line(client_fd, "OK   add img|vid|vid-once|screen|webcam <id> <x> <y> [<w> [<h>]] <file|geo|dev>");
        send_line(client_fd, "OK   remove <id>|all                      remove one or all overlays");
        send_line(client_fd, "OK   move   <id> [x=N|N] [y=N|N]          x=/y= or positional; 'keep'/'-' to skip");
        send_line(client_fd, "OK   resize <id> [w=N|N] [h=N|N]          w=/h= or positional; 'auto'/'-' = derive");
        send_line(client_fd, "OK   raise  <id>                          draw on top (last in list)");
        send_line(client_fd, "OK   lower  <id>                          draw at bottom (first in list)");
        send_line(client_fd, "OK   z      <id> <N>                      explicit z-index (0=bottom, -1=top)");
        send_line(client_fd, "OK   filter help                          list available filters");
        send_line(client_fd, "OK   filter list/add/remove/clear/set     manage filter chain");
        send_line(client_fd, "OK   quit                                 close the connection");
        return;
    }

    /* ── list ── */
    if (strcmp(cmd, "list") == 0) {
        pthread_mutex_lock(&reg->lock);
        if (reg->count == 0) {
            send_line(client_fd, "OK (no overlays)");
        } else {
            char buf[256];
            for (int i = 0; i < reg->count; i++) {
                OverlaySlot *s = &reg->slots[i];
                OverlayType t  = overlay_get_type(s->overlay);
                const char *type_str = t == OVERLAY_IMAGE  ? "img"    :
                                       t == OVERLAY_VIDEO  ? "vid"    :
                                       t == OVERLAY_WEBCAM ? "webcam" : "screen";
                int ox, oy, ow, oh;
                overlay_get_rect(s->overlay, &ox, &oy, &ow, &oh);
                snprintf(buf, sizeof buf,
                         "OK z=%d: [%s] id=%s pos=(%d,%d) size=%dx%d%s",
                         i, type_str, s->id, ox, oy, ow, oh,
                         (i == reg->count - 1) ? " (top)" :
                         (i == 0)              ? " (bottom)" : "");
                send_line(client_fd, buf);
            }
        }
        pthread_mutex_unlock(&reg->lock);
        return;
    }

    /* ── quit ── */
    if (strcmp(cmd, "quit") == 0) {
        send_line(client_fd, "OK bye");
        return; /* connection will be closed after return */
    }

    /* ── remove <id> ── */
    if (strcmp(cmd, "remove") == 0) {
        char *id = strtok_r(NULL, " \t", &save);
        if (!id) { send_line(client_fd, "ERR usage: remove <id>|all"); return; }

        if (strcmp(id, "all") == 0) {
            /* Snapshot the ID list under the lock, then remove each
             * (registry_remove takes the lock itself so we can't hold it). */
            pthread_mutex_lock(&reg->lock);
            int n = reg->count;
            char ids[MAX_OVERLAYS][64];
            for (int i = 0; i < n; i++) {
                strncpy(ids[i], reg->slots[i].id, sizeof ids[i] - 1);
                ids[i][sizeof ids[i] - 1] = '\0';
            }
            pthread_mutex_unlock(&reg->lock);
            for (int i = 0; i < n; i++) registry_remove(reg, ids[i]);
            char buf[64]; snprintf(buf, sizeof buf, "OK removed %d overlay(s)", n);
            send_line(client_fd, buf);
            return;
        }

        if (registry_remove(reg, id)) {
            char buf[128]; snprintf(buf, sizeof buf, "OK removed '%s'", id);
            send_line(client_fd, buf);
        } else {
            char buf[128]; snprintf(buf, sizeof buf, "ERR id '%s' not found", id);
            send_line(client_fd, buf);
        }
        return;
    }

    /* ── move <id> <args...> ──
     * Args can be positional (x [y]) or keyed (x=N y=N).
     * Each value can be a number, "keep" / "-" (default for move), or
     * blank.  Missing → keep current value.
     */
    if (strcmp(cmd, "move") == 0) {
        char *id = strtok_r(NULL, " \t", &save);
        if (!id) { send_line(client_fd, "ERR usage: move <id> [x=N|N] [y=N|N]"); return; }

        /* Collect remaining tokens */
        char *toks[8] = {0};
        int ntok = 0;
        char *t;
        while (ntok < 8 && (t = strtok_r(NULL, " \t", &save))) toks[ntok++] = t;

        /* Pull x=, y= out of any tokens that contain them; otherwise
         * positional (1st = x, 2nd = y). */
        char *x_str = NULL, *y_str = NULL;
        bool has_kv = false;
        for (int i = 0; i < ntok; i++) {
            if      (strncmp(toks[i], "x=", 2) == 0) { x_str = toks[i] + 2; has_kv = true; }
            else if (strncmp(toks[i], "y=", 2) == 0) { y_str = toks[i] + 2; has_kv = true; }
        }
        if (!has_kv) {
            if (ntok >= 1) x_str = toks[0];
            if (ntok >= 2) y_str = toks[1];
        }

        pthread_mutex_lock(&reg->lock);
        Overlay *o = registry_find(reg, id);
        if (!o) {
            pthread_mutex_unlock(&reg->lock);
            char buf[128]; snprintf(buf, sizeof buf, "ERR id '%s' not found", id);
            send_line(client_fd, buf); return;
        }
        int cx, cy, cw, ch; overlay_get_rect(o, &cx, &cy, &cw, &ch);

        /* Resolve "keep" / "-" / missing → current value */
        int x = cx, y = cy;
        if (x_str && *x_str && strcmp(x_str, "keep") != 0 && strcmp(x_str, "-") != 0)
            x = atoi(x_str);
        if (y_str && *y_str && strcmp(y_str, "keep") != 0 && strcmp(y_str, "-") != 0)
            y = atoi(y_str);

        overlay_move(o, x, y);
        pthread_mutex_unlock(&reg->lock);
        char buf[128]; snprintf(buf, sizeof buf, "OK moved '%s' to (%d,%d)", id, x, y);
        send_line(client_fd, buf);
        return;
    }

    /* ── resize <id> <args...> ──
     * Args can be positional (w [h]) or keyed (w=N h=N).
     * Each value can be a number, "keep", "auto" / "-" (default for
     * resize), or blank.  Missing → auto-derive from source aspect.
     */
    if (strcmp(cmd, "resize") == 0) {
        char *id = strtok_r(NULL, " \t", &save);
        if (!id) { send_line(client_fd, "ERR usage: resize <id> [w=N|N] [h=N|N]"); return; }

        char *toks[8] = {0};
        int ntok = 0;
        char *t;
        while (ntok < 8 && (t = strtok_r(NULL, " \t", &save))) toks[ntok++] = t;

        char *w_str = NULL, *h_str = NULL;
        bool has_kv = false;
        for (int i = 0; i < ntok; i++) {
            if      (strncmp(toks[i], "w=", 2) == 0) { w_str = toks[i] + 2; has_kv = true; }
            else if (strncmp(toks[i], "h=", 2) == 0) { h_str = toks[i] + 2; has_kv = true; }
        }
        if (!has_kv) {
            if (ntok >= 1) w_str = toks[0];
            if (ntok >= 2) h_str = toks[1];
        }

        pthread_mutex_lock(&reg->lock);
        Overlay *o = registry_find(reg, id);
        if (!o) {
            pthread_mutex_unlock(&reg->lock);
            char buf[128]; snprintf(buf, sizeof buf, "ERR id '%s' not found", id);
            send_line(client_fd, buf); return;
        }
        int cx, cy, cw, ch; overlay_get_rect(o, &cx, &cy, &cw, &ch);
        int src_w, src_h; overlay_get_src(o, &src_w, &src_h);

        /* Decode each arg into an explicit value:
         *   "keep" / unspecified-and-mode-is-positional → keep current
         *   "auto" / "-" / missing                       → derive from aspect
         *   <number>                                     → explicit
         */
        int w = -2, h = -2; /* -2 = unset, -1 = keep, 0 = auto, >0 = explicit */
        if (w_str) {
            if (*w_str == '\0' || strcmp(w_str, "auto") == 0 || strcmp(w_str, "-") == 0) w = 0;
            else if (strcmp(w_str, "keep") == 0) w = -1;
            else w = atoi(w_str);
        }
        if (h_str) {
            if (*h_str == '\0' || strcmp(h_str, "auto") == 0 || strcmp(h_str, "-") == 0) h = 0;
            else if (strcmp(h_str, "keep") == 0) h = -1;
            else h = atoi(h_str);
        }
        /* Default behavior for resize when arg is omitted = auto */
        if (w == -2) w = 0;
        if (h == -2) h = 0;
        /* Translate -1 (keep) into the current value */
        if (w == -1) w = cw;
        if (h == -1) h = ch;
        /* Translate 0 (auto) using source aspect */
        if (w <= 0 && h <= 0) {
            pthread_mutex_unlock(&reg->lock);
            send_line(client_fd, "ERR resize: at least one of w/h must be a number");
            return;
        }
        if (w <= 0 && src_h > 0) w = (int)((long)h * src_w / src_h);
        if (h <= 0 && src_w > 0) h = (int)((long)w * src_h / src_w);
        if (w <= 0 || h <= 0) {
            pthread_mutex_unlock(&reg->lock);
            send_line(client_fd, "ERR resize: invalid dimensions");
            return;
        }

        overlay_resize(o, w, h);
        pthread_mutex_unlock(&reg->lock);
        char buf[128]; snprintf(buf, sizeof buf, "OK resized '%s' to %dx%d", id, w, h);
        send_line(client_fd, buf);
        return;
    }

    /* ── raise <id> / lower <id> ── */
    if (strcmp(cmd, "raise") == 0 || strcmp(cmd, "lower") == 0) {
        char *id = strtok_r(NULL, " \t", &save);
        if (!id) {
            char b[128]; snprintf(b, sizeof b, "ERR usage: %s <id>", cmd);
            send_line(client_fd, b); return;
        }
        bool ok = (strcmp(cmd, "raise") == 0) ? registry_raise(reg, id)
                                              : registry_lower(reg, id);
        if (ok) {
            char b[128]; snprintf(b, sizeof b, "OK %sd '%s'", cmd, id);
            send_line(client_fd, b);
        } else {
            char b[128]; snprintf(b, sizeof b, "ERR id '%s' not found", id);
            send_line(client_fd, b);
        }
        return;
    }

    /* ── z <id> <N> ──
     * Move the overlay to position N in the draw order (0 = bottom,
     * count-1 = top). Negative N counts from the top (-1 = top).
     */
    if (strcmp(cmd, "z") == 0) {
        char *id = strtok_r(NULL, " \t", &save);
        char *sn = strtok_r(NULL, " \t", &save);
        if (!id || !sn) {
            send_line(client_fd, "ERR usage: z <id> <index>  (0=bottom, -1=top)");
            return;
        }
        int target = atoi(sn);
        if (!registry_set_z(reg, id, target)) {
            char b[128]; snprintf(b, sizeof b, "ERR id '%s' not found", id);
            send_line(client_fd, b); return;
        }
        char b[128]; snprintf(b, sizeof b, "OK '%s' moved in draw order", id);
        send_line(client_fd, b);
        return;
    }

    /* ── add <type> <id> <x> <y> [<w>] [<h>] <file|geometry> ── */
    if (strcmp(cmd, "add") == 0) {
        char *type = strtok_r(NULL, " \t", &save);
        char *id   = strtok_r(NULL, " \t", &save);
        char *sx   = strtok_r(NULL, " \t", &save);
        char *sy   = strtok_r(NULL, " \t", &save);

        if (!type || !id || !sx || !sy) {
            send_line(client_fd, "ERR usage: add <img|vid|screen> <id> <x> <y> [<w>] [<h>] <file>");
            return;
        }
        int x = atoi(sx), y = atoi(sy);
        int w = 0, h = 0;
        char *path = NULL;

        parse_size_and_path(&save, &w, &h, &path);

        if (!path && strcmp(type, "webcam") != 0) {
            send_line(client_fd, "ERR missing file/geometry argument");
            return;
        }

        Overlay *o = NULL;
        if (strcmp(type, "img") == 0) {
            o = overlay_create_image(path, x, y, w, h);
        } else if (strcmp(type, "vid") == 0) {
            o = overlay_create_video(path, x, y, w, h, true);   /* loop */
        } else if (strcmp(type, "vid-once") == 0) {
            o = overlay_create_video(path, x, y, w, h, false);  /* play once */
        } else if (strcmp(type, "screen") == 0) {
            o = overlay_create_screen(path, x, y, w, h);
        } else if (strcmp(type, "webcam") == 0) {
            o = overlay_create_webcam(path, x, y, w, h);
        } else {
            char buf[128]; snprintf(buf, sizeof buf, "ERR unknown type '%s'", type);
            send_line(client_fd, buf); return;
        }

        if (!o) {
            send_line(client_fd, "ERR failed to create overlay (check stderr)");
            return;
        }

        if (!registry_add(reg, id, o)) {
            overlay_free(o);
            char buf[128]; snprintf(buf, sizeof buf, "ERR id '%s' already exists", id);
            send_line(client_fd, buf);
            return;
        }

        char buf[256];
        int ox, oy, ow, oh;
        overlay_get_rect(o, &ox, &oy, &ow, &oh);
        snprintf(buf, sizeof buf, "OK added %s overlay '%s' at (%d,%d) size=%dx%d",
                 type, id, ox, oy, ow, oh);
        send_line(client_fd, buf);
        return;
    }

    char buf[256];
    snprintf(buf, sizeof buf, "ERR unknown command '%s'", cmd);
    send_line(client_fd, buf);
}

/* ── server thread ───────────────────────────── */
static void *ipc_thread(void *arg)
{
    IPCServer *srv = arg;

    while (srv->running) {
        struct sockaddr_un peer;
        socklen_t peer_len = sizeof peer;
        int client = accept(srv->server_fd, (struct sockaddr *)&peer, &peer_len);
        if (client < 0) {
            if (srv->running) perror("[ipc] accept");
            break;
        }

        char line[1024];
        int n;
        while ((n = read_line(client, line, sizeof line)) >= 0) {
            bool should_quit = (strcmp(line, "quit") == 0);
            handle_command(client, line, srv->registry);
            if (should_quit) break;
        }
        close(client);
    }
    return NULL;
}

/* ── public API ──────────────────────────────── */
bool ipc_start(IPCServer *srv, OverlayRegistry *reg)
{
    srv->registry = reg;
    srv->running  = true;

    unlink(VCAM_SOCK_PATH);

    srv->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv->server_fd < 0) { perror("[ipc] socket"); return false; }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, VCAM_SOCK_PATH, sizeof addr.sun_path - 1);

    if (bind(srv->server_fd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        perror("[ipc] bind"); close(srv->server_fd); return false;
    }
    chmod(VCAM_SOCK_PATH, 0600);

    if (listen(srv->server_fd, 4) < 0) {
        perror("[ipc] listen"); close(srv->server_fd); return false;
    }

    if (pthread_create(&srv->thread, NULL, ipc_thread, srv) != 0) {
        perror("[ipc] pthread_create"); close(srv->server_fd); return false;
    }

    fprintf(stderr, "[ipc] Listening on %s\n", VCAM_SOCK_PATH);
    return true;
}

void ipc_stop(IPCServer *srv)
{
    srv->running = false;
    shutdown(srv->server_fd, SHUT_RDWR);
    close(srv->server_fd);
    pthread_join(srv->thread, NULL);
    unlink(VCAM_SOCK_PATH);
}
