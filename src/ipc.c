/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

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
#include "vcam_cv.h"
#include "vcam_ort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

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
        if (n < 0 && errno == EINTR) continue;   /* signal — retry */
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
extern Recorder   *g_recorder; /* defined in main.c, may be NULL pre-init */
extern int         g_output_w;
extern int         g_output_h;
extern int         g_output_fps;


/* ── mode-name tables: the single source of truth for what each AI
 * filter's mode numbers mean. Surfaced via 'filter modes <name>' and
 * consumed by vcam-tune's picker. Keep in sync when adding modes. */
static const char *PERSONFX_MODES[] = {
    "blur-bg","spotlight","gradient-bg","image-bg","invisibility","neon-person",
    "clone-army","window-3d","chrono-clones","tiny-you","hologram","rainbow-aura",
    "painted-world","body-pixelate","gravity-flip","motion-echo","double-exposure",
    "glitch-body","sliced","disintegrate","light-paint","mirror-twin","crystal",
    "x-ray-scan","wanted-poster","stage-spotlight","haunted-shadow","cosmic-you",
    "paper-sticker","heat-signature","afterimage-burst","mandala","topo-aura",
    "ball-pit","orbit-rim-light","mind-vortex","stained-glass","upside-down-room",
    "snowfall","fire-silhouette","bubbles","thor-lightning","fireflies","stardust",
    "matrix-you","black-hole","pet-moons","portals","god-rays","tron-trace",
    "shockwave","rapture-beam","kaleido-body","liquid-floor","astral-soul",
    "mystic-fog","color-pop","cinema","bokeh-pro","studio-light","night-boost",
    "green-screen","brb-privacy","pop-out-3d","dolly-zoom","duotone-poster",
    "retro-sprite","acid-you","ascii-you","halftone-print","liquid-metal",
    "contour-stripes","static-you","golden-statue","comic-ink","venetian",
    "burning-edge","glass-body","blueprint","brick-tiles","shadow-caller",
    "glow-in-dark","graphite","pop-poster","riso-outlines","heat-haze",
    "security-cam","negative-you","vaporwave","emboss-relief","pixel-drip",
    "anaglyph-3d","chalk-outline","scan-bar","test-pattern","dither-1bit",
    "frost-edge","hollow-man","searchlight","pixel-sort","laser-grid",
    "oil-slick","neon-sign","rain-shadow","mirror-shatter","magnet-field",
    "presence-heat","grand-finale","keepy-uppy","bubble-pop","punching-bag",
    "goalkeeper","motion-paint","force-field","smoke-flow","ripple-pool",
    "cloth-net","swarm","sand-fall","bouncy-balls","liquid-pool","slime",
    "leaf-storm","iron-filings",
};
static const char *FACEFX_MODES[] = {
    "pixelate","ring-dots","googly-eyes","bighead","deal-with-it","laser-eyes",
    "bobblehead","third-eye","crown","mirror-face","cyclops","heart-eyes",
    "interrogation","target-lock","face-satellites","melting-face","sparkle-eyes",
    "halo","clown","auto-director","picasso","time-ripple","fisheye","golden-ratio",
    "face-image",
};
static const char *MOODFX_MODES[] = {
    "aura","hud","party","emoji","paintmood","auto-grade","mood-ring","fountain",
};
static const char *STYLEAI_MODES[] = {
    "mosaic","candy","rain-princess","udnie","pointilism",
};
static const char *DEPTHFX_MODES[] = {
    "depth-blur","depth-fog","depth-view","parallax",
};
static const char *POSEFX_MODES[] = {
    "skeleton","neon-bones","energy-hands","wrist-trails","light-wings",
    "puppet-strings",
};

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
            send_line(client_fd, "OK   filter list                    show active filters + all params");
            send_line(client_fd, "OK   filter add <name> <p> [<p2> [<p3> [<p4> [<p5>]]]]");
            send_line(client_fd, "OK                                    append a filter with up to 4 params");
            send_line(client_fd, "OK   filter set <index> <param>       update the primary param");
            send_line(client_fd, "OK   filter set <index> p2|p3|p4|p5 <v>  update a specific extra param (LIVE, no re-add)");
            send_line(client_fd, "OK   filter remove <index>          remove filter at index");
            send_line(client_fd, "OK   filter clear                   remove all filters");
            send_line(client_fd, "OK   filter help                    this message");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK filters marked (*) accept extra params — see the notes below each.");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK basic filters:");
            send_line(client_fd, "OK   grayscale                      Rec.601 luma");
            send_line(client_fd, "OK   sepia                          warm brown tone");
            send_line(client_fd, "OK   invert                         negative colors");
            send_line(client_fd, "OK   mirror                         horizontal flip");
            send_line(client_fd, "OK   flip                           vertical flip");
            send_line(client_fd, "OK   pixelate <block>               mosaic, default block=8");
            send_line(client_fd, "OK   blur <radius>                  box blur, radius 1..6, default 2");
            send_line(client_fd, "OK   vignette <strength> [<r> <warmth>](*) darken edges 0..1 (def 0.6); r=inner radius; warmth 0..1 = brown tint");
            send_line(client_fd, "OK   brightness <mult>              multiply brightness, default 1.2");
            send_line(client_fd, "OK   contrast <mult>                contrast around 128, default 1.4");
            send_line(client_fd, "OK   posterize <levels>             reduce color levels, default 4");
            send_line(client_fd, "OK   edges <strength>               Sobel edge detection, 0..2, default 1.0");
            send_line(client_fd, "OK   saturate <factor>              chroma boost, 0=gray, 1=normal, 0..3, default 1.5");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK fancy filters:");
            send_line(client_fd, "OK   chromashift <dx> [<dxb> <dy_r>](*) RGB split; dx=R shift (B=-dx unless dxb given); dy_r=R vertical shift");
            send_line(client_fd, "OK   glitch <intensity> [<slice_h> <shift_pct>](*) animated slice shift; slice_h in px, shift_pct=max shift as %-of-width");
            send_line(client_fd, "OK   kaleidoscope <segments> [<rot> <spin> <zoom>](*) radial mirror; spin deg/frame; zoom of the wedge");
            send_line(client_fd, "OK   swirl <strength> [<radius> <spin>](*) radial twist; radius 0.2..1 of frame; spin animates it");
            send_line(client_fd, "OK   thermal                        heatmap palette (luma → fire colors)");
            send_line(client_fd, "OK   oldfilm                        sepia + grain + scratches + vignette");
            send_line(client_fd, "OK   wave <amp> [<freq> <speed> <vert>](*) sinusoidal displacement; freq=waves/screen, speed=px/frame, vert=1 for vertical");
            send_line(client_fd, "OK   cartoon <edge_thresh> [<levels>](*) posterize + dark edges; levels 2..12 (def 5)");
            send_line(client_fd, "OK   solarize <threshold>           invert pixels above value, default 128");
            send_line(client_fd, "OK   rainbow <speed>                animated hue rotation, default 1.0");
            send_line(client_fd, "OK   crt <strength>                 CRT scanlines + barrel + RGB fringe, 0..1, default 0.7");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK soft & dreamy filters:");
            send_line(client_fd, "OK   dream <strength>               hazy soft glow, 0..1, default 0.5");
            send_line(client_fd, "OK   bloom <strength>               highlight glow, 0..1, default 0.6");
            send_line(client_fd, "OK   glow <intensity>               gentle gaussian glow, 0..1, default 0.4");
            send_line(client_fd, "OK   pinch <strength> [<cx> <cy>](*) radial inward pull; cx/cy = center 0..1 (off-center = gravity well)");
            send_line(client_fd, "OK   bulge <strength> [<cx> <cy>](*) fisheye push; cx/cy = center 0..1");
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
            send_line(client_fd, "OK");
            send_line(client_fd, "OK glitch art / generative:");
            send_line(client_fd, "OK   shuffle <intensity> [<size> <phase_s>](*) shuffle/unshuffle blocks; size 16..96 px; phase seconds 0.3..5");
            send_line(client_fd, "OK   pixelsort <density>            glitch-art row sort by luminance, 0..1, default 0.3");
            send_line(client_fd, "OK   rgbdelay <strength>            chromatic lag — G & R from past frames, 0..1, default 0.7");
            send_line(client_fd, "OK   drift <band_h> [<max_shift>](*) scanline drift; max shift 5..120 px");
            send_line(client_fd, "OK   rain <density> [<speed> <trail>](*) Matrix rain; speed mult 0.3..3; trail length 8..80 px");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK temporal / time-bending:");
            send_line(client_fd, "OK   echo <persistence>             motion trails (ghosting), 0..0.97, default 0.6");
            send_line(client_fd, "OK   slitscan <depth> [<dir>](*)    rows sample older frames; dir 1 = time flows upward");
            send_line(client_fd, "OK   mosh <sensitivity> [<keyint>](*) datamosh; keyint = frames between resyncs 30..600 (def 90)");
            send_line(client_fd, "OK   stutter <period>               freeze each frame for N frames, 2..30, default 6");
            send_line(client_fd, "OK   halftone <cell>                newspaper dot grid, cell 4..32 px, default 8");
            send_line(client_fd, "OK   neon <strength> [<thresh> <huespd> <glow>](*) glowing rainbow edges + real bloom; thresh 10..120");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK retro & warp:");
            send_line(client_fd, "OK   emboss <strength>              carved gray relief, 0.1..2, default 1");
            send_line(client_fd, "OK   dither <levels>                ordered Bayer dither, levels/channel 2..8, default 2 (retro 8-color)");
            send_line(client_fd, "OK   tiles <n>                      n x n mirror-tile wallpaper grid, 2..8, default 3");
            send_line(client_fd, "OK   zoompulse <strength> [<rate>](*) breathing zoom; rate = oscillation speed");
            send_line(client_fd, "OK   oilslick <strength> [<speed>](*) iridescent hue swirl; speed = drift rate");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK interactive & extravagant:");
            send_line(client_fd, "OK   feedback <decay> [<zoom> <rot>](*) infinite video-feedback tunnel; zoom/frame (def 1.04), deg/frame (def 1.5)");
            send_line(client_fd, "OK   ripples <sens> [<rain> <refract>](*) INTERACTIVE water sim; rain=drops/sec ambient, refract 0..8");
            send_line(client_fd, "OK   lightpaint <fade> [<thresh> <huespd>](*) INTERACTIVE painting; thresh 10..120; hue cycle rate");
            send_line(client_fd, "OK   hologram <strength> [<gap> <glitch>](*) sci-fi projection; scanline gap 2..8; glitch rate 0..1");
            send_line(client_fd, "OK   life <density> [<color> <speed>](*) Conway's Life on your edges; color 1=age-rainbow; speed=frames/gen");
            send_line(client_fd, "OK   mathwarp <amp>                 warp AND recolor by YOUR math");
            send_line(client_fd, "OK                                    warp:  filter expr <dx> <dy>   (vars x y w h t r a)");
            send_line(client_fd, "OK                                    color: filter color <r> <g> <b> (adds cr cg cb = src color)");
            send_line(client_fd, "OK                                    fns sin cos abs sqrt floor; ops + - * / % ; cmp; ternary c?a:b");
            send_line(client_fd, "OK                                    e.g. filter expr (x-w/2)*(sin(t*5)>0?0.08:0) 0");
            send_line(client_fd, "OK                                         filter color cg cb cr   (swap RGB channels)");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK demoscene & particles:");
            send_line(client_fd, "OK   plasma <blend> [<speed> <scale>](*) classic demoscene plasma over the video, blend 0..0.9");
            send_line(client_fd, "OK   contour <levels> [<dim>](*)    glowing topographic iso-lines of your luma, 3..24 bands");
            send_line(client_fd, "OK   fireworks <sens> [<burst>](*)  INTERACTIVE: motion detonates particle bursts w/ gravity");
            send_line(client_fd, "OK   infinity <scale> [<rot>](*)    recursive picture-in-picture (droste), scale 0.2..0.85");
            send_line(client_fd, "OK   moire <spacing> [<speed>](*)   two orbiting ring systems interfere over the image");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK text, tone & tracking:");
            send_line(client_fd, "OK   ascii <cell> [<mode>](*)       REAL ASCII-art from an 8x8 font; cell 8|16; mode 0=green 1=color 2=white");
            send_line(client_fd, "OK   duotone <hueA> [<hueB> <mix>](*) two-color gradient map; hues in degrees (def 220/30 = teal-orange)");
            send_line(client_fd, "OK   sketch <detail> [<darkness>](*) pencil sketch via dodge blend; detail 1..4");
            send_line(client_fd, "OK   spotlight <radius> [<dark> <speed>](*) INTERACTIVE: the light FOLLOWS your motion");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK the AI-craze set:");
            send_line(client_fd, "OK   predator <vis> [<shimmer> <adapt>](*) camouflage cloak — hold still and VANISH; motion shimmers");
            send_line(client_fd, "OK   shards <count> [<offset>](*)  broken mirror: voronoi shards with crack lines, 6..64");
            send_line(client_fd, "OK   portal <radius> [<sep>](*)    two ringed portals that swap their contents (orange/blue)");
            send_line(client_fd, "OK   heattrail <gain> [<cool>](*)  motion leaves cooling thermal glow (fire palette)");
            send_line(client_fd, "OK   wormhole <depth> [<speed>](*) fly down a tunnel textured with your own live image");
            send_line(client_fd, "OK   magnet <strength> [<follow>](*) the frame is pulled toward the brightest light");
            send_line(client_fd, "OK   liquid <amp> [<scale> <speed>](*) the whole image flows like disturbed water");
            send_line(client_fd, "OK   nightvision <gain> [<noise>](*) gen-3 intensifier: green amp + noise + vignette");
            send_line(client_fd, "OK   badtv <amount> [<roll>](*)   dying CRT: roll, tearing, static bursts, chroma wobble");
            send_line(client_fd, "OK   ghosts <delay> [<gain>](*)   motion extraction: static = gray, movement = vivid spectres");
            send_line(client_fd, "OK");
            send_line(client_fd, "OK AI-craze volume 2:");
            send_line(client_fd, "OK   clones <count> [<interval> <alpha>](*) freeze-frames of you populate the room (multiplicity)");
            send_line(client_fd, "OK   meltdown <speed> [<chaos>](*) Doom screen-melt: the frozen frame slides off, revealing live");
            send_line(client_fd, "OK   rotozoom <speed> [<zoomamp>](*) demoscene rotozoomer: infinite rotating tiled plane of you");
            send_line(client_fd, "OK   starfield <density> [<speed>](*) hyperspace stars streak over the video");
            send_line(client_fd, "OK   lightning <rate> [<branches>](*) bolts STRIKE WHERE YOU MOVE, with flash");
            send_line(client_fd, "OK   scatter <power> [<heal>](*)   motion shatters pixels outward; the frame heals itself");
            send_line(client_fd, "OK   vhs <wear> [<timecode>](*)    chroma bleed + head-switch bar + live counting timecode");
            send_line(client_fd, "OK   outline <thick> [<huespd>](*) marching-ants lasso traces whatever MOVES");
            send_line(client_fd, "OK   symmetry <mode>               face symmetry: 0=left mirrored, 1=right, 2=quad");
            send_line(client_fd, "OK   timebubble <delay_s> [<radius>](*) a drifting bubble shows THE PAST inside it");
            send_line(client_fd, "OK");
            if (cv_available()) {
                char cvline[256];
                snprintf(cvline, sizeof cvline, "OK OpenCV-backed (%s):", cv_engine_info());
                send_line(client_fd, cvline);
            } else {
                send_line(client_fd, "OK OpenCV-backed (NOT in this build — rebuild with 'make OPENCV=1' to enable):");
            }
            send_line(client_fd, "OK   cvcartoon <strength>          edge-preserving cartoon with ink outlines");
            send_line(client_fd, "OK   cvpencil <color>              pencil sketch: 1=color, 0=graphite");
            send_line(client_fd, "OK   cvstylize <strength>          watercolor / poster stylization");
            send_line(client_fd, "OK   opticalflow <mix>             motion visualized as flowing color");
            send_line(client_fd, "OK   bgsub <sensitivity>           moving foreground vivid, static background dimmed");
            send_line(client_fd, "OK   facefx <mode> [<amount> <speed>] YuNet faces: 0=blur 1=ring+dots 2=googly-eyes (needs model)");
            send_line(client_fd, "OK                                    amount 0.3..2.5 = effect size; speed 0.2..3 = animation rate");
            send_line(client_fd, "OK   personfx <mode> [<edge> <smooth> <size>] person matting: 0=blur-bg 1=spotlight 2=gradient-bg");
            send_line(client_fd, "OK                                    edge 0.05..0.5 (lower=more hair, higher=tighter cut, def 0.22)");
            send_line(client_fd, "OK                                    smooth 0.4..1 (mask responsiveness, def 0.8)");
            send_line(client_fd, "OK                                    size 256..640 (RVM detail vs speed, def 512; try 384 on slower CPUs)");
            send_line(client_fd, "OK                                    3=YOUR-image-bg (filter bgimage <path>) 4=INVISIBILITY CLOAK 5=neon person");
            send_line(client_fd, "OK   styleai <style> [<blend> <size>] LIVE NEURAL STYLE TRANSFER: 0=mosaic 1=candy 2=rain-princess 3=udnie 4=pointilism");
            send_line(client_fd, "OK                                    blend 0..1 mixes with original; size 96..384 = quality vs speed");
            send_line(client_fd, "OK   moodfx <mode> [<gain> <extra>] EMOTION-REACTIVE: model reads your expression, effect responds");
            send_line(client_fd, "OK                                    0=aura 1=HUD 2=party 3=EMOJI 4=PAINTMOOD 5=AUTO-GRADE 6=MOOD-RING 7=EMOTION-FOUNTAIN");
            send_line(client_fd, "OK   paintflow <strength> [<decay>] motion smears reality like wet paint (dense optical flow)");
            send_line(client_fd, "OK   facefx extra modes: 3=BIGHEAD (inflated tracked head) 4=deal-with-it sunglasses");
            send_line(client_fd, "OK   personfx extra modes: 6=CLONE ARMY 7=WINDOW-3D (head-coupled parallax) 8=CHRONO CLONES (past-selves trail)");
            send_line(client_fd, "OK   facefx extra modes: 5=LASERS 6=BOBBLE 7=3RD-EYE 8=CROWN 9=MIRROR 10=CYCLOPS 11=HEART-EYES 12=INTERROGATE 13=TARGET-LOCK");
            send_line(client_fd, "OK                       14=FACE-SATELLITES 15=MELTING-FACE 16=SPARKLE-EYES 17=HALO 18=CLOWN");
            send_line(client_fd, "OK                       19=AUTO-DIRECTOR (camera follows + frames your face; amount=tightness speed=follow)");
            send_line(client_fd, "OK                       20=PICASSO 21=TIME-RIPPLE 22=FISHEYE 23=GOLDEN-RATIO 24=FACE-IMAGE (your PNG tracks your head)");
            send_line(client_fd, "OK   filter faceimage <path|off>    set the PNG for facefx 24 — ANY animal face / mask / meme with transparency");
            send_line(client_fd, "OK   genai [<blend> <size>]        NEURAL img2img: run ANY 1-in/1-out ONNX (AnimeGAN, CartoonGAN...) on every frame");
            send_line(client_fd, "OK                                    via ONNX Runtime; model: filter genmodel <path> (or <models>/img2img.onnx)");
            send_line(client_fd, "OK                                    blend 0.2..1 (def 1)  size 128..512 (def 288)  p3: 0=[-1,1] GAN, 1=[0,1] models");
            send_line(client_fd, "OK                                    p3=1 fits AI low-light nets (Zero-DCE class): filter genmodel <path>, add genai 1 288 1");
            send_line(client_fd, "OK   depthfx <mode> [<fx>]         MONOCULAR DEPTH (MiDaS): every pixel gets a distance (vcam-setup --models-depth)");
            send_line(client_fd, "OK                                    0=DEPTH-BLUR (true DoF) 1=DEPTH-FOG 2=DEPTH-VIEW 3=PARALLAX-WIGGLE; fx=strength");
            send_line(client_fd, "OK   enhance [<strength> <denoise>] LOW-LIGHT RESCUE: retinex shadow lift + auto white balance + shadow denoise");
            send_line(client_fd, "OK                                    strength 0.2..2 (def 1), denoise 0..1 — fixes bad lighting, no model needed");
            send_line(client_fd, "OK   posefx <mode> [<fx>]          BODY LANDMARKS (BlazePose): vcam sees your skeleton (vcam-setup --models-pose)");
            send_line(client_fd, "OK                                    0=SKELETON 1=NEON-BONES 2=ENERGY-HANDS (merge palms!) 3=WRIST-TRAILS");
            send_line(client_fd, "OK                                    4=LIGHT-WINGS (beating, on your shoulders) 5=PUPPET-STRINGS");
            send_line(client_fd, "OK   personfx extra modes: 9=TINY 10=HOLOGRAM 11=RAINBOW-AURA 12=PAINTED-WORLD 13=BODY-PIXELATE 14=GRAVITY-FLIP");
            send_line(client_fd, "OK                                    15=MOTION-ECHO 16=DOUBLE-EXPOSURE 17=GLITCH-BODY 18=SLICED 19=DISINTEGRATE");
            send_line(client_fd, "OK                                    20=LIGHT-PAINT 21=MIRROR-TWIN 22=CRYSTAL 23=X-RAY-SCAN 24=WANTED-POSTER");
            send_line(client_fd, "OK                                    25=STAGE-SPOTLIGHT 26=HAUNTED-SHADOW (your shadow lags 1s) 27=COSMIC-YOU");
            send_line(client_fd, "OK                                    28=PAPER-STICKER 29=HEAT-SIGNATURE 30=AFTERIMAGE-BURST (dodge fast!)");
            send_line(client_fd, "OK                                    31=MANDALA (kaleidoscoped outline) 32=TOPO-AURA 33=BALL-PIT (balls bounce off YOU!)");
            send_line(client_fd, "OK                                    34=ORBIT-RIM-LIGHT 35=MIND-VORTEX 36=STAINED-GLASS 37=UPSIDE-DOWN-ROOM");
            send_line(client_fd, "OK                                    38=SNOWFALL (piles up on you!) 39=FIRE-SILHOUETTE 40=BUBBLES (pop them!) 41=THOR");
            send_line(client_fd, "OK                                    42=FIREFLIES 43=STARDUST 44=MATRIX-YOU 45=BLACK-HOLE 46=PET-MOONS 47=PORTALS");
            send_line(client_fd, "OK                                    48=GOD-RAYS 49=TRON-TRACE 50=SHOCKWAVE (move fast!) 51=RAPTURE-BEAM");
            send_line(client_fd, "OK                                    52=KALEIDO-BODY 53=LIQUID-FLOOR 54=ASTRAL-SOUL 55=MYSTIC-FOG");
            send_line(client_fd, "OK                                    PRO LOOKS: 56=COLOR-POP 57=CINEMA 58=BOKEH-PRO 59=STUDIO-LIGHT 60=NIGHT-BOOST");
            send_line(client_fd, "OK                                    61=GREEN-SCREEN-OUT 62=BRB-PRIVACY 63=POP-OUT-3D 64=DOLLY-ZOOM 65=DUOTONE");
            send_line(client_fd, "OK                                    66=RETRO-SPRITE 67=ACID-YOU 68=ASCII-YOU 69=HALFTONE-PRINT 70=LIQUID-METAL");
            send_line(client_fd, "OK                                    71=CONTOUR-STRIPES 72=STATIC-YOU (signal lost) 73=GOLDEN-STATUE 74=COMIC-INK");
            send_line(client_fd, "OK                                    75=VENETIAN 76=BURNING-EDGE 77=GLASS-BODY 78=BLUEPRINT 79=BRICK-TILES");
            send_line(client_fd, "OK                                    80=SHADOW-CALLER 81=GLOW-IN-DARK 82=GRAPHITE 83=POP-POSTER 84=RISO-OUTLINES");
            send_line(client_fd, "OK                                    85=HEAT-HAZE 86=SECURITY-CAM 87=NEGATIVE-YOU 88=VAPORWAVE 89=EMBOSS-RELIEF");
            send_line(client_fd, "OK                                    90=PIXEL-DRIP 91=ANAGLYPH-3D 92=CHALK-OUTLINE 93=SCAN-BAR 94=TEST-PATTERN");
            send_line(client_fd, "OK                                    95=DITHER-1BIT 96=FROST-EDGE 97=HOLLOW-MAN 98=SEARCHLIGHT (hunts you)");
            send_line(client_fd, "OK                                    99=PIXEL-SORT 100=LASER-GRID 101=OIL-SLICK 102=NEON-SIGN 103=RAIN-SHADOW");
            send_line(client_fd, "OK                                    104=MIRROR-SHATTER 105=MAGNET-FIELD 106=PRESENCE-HEAT 107=GRAND-FINALE");
            send_line(client_fd, "OK   -- INTERACTIVE (your body is the collider, your motion is the force) --");
            send_line(client_fd, "OK                                    108=KEEPY-UPPY: head/hand the ball, keep it up (fx=gravity)");
            send_line(client_fd, "OK                                    109=BUBBLE-POP: pop rising bubbles, chain combos (fx=rate)");
            send_line(client_fd, "OK                                    110=PUNCHING-BAG: punch it, it swings; POWER meter (fx=bag weight)");
            send_line(client_fd, "OK                                    111=GOALKEEPER: block incoming shots with your body (fx=shot rate)");
            send_line(client_fd, "OK                                    112=MOTION-PAINT: your movement paints the room (fx=drying)");
            send_line(client_fd, "OK                                    113=FORCE-FIELD: shove a particle field around (fx=strength)");
            send_line(client_fd, "OK                                    114=SMOKE-FLOW: real fluid sim — stir currents with your hands");
            send_line(client_fd, "OK                                    115=RIPPLE-POOL: wave equation — your moves drop ripples that refract the room");
            send_line(client_fd, "OK                                    116=CLOTH-NET: a Verlet spring net you push, stretch and wobble");
            send_line(client_fd, "OK                                    117=SWARM: boids that flock, and scatter from your hands");
            send_line(client_fd, "OK                                    118=SAND-FALL: grains pile on your shoulders — shake them off");
            send_line(client_fd, "OK                                    119=BOUNCY-BALLS: 64 balls, real ball-to-ball collisions, bat them around");
            send_line(client_fd, "OK                                    120=LIQUID-POOL: liquid that pools, sloshes and splashes — scoop it");
            send_line(client_fd, "OK                                    121=SLIME: sticky goo clings to you, sags in strands, flings off when you move");
            send_line(client_fd, "OK                                    122=LEAF-STORM: leaves caught in the wake of your movement; kick the piles up");
            send_line(client_fd, "OK                                    123=IRON-FILINGS: you are the magnet — filings cling to your field and flow around you");
            send_line(client_fd, "OK   filter modes <name>            list ALL modes of personfx/facefx/moodfx/styleai BY NAME");
            send_line(client_fd, "OK   personfx p5 = fx  per-mode effect strength 0.2..3 (glitch amp, fire, rays, lift, fog...) — tune LIVE:");
            send_line(client_fd, "OK                       vcam-ctl filter set 0 p5 1.8");
            send_line(client_fd, "OK   speedlines <sens> [<density>] anime motion lines streak in when you move fast");
            send_line(client_fd, "OK   filter bgimage <path|off>      set the background image for personfx mode 3");
            send_line(client_fd, "OK   (face/person models: run 'vcam-setup --models' to download)");
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
                    snprintf(buf, sizeof buf,
                        "OK %d: %s param=%.2f p2=%.2f p3=%.2f p4=%.2f p5=%.2f",
                        i, filter_type_to_name(f->type),
                        f->param, f->param2, f->param3, f->param4, f->param5);
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
            /* Parse 0–4 numeric params. Missing ones default: primary
             * → filter_default_param(t); the rest → 0. */
            float params[FILTER_MAX_PARAMS] = {
                filter_default_param(t), 0.0f, 0.0f, 0.0f
            };
            int n_parsed = 0;
            for (int p = 0; p < FILTER_MAX_PARAMS; p++) {
                char *pstr = strtok_r(NULL, " \t", &save);
                if (!pstr) break;
                char *ep; float v = strtof(pstr, &ep);
                if (*ep != '\0') {
                    char b[128];
                    snprintf(b, sizeof b, "ERR filter add: '%s' isn't a number", pstr);
                    send_line(client_fd, b);
                    return;
                }
                params[p] = v;
                n_parsed++;
            }
            if (n_parsed == 0) n_parsed = 1;  /* use the default */

            if (filter_chain_add_multi(&g_filters, t, params, n_parsed)) {
                char b[192];
                snprintf(b, sizeof b,
                    "OK added filter '%s' param=%.2f p2=%.2f p3=%.2f p4=%.2f p5=%.2f",
                    filter_type_to_name(t),
                    params[0], params[1], params[2], params[3], params[4]);
                send_line(client_fd, b);
            } else {
                send_line(client_fd, "ERR filter chain full");
            }
            return;
        }

        if (strcmp(sub, "expr") == 0) {
            /* filter expr <dx-expression> <dy-expression>
             * NO SPACES within an expression. Vars: x y w h t r a.
             * Fns: sin cos abs sqrt. Applies to the mathwarp filter. */
            char *dxs = strtok_r(NULL, " \t", &save);
            char *dys = strtok_r(NULL, " \t", &save);
            if (!dxs || !dys) {
                send_line(client_fd, "ERR usage: filter expr <dx> <dy>   (no spaces inside expressions)");
                send_line(client_fd, "ERR   vars: x y w h t r a   fns: sin cos abs sqrt floor");
                send_line(client_fd, "ERR   ops: + - * / % ( )  comparisons > < >= <= == !=  ternary c?a:b");
                send_line(client_fd, "ERR   example: filter expr (x-w/2)*(sin(t*5)>0?0.08:0) (y-h/2)*(sin(t*5)>0?0.08:0)");
                return;
            }
            char err[128] = "";
            if (!mathwarp_set_expr(dxs, dys, err, sizeof err)) {
                char b[192];
                snprintf(b, sizeof b, "ERR expression: %s", err);
                send_line(client_fd, b);
                return;
            }
            send_line(client_fd, "OK mathwarp expressions updated (add the 'mathwarp' filter to see them)");
            return;
        }

        if (strcmp(sub, "color") == 0) {
            /* filter color <r-expr> <g-expr> <b-expr>   → recolor pixels
             * filter color off                          → disable recolor
             * Each expr returns a channel value 0..255 and may read the
             * source pixel color as cr cg cb (plus x y w h t r a). */
            char *rs = strtok_r(NULL, " \t", &save);
            char *gs = strtok_r(NULL, " \t", &save);
            char *bs = strtok_r(NULL, " \t", &save);
            if (rs && !gs && (!strcmp(rs, "off") || !strcmp(rs, "none"))) {
                mathwarp_clear_color();
                send_line(client_fd, "OK mathwarp recolor disabled");
                return;
            }
            if (!rs || !gs || !bs) {
                send_line(client_fd, "ERR usage: filter color <r> <g> <b>   (each 0..255, no spaces in exprs)");
                send_line(client_fd, "ERR   extra vars: cr cg cb = this pixel's source R/G/B (0..255)");
                send_line(client_fd, "ERR   plus the usual x y w h t r a; fns sin cos abs sqrt floor; ops + - * / % ?:");
                send_line(client_fd, "ERR   'filter color off' disables recolor");
                send_line(client_fd, "ERR   example: filter color cg cb cr   (rotate channels)");
                return;
            }
            char err[128] = "";
            if (!mathwarp_set_color(rs, gs, bs, err, sizeof err)) {
                char b[192];
                snprintf(b, sizeof b, "ERR color expression: %s", err);
                send_line(client_fd, b);
                return;
            }
            send_line(client_fd, "OK mathwarp recolor updated (needs the 'mathwarp' filter active)");
            return;
        }

        if (strcmp(sub, "modes") == 0) {
            /* filter modes <name> — list an AI filter's modes by name */
            char *fname = strtok_r(NULL, " \t", &save);
            const char **tbl = NULL; int n2 = 0;
            if (fname && (!strcmp(fname,"personfx")||!strcmp(fname,"portrait")))
                { tbl = PERSONFX_MODES; n2 = (int)(sizeof PERSONFX_MODES/sizeof *PERSONFX_MODES); }
            else if (fname && (!strcmp(fname,"facefx")||!strcmp(fname,"faces")))
                { tbl = FACEFX_MODES; n2 = (int)(sizeof FACEFX_MODES/sizeof *FACEFX_MODES); }
            else if (fname && (!strcmp(fname,"moodfx")||!strcmp(fname,"emotion")))
                { tbl = MOODFX_MODES; n2 = (int)(sizeof MOODFX_MODES/sizeof *MOODFX_MODES); }
            else if (fname && (!strcmp(fname,"styleai")||!strcmp(fname,"neuralstyle")))
                { tbl = STYLEAI_MODES; n2 = (int)(sizeof STYLEAI_MODES/sizeof *STYLEAI_MODES); }
            else if (fname && !strcmp(fname,"depthfx"))
                { tbl = DEPTHFX_MODES; n2 = (int)(sizeof DEPTHFX_MODES/sizeof *DEPTHFX_MODES); }
            else if (fname && !strcmp(fname,"posefx"))
                { tbl = POSEFX_MODES; n2 = (int)(sizeof POSEFX_MODES/sizeof *POSEFX_MODES); }
            if (!tbl) {
                send_line(client_fd, "ERR usage: filter modes <personfx|facefx|moodfx|styleai|depthfx|posefx>");
                return;
            }
            char b[96];
            for (int k2 = 0; k2 < n2; k2++) {
                snprintf(b, sizeof b, "OK %2d %s", k2, tbl[k2]);
                send_line(client_fd, b);
            }
            return;
        }

        if (strcmp(sub, "posemodel") == 0) {
            char *path = strtok_r(NULL, "", &save);
            while (path && (*path == ' ' || *path == '\t')) path++;
            if (!path || !*path) {
                send_line(client_fd, "ERR usage: filter posemodel </abs/path/pose.onnx>");
                return;
            }
            if (cv_set_pose_model(path))
                send_line(client_fd, "OK pose model set — 'filter add posefx <0-3>'");
            else
                send_line(client_fd, "ERR file not found");
            return;
        }

        if (strcmp(sub, "depthmodel") == 0) {
            char *path = strtok_r(NULL, "", &save);
            while (path && (*path == ' ' || *path == '\t')) path++;
            if (!path || !*path) {
                send_line(client_fd, "ERR usage: filter depthmodel </abs/path/model-small.onnx>");
                return;
            }
            if (cv_set_depth_model(path))
                send_line(client_fd, "OK depth model set — 'filter add depthfx <0-3>'");
            else
                send_line(client_fd, "ERR file not found");
            return;
        }

        if (strcmp(sub, "genmodel") == 0) {
            /* filter genmodel <path.onnx> — img2img model for 'genai'
             * filter genmodel off        — unload */
            char *path = strtok_r(NULL, "", &save);
            while (path && (*path == ' ' || *path == '\t')) path++;
            if (!path || !*path) {
                send_line(client_fd, "ERR usage: filter genmodel </abs/path/model.onnx | off>");
                send_line(client_fd, "ERR   then: filter add genai [blend 0.2-1] [size 128-512]");
                return;
            }
            if (!strcmp(path, "off") || !strcmp(path, "none")) {
                cv_clear_gen_model();
                send_line(client_fd, "OK genai model unloaded");
                return;
            }
            if (cv_set_gen_model(path)) {
                send_line(client_fd, "OK genai model set — 'filter add genai' transforms every frame through it");
            } else {
                send_line(client_fd, "ERR file not found (absolute path to a .onnx)");
            }
            return;
        }

        if (strcmp(sub, "faceimage") == 0) {
            /* filter faceimage <path> — overlay PNG for facefx mode 24
             * filter faceimage off    — remove it */
            char *path = strtok_r(NULL, "", &save);
            while (path && (*path == ' ' || *path == '\t')) path++;
            if (!path || !*path) {
                send_line(client_fd, "ERR usage: filter faceimage </abs/path/animal.png | off>");
                send_line(client_fd, "ERR   used by: filter add facefx 24  (PNG tracks your face; alpha respected)");
                return;
            }
            if (!strcmp(path, "off") || !strcmp(path, "none")) {
                cv_clear_face_image();
                send_line(client_fd, "OK face image cleared");
                return;
            }
            if (!cv_available()) {
                send_line(client_fd, "ERR this build has no OpenCV (rebuild with 'make OPENCV=1')");
                return;
            }
            if (cv_set_face_image(path)) {
                send_line(client_fd, "OK face image loaded — add/keep 'facefx 24' and it tracks your head");
            } else {
                send_line(client_fd, "ERR could not load image (PNG with transparency works best)");
            }
            return;
        }

        if (strcmp(sub, "bgimage") == 0) {
            /* filter bgimage <path> — background image for personfx mode 3
             * filter bgimage off    — remove it (mode 3 falls back to gradient) */
            char *path = strtok_r(NULL, "", &save);   /* rest of line (paths may have spaces) */
            while (path && (*path == ' ' || *path == '\t')) path++;
            if (!path || !*path) {
                send_line(client_fd, "ERR usage: filter bgimage </abs/path/to/image.jpg | off>");
                send_line(client_fd, "ERR   used by: filter add personfx 3   (person stays, background becomes your image)");
                return;
            }
            if (!strcmp(path, "off") || !strcmp(path, "none")) {
                cv_clear_bg_image();
                send_line(client_fd, "OK background image cleared (personfx 3 shows the gradient)");
                return;
            }
            if (!cv_available()) {
                send_line(client_fd, "ERR this build has no OpenCV (rebuild with 'make OPENCV=1')");
                return;
            }
            if (cv_set_bg_image(path)) {
                send_line(client_fd, "OK background image loaded — add/keep 'personfx 3' to use it");
            } else {
                send_line(client_fd, "ERR could not load image (check the path; PNG/JPEG supported)");
            }
            return;
        }

        if (strcmp(sub, "set") == 0) {
            /* Usage:
             *   filter set <index> <param>              → primary
             *   filter set <index> p2|p3|p4 <param>     → specific extra
             */
            char *istr = strtok_r(NULL, " \t", &save);
            char *first = strtok_r(NULL, " \t", &save);
            if (!istr || !first) {
                send_line(client_fd, "ERR filter set: usage 'filter set <index> [p2|p3|p4|p5] <param>'");
                return;
            }
            int idx = atoi(istr);
            int which = 1;
            char *pstr = first;
            if (!strcmp(first, "p2") || !strcmp(first, "p3") || !strcmp(first, "p4") || !strcmp(first, "p5")) {
                which = first[1] - '0';
                pstr = strtok_r(NULL, " \t", &save);
                if (!pstr) { send_line(client_fd, "ERR filter set: missing value"); return; }
            }
            char *ep; float v = strtof(pstr, &ep);
            if (*ep != '\0') { send_line(client_fd, "ERR filter set: bad number"); return; }
            pthread_mutex_lock(&g_filters.lock);
            if (idx < 0 || idx >= g_filters.count) {
                pthread_mutex_unlock(&g_filters.lock);
                send_line(client_fd, "ERR filter set: index out of range");
                return;
            }
            switch (which) {
            case 1: g_filters.filters[idx].param  = v; break;
            case 2: g_filters.filters[idx].param2 = v; break;
            case 3: g_filters.filters[idx].param3 = v; break;
            case 4: g_filters.filters[idx].param4 = v; break;
            case 5: g_filters.filters[idx].param5 = v; break;
            }
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
                         "OK %d: [%s] id=%s pos=(%d,%d) size=%dx%d%s",
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

    /* ── opencv — report build info + DNN engine ── */
    if (strcmp(cmd, "opencv") == 0) {
        char buf[256];
        snprintf(buf, sizeof buf, "OK %s", cv_engine_info());
        send_line(client_fd, buf);
        if (cv_available()) {
            snprintf(buf, sizeof buf, "OK model dir: %s", cv_model_dir());
            send_line(client_fd, buf);
        }
        if (ort_available()) {
            snprintf(buf, sizeof buf,
                     "OK ONNX Runtime backend: v%s (RVM matting runs here)",
                     ort_version());
            send_line(client_fd, buf);
        } else {
            send_line(client_fd, "OK ONNX Runtime backend: not built in "
                      "(vcam-setup --ort, then make OPENCV=1 ORT=1)");
        }
        return;
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

    /* ── record start [file=<path>] [mic=<dev>] [crf=N] [preset=<p>] [vcodec=<v>]
     *    record stop
     *    record status
     *    record info
     *
     * Spawns ffmpeg as a child of vcam and pipes our composed BGR24
     * frames into it. This avoids all v4l2loopback contention because
     * we don't read /dev/video0 at all on the recording side.
     */
    if (strcmp(cmd, "record") == 0) {
        char *sub = strtok_r(NULL, " \t", &save);
        if (!sub) { send_line(client_fd, "ERR usage: record start|stop|status|info"); return; }

        if (strcmp(sub, "status") == 0) {
            if (recorder_active(g_recorder))
                send_line(client_fd, "OK recording");
            else
                send_line(client_fd, "OK stopped");
            return;
        }

        if (strcmp(sub, "info") == 0) {
            if (!recorder_active(g_recorder)) {
                send_line(client_fd, "OK stopped");
                return;
            }
            char buf[256];
            snprintf(buf, sizeof buf,
                     "OK recording pid=%d file=%s",
                     (int)recorder_pid(g_recorder),
                     recorder_file(g_recorder));
            send_line(client_fd, buf);
            return;
        }

        if (strcmp(sub, "stop") == 0) {
            if (!recorder_active(g_recorder)) {
                send_line(client_fd, "OK already stopped");
                return;
            }
            recorder_stop(g_recorder);
            send_line(client_fd, "OK stopped");
            return;
        }

        if (strcmp(sub, "start") == 0) {
            if (!g_recorder) {
                send_line(client_fd, "ERR recorder not initialised");
                return;
            }
            if (recorder_active(g_recorder)) {
                send_line(client_fd, "ERR already recording");
                return;
            }

            /* Parse key=value tokens for file=, mic=, crf=, preset=, vcodec=, aoffset= */
            char file[512] = {0};
            char mic[128]  = {0};
            char vcodec[64] = "libx264";
            char vpreset[32] = "ultrafast";
            int  crf = 23;
            float aoffset = 0.2f;   /* default: shift audio 0.2s later to
                                     * match the video path's buffering
                                     * latency; tune with aoffset= */
            char *tok;
            while ((tok = strtok_r(NULL, " \t", &save))) {
                if      (strncmp(tok, "file=",   5) == 0)
                    strncpy(file, tok + 5, sizeof file - 1);
                else if (strncmp(tok, "mic=",    4) == 0)
                    strncpy(mic, tok + 4, sizeof mic - 1);
                else if (strncmp(tok, "crf=",    4) == 0)
                    crf = atoi(tok + 4);
                else if (strncmp(tok, "vcodec=", 7) == 0)
                    strncpy(vcodec, tok + 7, sizeof vcodec - 1);
                else if (strncmp(tok, "preset=", 7) == 0)
                    strncpy(vpreset, tok + 7, sizeof vpreset - 1);
                else if (strncmp(tok, "aoffset=", 8) == 0)
                    aoffset = strtof(tok + 8, NULL);
            }
            if (!file[0]) {
                send_line(client_fd, "ERR record start: file= argument required");
                return;
            }

            if (!recorder_start(g_recorder, file,
                                mic[0] ? mic : NULL,
                                g_output_w, g_output_h, g_output_fps,
                                crf, vcodec, vpreset, aoffset)) {
                send_line(client_fd, "ERR ffmpeg failed to start (see /tmp/vcam-record.log)");
                return;
            }

            char buf[256];
            snprintf(buf, sizeof buf,
                     "OK recording pid=%d file=%s",
                     (int)recorder_pid(g_recorder), file);
            send_line(client_fd, buf);
            return;
        }

        send_line(client_fd, "ERR record: unknown subcommand (use start|stop|status|info)");
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

    /* Block SIGCHLD in this thread. When ffmpeg (the recorder child)
     * exits, SIGCHLD fires on whichever thread the kernel picks; if
     * that's the IPC thread mid-accept(), accept() returns EINTR.
     * We now retry on EINTR (see below), but blocking the signal here
     * means the main thread handles it and the IPC thread stays
     * quietly in accept(). Belt and braces. */
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGCHLD);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, NULL);

    while (srv->running) {
        struct sockaddr_un peer;
        socklen_t peer_len = sizeof peer;
        int client = accept(srv->server_fd, (struct sockaddr *)&peer, &peer_len);
        if (client < 0) {
            /* Signals (SIGCHLD when a child ffmpeg exits, etc.) can
             * interrupt accept(). Don't kill the server thread just
             * because a child died — retry. */
            if (errno == EINTR) continue;
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
