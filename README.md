# vcam — virtual webcam pipeline

Real-time virtual webcam written in C. Reads from a physical webcam, runs
your frames through a chain of pixel effects, composites images / looping
videos / live screen regions on top, and writes the result to a
v4l2loopback device that browsers and video-call apps see as a normal
camera.

```
/dev/video1 ──► [capture] ──► [filters] ──► [composite] ──► /dev/video0
   (real cam)                                ▲              (loopback)
                            PNG/JPEG · looping video · live screen region
                                runtime add/remove via vcam-ctl
```

Works in Firefox, Chrome/Chromium, qutebrowser, Brave, Discord, OBS,
Zoom — anything that consumes a V4L2 capture device.

---

## Tools at a glance

| Command | What it does |
|---|---|
| `vcam`        | Main process — runs the capture → filter → composite → output pipeline. |
| `vcam-ctl`    | Send commands to a running vcam (overlays, filters, screen capture). |
| `vcam-setup`  | One-time install of `/etc/modprobe.d/vcam.conf` for browser compatibility. |
| `vcam-tune`   | Interactively dial filter parameters with arrow keys. |

Every tool has `--help`. Run any of them with no arguments or `-h` for full usage.

---

## Quick start

```bash
# 1. Dependencies (Debian / Ubuntu — package names differ on other distros)
sudo apt install \
    libv4l-dev libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavdevice-dev libpng-dev \
    v4l2loopback-dkms pkg-config build-essential \
    wf-recorder slurp ncat

# 2. Build & install
git clone <repo-url> vcam && cd vcam
make
sudo make install      # vcam, vcam-ctl, vcam-setup, vcam-tune

# 3. ONE-TIME: make v4l2loopback land at /dev/video0 (Chromium needs this)
sudo vcam-setup --install

# Then reboot, or do the manual module reload it prints:
sudo modprobe -r uvcvideo
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback
sudo modprobe uvcvideo

# 4. Verify
vcam-setup
# → "OK — loopback at /dev/video0 comes before real cam at /dev/video1"

# 5. Start it (defaults: input /dev/video1, output /dev/video0)
vcam

# 6. Test the output
ffplay /dev/video0
```

In your video-call app, pick "vcam" as the camera.

---

## Why two device numbers? (the Chromium fix)

Chromium-family browsers pick the first `/dev/video*` they see. If that's
your real webcam and vcam is already streaming from it, the browser gets
`EBUSY` → `NotReadableError` and the page errors out before the user can
even pick a different device. Firefox doesn't have this problem.

`vcam-setup --install` writes `/etc/modprobe.d/vcam.conf` so the
v4l2loopback module claims `/dev/video0` *before* USB cameras enumerate.
Your real cam then sits at `/dev/video1+` and the loopback is what
Chromium auto-picks. The `exclusive_caps=1` parameter it sets is also
mandatory for Firefox.

---

## vcam — main process

```
vcam [OPTIONS]

Pipeline:
  -i <device>     Input webcam        (default /dev/video1)
  -o <device>     Output loopback     (default /dev/video0)
  -W <px>         Output width        (default: camera native)
  -H <px>         Output height       (default: camera native)
  -f <fps>        Frame rate
  --no-input      Skip the webcam (screen overlay becomes the source).
                  Requires -W and -H.

Startup overlays:
  --img      <id> <x> <y> [<w> [<h>]] <file>       PNG or JPEG
  --vid      <id> <x> <y> [<w> [<h>]] <file>       video, loops
  --vid-once <id> <x> <y> [<w> [<h>]] <file>       video, plays once
  --screen   <id> <x> <y> [<w> [<h>]] <WxH+X+Y>    live screen region

Startup filters (chain order = listed order):
  --filter <name> [<param>]
```

If you pass only `<w>`, height is computed from the source aspect ratio;
same for `<h>` alone (use `0 <h>` to skip width).

### Examples

```bash
vcam                                                    # plain forward
vcam -W 1280 -H 720 -f 30                               # force 720p30
vcam --img logo 1100 10 160 /assets/logo.png            # top-right logo
vcam --vid intro 50 50 320 180 /clips/intro.mp4
vcam --filter modern 0.8 --filter vignette 0.6          # cinematic
vcam --no-input -W 1920 -H 1080 \
     --screen scr 0 0 1920 1080 1920x1080+0+0           # pure screencast
```

---

## vcam-ctl — runtime control

```bash
vcam-ctl --help                                # script help
vcam-ctl help                                  # all IPC commands

vcam-ctl list                                  # active overlays (draw order)
vcam-ctl add img      <id> <x> <y> [<w> [<h>]] <file>
vcam-ctl add vid      <id> <x> <y> [<w> [<h>]] <file>      # loops
vcam-ctl add vid-once <id> <x> <y> [<w> [<h>]] <file>      # plays once
vcam-ctl add screen   <id> <x> <y> [<w> [<h>]] <WxH+X+Y>
vcam-ctl add webcam   <id> <x> <y> [<w> [<h>]] <device>    # /dev/video1, …
vcam-ctl remove <id>                           # one overlay
vcam-ctl remove all                            # every overlay
vcam-ctl move   <id> [x=N|N] [y=N|N]           # positional or keyed
                                               # value can be a number, "keep", or "-"
                                               # missing = keep
vcam-ctl resize <id> [w=N|N] [h=N|N]           # positional or keyed
                                               # value can be a number, "keep", "auto" or "-"
                                               # missing = auto (derive from aspect)
vcam-ctl raise  <id>                           # bring to top
vcam-ctl lower  <id>                           # send to bottom
vcam-ctl z      <id> <N>                       # explicit z-index
                                               # 0 = bottom, -1 = top, -2 = second-from-top

vcam-ctl webcam tr             # quick: webcam PiP at top-right
vcam-ctl webcam br 480         # bigger, bottom-right
vcam-ctl webcam center         # centered
vcam-ctl webcam full           # fullscreen webcam overlay
vcam-ctl webcam tl 200 cam2    # second webcam overlay (different id)
vcam-ctl webcam off            # remove the default 'cam'

vcam-ctl filter help                           # full filter list
vcam-ctl filter list
vcam-ctl filter add <name> [<param>]
vcam-ctl filter set <index> <param>            # update parameter live
vcam-ctl filter remove <index>
vcam-ctl filter clear
```

The socket lives at `/tmp/vcam.sock`. Override with `VCAM_SOCK=...` if you
ever need to.

---

## vcam-ctl screen — live screen capture

(This used to be a separate `vcam-screen` script; it's now a subcommand
of `vcam-ctl` so everything lives in one tool.)

```
vcam-ctl screen [OPTIONS]

  -i <id>          Overlay ID (default: auto — picks screen, screen2, …)
  -x <px>          Destination X on output           (default: 0)
  -y <px>          Destination Y on output           (default: 0)
  -w <px>          Destination width  (only -w preserves source aspect)
  -h <px>          Destination height
  --full           Capture the whole monitor. If multiple monitors are
                   connected and --monitor wasn't passed, asks which to use.
  --monitor <N>    For --full: pick monitor by connector name (DP-1,
                   HDMI-A-1, …) or 1-based index. Suppresses the prompt.
  --monitors       List connected monitors and exit.
  -r, --replace    Remove existing overlay with the same ID first
  --list           List active screen overlays
  --remove [id]    Remove a screen overlay (default id "screen")
  --remove-all     Remove every active screen overlay
```

Multiple captures are first-class: calling `vcam-ctl screen` repeatedly
without `-i` creates `screen`, `screen2`, `screen3`, … so they don't
collide.

If the captured region is larger than the vcam output (very common with
`--full` on a 4K monitor while vcam runs at 720p), the overlay is
automatically scaled down to fit while preserving aspect — no more
top-left-quadrant clipping.

For low capture latency, vcam sets the kernel pipe size to exactly one
frame's worth of bytes (`F_SETPIPE_SZ`). This tightly couples
wf-recorder's output rate to vcam's consumption rate so frames can't
build up a backlog. On most systems on-screen lag drops to roughly one
frame after the change.

Examples:

```bash
vcam-ctl screen                          # auto-ID, draw a region with slurp/slop
vcam-ctl screen                          # second call → ID "screen2"
vcam-ctl screen --full                   # whole monitor (picker if multiple)
vcam-ctl screen --full -w 1280           # whole monitor, scaled to 1280 wide
vcam-ctl screen --full --monitor DP-1    # specific monitor, no prompt
vcam-ctl screen --full --monitor 2       # 2nd monitor in --monitors list
vcam-ctl screen --monitors               # list connected monitors
vcam-ctl screen -i clip -x 100 -y 100 -w 320
vcam-ctl screen --list
vcam-ctl screen --remove screen2
vcam-ctl screen --remove-all

# move/resize examples — keep / auto syntax
vcam-ctl move   logo 500              # x=500, keep current y
vcam-ctl move   logo - 300            # keep current x, y=300
vcam-ctl resize logo 400              # w=400, h derived from source aspect
vcam-ctl resize logo - 300            # h=300, w derived from source aspect
```

### How screen capture works internally

On Wayland we spawn `wf-recorder` (uses the `wlr-screencopy` protocol —
no root, no `CAP_SYS_ADMIN` needed). On X11 we use `ffmpeg -f x11grab`.
Each screen overlay gets its own reader thread that drains the capture
pipe at full speed; the main loop just snapshots the latest frame, so
capture latency is minimal regardless of output framerate.

Region dimensions are snapped down to multiples of 16 (width) and 2
(height) to match compositor stride alignment — otherwise the captured
image appears to scroll diagonally because frame N starts a few bytes
into frame N+1's data.

A historic bug worth knowing: `wf-recorder` prompts "Overwrite?" when its
output file exists. We pre-load `y\n` into its stdin so the prompt is
auto-answered, otherwise the whole pipeline hangs silently and you'd see
a black overlay forever.

`--full` detects connected monitors by reading `/sys/class/drm/card*-*/status`
and `modes` directly from the kernel — no external tools required. If
`swaymsg` is available (i.e. you're running sway) it's used in addition
for accurate per-monitor X/Y positions on multi-monitor setups; without
it we report `0,0` for all monitors, which is correct for single-monitor
setups but may be wrong for non-primary monitors on multi-monitor.

---

## vcam-setup — boot-time configuration

```bash
vcam-setup                  # diagnose current /dev/video* layout
sudo vcam-setup --install   # write /etc/modprobe.d/vcam.conf
sudo vcam-setup --uninstall # revert
vcam-setup --help
```

The script detects systemd / OpenRC / runit / manual init and prints
init-system-specific instructions for activating the new module load
order without a reboot.

---

## Webcam as an overlay

The webcam is captured once by the main pipeline. Any number of webcam
overlays can sample from that single shared capture, so you can have
the cam fullscreen, in a corner, several times at different sizes, etc.
— V4L2's one-streamer-per-device rule doesn't apply because we never
open the device more than once.

Filters apply to the captured cam stream before any compositing, so a
cam-in-corner picture-in-picture shows whatever filters are active.

Typical screencast-with-webcam-corner workflow:

```bash
vcam                                # captures cam, uses it as base
vcam-ctl screen --full              # screen covers the cam base
vcam-ctl webcam br 400              # cam reappears in bottom-right
vcam-ctl raise cam                  # make sure it stays on top
vcam-ctl filter add modern 0.8      # cinema look on the cam image
```

Multiple corners are fine — pass distinct IDs:

```bash
vcam-ctl webcam tl 200 cam_tl
vcam-ctl webcam br 200 cam_br
vcam-ctl filter add mirror          # both corners are mirrored
```

In `--no-input` mode (no cam captured) webcam overlays simply show
black. If you want screencast + webcam corner, run vcam *without*
`--no-input` and cover the base with a fullscreen screen overlay as
shown above.

---

## Filters (49 total, 6 categories)

Filters run in chain order on every frame from the webcam, *before*
overlays are composited.

### Basic

| Name | Param | Description |
|---|---|---|
| `grayscale`  | —             | Rec.601 luma |
| `sepia`      | —             | warm brown tone |
| `invert`     | —             | negative colours |
| `mirror`     | —             | horizontal flip |
| `flip`       | —             | vertical flip |
| `pixelate`   | block px (8)  | average colour per block |
| `blur`       | radius 1–6 (2)| two-pass box blur |
| `vignette`   | 0–1 (0.6)     | darken edges radially |
| `brightness` | mult (1.2)    | clamped multiply |
| `contrast`   | mult (1.4)    | contrast around mid-grey |
| `posterize`  | levels (4)    | reduce colour levels per channel |
| `edges`      | 0–2 (1.0)     | Sobel edge detection, strength multiplier |
| `saturate`   | 0–3 (1.5)     | chroma boost (0=gray, 1=normal, 2=vivid) |

### Fancy

| Name | Param | Description |
|---|---|---|
| `chromashift`  | offset px (5)       | VHS-style RGB channel separation |
| `glitch`       | intensity 0–1 (0.3) | animated slice displacement |
| `kaleidoscope` | segments (6)        | radial mirror |
| `swirl`        | strength (1.0)      | radial twist |
| `thermal`      | —                   | luma → heatmap palette |
| `oldfilm`      | —                   | sepia + grain + scratches + vignette |
| `wave`         | amplitude px (10)   | animated sin-wave x-shift per row |
| `cartoon`      | edge thresh (50)    | posterize + dark edges |
| `solarize`     | threshold (128)     | invert pixels above value |
| `rainbow`      | speed (1.0)         | animated additive hue rotation |
| `crt`          | 0–1 (0.7)           | CRT scanlines + barrel + R/B chromatic aberration |

### Soft & dreamy

| Name | Param | Description |
|---|---|---|
| `dream`     | 0–1 (0.5) | hazy soft glow |
| `bloom`     | 0–1 (0.6) | highlight bloom |
| `glow`      | 0–1 (0.4) | gentle gaussian glow |
| `pinch`     | 0–1 (0.5) | radial inward pull |
| `bulge`     | 0–1 (0.5) | fisheye-like outward push |
| `painterly` | 1–5 (3)   | oil-painting effect |
| `velvet`    | 0–1 (0.6) | soft desat + warm tint + blur — flattering portrait look |

### Themed looks

| Name | Param | Description |
|---|---|---|
| `modern`    | 0–1 (0.7)         | teal-orange cinematic grade |
| `cyberpunk` | 0–1 (0.7)         | neon contrast + chromashift |
| `romantic`  | 0–1 (0.7)         | soft warm haze + blur + vignette |
| `vangogh`   | radius 1–5 (3)    | heavy painterly + saturation |
| `picasso`   | segments 4–16 (8) | cubist: kaleidoscope + posterize + edges |
| `noir`      | 0–1 (0.8)         | high-contrast B&W with split-tone shadows/highlights |

### Mathematical patterns

| Name | Param | Description |
|---|---|---|
| `sine`    | amplitude (15)    | animated sin-wave x/y distortion |
| `xor`     | 0–1 (0.5)         | `(x ^ y)` bitwise overlay |
| `modulo`  | divisor 2–128 (64)| colour mod N |
| `polar`   | 0–1 (0.7)         | polar-coordinate warp |
| `voronoi` | cells 4–64 (20)   | voronoi-cell mosaic |
| `julia`   | 0–1 (0.5)         | animated Julia fractal coloured overlay |

### Pixel-crazy / animated

| Name | Param | Description |
|---|---|---|
| `bitcrush` | bits 1–8 (3)   | quantize each channel to N bits — chunky banding |
| `melt`     | 0–1 (0.5)      | columns drip downward, animated over time |
| `popart`   | 0–1 (1.0)      | Warhol 2×2 quadrants with rotating colour tints |
| `displace` | 0–1 (0.5)      | bright pixels shift right, dark pixels left |
| `ripple`   | 0–1 (0.5)      | animated water ripple radiating from the center |
| `scanlines`| gap 2–16 (3)   | hard black scanline every N rows — CRT/terminal look |

### Combinations to try

```bash
vcam-ctl filter add modern 0.8                                  # cinema look
vcam-ctl filter add cyberpunk 0.9                               # neon
vcam-ctl filter add romantic 0.7                                # soft skin
vcam-ctl filter add vangogh 3                                   # painting
vcam-ctl filter add picasso 8                                   # cubism
vcam-ctl filter add thermal                                     # heatvision
vcam-ctl filter add cartoon                                     # animated
vcam-ctl filter add glitch 0.4 && vcam-ctl filter add chromashift 6
vcam-ctl filter add voronoi 30 && vcam-ctl filter add xor 0.4   # math art
```

---

## vcam-tune — interactive parameter dial

Walk mode (no args) navigates every active filter; the highlighted one is
the one whose parameter the arrows are tweaking. Single-filter mode adds
(or selects) one filter and dials just it.

```
vcam-tune                  walk mode — navigate all active filters
vcam-tune <name> [<step>]  add the filter and dial it
vcam-tune <index> [<step>] dial an existing filter at <index>
vcam-tune --help
```

| Key | Action |
|---|---|
| `↑` / `k` | increase parameter |
| `↓` / `j` | decrease parameter |
| `r`       | reset to default |
| `x`       | remove current filter (stays in walk mode) |
| `a`       | open a categorised picker to add a new filter |
| `q` / Esc | quit |
| `→` / `l` / Tab | next filter (walk mode) |
| `←` / `h`       | previous filter (walk mode) |

If you launch `vcam-tune` with no filters active, it opens the picker
automatically so you can start composing a chain from scratch.

Examples:

```bash
vcam-tune modern         # add modern, then arrows to dial strength
vcam-tune pinch 0.02     # smaller step for finer control
vcam-tune                # walk everything you already added
vcam-tune 2              # dial existing filter at index 2
```

---

## Architecture

```
src/
  main.c       – CLI parsing, signal handling, main compositing loop
  capture.c    – V4L2 mmap capture (auto-picks YUYV/MJPG/NV12 at best
                 resolution), MJPG decoded via libavcodec, converted to BGR24
  output.c     – V4L2 streaming I/O to loopback with mmap output buffers,
                 BGR24 → YUYV422 conversion, BT.709 (HD) / BT.601 (SD)
                 colourspace tags
  frame.c      – BGR24 frame allocation
  overlay.c    – Image (PNG/JPEG), looping/once video, live screen overlays
                 (one reader thread per screen)
  registry.c   – Mutex-protected overlay list (thread-safe runtime mutation)
  ipc.c        – Unix-domain-socket server (separate thread)
  filter.c     – 38-effect filter chain; mutex-protected
  test_screen.c – Standalone diagnostic for the screen-capture path
include/
  vcam.h       – Public API
scripts/
  vcam-ctl     – Complete runtime control: overlays, filters, and the
                 'screen' subcommand (slurp/slop region picker, --full
                 monitor capture via /sys/class/drm, list/remove screens)
  vcam-setup   – v4l2loopback boot configuration
  vcam-tune    – Interactive parameter dial
```

**Threading model**

| Thread | Job |
|---|---|
| main | capture → filter chain → composite → output, paced at fps |
| IPC  | accepts UDS connections, mutates registry + filter chain under their locks |
| 1 per screen overlay | drains FIFO/pipe as fast as data arrives, keeps `latest_frame` mutex-protected |

**Pixel format**

- Internal: BGR24, packed, row-major.
- Capture: V4L2 native → swscale → BGR24 (lazy: built from the first
  received frame's actual `pix_fmt`, not the pre-decode guess — this is
  what fixed the historic "green block" bug).
- Output: swscale BGR24 → YUYV422 directly into mmap'd output buffers,
  with proper `V4L2_COLORSPACE_*` / `ycbcr_enc` tags so browsers don't
  reject the device.

---

## Working on the source

```bash
make              # build vcam and test_screen with -Wall -Wextra -Wpedantic -O2
make clean
sudo make install # all four scripts + vcam to /usr/local/bin
```

To add a new source file, drop it in `src/` and append to `SRCS` in the
Makefile. No autotools, no generated headers.

### test_screen

Standalone diagnostic for screen capture. Prints every step (env vars,
exact wf-recorder command, child status, byte-by-byte arrival progress),
saves a raw frame, and tells you how to view it as PNG:

```bash
./test_screen 1280x720+0+0 /tmp/frame.raw
ffmpeg -f rawvideo -pix_fmt bgr0 -video_size 1280x720 -i /tmp/frame.raw \
       -frames:v 1 -update 1 -y /tmp/frame.png
xdg-open /tmp/frame.png
```

If `test_screen` produces a good PNG but vcam itself still shows black,
the bug is in the compositing path, not capture. If `test_screen` itself
times out, your compositor probably doesn't support `wlr-screencopy`
(GNOME/Plasma don't; Sway/Hyprland/river/niri/wayfire do).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot open /dev/video0` | v4l2loopback not loaded | `sudo vcam-setup --install`, then reboot or manual reload |
| `NotReadableError` in Chromium | Loopback isn't at /dev/video0 | `sudo vcam-setup --install` |
| Firefox shows no camera | v4l2loopback without `exclusive_caps=1` | vcam-setup writes it correctly |
| Screen overlay is black | (was) wf-recorder hung on Overwrite prompt | rebuild — pre-loaded `y\n` is in stdin now |
| Screen overlay still black | Compositor doesn't support wlr-screencopy | Use Sway, Hyprland, river, niri, wayfire |
| Screen image scrolls diagonally | (was) stride misalignment | rebuild — dims snap to 16/2 now |
| Video overlay is a green block | (was) pre-decode `pix_fmt` was NONE | rebuild — sws is lazy now |
| Video plays jerky/jumps back | (was) spin limit too tight | rebuild — limit raised, `vid_eof_sent` flag added |
| `vcam-ctl: ncat: command not found` | ncat not installed | `sudo apt install ncat` |
| `shell-init: getcwd error retrieving current directory` | Your *shell* sits in a deleted directory — **not a vcam bug** | `cd ~` in that terminal |
| High CPU at 1080p | `blur` / `painterly` / `voronoi` are O(W·H·r²) | smaller param, drop to 720p, or pick cheaper filters |
