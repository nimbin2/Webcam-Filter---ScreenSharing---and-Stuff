# vcam — virtual webcam pipeline

> 🤖 **This is 100% Vibecode by ClaudeCode.** Every line of C, every shell
> script, and every word of this README was written by Claude through a
> long iterative back-and-forth. No human authored any of it directly —
> only the prompts that shaped it.

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
| `vcam-ctl`    | Send commands to a running vcam (overlays, filters, screen capture). POSIX `sh`. |
| `vcam-setup`  | One-time install of `/etc/modprobe.d/vcam.conf` for browser compatibility. POSIX `sh`. |
| `vcam-tune`   | Interactively dial filter parameters with arrow keys. Requires `bash` (single-keypress input). |

Every tool has `--help`. Run any of them with no arguments or `-h` for full usage.

---

## Quick start

Full details — dependencies, build options, the one-time device setup —
are in **[INSTALL.md](INSTALL.md)**. The short version:

```bash
# 1. Dependencies (Debian / Ubuntu)
sudo apt install \
    libv4l-dev libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavdevice-dev libpng-dev \
    v4l2loopback-dkms pkg-config build-essential \
    wf-recorder slurp ncat

# 2. Build & install
make && sudo make install

# 3. ONE-TIME: make v4l2loopback land at /dev/video0 (Chromium needs this)
sudo vcam-setup --install     # then reboot, or reload modules as it prints

# 4. Run it, and test the output
vcam
ffplay /dev/video0
```

In your video-call app, pick **vcam** as the camera. If the browser can't
see it, that's almost always the device-number issue —
[INSTALL.md](INSTALL.md#one-time-device-setup-the-chromium-fix) explains
why and how `vcam-setup` fixes it.

---

## vcam — main process

```
vcam [OPTIONS]

Pipeline:
  -i <device>     Input webcam        (default /dev/video1)
  -o <device>     Output loopback     (default /dev/video0)
  --res WxH       Output resolution   (e.g. --res 1280x720; default: camera native)
  -f <fps>        Frame rate
  --list-formats  List the resolutions the input device supports, then exit
  --no-input      Skip the webcam (screen overlay becomes the source).
                  Combine with --res.

Startup overlays:
  --img      <id> <x> <y> [<w> [<h>]] <file>       PNG or JPEG
  --vid      <id> <x> <y> [<w> [<h>]] <file>       video, loops
  --vid-once <id> <x> <y> [<w> [<h>]] <file>       video, plays once
  --screen   <id> <x> <y> [<w> [<h>]] <WxH+X+Y>    live screen region

Startup filters (chain order = listed order):
  --filter <name> [<param>]
```

Not sure what your camera supports? Ask it:

```bash
vcam --list-formats           # prints every format / resolution / fps it offers
```

If you pass only `<w>` to an overlay, height is computed from the source
aspect ratio; same for `<h>` alone (use `0 <h>` to skip width).

### Examples

```bash
vcam                                                    # plain forward
vcam --list-formats                                     # what can my camera do?
vcam --res 1280x720 -f 30                               # force 720p30
vcam --img logo 1100 10 160 /assets/logo.png            # top-right logo
vcam --vid intro 50 50 320 180 /clips/intro.mp4
vcam --filter modern 0.8 --filter vignette 0.6          # cinematic
vcam --no-input --res 1920x1080 \
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

## Filters (111 total, 12 categories)

Filters run in chain order on every frame from the webcam, *before*
overlays are composited. Every filter has a primary parameter; many
have up to three extra tunable knobs (marked in `filter help` with
`(*)`). All parameters can be set at add time, changed live over IPC,
or dialed interactively with `vcam-tune`.

```bash
vcam-ctl filter help                        # authoritative list + all knobs
vcam-ctl filter add <name> [p1 [p2 [p3 [p4]]]]
vcam-ctl filter set <index> <value>         # primary param
vcam-ctl filter set <index> p2|p3|p4 <v>    # extra knobs
vcam-ctl filter list                        # active chain with all params
```

### Basic
`grayscale sepia invert mirror flip pixelate blur vignette brightness
contrast posterize edges saturate` — the workhorses. `vignette` takes
optional radius + warmth for a film look.

### Fancy
`chromashift glitch kaleidoscope swirl thermal oldfilm wave cartoon
solarize rainbow crt` — kaleidoscope has rotation, per-frame spin and
wedge zoom; wave does vertical/horizontal with frequency and speed;
chromashift splits all three channels independently.

### Soft & dreamy
`dream bloom glow pinch bulge painterly velvet` — pinch/bulge accept
an off-center point (gravity wells).

### Themed
`modern cyberpunk romantic vangogh picasso noir` — one-shot looks
composed from the primitives.

### Math
`sine xor modulo polar voronoi julia` — pure math over your pixels.

### Pixel-crazy
`bitcrush melt popart displace ripple scanlines`

### Glitch-art (animated, stateful)
| Name | Knobs | Effect |
|---|---|---|
| `shuffle`   | intensity, block size, phase seconds | regions scramble over seconds, then UNSCRAMBLE pixel-perfectly |
| `pixelsort` | density | luminance-sorted spans, the classic glitch smear |
| `rgbdelay`  | strength | R and G channels lag frames behind B — chromatic ghosts |
| `drift`     | band height, max shift | VHS-style horizontal band misalignment |
| `rain`      | density, speed, trail | Matrix digital rain over the image |

### Temporal (frame-history effects)
| Name | Knobs | Effect |
|---|---|---|
| `echo`     | persistence | motion trails / light painting |
| `slitscan` | depth, direction | each row samples an older frame — time taffy |
| `mosh`     | sensitivity, keyframe interval | fake datamosh; static freezes, motion smears |
| `stutter`  | period | hold each frame N frames — stop-motion |
| `halftone` | cell | newspaper dot print |
| `neon`     | strength, threshold, hue speed, glow | rainbow edges with real bloom |

### Interactive — these respond to YOUR MOTION on camera
| Name | Knobs | Effect |
|---|---|---|
| `ripples`    | sensitivity, rain, refraction | real water simulation — wave a hand, ripples flow away from it |
| `lightpaint` | fade, threshold, hue speed | movement deposits glowing hue-cycling trails |
| `fireworks`  | sensitivity, burst | motion detonates gravity-arcing particle bursts |
| `spotlight`  | radius, darkness, follow speed | a theatrical spotlight CHASES you across the frame |
| `feedback`   | decay, zoom, rotation | infinite video-feedback tunnel |
| `hologram`   | strength, scanline gap, glitch rate | sci-fi projection |
| `life`       | density, color mode, speed | Conway's Game of Life colonising your edges; color 1 = age rainbow |
| `mathwarp`   | amplitude | **your own math** — see below |

### Demoscene & artistic
`plasma` (animated sine palette), `contour` (glowing topographic
iso-lines), `infinity` (recursive droste picture-in-picture),
`moire` (orbiting interference rings), `ascii` (REAL ASCII art from an
embedded 8×8 font — cell 8/16, green/color/white modes), `duotone`
(two-hue gradient map, hues in degrees), `sketch` (pencil via dodge
blend), plus `emboss dither tiles zoompulse oilslick`.

### The AI-craze set
| Name | Knobs | Effect |
|---|---|---|
| `predator`    | visibility, shimmer, adapt | camouflage cloak: hold still and VANISH; motion shows a shimmering outline |
| `shards`      | count, offset | broken mirror — voronoi shards with bright crack lines |
| `portal`      | radius, separation | two ringed portals (orange/blue) that swap their contents |
| `heattrail`   | gain, cooling | motion leaves a cooling thermal wake (fire palette) |
| `wormhole`    | depth, speed | fly down a demoscene tunnel textured with your live image |
| `magnet`      | strength, follow | the frame bends toward the brightest light — drag it with a flashlight |
| `liquid`      | amplitude, scale, speed | the whole image flows like disturbed water |
| `nightvision` | gain, noise | gen-3 intensifier: green amplification, sensor noise, hard vignette |
| `badtv`       | amount, roll | dying CRT: vertical roll, tearing, static bursts, chroma wobble |
| `ghosts`      | delay, gain | motion extraction — static scenery goes gray, movement appears as vivid spectres |

### The AI-craze set, volume 2
| Name | Knobs | Effect |
|---|---|---|
| `clones`     | count, interval, alpha | freeze-frames of you captured every few seconds populate the room — *Multiplicity* |
| `meltdown`   | speed, chaos | Doom's screen-melt, live: the frozen frame's columns slide off, revealing the feed underneath; loops |
| `rotozoom`   | speed, zoom | demoscene rotozoomer — an infinite mirror-tiled plane of you, rotating and breathing |
| `starfield`  | density, speed | hyperspace: stars fly out of center with speed streaks over the video |
| `lightning`  | rate, branches | jagged bolts STRIKE wherever you move, with branch forks and a full-frame flash |
| `scatter`    | power, heal | motion shatters pixels outward with an impulse; they spring back — explode and heal |
| `vhs`        | wear, timecode | the full tape look: chroma bleed, luma noise, head-switch bar, live counting timecode |
| `outline`    | thickness, hue speed | animated marching-ants lasso traces whatever is moving |
| `symmetry`   | mode | face-symmetry mirrors — 0 left→right, 1 right→left, 2 quad kaleidoscope |
| `timebubble` | delay, radius | a glowing bubble drifts around; inside it you see seconds into the past |

### OpenCV-backed set (optional — build with `make OPENCV=1`)

These seven use OpenCV's computer-vision algorithms for effects the
hand-rolled filters can't do: real motion analysis, background
segmentation, and face detection. They're **opt-in at build time** —
the default build has no OpenCV dependency and these filters simply
aren't present.

They're written **OpenCV-5-first**: built against OpenCV 5, the DNN
filters load their models on the **new graph-based DNN engine**
(`ENGINE_AUTO` — 80%+ ONNX operator coverage, shape inference, operator
fusion) and the person-matte blend runs in **first-class FP16**
(`cv::hfloat`), both new in 5.0. Against OpenCV 4.5+ the same code
compiles a classic-engine/FP32 fallback. Check which path your build
uses with `vcam-ctl opencv`. See
[INSTALL.md](INSTALL.md#optional-opencv-filters) to enable the filters
and fetch the (free, tiny) model files.

| Name | Knob | Effect | Needs model |
|---|---|---|---|
| `cvcartoon`   | strength | edge-preserving bilateral smoothing + adaptive ink outlines — a proper cartoon, not a posterize | no |
| `cvpencil`    | color (1/0) | OpenCV's non-photorealistic pencil sketch; 1 keeps color, 0 is graphite | no |
| `cvstylize`   | strength | watercolor / poster stylization (edge-aware) | no |
| `opticalflow` | mix | dense Farnebäck optical flow — motion becomes flowing color (direction = hue, speed = brightness) | no |
| `bgsub`       | sensitivity | MOG2 background subtraction: whatever moves stays vivid, the learned static background dims and desaturates | no |
| `facefx`      | mode, amount, speed | YuNet face detection (`amount` scales the effect — bighead size, laser thickness, halo radius…; `speed` drives every animation) — 0 privacy-pixelate, 1 glossy ring + landmark dots, 2 googly eyes, 3 **bighead**, 4 **deal-with-it sunglasses**, 5 **laser eyes**, 6 **bobblehead**, 7 **third eye**, 8 **crown**, 9 **mirror face**, 10 **cyclops**, 11 **heart eyes**, 12 **interrogation spotlight**, 13 **target lock**, 14 **face satellites**, 15 **melting face**, 16 **sparkle eyes** (anime glints twinkling at eyes and cheeks), 17 **halo** (floating, bobbing, glowing), 18 **clown**, 20 **picasso** (your eyes, nose and mouth swap places — cubism by patch surgery; `amount` = how far they stray), 21 **time ripple**, 22 **fisheye face**, 23 **golden ratio**, 24 **face image** — become an animal: `vcam-ctl filter faceimage ~/pics/cat.png` then `filter add facefx 24`; any PNG with transparency tracks your head (scaled by face width, rotated with your eye angle; `amount` = overlay size). Works for animal faces, masks, helmets, meme faces, 19 **auto-director** (the camera dollies to keep your face centered at constant size — Center Stage for any app; amount = framing tightness, speed = follow) | **yes** |
| `personfx`    | mode, edge, smooth, size, **fx** | person matting, best engine wins (adaptive pacing keeps it realtime — inference cost is measured live and mask updates are spaced to fit your CPU; `size` 256–640 trades matte detail vs speed; **`fx` (p5, 0.2–3) is the per-mode effect-strength dial** — glitch amplitude, fire intensity, ray strength, levitation height, fog density… — adjustable **live**: `vcam-ctl filter set 0 p5 1.8`): **RVM on ONNX Runtime** (`vcam-setup --ort` + `--models-rvm`, build `ORT=1` — correct, temporally-stable, hair-level mattes) → RVM on OpenCV-DNN (auto-disabled when the engine miscomputes it) → PP-HumanSeg — 0 = blur background, 1 = spotlight, 2 = moving gradient, 3 = **your own image** (`vcam-ctl filter bgimage /path/pic.jpg`), 4 = **invisibility cloak**, 5 = **neon person**, 6 = **clone army**, 7 = **window-3D**, 8 = **chrono clones**, 9 = **tiny you**, 10 = **hologram you**, 11 = **rainbow aura** (hue-cycling glow rings around your body), 12 = **painted world** (neural style transfers the *background* while you stay photographic), 13 = **body pixelate** (privacy for everything but the room), 14 = **gravity flip**, 15 = **motion echo**, 16 = **double exposure**, 17 = **glitch body** (only you get datamoshed — band tearing + RGB split; the room stays clean), 18 = **sliced portrait** (you split into slabs sliding apart, gallery-art style), 19 = **disintegration** (you dissolve into drifting embers and re-form, on a loop), 20 = **light-paint silhouette** (your outline writes lingering hue-cycling light trails), 21 = **mirror twin** (you and your flipped double side by side — wave and it waves back with the other hand), 22 = **crystal you** (your body as a flat-shaded low-poly mesh), 23 = **X-ray scanner**, 24 = **wanted poster**, 25 = **stage spotlight** (a theater beam tracks you; dust motes drift in the light), 26 = **haunted shadow** (your wall shadow is your silhouette from one second ago — it lags you and catches up), 27 = **cosmic you** (your body is a window into an animated starfield), 28 = **paper sticker** (thick white cut-out border + drop shadow), 29 = **heat signature**, 30 = **afterimage burst**, 31 = **mandala you** (your outline contour, kaleidoscoped into six spinning neon copies while the real you sits in the center), 32 = **topo aura** (animated topographic rings radiate from your body — a living radar map via distance transform), 33 = **BALL PIT** (physics balls rain down and *bounce off your silhouette* — the matte is the collider, its gradient the surface normal; bat them with your hands), 34 = **orbit rim light** (a virtual studio light circles you; your edge catches it exactly where physics says), 35 = **mind vortex** (the room spirals into your head; you stay solid), 36 = **stained glass**, 37 = **upside-down room**, 38 = **snowfall** (snow lands on your shoulders and head — it piles up on upward-facing surfaces of your silhouette, and tumbles off when you move), 39 = **fire silhouette** (doom-fire heat sim seeded at your outline; flames rise off your body), 40 = **bubbles** (they drift up; touch one and it pops into droplets), 41 = **THOR** (lightning periodically strikes your highest point — raise a hand and the sky answers), 42 = **fireflies** (glowing agents drawn to your outline; they hover around you and scatter if you push into them), 43 = **stardust** (your moving edges shed twinkling glitter), 44 = **matrix you** (digital rain falls *inside* your silhouette), 45 = **black hole you** (the room stretches inward toward you, with an accretion glow ring), 46 = **pet moons**, 47 = **portals**, 48 = **god rays** (volumetric light streams past you — screen-space crepuscular rays, occluded by your silhouette), 49 = **tron trace** (your outline as glowing neon with a bright pulse racing around the perimeter), 50 = **shockwave** (move fast and an expanding refraction ring bursts out of you), 51 = **rapture beam** (a pillar of light takes you — you levitate inside a golden beam with rising sparks), 52 = **kaleido-body** (your matted body folded into eight mirror-wedges — a living mandala of flesh), 53 = **liquid floor** (the bottom of the frame becomes rippling water reflecting the scene), 54 = **astral soul** (a translucent cyan copy of you drifts out of your body and sinks back), 55 = **mystic fog**, and the **pro looks**: 56 = **color pop** (muted mono room, full-color you), 57 = **cinema** (2.39:1 letterbox + teal-orange split-tone + S-curve + grain), 58 = **bokeh pro** (background highlights bloom into real lens discs), 59 = **studio light** (warm key on you, cool rim, eased room + vignette — the look-better button), 60 = **night boost** (shadow lift + gentle denoise on you, room keeps its mood), 61 = **green-screen out** (you over solid chroma for OBS; fx≥1.5 = blue), 62 = **BRB privacy** (leave frame → feed auto-pixelates with a BRB card; return → instant clear), 63 = **pop-out 3D** (soft contact shadow falls from you onto the real room), 64 = **dolly zoom** (the Hitchcock shot — the room breathes scale around your head; you stay fixed), 65 = **duotone poster** (curated two-color gradient maps; fx 1–4 picks the palette) | **yes** |
| `styleai`     | style, blend, size | **live neural style transfer** — your webcam through actual neural nets: 0 mosaic, 1 candy, 2 rain-princess, 3 udnie, 4 pointilism. `blend` mixes with the original; `size` (96–384) is the quality↔speed dial | **yes** (`--models-style`) |
| `moodfx`      | mode, gain, extra | **emotion-reactive**: an expression model reads your face (7 emotions, ~2 ms) and the effect responds — 0 aura, 1 HUD label, 2 party (happy→confetti, sad→rain, surprised→flash, angry→screen-shake), 3 **emoji face**, 4 **paintmood**, 5 **auto-grade**, 6 **mood ring**, 7 **emotion fountain** (particles pour from behind your head, their physics written by your mood — happy erupts golden, sad drips, angry bursts sparks, surprised rings outward; `extra` = flow), — paintmood: two neural nets chained: the emotion model *selects the neural style* live (happy→candy, sad→rain-princess, angry→mosaic, surprised→pointillism) and your confidence drives the blend; needs the `--models-style` pack | **yes** |
| `speedlines`  | sens, density | **anime speed lines**: mean optical flow past a threshold fires white motion streaks from the border opposite your movement — dodge left, lines fly right | no |
| `paintflow`   | strength, decay | **motion paints**: dense optical flow drags a persistent canvas — wave and you smear reality like wet paint; hold still and it sharpens back | no |

They're laptop-friendly: classic effects run on the CPU at 720p, and the
two DNN filters run inference on a downscaled frame every few frames
(the models are ~85 KB and ~6 MB, milliseconds per inference). If a model
file is missing, that filter passes the frame through untouched and prints
a one-time hint to run `vcam-setup --models`.

```bash
vcam-ctl filter add cvcartoon 0.7
vcam-ctl filter add opticalflow 0.7      # wave your hand, watch it flow
vcam-ctl filter add facefx 2             # googly eyes
vcam-ctl filter add personfx 0           # blur everything but you
vcam-ctl filter bgimage ~/pics/beach.jpg # then:
vcam-ctl filter add personfx 3           # you, at the beach
vcam-ctl filter add styleai 2 0.9 256    # live rain-princess painting, high quality
vcam-ctl filter add moodfx 2             # smile → confetti rains down
vcam-ctl filter add personfx 4           # invisibility cloak
vcam-ctl filter add personfx 5           # neon silhouette in a black room
vcam-ctl filter add personfx 6           # CLONE ARMY: three of you
vcam-ctl filter add facefx 3             # bighead mode
vcam-ctl filter add facefx 4             # deal-with-it sunglasses
vcam-ctl filter add moodfx 3             # you are the emoji now
vcam-ctl filter add paintflow 1.5        # smear reality by moving
vcam-ctl filter add personfx 7           # WINDOW-3D: move your head, see parallax
vcam-ctl filter add personfx 8           # your past selves follow you
vcam-ctl filter add facefx 5             # laser eyes
vcam-ctl filter add facefx 6             # bobblehead
vcam-ctl filter add facefx 7             # third eye
vcam-ctl filter add personfx 10          # hologram you
vcam-ctl filter add speedlines 1.5       # anime dodge lines
```



`mathwarp` has two independent expression sets. **Warp** moves pixels;
**color** repaints them. Use either or both.

```bash
vcam-ctl filter add mathwarp 1.0

# WARP: displace each pixel by (dx, dy)
vcam-ctl filter expr '(x-w/2)*(sin(t*5)>0?0.08:0)' '(y-h/2)*(sin(t*5)>0?0.08:0)'

# COLOR: repaint each pixel; r/g/b return 0..255
vcam-ctl filter color 'cg' 'cb' 'cr'          # rotate the RGB channels
vcam-ctl filter color off                     # back to normal colors
```

Both are compiled on the fly by a built-in parser. Warp is evaluated on
a coarse grid + bilinear upsampling; color is evaluated per pixel — both
comfortably hold 720p30.

* Variables (everywhere): `x y` (pixel), `w h` (frame size), `t` (seconds),
  `r a` (radius / angle from center)
* Color expressions also get `cr cg cb` — this pixel's source R/G/B (0–255)
* Functions: `sin cos abs sqrt floor`
* Operators: `+ - * / %`, comparisons `> < >= <= == !=`,
  ternary `cond ? a : b`, parentheses, unary minus

```bash
# brick-shatter: alternating 40px bands slide opposite ways
vcam-ctl filter expr '(floor(y/40)%2==0?1:-1)*sin(t*3)*15' '0'
# photographic negative
vcam-ctl filter color '255-cr' '255-cg' '255-cb'
# a spectrum that scrolls across the frame forever
vcam-ctl filter color 'sin(x/40+t)*127+128' 'sin(x/40+t+2)*127+128' 'sin(x/40+t+4)*127+128'
```

**→ See [EXPRESSIONS.md](EXPRESSIONS.md) for a full cookbook** — black
holes, tornados, thermal maps, psychedelic wormholes, and dozens more
verified recipes for warp, color, and the two combined.

The `mathwarp` primary parameter is a global amplitude multiplier, so
you can fade your formula in and out live from `vcam-tune`.

## Recording the vcam output

`vcam-ctl record` drives a detached `ffmpeg` child to capture whatever
vcam is currently outputting on `/dev/video0`. Because v4l2loopback
allows multiple readers, you can record at the same time a browser or
OBS or whatever is also using the cam — no contention.

```bash
vcam-ctl record start                                # auto-picks a mic if one's
                                                     # available (ALSA "default"
                                                     # if present, else first
                                                     # hw:N,M from arecord -l).
                                                     # Falls back to video-only.
vcam-ctl record start --no-audio                     # force video-only
vcam-ctl record start -m hw:1,0 -o ~/Videos/out.mkv  # explicit mic + filename
vcam-ctl record start --crf 18 --preset slower       # higher quality
vcam-ctl record pause                                # SIGSTOP ffmpeg
vcam-ctl record resume                               # SIGCONT ffmpeg
vcam-ctl record stop                                 # flush + close cleanly
vcam-ctl record status                               # → "recording" | "paused" | "stopped"
vcam-ctl record info                                 # detailed + elapsed time
vcam-ctl record mics                                 # list available ALSA devices
```

Audio policy: ALSA only by design (no Pulse/PipeWire daemon dependency).
On systems running PipeWire, the pipewire-alsa compatibility plugin
generally exposes `default` as a working capture device, which `record
start` will pick automatically.

Mic identifiers accepted by `--mic`:

| Form | Behaviour |
|---|---|
| *(omitted)*    | auto-pick: `default` if present, else first `plughw:N,M` from `arecord -l` |
| `N,M`          | expanded to `plughw:N,M` for you |
| `plughw:N,M`   | ALSA auto-converts sample format/channels (recommended) |
| `hw:N,M`       | raw access — only if you know the device's native format |
| `default`      | system default |

Run `vcam-ctl record mics` for a ready-to-paste table of what's
available on your machine.

### Recording while vcam is already streaming

This now works without contention. Two things had to be right:

1. **v4l2loopback needs enough buffers.** Older `vcam-setup` defaulted
   to `max_buffers=2`; the current default is `8`. If you set this up
   previously and `record start` fails with `Device or resource busy`,
   re-apply and reload:

   ```bash
   sudo vcam-setup --install      # rewrites /etc/modprobe.d/vcam.conf
   sudo modprobe -r v4l2loopback
   sudo modprobe v4l2loopback
   ```

2. **The ffmpeg command must not call `VIDIOC_S_PARM`.** v4l2loopback
   rejects framerate changes from any consumer while a producer is
   active, returning EBUSY. `vcam-ctl record start` deliberately
   omits the `-framerate` flag for this reason — it lets the loopback
   keep whatever framerate vcam set. (This is why `ffplay` worked but
   a hand-rolled `ffmpeg -framerate 30 -f v4l2 -i /dev/video0` would
   not.) If you ever invoke ffmpeg by hand against vcam, leave out
   `-framerate`.

With both in place: vcam writing + a browser reading + a recording
capturing all run simultaneously without any contention.

### If audio crackles

The most common cause is ffmpeg's default `thread_queue_size` of 8
packets being too small for live capture; `record start` passes
`-thread_queue_size 1024` plus `aresample=async=1000` to mitigate
this. If you're still hearing crackles:

* Try a different mic with `record mics` + `--mic plughw:N,M`. Laptop
  DMICs (AMD acp etc.) are often the culprit.
* If your system runs PipeWire, the `pipewire-alsa` shim sometimes
  introduces extra resampling latency. Routing through `default` may
  be smoother than `plughw:N,M` in that case.

The recording lives in a separate process from vcam, so it survives a
vcam restart. State is published in three plain files under `/tmp` so any
status bar can read them with `cat`:

| File | Contents |
|---|---|
| `/tmp/vcam-record.pid`    | PID of the ffmpeg child |
| `/tmp/vcam-record.status` | `recording` or `paused` (absent when stopped) |
| `/tmp/vcam-record.info`   | `file=`, `mic=`, `started=`, `format=`, etc. |
| `/tmp/vcam-record.log`    | ffmpeg's own stderr |

### Swaybar integration

`vcam-ctl record bar` emits one line of swaybar-compatible JSON:

```json
{"text":"● REC 00:01:23","class":"recording","color":"#ff5555"}
```

Drop into `~/.config/sway/config`:

```
bar {
    status_command while date +'%T'; do
        printf '%s | %s\n' "$(vcam-ctl record bar)" "$(date +%T)"
        sleep 1
    done
}
```

Or feed it into [waybar](https://github.com/Alexays/Waybar) / i3blocks /
[i3status-rust](https://github.com/greshake/i3status-rust). The
simplest possible integration is `vcam-ctl record status` — exits 0 if
recording, 1 if not.

### Pause caveat

`pause` sends `SIGSTOP` to the ffmpeg process — it literally halts in
place. Video resumes cleanly because v4l2loopback just queues frames,
but the ALSA capture device may buffer a small amount of audio data
while stopped and produce a tiny pop or skip on resume. For long pauses
prefer `stop` + a fresh `start` (the auto-generated filename includes a
timestamp so you won't collide).

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
sudo make install                       # → /usr/local/bin
make install PREFIX=$HOME/.local        # user-local, no root needed
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

Installation and device-setup problems (loopback not found, browser can't
see the camera, missing build tools) are covered in
**[INSTALL.md](INSTALL.md#install-time-troubleshooting)**. Runtime issues:

| Symptom | Likely cause | Fix |
|---|---|---|
| Screen overlay is black | (was) wf-recorder hung on Overwrite prompt | rebuild — pre-loaded `y\n` is in stdin now |
| Screen overlay still black | Compositor doesn't support wlr-screencopy | Use Sway, Hyprland, river, niri, wayfire |
| Screen image scrolls diagonally | (was) stride misalignment | rebuild — dims snap to 16/2 now |
| Video overlay is a green block | (was) pre-decode `pix_fmt` was NONE | rebuild — sws is lazy now |
| Video plays jerky/jumps back | (was) spin limit too tight | rebuild — limit raised, `vid_eof_sent` flag added |
| `shell-init: getcwd error retrieving current directory` | Your *shell* sits in a deleted directory — **not a vcam bug** | `cd ~` in that terminal |
| High CPU at 1080p | `blur` / `painterly` / `voronoi` are O(W·H·r²) | smaller param, drop to 720p, or pick cheaper filters |
