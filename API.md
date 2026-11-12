# vcam Control API

> 100% Vibecode, but tested.
> Self-contained reference. Hand this file to anyone (or any AI chat)
> who writes a program that controls vcam.

<!-- Created-by: Claude Opus 5.5 -->

## What it controls

vcam is a virtual webcam. It reads the real webcam, composes overlays,
and writes to a v4l2loopback device that video-call apps use as a camera.

Through the API a program can:

- start / stop a **fullscreen screen share** (the screen replaces the camera image),
- show the **webcam as a picture-in-picture** (PiP) and move / resize it,
- **switch the camera** format, resolution and fps live,
- **read the state**: is sharing on, where is the cam, is the cam over the screen.

vcam must already be running. The API does not start it.

## Connection

| | |
|---|---|
| Transport | Unix domain socket, `SOCK_STREAM` |
| Path | `$VCAM_SOCK` if set, else `/tmp/vcam.sock` |
| Access | file mode `0600`, so only the user running vcam can connect |
| Clients | up to 16 at once; the 17th gets `ERR too many clients` |
| Lifetime | keep the connection open, or connect per command; both work |

vcam is running when the socket exists and `connect()` succeeds.
When vcam exits, open connections get EOF.

## Protocol

- Text (UTF-8). One command per line, ending in `\n`. `\r` is ignored.
- A line is at most 1023 bytes.
- Words are separated by spaces. Commands are case-sensitive.
- Every reply line starts with `OK` or `ERR`.
  One exception: `status json` replies with one JSON line starting with `{`.
- Commands run one at a time across all clients, in the order received.
- vcam never sends anything unprompted (no events). Poll `status json` instead.
- `quit` replies `OK bye`, then the server closes the connection.

### Number of reply lines

All commands in this file reply with **exactly one line**, except `monitors`,
which replies with one line per monitor.

To read a reply of unknown length, send a marker line `#` after the
command. vcam answers `#` with `OK noop`. Read lines until you see
`OK noop`. Everything before it is the reply.

```
→ monitors
→ #
← OK 1 DP-1 1920x1080+0+0
← OK 2 HDMI-A-1 2560x1440+1920+0
← OK noop
```

### Timing

Most commands answer in about a millisecond. `share start` takes around
0.3 s (it starts the screen capture). `share stop` can take up to about
1 s (it waits for the capture process to end). `capture set` takes
0.2 to 1 s (the camera is re-opened). Use a read timeout of at
least **3 s**.

## Coordinates

- All positions and sizes are in **output pixels** (`output_w` × `output_h`,
  e.g. 1280×720), with the origin at the top-left.
- Overlays may stick out of the frame. The part outside is clipped.
- Monitor geometries (`WxH+X+Y`) are in **desktop** coordinates, as your
  compositor reports them (scale already applied).

## Commands

### `status` and `status json`

The whole state in one line.

```
→ status
← OK version=1.1.0 build=abc1234 input=webcam output_w=1280 output_h=720 fps=30 share=yes … recording=no
→ status json
← {"ok":true,"version":"1.1.0","build":"abc1234","input":"webcam","output_w":1280,…,"recording":false}
```

Plain form: `key=value` pairs separated by spaces, with booleans as `yes` / `no`.
JSON form: one flat object, with booleans as `true` / `false` and integers as numbers.
Values never contain spaces.

| Key | Type | Meaning |
|---|---|---|
| `version` | string | vcam version, e.g. `1.1.0` |
| `build` | string | md5 build hash (7 chars) |
| `input` | string | `webcam`, or `none` if vcam was started with `--no-input` |
| `output_w`, `output_h` | int | output frame size |
| `fps` | int | output frame rate |
| `capture` | bool | a camera is open and delivering |
| `capture_device` | string | camera device, e.g. `/dev/video1`; `-` with `--no-input` |
| `capture_fmt` | string | camera format, e.g. `MJPG`; `-` when off |
| `capture_w`, `capture_h` | int | camera resolution (0 when off) |
| `capture_fps` | int | camera frame rate (0 when off) |
| `capture_fit` | string | `crop` or `stretch`; `-` when off |
| `share` | bool | a screen overlay exists |
| `share_id` | string | its overlay id (`share` when started via the API), `-` if none |
| `share_fullscreen` | bool | the screen overlay covers the whole output |
| `share_monitor` | string | monitor name, `-` if unknown or not started via `share start` |
| `share_geometry` | string | captured area `WxH+X+Y`, `-` if unknown |
| `share_x`, `share_y`, `share_w`, `share_h` | int | screen overlay rect (0 when off) |
| `cam` | bool | a webcam overlay (PiP) exists |
| `cam_id` | string | its overlay id (`cam` when made via the API), `-` if none |
| `cam_pos` | string | `tl` `tr` `bl` `br` `center` `full` `custom`, or `-` when off |
| `cam_x`, `cam_y`, `cam_w`, `cam_h` | int | cam rect (0 when off) |
| `cam_z` | int | draw order index, 0 = bottom; `-1` when off |
| `cam_margin` | int | gap (px) used for the corner positions |
| `cam_over_screen` | bool | a cam is drawn **above** the screen, overlaps it, and is visible |
| `overlays` | int | total overlays (images, videos, screens, cams) |
| `filters` | int | active filters |
| `recording` | bool | vcam is recording to a file |

Notes:

- `share` and `cam` also see overlays created by other commands
  (e.g. `vcam-ctl screen --full`, `vcam-ctl webcam br`). If there are
  several, the one with id `share` / `cam` wins, otherwise the topmost one.
- `cam_pos` is worked out from the current rect: `tr` means the cam sits
  exactly in the top-right corner (minus `cam_margin`). Anything else is
  `custom`.
- The JSON form may gain new keys later. Ignore keys you don't know.

### `get <key>`

One value from `status`, in plain form.

```
→ get cam_pos          ← OK br
→ get cam_over_screen  ← OK yes
→ get nope             ← ERR unknown key 'nope'
```

### `version`

```
→ version   ← OK vcam 1.1.0 (build abc1234)
```

### `monitors`

One line per monitor: `OK <n> <name> <WxH+X+Y>`. `n` starts at 1.

```
→ monitors
← OK 1 DP-1 1920x1080+0+0
← OK 2 HDMI-A-1 2560x1440+1920+0
```

Error: `ERR no monitor detected`. Detection uses sway, Hyprland,
wlr-randr or xrandr, and falls back to `/sys/class/drm`, which knows
sizes but reports every position as `0,0`.

### `share start [<target>] [options]`

Captures a monitor and shows it over the whole output.
If a share is already running, it is replaced.

`<target>`, all optional:

| Form | Example | Meaning |
|---|---|---|
| (none) | `share start` | first monitor from `monitors` |
| number | `share start 2` | monitor #2 |
| name | `share start DP-1` | monitor by name |
| geometry | `share start 1920x1080+0+0` | any desktop area |
| `monitor=` / `geo=` | `share start monitor=DP-1` | same, explicit |

Options:

| Option | Default | Meaning |
|---|---|---|
| `fit=stretch` / `fit=contain` | `stretch` | `contain` keeps the aspect ratio and centers it; the bars show the webcam |
| `cam=<pos>` | — | also show the cam PiP at `<pos>` (see `cam on`) |
| `cam=off` | — | also remove the cam PiP |
| `camw=<px>` | current or 320 | cam width; on its own it also turns the cam on |

Without `cam=` / `camw=`, an existing cam stays as it is and is raised above the screen.

Reply: the share keys plus `cam`, `cam_pos` and `cam_over_screen`:

```
→ share start 2 cam=br
← OK share=yes share_id=share share_fullscreen=yes share_monitor=HDMI-A-1 share_geometry=2560x1440+1920+0 share_x=0 share_y=0 share_w=1280 share_h=720 cam=yes cam_pos=br cam_over_screen=yes
```

Errors:

- `ERR no monitor detected — pass a geometry: share start WxH+X+Y`
- `ERR unknown monitor (see 'monitors')`
- `ERR screen capture failed (…)`: `wf-recorder` (Wayland) or `ffmpeg` (X11) is missing or failed
- `ERR share on, but cam: <reason>`: the share **is** running, only the cam part failed

### `share stop` · `share toggle` · `share status`

| Command | Effect |
|---|---|
| `share stop` (or `share off`) | removes the screen overlay; the webcam image is back |
| `share toggle [<target>] [options]` | stop if sharing, else `share start …` |
| `share status` (or just `share`) | no change, only the reply |

All of them reply in the same format as `share start`.

### `cam …`: webcam picture-in-picture

Positions (`<pos>`): `tl` `tr` `bl` `br` (corners), `center`, `full` (whole output).

| Command | Effect |
|---|---|
| `cam on [<pos>] [<w>]` | show the PiP, or re-place it if already shown. Defaults: current position, else `tr`; current width, else 320 |
| `cam pos <pos> [<w>]` | like `cam on`, but `<pos>` is required |
| `cam move <x> <y>` | free position (top-left corner of the PiP), so `cam_pos` becomes `custom` |
| `cam size <w> [<h>]` | resize but keep the position; `<h>` defaults to the camera's aspect ratio; resizing a `full` cam moves it to `tr` |
| `cam margin [<px>]` | gap to the output edge for the corner positions (default 0); a cam sitting in a corner is moved at once; without `<px>`, replies `OK margin=<n>` |
| `cam raise` | draw the cam above every other overlay |
| `cam off` | remove the PiP, reply `OK cam=off` |
| `cam toggle` | off if shown, else `cam on` |
| `cam get` (or just `cam`) | no change, only the reply |

The PiP is always drawn above the shared screen.

Reply (all except `off` and `margin` without a value):

```
OK cam=on id=cam pos=br x=960 y=540 w=320 h=180 z=1 over_screen=yes
OK cam=off
```

Errors: `ERR cam is off …` (from `move`, `size` and `raise`),
`ERR unknown position (use tl|tr|bl|br|center|full)`, `ERR cam size too small`.

### `capture …`: camera format, resolution and fps

Switch the camera while vcam runs. The **output size never changes**:
it is fixed at startup (`vcam --res WxH`), because apps reading the
virtual camera depend on it. The camera picture is scaled to it.

| Command | Effect |
|---|---|
| `capture` / `capture get` | current mode, no change |
| `capture formats` | every mode, one line each: `OK <fmt> <WxH> <max fps>` (use the `#` marker) |
| `capture formats json` | the same as one line: `{"ok":true,"formats":[{"fmt":"MJPG","w":1280,"h":720,"fps":30},…]}` |
| `capture set [<fmt>] [<WxH>\|max] [<fps>] [fit=crop\|stretch]` | switch the camera live |
| `capture fit crop\|stretch` | change only how the picture is fitted (no camera restart) |

`capture set` takes its arguments in any order, and at least one is needed:

| Argument | Values |
|---|---|
| `<fmt>` or `fmt=` | `MJPG` `YUYV` `NV12` `YU12` `BGR3` `RGB3` (any case, `MJPEG` also works), or `auto` = best for the size and fps |
| `<WxH>` or `res=` | an exact size from `capture formats`, or `max` = the largest size that still reaches the fps |
| `<fps>` or `fps=` | frame rate |
| `fit=` | `crop` (default) keeps the aspect ratio and cuts the edges; `stretch` distorts instead |

Rules:

- A value you leave out keeps its current value, if the camera has that combination.
  Otherwise vcam picks the best match for the values you did give.
  A kept fps that the camera can't reach becomes the highest fps it can.
- A value you give is **never** changed. If it's impossible, you get an error.

Reply:

```
OK capture=on device=/dev/video1 fmt=YUYV w=640 h=480 fps=30 fit=crop output=1280x720
```

Examples, for a camera that offers YUYV 1280x720 at 10 fps max:

```
→ capture set YUYV 640x480 30   ← OK capture=on … fmt=YUYV w=640 h=480 fps=30 …
→ capture set YUYV 1280x720 30  ← ERR YUYV 1280x720 supports at most 10 fps
→ capture set YUYV              ← OK capture=on … fmt=YUYV w=1280 h=720 fps=10 …   (size kept)
→ capture set 15                ← OK capture=on … fps=15 …                          (fps only)
```

Errors:

- `ERR camera has no mode … (see 'capture formats')`
- `ERR … supports at most N fps`
- `ERR unknown format '…'`
- `ERR camera refused …`: the camera rejected the mode; the **old mode is back**
- `ERR vcam runs with --no-input (no camera)`

A switch takes about 0.2 to 1 s, and the picture pauses briefly.
A mode smaller than the output is upscaled, so it looks softer, but it
needs less USB bandwidth and CPU.

## Behaviour to know

- **State is not saved.** After a vcam restart there is no share and no cam,
  the margin is back to 0, and the camera uses the startup mode again.
- **Other clients can change things** at any time (`vcam-ctl`, hotkeys,
  other apps). Don't cache state; read `status json` before deciding.
- **`input=none`** (vcam started with `--no-input`): there is no camera,
  so the PiP shows black. Normal use is to start vcam *with* the webcam.
- The lower-level commands (`list`, `add`, `move`, `resize`, `remove`,
  `raise`, `lower`, `z`, `filter …`, `record …`) also exist, see
  `vcam-ctl --help`. Some reply with several lines, so use the `#` marker.

## Example: Python client

Standard library only. Handles multi-line replies, timeouts and errors.

```python
import json, os, socket

class VcamError(Exception):
    pass

class Vcam:
    def __init__(self, path=None, timeout=3.0):
        self.path = path or os.environ.get("VCAM_SOCK", "/tmp/vcam.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(self.path)          # raises if vcam is not running
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")

    def cmd(self, line):
        """Send one command, return its reply lines (without 'OK ')."""
        self.f.write(line + "\n#\n")          # '#' → 'OK noop' marks the end
        self.f.flush()
        out = []
        while True:
            r = self.f.readline()
            if not r:
                raise VcamError("vcam closed the connection")
            r = r.rstrip("\n")
            if r == "OK noop":
                break
            out.append(r)
        for r in out:
            if r.startswith("ERR"):
                raise VcamError(r[4:])
        return [r[3:] if r.startswith("OK ") else r for r in out]

    def status(self):
        return json.loads(self.cmd("status json")[0])

    def get(self, key):
        return self.cmd(f"get {key}")[0]

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    v = Vcam()
    print(v.cmd("version")[0])
    print("monitors:", v.cmd("monitors"))
    v.cmd("share start cam=br")
    s = v.status()
    print("sharing:", s["share"], "cam at:", s["cam_pos"], "over screen:", s["cam_over_screen"])
    v.cmd("cam pos tl")
    v.cmd("share stop")
    v.close()
```

Other languages: anything that opens a Unix stream socket works
(Node `net.createConnection("/tmp/vcam.sock")`, Go `net.Dial("unix", …)`,
C `socket(AF_UNIX, SOCK_STREAM, 0)`). Use the same line protocol.

## Example: shell

`vcam-ctl` wraps the API. It prints bare values (no `OK`) and exits with 1 on `ERR`.

```bash
vcam-ctl share start cam=br          # share monitor 1, PiP bottom-right
vcam-ctl get cam_pos                 # → br
vcam-ctl get cam_over_screen         # → yes
vcam-ctl cam pos tl
vcam-ctl status                      # one key=value per line
vcam-ctl status --json               # one JSON line
vcam-ctl share stop
```

`vcam-ctl status` prints `key=value` lines, so a shell can load them directly:

```bash
eval "$(vcam-ctl status)"
[ "$share" = yes ] && echo "sharing $share_monitor, cam at $cam_pos"
```

Raw socket, without vcam-ctl:

```bash
printf 'status json\nquit\n' | ncat -U /tmp/vcam.sock     # or: socat - UNIX-CONNECT:/tmp/vcam.sock
```

## Recipes

**Share button.** Call `share toggle cam=br`, then show `share` from the reply.

**Status indicator** (tray, bar). Poll `status json` every 0.5 to 1 s and
read `share`, `cam` and `cam_over_screen`.

**Corner hotkeys.** Bind `cam pos tl` / `tr` / `bl` / `br`.

**Quality picker.** Fill a menu from `capture formats json`, then call
`capture set <fmt> <WxH> <fps>` with the user's choice.

**Wait for vcam.** Retry `connect()` every 0.5 s until it succeeds.

**Share a single window or area.** Get its geometry from your compositor
(e.g. `slurp` on Wayland, `slop` on X11), then call `share start WxH+X+Y`.
