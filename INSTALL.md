# Installing vcam

vcam is a Linux program. It needs a C toolchain, the FFmpeg + libpng
development headers, and the `v4l2loopback` kernel module for its virtual
camera output. Screen-capture overlays additionally use `wf-recorder` on
Wayland.

- [Dependencies](#dependencies)
- [Build & install](#build--install)
- [Optional: OpenCV filters](#optional-opencv-filters)
- [One-time device setup (the Chromium fix)](#one-time-device-setup-the-chromium-fix)
- [Verify](#verify)
- [Uninstall](#uninstall)
- [Install-time troubleshooting](#install-time-troubleshooting)

---

## Dependencies

On Debian / Ubuntu (package names differ on other distros):

```bash
sudo apt install \
    libv4l-dev libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavdevice-dev libpng-dev \
    v4l2loopback-dkms pkg-config build-essential \
    wf-recorder slurp ncat
```

What each group is for:

- **libav\* / libpng** — the pipeline links these to decode the webcam's
  MJPEG, scale frames, and load PNG/JPEG overlays.
- **v4l2loopback-dkms** — provides the virtual camera device that browsers
  and video-call apps read from.
- **wf-recorder, slurp** — Wayland screen capture for the `--screen`
  overlay (`slurp` lets you select a region interactively).
- **ncat** — `vcam-ctl` uses it to talk to the running vcam over a Unix
  socket.

Runtime helpers vcam will use if present but that aren't hard build
deps: `swaymsg` (monitor layout), `v4l2-ctl` (device inspection in
`vcam-setup`), `grim`, `xdg-open`. See [NOTICE](NOTICE) for the full list
and their licenses.

`vcam-ctl` and `vcam-setup` are POSIX `sh`. `vcam-tune` requires **bash**
(it uses single-keypress input that has no POSIX equivalent).

---

## Build & install

```bash
git clone <repo-url> vcam && cd vcam
make
sudo make install                          # default → /usr/local/bin
```

The build is warning-clean under `-Wall -Wextra -Wpedantic`. `make`
produces two binaries: `vcam` and the `test_screen` diagnostic.

Choose a different destination if you prefer:

```bash
make install PREFIX=$HOME/.local             # ~/.local/bin (no sudo)
make install PREFIX=/usr                      # system-wide
make install DESTDIR=/tmp/pkg PREFIX=/usr     # staged, for packaging
```

`install` also drops LICENSE, NOTICE, README.md, and EXPRESSIONS.md into
`PREFIX/share/doc/vcam`. Make sure your chosen `bin` directory is on
`PATH`.

---

## Optional: OpenCV filters

vcam has seven extra filters powered by **OpenCV 5** — cartoon,
pencil-sketch, watercolor, optical-flow, background subtraction, face
effects, and person-segmentation background effects (see the OpenCV
section in [README.md](README.md)). They are **off by default**: the
normal build has no OpenCV dependency at all, and these filters simply
aren't compiled in.

**OpenCV 5 is the recommended version** — the filters are written
OpenCV-5-first. Built against 5.x, they use the features new in that
release:

- the **new graph-based DNN engine** (`ENGINE_AUTO`): 80%+ ONNX operator
  coverage (vs ~22% in 4.x), shape inference, constant folding, and
  operator fusion — the segmentation and face models load through it
- **first-class FP16** (`cv::hfloat`/`CV_16F`): the person-matte blend in
  `personfx` runs at half precision, halving the memory traffic of its
  heaviest per-frame step

OpenCV 4.5–4.x also works; the same code then compiles a
classic-engine/FP32 fallback path, selected at compile time. Ask a
running vcam which path it has with `vcam-ctl opencv`.

```bash
# OpenCV 5 dev files. If your distro still packages 4.x as
# libopencv-dev, build 5.x from source (github.com/opencv/opencv,
# branch 5.x) or use your distro's opencv5 package when available.
sudo apt install libopencv-dev

# Build with OpenCV support
make OPENCV=1
sudo make install     # remembers the OPENCV=1 flavor automatically
```

**After any rebuild, restart the running vcam** — `vcam-ctl` talks to the
already-running process, which keeps serving the old binary until you
replace it:

```bash
pkill vcam
vcam &
vcam-ctl opencv       # should now report your OpenCV version + engine
```

The build finds OpenCV via pkg-config, probing the module names
`opencv5`, `opencv4`, and `opencv` in that order — note that an OpenCV 5
install often still ships its pkg-config file under the `opencv4` name
for compatibility, which is fine: the version, not the module name,
decides which code path is compiled. If nothing is found, `make
OPENCV=1` stops with a clear message; you can force a name with
`OPENCV_PC=<name>`.

### Best matting quality: the ONNX Runtime backend

OpenCV 5's new DNN engine *loads* the RVM video-matting model but
currently **miscomputes** it (verified: empty mattes). ONNX Runtime is
RVM's reference runtime and computes it correctly. Enabling it gives
`personfx` its best-quality engine — temporally stable, hair-level
mattes:

```bash
vcam-setup --ort              # fetch onnxruntime (MIT, ~60 MB, one-time)
vcam-setup --models-rvm       # fetch the RVM model if you haven't
make OPENCV=1 ORT=1
sudo make install             # the build remembers ORT=1 afterwards
pkill vcam; vcam &
vcam-ctl opencv               # should report the ONNX Runtime backend
```

vcam picks matting engines in order: ONNX-Runtime RVM → OpenCV-DNN RVM
(auto-disabled if it produces empty mattes) → PP-HumanSeg. Every step
falls back automatically, so a missing library or model never breaks
the filter. `ORT_HOME=<dir>` overrides the library location.

### Model files (for `facefx` and `personfx`)

The five classic OpenCV filters work immediately. The two DNN-based ones
need small ONNX model files, which you download once:

```bash
vcam-setup --models          # → ~/.local/share/vcam/models
# or system-wide:
sudo vcam-setup --models --system
```

This fetches two models, both free and permissively licensed:

- **YuNet** face detector — ~230 KB, MIT license
- **PP-HumanSeg** person segmentation — a few MB, Apache-2.0 license
- **MobileFaceNet FER** expression recognition (for `moodfx`) — Apache-2.0

The `styleai` neural-style filter has its own optional pack (five
fast-neural-style networks, MIT license, ~6.5 MB each, from the ONNX
Model Zoo mirror on HuggingFace):

```bash
vcam-setup --models-style
```

They're pulled from the official OpenCV model repository on HuggingFace.
If you'd rather place them yourself, drop the `.onnx` files in any of
`$VCAM_MODEL_DIR`, `~/.local/share/vcam/models`, or
`/usr/local/share/vcam/models` — vcam checks those in order. Until the
models are present, `facefx` and `personfx` pass the frame through
unchanged and print a one-time hint.

Nothing about these filters is required: if you never build with
`OPENCV=1`, vcam is exactly the pure-C program it always was.

---

## One-time device setup (the Chromium fix)

Chromium-family browsers pick the first `/dev/video*` they see. If that's
your real webcam and vcam is already streaming from it, the browser gets
`EBUSY` → `NotReadableError` and the page errors out before you can even
pick a different device. Firefox doesn't have this problem.

`vcam-setup --install` writes `/etc/modprobe.d/vcam.conf` so the
v4l2loopback module claims `/dev/video0` *before* USB cameras enumerate.
Your real cam then sits at `/dev/video1+`, and the loopback is what
Chromium auto-picks. The `exclusive_caps=1` parameter it sets is also
mandatory for Firefox to see the device.

```bash
sudo vcam-setup --install
```

Then reboot, or do the manual module reload it prints:

```bash
sudo modprobe -r uvcvideo
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback
sudo modprobe uvcvideo
```

---

## Verify

```bash
vcam-setup
# → "OK — loopback at /dev/video0 comes before real cam at /dev/video1"

vcam                 # start it (defaults: input /dev/video1, output /dev/video0)
ffplay /dev/video0   # in another terminal, watch the output
```

In your video-call app, pick **vcam** as the camera.

Not sure what your webcam supports? Ask it:

```bash
vcam --list-formats
```

---

## Uninstall

```bash
sudo make uninstall          # uses the same PREFIX/DESTDIR you installed with
```

To also undo the device setup, remove `/etc/modprobe.d/vcam.conf` and
reload the modules (or reboot).

---

## Install-time troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot open /dev/video0` | v4l2loopback not loaded | `sudo vcam-setup --install`, then reboot or manual reload |
| `NotReadableError` in Chromium | Loopback isn't at /dev/video0 | `sudo vcam-setup --install` |
| Firefox shows no camera | v4l2loopback without `exclusive_caps=1` | vcam-setup writes it correctly |
| `vcam-ctl: ncat: command not found` | ncat not installed | `sudo apt install ncat` |
| Screen overlay is black | Compositor doesn't support wlr-screencopy | Use Sway, Hyprland, river, niri, or wayfire |
| `pkg-config: command not found` during build | build tools missing | `sudo apt install pkg-config build-essential` |

For usage and runtime issues (filters, recording, screen capture), see
the Troubleshooting section in [README.md](README.md).
