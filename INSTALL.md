# Installing vcam

vcam is a Linux program. It needs a C toolchain, the FFmpeg + libpng
development headers, and the `v4l2loopback` kernel module for its virtual
camera output. Screen-capture overlays additionally use `wf-recorder` on
Wayland.

- [Dependencies](#dependencies)
- [Build & install](#build--install)
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
