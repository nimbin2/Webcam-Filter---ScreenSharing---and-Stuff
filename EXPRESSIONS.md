# The mathwarp Expression Cookbook

`mathwarp` turns the webcam into a programmable canvas. You write math,
and every pixel obeys it — in real time, recompiled the instant you hit
Enter. Two independent expression sets drive it:

- **warp** — `filter expr <dx> <dy>` — moves each pixel by `(dx, dy)`
- **color** — `filter color <r> <g> <b>` — recolors each pixel (0–255 per channel)

Use either alone, or both together. Add the filter first:

```sh
vcam-ctl filter add mathwarp 1.0     # the 1.0 is a global warp-amplitude knob
```

Then paste any recipe below. Every expression here is **verified to
compile**. There are no spaces inside an expression (spaces separate the
dx/dy or r/g/b arguments).

---

## The language

**Variables available everywhere**

| var | meaning |
|-----|---------|
| `x` `y` | pixel coordinates (0,0 = top-left) |
| `w` `h` | frame width / height |
| `t` | time in seconds (this is what makes things move) |
| `r` | distance from the center |
| `a` | angle from the center, in radians (−π … π) |

**Extra variables in color expressions only**

| var | meaning |
|-----|---------|
| `cr` `cg` `cb` | this pixel's source Red / Green / Blue (0–255) |

**Functions:** `sin` `cos` `abs` `sqrt` `floor`
**Operators:** `+ - * / %` · comparisons `> < >= <= == !=` · ternary `cond?a:b` · parentheses · unary `-`

A comparison returns `1.0` or `0.0`, so you can multiply by it (`sin(t)*(r<200)`)
or branch with the ternary (`r<200?sin(t)*20:0`). Division and modulo by
zero are guarded and yield 0 — expressions can never crash.

---

## WARP recipes — `filter expr <dx> <dy>`

### Gravity & vortex

```sh
# Black hole — everything is sucked toward the center, harder up close
vcam-ctl filter expr (x-w/2)*(-0.3)*sqrt(w*w/4/(r*r+1)) (y-h/2)*(-0.3)*sqrt(w*w/4/(r*r+1))

# Twist tornado — a rotational shear that grows with radius
vcam-ctl filter expr sin(a*2+r/20-t*3)*r*0.06 cos(a*2+r/20-t*3)*r*0.04

# Breathing vortex — a whirlpool that inhales and exhales
vcam-ctl filter expr sin(a*3+t*2)*(sin(t)*20+20)*sin(r/50) cos(a*3+t*2)*(sin(t)*20+20)*sin(r/50)

# Pinch pulse — the whole image clenches to a point and releases
vcam-ctl filter expr (x-w/2)*sin(t*3)*0.12 (y-h/2)*sin(t*3)*0.12
```

### Waves & ripples

```sh
# Ripple pond — concentric waves flowing out from the center
vcam-ctl filter expr sin(sqrt((x-w/2)*(x-w/2)+(y-h/2)*(y-h/2))/8-t*4)*12 0

# Shockwave burst — a ring of distortion that only exists within radius 200
vcam-ctl filter expr cos(r/12-t*6)*(r<200?20:0) sin(r/12-t*6)*(r<200?20:0)

# Wobble grid — two interfering sine fields, liquid glass
vcam-ctl filter expr sin(y/15+t*3)*8+cos(x/25-t)*5 cos(x/15+t*3)*8+sin(y/25-t)*5

# Pulsing rings with angular petals — a living flower
vcam-ctl filter expr sin(r/10-t*5)*cos(a*3)*15 sin(r/10-t*5)*sin(a*3)*15
```

### Geometric & tiled

```sh
# Checkerslide — alternating 50px tiles drift in opposite directions
vcam-ctl filter expr (floor(x/50)%2==0?1:-1)*sin(t*2)*20 (floor(y/50)%2==0?1:-1)*cos(t*2)*20

# Quadrant spin — each screen quarter shears independently
vcam-ctl filter expr (x>w/2?1:-1)*(y>h/2?1:-1)*sin(t*4)*15 (x>w/2?-1:1)*sin(t*4)*15

# Kaleidofold — 8-fold angular ripples modulated by radius
vcam-ctl filter expr sin(a*8+t)*20*sin(r/40) cos(a*8+t)*20*sin(r/40)

# DNA helix — horizontal sine displacement scrolling down the frame
vcam-ctl filter expr sin(y/20+t*2)*30 0
```

### Chaos & motion

```sh
# Earthquake — a violent horizontal shudder, opposite on each half
vcam-ctl filter expr sin(t*30+y*0.3)*(x>w/2?8:-8) cos(t*25+x*0.2)*6

# Melting wax — the image sags and drips downward
vcam-ctl filter expr sin(x/30+t)*6 abs(sin(x/20-t*2))*sin(y/40)*25

# Radial strobe — snaps inward on the beat, then frees itself
vcam-ctl filter expr (x-w/2)*(sin(t*8)>0?0.15:0) (y-h/2)*(sin(t*8)>0?0.15:0)
```

---

## COLOR recipes — `filter color <r> <g> <b>`

Turn recolor off any time with `vcam-ctl filter color off`.

### Channel play

```sh
# Channel rotate — R←G, G←B, B←R (alien palette)
vcam-ctl filter color cg cb cr

# Invert — photographic negative
vcam-ctl filter color 255-cr 255-cg 255-cb

# Matrix green — collapse to luminance, keep only green
vcam-ctl filter color 0 (cr+cg+cb)/3 0

# Duotone teal-orange — the cinematic blockbuster grade
vcam-ctl filter color cr*0.5+cb*0.3 cg*0.6+cr*0.2 cb*0.8
```

### Posterize & threshold

```sh
# Posterize — crush each channel to 4 levels, poster-print look
vcam-ctl filter color floor(cr/64)*64 floor(cg/64)*64 floor(cb/64)*64

# Solarize — invert only the bright half of each channel (Sabattier effect)
vcam-ctl filter color cr>128?255-cr:cr cg>128?255-cg:cg cb>128?255-cb:cb

# Neon glow — push saturated reds and blues to full blast
vcam-ctl filter color cr>100?255:cr cg cb>100?255:cb

# Heatmap — hot areas glow, cold areas go blue
vcam-ctl filter color cr+cg+cb>380?255:cr (cr+cg+cb)/3 cr+cg+cb<200?150:cb
```

### Time & space driven

```sh
# Rainbow by x — a spectrum sweeps across the frame, scrolling forever
vcam-ctl filter color sin(x/40+t)*127+128 sin(x/40+t+2)*127+128 sin(x/40+t+4)*127+128

# Strobe flash — the whole frame blinks white on the beat
vcam-ctl filter color sin(t*10)>0?255:cr cg cb

# Chroma pulse — red and blue breathe out of phase
vcam-ctl filter color cr*(sin(t*3)*0.5+0.5) cg cb*(cos(t*3)*0.5+0.5)

# Radial tint — warm in the center, cool at the edges
vcam-ctl filter color cr*(r/400) cg cb*(1-r/400)

# Scanline color — every 4th row burns red, CRT-style
vcam-ctl filter color y%4==0?255:cr cg cb
```

---

## Combining warp + color

The two run in sequence (warp first, then color reads the warped pixel),
so they stack into something greater than either. A few showpieces:

```sh
# ── Psychedelic wormhole ──
vcam-ctl filter expr  sin(a*3+t*2)*(sin(t)*20+20)*sin(r/50) cos(a*3+t*2)*(sin(t)*20+20)*sin(r/50)
vcam-ctl filter color sin(r/30+t)*127+128 sin(r/30+t+2)*127+128 sin(r/30+t+4)*127+128

# ── Earthquake TV ── (violent shake + channel-split glitch color)
vcam-ctl filter expr  sin(t*30+y*0.3)*(x>w/2?8:-8) cos(t*25+x*0.2)*6
vcam-ctl filter color cg cr cb

# ── Liquid chrome ── (glassy wobble + solarized metal)
vcam-ctl filter expr  sin(y/15+t*3)*8+cos(x/25-t)*5 cos(x/15+t*3)*8+sin(y/25-t)*5
vcam-ctl filter color cr>128?255-cr:cr cg>128?255-cg:cg cb>128?255-cb:cb

# ── Black-hole rainbow ── (gravitational pull + spectrum by radius)
vcam-ctl filter expr  (x-w/2)*(-0.3)*sqrt(w*w/4/(r*r+1)) (y-h/2)*(-0.3)*sqrt(w*w/4/(r*r+1))
vcam-ctl filter color cr*(r/400) sin(r/40+t)*127+128 cb*(1-r/400)
```

---

## Tips

- The **global amplitude knob** (`filter add mathwarp <amp>`, or
  `filter set <index> <amp>` live) scales *all* warp displacement — so you
  can build a wild expression and then fade it in from 0, or ride it in
  `vcam-tune` while the formula stays fixed.
- Warp is evaluated on a coarse grid and smoothly upsampled (cheap); color
  is evaluated per-pixel (a little heavier). Both comfortably hold 720p30.
- Keep displacement roughly under ±40 px or the image tears at the edges —
  which is itself a look, if you want it.
- `t` never resets, so slow terms like `t*0.2` drift gently while fast ones
  like `t*30` strobe. Mixing timescales in one expression is where the
  magic is.
