#!/usr/bin/env python3
"""Voicefield — animated dashboard background renderer for VoiceAgent.

Renders the same visual as dashboard-bg/voicefield.js (the live canvas), as a
seamless 8-second loop, then encodes:
    assets/voicefield-loop.mp4   1920x1080 @ 30fps  (muted, loops forever)
    assets/voicefield-loop.gif    800x450  @ 15fps  (palettegen, drop-in)
    assets/poster.png            1920x1080 still (hero/poster frame)

The loop is mathematically seamless: every time-dependent term is a sum of
sin(2*pi*k*t/T) with integer k, so frame 0 == frame T exactly.

Usage:  python3 render.py [--frames-dir /tmp/vf_frames]
Deps:   pip install pillow numpy imageio imageio-ffmpeg
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from imageio import v3 as imageio_v3
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Design tokens (keep in sync with voicefield.js)
# ----------------------------------------------------------------------------
T = 8.0                      # loop length, seconds
RIBBONS = 7                  # waveform ribbons
SEGMENTS = 260               # points per ribbon
DOTS = 46                    # particles riding the waves
RIPPLES = [                  # sonar rings: (birth as fraction of T, x, y, ribbon)
    (0.16, 0.24, None, 1),
    (0.58, 0.76, None, 5),
]

LIGHT = {
    "bg_top": (250, 251, 255),
    "bg_bottom": (242, 243, 252),
    "vignette": (237, 239, 252),
    "aurora_a": (224, 231, 255),
    "aurora_b": (232, 225, 253),
    "aurora_a2": (219, 244, 253),
    "dot_lattice": (99, 102, 241),
    "dot_lattice_a": 16,
    "stops": [(99, 102, 241), (139, 92, 246), (34, 211, 238)],  # indigo/violet/cyan
    "line_a": 175,       # core line alpha
    "glow_a": 80,        # glow alpha
    "line_w": 2.3,
    "glow_w": 9.0,
    "dot_a": 210,
}
DARK = {
    "bg_top": (12, 15, 30),
    "bg_bottom": (6, 8, 19),
    "vignette": (22, 26, 52),
    "aurora_a": (38, 44, 96),
    "aurora_b": (52, 38, 100),
    "aurora_a2": (20, 60, 92),
    "dot_lattice": (148, 163, 255),
    "dot_lattice_a": 22,
    "stops": [(129, 140, 248), (167, 139, 250), (34, 211, 238)],
    "line_a": 195,
    "glow_a": 100,
    "line_w": 2.4,
    "glow_w": 10.0,
    "dot_a": 235,
}


# ----------------------------------------------------------------------------
# Scene math (all periodic in T -> seamless loop)
# ----------------------------------------------------------------------------
def envelope(t: float, phase: float = 0.0) -> float:
    """Speech-envelope: agent breathes LISTEN -> THINK -> SPEAK -> LISTEN."""
    th = 2 * math.pi * (t / T) + phase
    e = 0.42 + 0.30 * math.sin(th) + 0.20 * math.sin(2 * th + 1.7) \
        + 0.12 * math.sin(3 * th + 4.2)
    return max(0.12, min(1.0, e))


def _center_weight(i: int) -> float:
    """Amplitude profile across ribbons — field swells at the middle."""
    x = i / (RIBBONS - 1)
    return 0.35 + 0.65 * math.sin(math.pi * x) ** 1.5


def _xenv(u: np.ndarray, t: float, seed: float) -> np.ndarray:
    """Traveling loud/quiet windows along x — syllable bursts, loop-safe."""
    th = 2 * math.pi * (t / T)
    a = 0.55 + 0.45 * np.sin(2 * math.pi * (2.0 * u) - 2 * th + seed)
    b = 0.70 + 0.30 * np.sin(2 * math.pi * (3.0 * u) + 3 * th + 1.3 * seed)
    m = 0.62 * a + 0.38 * b
    return np.clip(0.22 + 1.15 * np.power(m, 1.7), 0.06, 1.55)


def ribbon_y(i: int, xs: np.ndarray, t: float, W: float, H: float) -> np.ndarray:
    """y(x, t) for ribbon i — 3 traveling harmonics, integer time freqs."""
    u = xs / W
    th = 2 * math.pi * (t / T)
    seed = i * 2.399963  # golden-angle-ish per-ribbon phases
    y_base = H * (0.14 + 0.72 * i / (RIBBONS - 1)) \
        + H * 0.018 * math.sin(th + seed)          # slow vertical breathing
    amp = H * (0.030 + 0.135 * _center_weight(i)) * (0.7 + 0.3 * math.sin(seed * 1.9))
    env = envelope(t, phase=seed)
    xe = _xenv(u, t, seed)
    wave = (
        0.46 * np.sin(2 * math.pi * (1.6 * u + 0.07 * i) + 3 * th + seed) +
        0.30 * np.sin(2 * math.pi * (3.7 * u + 0.19 * i) - 5 * th + 2.1 * i) +
        0.24 * xe * np.sin(2 * math.pi * (17.0 * u + 0.31 * i) + 9 * th + 4.7 * i)
    )
    return y_base + amp * env * wave * xe


def dot_positions(t: float, W: float, H: float):
    """Particles riding ribbons; x loops integer times per cycle."""
    out = []
    th = 2 * math.pi * (t / T)
    for d in range(DOTS):
        f = (d * 0.61803398875) % 1.0            # fractional track position
        ri = d % RIBBONS
        laps = 1 + (d % 2)                       # 1 or 2 width-laps per loop
        x = ((f + laps * (t / T)) % 1.0) * W
        y = float(ribbon_y(ri, np.array([x]), t, W, H)[0])
        tw = 0.55 + 0.45 * math.sin(th * (1 + d % 3) + d)
        r = 2.0 + 1.8 * ((d * 7) % 5) / 4.0
        out.append((x, y, r, max(0.0, min(1.0, tw))))
    return out


def ripple_state(t: float, W: float, H: float):
    """Expand-and-fade sonar rings; each fully dies before T (loop-safe)."""
    life = 0.26 * T
    out = []
    for birth_f, xf, _yf, ri in RIPPLES:
        tb = birth_f * T
        if tb <= t < tb + life:
            p = (t - tb) / life                   # 0..1
            x = xf * W
            y = float(ribbon_y(ri, np.array([x]), t, W, H)[0])
            r = (0.05 + 0.20 * p) * W
            a = 1.0 - p
            out.append((x, y, r, a))
    return out


def color_at(u: float, stops) -> tuple:
    """Gradient across x through the palette stops."""
    n = len(stops) - 1
    s = min(max(u, 0.0), 1.0) * n
    k = min(int(s), n - 1)
    f = s - k
    a, b = stops[k], stops[k + 1]
    return tuple(int(a[c] + (b[c] - a[c]) * f) for c in range(3))


# ----------------------------------------------------------------------------
# Frame painter
# ----------------------------------------------------------------------------
def paint(w: int, h: int, t: float, theme: dict) -> Image.Image:
    W, H = float(w), float(h)

    # --- background: vertical gradient + soft radial vignette ---------------
    top = np.array(theme["bg_top"], float)
    bot = np.array(theme["bg_bottom"], float)
    grad = top[None, None, :] + (bot - top)[None, None, :] * (
        np.linspace(0, 1, h, dtype=float)[:, None, None])
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = W * 0.5, H * 0.46
    d2 = ((xx - cx) / (W * 0.62)) ** 2 + ((yy - cy) / (H * 0.62)) ** 2
    vin = np.array(theme["vignette"], float)[None, None, :]
    k = np.clip(1.0 - d2, 0, 1)[..., None] * 0.9
    rgb = grad * (1 - k) + vin * k
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")

    # --- aurora wash: 3 slow drifting color blobs behind everything ---------
    aurora = Image.new("RGB", (w, h), theme["bg_top"])
    ad = ImageDraw.Draw(aurora)
    th = 2 * math.pi * (t / T)
    blobs = [
        (0.28 + 0.04 * math.sin(th), 0.46 + 0.05 * math.sin(2 * th + 1.0),
         0.42, theme["aurora_a"]),
        (0.70 + 0.05 * math.sin(2 * th + 2.5), 0.34 + 0.06 * math.sin(th + 3.1),
         0.36, theme["aurora_b"]),
        (0.52 + 0.06 * math.sin(3 * th + 4.0), 0.68 + 0.04 * math.sin(2 * th + 5.2),
         0.34, theme["aurora_a2"]),
    ]
    for bxf, byf, brf, col in blobs:
        bx, by, br = bxf * w, byf * h, brf * w
        ad.ellipse((bx - br, by - br, bx + br, by + br), fill=col)
    aurora = aurora.filter(ImageFilter.GaussianBlur(radius=max(40, w / 8)))
    img = Image.blend(img, aurora, 0.55)

    # --- dot lattice (texture for the empty white) --------------------------
    lattice = Image.new("L", (w, h), 0)
    ld = ImageDraw.Draw(lattice)
    step = max(34, int(W / 46))
    for gy in range(step // 2, h, step):
        for gx in range(step // 2, w, step):
            ld.ellipse((gx - 1, gy - 1, gx + 1, gy + 1), fill=theme["dot_lattice_a"])
    tint = Image.new("RGB", (w, h), theme["dot_lattice"])
    img = Image.composite(tint, img, lattice)

    xs = np.linspace(0, W, SEGMENTS)

    # --- glow layer: all ribbons wide+blurred --------------------------------
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(RIBBONS):
        ys = ribbon_y(i, xs, t, W, H)
        pts = list(zip(xs.tolist(), ys.tolist()))
        for j in range(SEGMENTS - 1):
            u = (xs[j] + xs[j + 1]) / 2 / W
            r, g, b = color_at(u, theme["stops"])
            gd.line([pts[j], pts[j + 1]], fill=(r, g, b, theme["glow_a"]),
                    width=max(3, int(theme["glow_w"] * h / 1080 * 2)))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(4, 9 * w / 1920)))
    img.paste(glow, (0, 0), glow)

    # --- crisp core lines ----------------------------------------------------
    core = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    lw = max(1, round(theme["line_w"] * h / 1080))
    for i in range(RIBBONS):
        ys = ribbon_y(i, xs, t, W, H)
        pts = list(zip(xs.tolist(), ys.tolist()))
        for j in range(SEGMENTS - 1):
            u = (xs[j] + xs[j + 1]) / 2 / W
            r, g, b = color_at(u, theme["stops"])
            cd.line([pts[j], pts[j + 1]], fill=(r, g, b, theme["line_a"]), width=lw)
    img.paste(core, (0, 0), core)

    # --- sonar ripples -------------------------------------------------------
    for (x, y, r, a) in ripple_state(t, W, H):
        ring = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        col = color_at(x / W, theme["stops"])
        rw = max(1, int(2.5 * h / 1080))
        rd.ellipse((x - r, y - r * 0.62, x + r, y + r * 0.62),
                   outline=col + (int(120 * a),), width=rw)
        rd.ellipse((x - r * 0.55, y - r * 0.34, x + r * 0.55, y + r * 0.34),
                   outline=col + (int(60 * a),), width=rw)
        ring = ring.filter(ImageFilter.GaussianBlur(2))
        img.paste(ring, (0, 0), ring)

    # --- phoneme particles (halo + core) -------------------------------------
    dl = ImageDraw.Draw(img, "RGBA")
    for (x, y, r, tw) in dot_positions(t, W, H):
        col = color_at(x / W, theme["stops"])
        a = int(theme["dot_a"] * tw)
        hr = r * 3.2
        dl.ellipse((x - hr, y - hr, x + hr, y + hr), fill=col + (int(55 * tw),))
        dl.ellipse((x - r, y - r, x + r, y + r), fill=col + (a,))

    return img


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_pass(w, h, fps, seconds, theme, outdir: Path, tag: str, log=print):
    outdir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fps)
    all_args = [(w, h, f / fps, theme, outdir / f"{tag}_{f:04d}.png")
                for f in range(n)]
    # resume: skip frames that already exist with content
    args = [a for a in all_args
            if not (a[4].exists() and a[4].stat().st_size > 1000)]
    if len(args) < len(all_args):
        log(f"  [{tag}] reusing {len(all_args) - len(args)} existing frames")
    if args:
        if len(args) >= 8 and os.cpu_count() and os.cpu_count() > 1:
            import multiprocessing as mp
            with mp.Pool(min(os.cpu_count(), 4)) as pool:
                for done, _ in enumerate(pool.imap_unordered(_paint_one, args), 1):
                    if done % 60 == 0:
                        log(f"  [{tag}] {done}/{len(args)}")
        else:
            for k, a in enumerate(args):
                _paint_one(a)
                if k % 30 == 0:
                    log(f"  [{tag}] frame {k}/{len(args)}")
    return [a[4] for a in all_args]


def _paint_one(a):
    w, h, t, theme, path = a
    paint(w, h, t, theme).save(path, compress_level=1)
    return str(path)


def _pattern(frames: list[Path]) -> str:
    """'dir/hd_%04d.png' from the first frame path."""
    return str(frames[0].parent / (frames[0].name[:-8] + "%04d.png"))


def encode_mp4(frames: list[Path], fps: int, out: Path, log=print):
    ff = ffmpeg_exe()
    log(f"  encoding {out.name} …")
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", _pattern(frames),
         "-c:v", "libx264", "-preset", "slow", "-crf", "19",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True)


def encode_gif(frames: list[Path], fps: int, out: Path, log=print):
    ff = ffmpeg_exe()
    log(f"  encoding {out.name} (palette) …")
    pat = _pattern(frames)
    pal = frames[0].parent / "palette.png"
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", pat, "-vf", "palettegen=max_colors=128", str(pal)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", pat, "-i", str(pal),
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--skip-gif", action="store_true")
    args = ap.parse_args()

    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    tmp = Path(args.frames_dir) if args.frames_dir else Path(tempfile.mkdtemp(prefix="vf_"))
    log = print

    log("== MP4 pass: 1920x1080 @ 30fps ==")
    f1080 = render_pass(1920, 1080, 30, T, LIGHT, tmp, "hd", log)
    encode_mp4(f1080, 30, assets / "voicefield-loop.mp4", log)

    # poster: a mid-speech burst frame
    paint(1920, 1080, 3.05, LIGHT).save(assets / "poster.png")
    log("  poster.png ✓")

    if not args.skip_gif:
        log("== GIF pass: 800x450 @ 15fps ==")
        fgif = render_pass(800, 450, 15, T, LIGHT, tmp, "gif", log)
        encode_gif(fgif, 15, assets / "voicefield-loop.gif", log)

    for p in sorted(assets.iterdir()):
        log(f"{p.name:24s} {p.stat().st_size/1e6:.2f} MB")
    log(f"(frames in {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
