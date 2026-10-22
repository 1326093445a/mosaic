#!/usr/bin/env python
"""Render the mechanism animation: discrete search, bindcraft relaxation, the cliff.

Why this exists. Section 33 reports that a relaxation which descends 28.25 on
its own loss hands the discrete search a sequence worth 0.0002. That is the
whole result, and it is the one thing a table cannot show: the descent and the
collapse happen in two different spaces, so a reader sees two unrelated numbers
rather than one cliff. This draws them on a single timeline.

Quantities on screen are the measured ones -- the 40.5 -> 12.25 soft loss, the
0.0002 hard score, the 0.5801 the control reaches, the 240/120/216/24 stage
split bindcraft actually ran at 600 steps, 15 wanted substitutions against a
budget of 5. Residue identities, which positions move, and the order of
proposals are illustrative: nothing downstream depends on them and the real
ones would make the strip unreadable.

Writes an H.264 mp4 and a palette-optimised gif beside each other. Needs
ffmpeg, which is also what the gif pass uses.

    python docs/make_mechanism_video.py [--fps 25] [--out docs/build]
"""
from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

PAPER = "#eef1f4"
SURFACE = "#f7f9fb"
INK = "#15191e"
INK_SOFT = "#4d5760"
INK_FAINT = "#7d878f"
RULE = "#cfd6dd"
ACCENT = "#2f5d82"
ACCENT_DIM = "#8fa9bf"
NULL = "#9c4a35"
PASS = "#3f6b54"

SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]

N = 29
BUDGET = 5
WANTED = 15
SOFT_FROM, SOFT_TO = 40.5, 12.25
HARD_SCORE = 0.0002
DISC_FINAL = 0.5801
STAGES = [("logits", 240), ("logits", 120), ("soft", 216), ("hard", 24)]

_rng = np.random.default_rng(17)
PULL = _rng.uniform(0.15, 1.0, N)
ORDER = np.argsort(-PULL)
KEEP = sorted(ORDER[:BUDGET].tolist())
WANT = sorted(ORDER[:WANTED].tolist())

DISC = [
    dict(pos=4, ok=True, score=0.09), dict(pos=19, ok=False, score=0.09),
    dict(pos=11, ok=True, score=0.21), dict(pos=25, ok=False, score=0.21),
    dict(pos=2, ok=True, score=0.36), dict(pos=16, ok=True, score=0.47),
    dict(pos=22, ok=False, score=0.47), dict(pos=8, ok=True, score=0.55),
    dict(swap=(4, 13), score=0.57), dict(swap=(11, 27), score=DISC_FINAL),
]


def ease(t: float) -> float:
    """Smoothstep, so stage transitions do not snap."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


class Scene:
    """The frame-by-frame state machine.

    Every scene is a closed frame range and `state(f)` is a pure function of
    the frame index, so any frame can be rendered alone -- which is what makes
    the render reproducible and lets a still be pulled for a slide.
    """

    def __init__(self, fps: int):
        self.fps = fps
        s = lambda sec: max(1, int(round(sec * fps)))
        self.cuts = []
        t = 0
        for name, sec in [
            ("title", 2.6), ("discrete", 10.4), ("relax", 10.8),
            ("project", 4.0), ("cliff", 4.6), ("outcome", 5.2),
        ]:
            self.cuts.append((name, t, t + s(sec)))
            t += s(sec)
        self.total = t

    def scene_at(self, f):
        for name, a, b in self.cuts:
            if a <= f < b:
                return name, (f - a) / max(1, b - a - 1)
        return self.cuts[-1][0], 1.0

    def state(self, f):
        """Return everything the painter needs for frame `f`."""
        name, u = self.scene_at(f)
        st = dict(
            scene=name, title="", sub="", caption="", note="",
            par=np.ones(N), alt=np.zeros(N),
            dim=np.zeros(N, bool), probe=-1, reject=-1, revert=np.zeros(N, bool),
            spent=0, meters=[], big=None, schedule=None,
        )

        if name == "title":
            st["title"] = "Two ways to search 29 positions"
            st["sub"] = "P17 nanobody against JN1 · edit budget 5"
            st["caption"] = ("Both stages change residues at the same 29 designable "
                             "positions and are scored by the same frozen predictor.")
            st["dim"][:] = True

        elif name == "discrete":
            st["title"] = "1 · Discrete search"
            st["sub"] = "one whole substitution per proposal, scored and kept or thrown away"
            k = min(len(DISC) - 1, int(u * len(DISC)))
            frac = (u * len(DISC)) - k
            spent, score = 0, 0.0
            held = {}
            for i, mv in enumerate(DISC[:k]):
                if "swap" in mv:
                    held.pop(mv["swap"][0], None)
                    held[mv["swap"][1]] = True
                elif mv["ok"]:
                    held[mv["pos"]] = True
                score = mv["score"]
            spent = len(held)
            mv = DISC[k]
            if "swap" in mv:
                if frac > 0.45:
                    held.pop(mv["swap"][0], None)
                    held[mv["swap"][1]] = True
                    score = mv["score"]
                st["note"] = "budget full — exchange: revert one, add one"
                st["revert"][mv["swap"][0]] = frac <= 0.45
            else:
                if frac < 0.42:
                    st["probe"] = mv["pos"]
                    st["note"] = "proposing a substitution"
                elif mv["ok"]:
                    held[mv["pos"]] = True
                    score = mv["score"]
                    st["note"] = f"accepted — {len(held)} of {BUDGET} edits spent"
                else:
                    st["reject"] = mv["pos"]
                    st["note"] = "rejected — the score did not improve"
            for i in held:
                st["par"][i], st["alt"][i] = 0.0, 1.0
            st["spent"] = len(held)
            st["meters"] = [("score", score / 0.6, f"{score:.4f}", ACCENT)]
            st["caption"] = ("200 score calls per seed, about 51 accepted. It stops on "
                             "the call budget, never on edits — and every winner "
                             "spends the whole edit budget.")

        elif name == "relax":
            st["title"] = "2 · Continuous relaxation — the bindcraft schedule"
            st["sub"] = "every position holds a distribution over 20 residues, and they all move at once"
            g = ease(u)
            for i in range(N):
                a = PULL[i] * g * 0.88
                st["par"][i], st["alt"][i] = 1.0 - a, a
            loss = SOFT_FROM - g * (SOFT_FROM - SOFT_TO)
            st["meters"] = [("soft loss", g, f"{loss:.2f}", PASS)]
            st["schedule"] = u
            st["note"] = f"descended {SOFT_FROM - loss:.2f} of {SOFT_FROM - SOFT_TO:.2f}"
            st["caption"] = ("600 steps on BindCraft's four-stage schedule, rescaled from "
                             "50:25:45:5. By the end, 15 positions prefer a residue that "
                             "is not the parent's — against a budget of 5.")

        elif name == "project":
            st["title"] = "3 · The projection back to whole residues"
            st["sub"] = "keep the five highest-margin substitutions, discard the other ten"
            g = ease(u)
            for i in range(N):
                a = PULL[i] * 0.88
                if i in KEEP:
                    st["par"][i] = (1 - a) * (1 - g)
                    st["alt"][i] = a + (1 - a) * g
                else:
                    st["par"][i] = (1 - a) + a * g
                    st["alt"][i] = a * (1 - g)
                    st["dim"][i] = g > 0.6
            st["spent"] = BUDGET if g > 0.5 else 0
            st["meters"] = [("soft loss", 1.0, f"{SOFT_TO:.2f}", PASS)]
            st["note"] = "10 of the 15 wanted substitutions are discarded"
            st["caption"] = ("The search cannot hold a distribution, so the relaxation's "
                             "result has to be rounded to whole residues inside the "
                             "budget. This is where the two spaces meet.")

        elif name == "cliff":
            st["title"] = "4 · Now score the sequence the search receives"
            st["sub"] = "the same predictor, on the rounded sequence"
            for i in range(N):
                if i in KEEP:
                    st["par"][i], st["alt"][i] = 0.0, 1.0
                else:
                    st["dim"][i] = True
            st["spent"] = BUDGET
            g = ease(min(1.0, u * 2.2))
            st["meters"] = [
                ("soft loss, descended", 1.0, f"−{SOFT_FROM - SOFT_TO:.2f}", PASS),
                ("score the search sees", HARD_SCORE / 0.6 * g,
                 f"{HARD_SCORE:.4f}" if 0.25 < u <= 0.45 else "", NULL),
            ]
            if u > 0.45:
                st["big"] = "0.0002"
            st["note"] = "the parent scores 0.0000"
            st["caption"] = ("The descent does not survive the rounding. The seed arrives "
                             "at the parent's score with all five edits already spent, so "
                             "it has nothing to offer and no moves left.")

        else:
            st["title"] = "5 · What the search does with it"
            st["sub"] = "revert the seed, climb from the parent, land where it always lands"
            g = ease(u)
            for i in range(N):
                if i in KEEP:
                    keep_it = i in KEEP[:2]
                    st["par"][i] = 0.0 if keep_it else g
                    st["alt"][i] = 1.0 if keep_it else 1.0 - g
                    st["revert"][i] = (not keep_it) and g < 0.9
                else:
                    st["dim"][i] = True
            for n, i in enumerate(ORDER[BUDGET:BUDGET + 3]):
                st["par"][i], st["alt"][i] = 1.0 - g, g
                st["dim"][i] = False
            st["spent"] = BUDGET
            score = HARD_SCORE + g * (0.5335 - HARD_SCORE)
            st["meters"] = [("score", score / 0.6, f"{score:.4f}", ACCENT)]
            st["note"] = "44 of 100 seeded-edit slots reverted across the 20 winners"
            st["caption"] = ("Seeded and unseeded cells reach the same plateau: 0.53 to "
                             "0.58, interface RMSD about 7 Å, every structure "
                             "CAPRI-incorrect. Seeding changes which solution, not how good.")
        return st


class Painter:
    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax

    def draw(self, st):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.055, 0.905, st["title"], fontsize=23, fontweight="bold",
                color=INK, family=SANS, va="bottom")
        if st["sub"]:
            ax.text(0.055, 0.862, st["sub"], fontsize=12.5, color=INK_SOFT,
                    family=SANS, va="bottom")

        if st["schedule"] is not None:
            self._schedule(st["schedule"])

        self._strip(st)

        if st["note"]:
            ax.text(0.055, 0.408, st["note"], fontsize=11, color=INK_FAINT,
                    family=MONO, va="top")

        y = 0.330
        for label, frac, value, colour in st["meters"]:
            self._meter(y, label, frac, value, colour)
            y -= 0.070
        self._slots(st["spent"], y - 0.018)

        if st["big"]:
            ax.text(0.855, 0.330 - 0.070 * (len(st["meters"]) - 1) + 0.010,
                    st["big"], fontsize=42, fontweight="bold",
                    color=NULL, family=MONO, ha="center", va="center")

        cap = "\n".join(textwrap.wrap(st["caption"], width=96)[:3])
        ax.text(0.055, 0.125, cap, fontsize=12.5, color=INK_SOFT,
                family=SANS, va="top", linespacing=1.5)
        ax.text(0.055, 0.030, "P17 / JN1 \u00b7 OpenDDE gradient probe \u00b7 project record \u00a733",
                fontsize=9, color=INK_FAINT, family=MONO, va="bottom")

    def _schedule(self, u):
        ax = self.ax
        x0, x1, y, h = 0.055, 0.945, 0.775, 0.030
        total = sum(n for _, n in STAGES)
        done = u * total
        x = x0
        for name, n in STAGES:
            w = (x1 - x0) * n / total
            filled = min(1.0, max(0.0, (done - (x - x0) / (x1 - x0) * total) / n))
            ax.add_patch(Rectangle((x, y), w, h, facecolor=SURFACE,
                                   edgecolor=RULE, linewidth=0.8, zorder=2))
            if filled > 0:
                ax.add_patch(Rectangle((x, y), w * filled, h, facecolor=ACCENT,
                                       edgecolor="none", alpha=0.85, zorder=3))
            ax.text(x + w / 2, y + h + 0.012, f"{name} {n}", fontsize=9.5,
                    color=INK_FAINT, family=MONO, ha="center", va="bottom", zorder=4)
            x += w

    def _strip(self, st):
        ax = self.ax
        x0, x1 = 0.055, 0.945
        y0, h = 0.470, 0.225
        gap = 0.0035
        w = (x1 - x0 - gap * (N - 1)) / N
        for i in range(N):
            x = x0 + i * (w + gap)
            alpha = 0.3 if st["dim"][i] else 1.0
            ax.add_patch(Rectangle((x, y0), w, h, facecolor=SURFACE,
                                   edgecolor=RULE, linewidth=0.7,
                                   alpha=alpha, zorder=2))
            ah = h * float(st["alt"][i])
            if ah > 0.0005:
                colour = NULL if i == st["reject"] else ACCENT
                ax.add_patch(Rectangle((x, y0), w, ah, facecolor=colour,
                                       edgecolor="none", alpha=alpha, zorder=4))
            ph = h * float(st["par"][i])
            if ph > 0.0005:
                ax.add_patch(Rectangle((x, y0 + ah), w, ph, facecolor=ACCENT_DIM,
                                       edgecolor="none", alpha=0.55 * alpha, zorder=3))
            if i == st["probe"]:
                ax.add_patch(Rectangle((x, y0), w, h, facecolor="none",
                                       edgecolor=INK, linewidth=2.0, zorder=6))
            if st["revert"][i]:
                ax.add_patch(Rectangle((x, y0), w, h, facecolor="none",
                                       edgecolor=NULL, linewidth=1.8,
                                       linestyle=(0, (2, 1.6)), zorder=6))
        ax.text(x0, y0 - 0.014, f"{N} designable positions", fontsize=9.5,
                color=INK_FAINT, family=MONO, va="top")
        lx = 0.40
        for colour, a, lab in ((ACCENT, 1.0, "competing residue"),
                               (ACCENT_DIM, 0.55, "parent residue")):
            ax.add_patch(Rectangle((lx, y0 - 0.030), 0.012, 0.012, facecolor=colour,
                                   edgecolor="none", alpha=a, zorder=4))
            ax.text(lx + 0.018, y0 - 0.024, lab, fontsize=9.5, color=INK_FAINT,
                    family=MONO, va="center")
            lx += 0.185

    def _meter(self, y, label, frac, value, colour):
        ax = self.ax
        bx0, bx1, h = 0.255, 0.700, 0.020
        ax.text(0.055, y + h / 2, label, fontsize=11, color=INK_FAINT,
                family=MONO, va="center")
        ax.add_patch(FancyBboxPatch((bx0, y), bx1 - bx0, h,
                                    boxstyle="round,pad=0,rounding_size=0.010",
                                    facecolor=SURFACE, edgecolor=RULE,
                                    linewidth=0.8, zorder=2))
        f = min(1.0, max(0.0, float(frac)))
        if f > 0:
            w = max(0.004, (bx1 - bx0) * f)
            ax.add_patch(FancyBboxPatch((bx0, y), w, h,
                                        boxstyle="round,pad=0,rounding_size=0.010",
                                        facecolor=colour, edgecolor="none", zorder=3))
        if value:
            ax.text(bx1 + 0.018, y + h / 2, value, fontsize=13, fontweight="bold",
                    color=INK, family=MONO, va="center")

    def _slots(self, spent, y):
        ax = self.ax
        s, gap = 0.015, 0.006
        ax.text(0.055, y + 0.013, "edits spent", fontsize=11, color=INK_FAINT,
                family=MONO, va="center")
        x = 0.255
        for j in range(BUDGET):
            on = j < spent
            ax.add_patch(Rectangle((x + j * (s + gap), y), s, s * (16 / 9),
                                   facecolor=ACCENT if on else SURFACE,
                                   edgecolor=ACCENT if on else RULE,
                                   linewidth=0.9, zorder=3))
        ax.text(0.255 + BUDGET * (s + gap) + 0.012, y + 0.013,
                f"{spent} / {BUDGET}", fontsize=12, fontweight="bold",
                color=INK, family=MONO, va="center")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("docs/build"))
    ap.add_argument("--gif", action="store_true", default=True)
    ap.add_argument("--no-gif", dest="gif", action="store_false")
    ap.add_argument("--stills", action="store_true",
                    help="also write one PNG per scene, for slides")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.fps)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(PAPER)
    painter = Painter(fig, ax)

    def update(f):
        painter.draw(scene.state(f))
        return []

    mp4 = args.out / "mechanism.mp4"
    anim = animation.FuncAnimation(fig, update, frames=scene.total, interval=1000 / args.fps)
    writer = animation.FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart"],
    )
    print(f"rendering {scene.total} frames at {args.fps} fps "
          f"({scene.total / args.fps:.1f} s) -> {mp4}")
    anim.save(str(mp4), writer=writer, savefig_kwargs={"facecolor": PAPER})
    print(f"wrote {mp4}")

    if args.stills:
        for name, a, b in scene.cuts:
            painter.draw(scene.state(b - 1))
            p = args.out / f"mechanism_{name}.png"
            fig.savefig(p, facecolor=PAPER, dpi=150)
            print(f"wrote {p}")

    plt.close(fig)

    if args.gif:
        gif = args.out / "mechanism.gif"
        pal = args.out / "_palette.png"
        vf = "fps=12,scale=900:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-vf", vf + ",palettegen=stats_mode=diff", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-i", str(pal), "-lavfi",
                        vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(gif)], check=True)
        pal.unlink(missing_ok=True)
        print(f"wrote {gif} ({gif.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
