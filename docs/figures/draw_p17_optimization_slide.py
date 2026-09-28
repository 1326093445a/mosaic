"""Render a minimal 16:9 presentation slide of the P17 optimization loop."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 160), ylim=(0, 90))
ax.axis("off")
INK, MUTED = "#182d48", "#586b81"
BLUE, GREEN, AMBER = "#2468a6", "#087d70", "#a96012"


def label(x, y, text, size=18, color=INK, weight="normal", ha="center"):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight,
            ha=ha, va="center", linespacing=1.6, zorder=4)


def arrow(points, color=MUTED, width=2.3):
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=22,
                                color=color, linewidth=width, zorder=3))


label(8, 79, "Gradient-guided antibody optimization", 30, weight="bold", ha="left")
label(8, 71, "Start from P17. Improve interface confidence through repeated CDR edits.",
      18, MUTED, ha="left")

cards = [
    (8, "1", "Guide", "OpenDDE + AbLang2", "RMSD + confidence gradients", BLUE, "#eef5fc"),
    (46, "2", "Propose", "Small CDR edits", "Framework stays fixed", BLUE, "#eef5fc"),
    (84, "3", "Score", "Full OpenDDE", "Evaluate ipSAE", GREEN, "#eaf7f3"),
    (122, "4", "Select", "Choose next parents", "Confidence + search policy", AMBER, "#fff5e8"),
]
for x, number, title, model, detail, color, face in cards:
    ax.add_patch(FancyBboxPatch((x, 35), 30, 26,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=face, edgecolor="none", zorder=2))
    label(x + 15, 57, number, 13, color, "bold")
    label(x + 15, 51, title, 25, color, "bold")
    label(x + 15, 43, model, 16, weight="bold")
    label(x + 15, 38, detail, 13.5, MUTED)

for start in (38, 76, 114):
    arrow([(start + 1, 48), (start + 7, 48)])

arrow([(137, 35), (137, 25), (23, 25), (23, 35)], AMBER)
label(80, 28.5, "Repeat with the retained candidates", 17, AMBER, "bold")
label(80, 15, "Compare: independent searches  vs.  a diverse population", 18, weight="bold")
label(80, 7, "Model weights stay frozen  •  ipSAE measures confidence, not binding affinity", 14, MUTED)

for extension in ("svg", "pdf", "png"):
    path = OUT / f"p17_optimization_slide.{extension}"
    fig.savefig(path, dpi=180, facecolor="white")
    print(path)
