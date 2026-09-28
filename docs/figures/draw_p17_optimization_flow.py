"""Render the model/search flow as editable SVG, PDF, and PNG (no model calls)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(18, 11.2))
fig.patch.set_facecolor("#f8fafc")
ax.set(xlim=(0, 180), ylim=(0, 112))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
INK, MUTED = "#172b45", "#52657c"
BLUE, GREEN, AMBER = "#2765a6", "#087d70", "#a96012"


def text(x, y, label, size=12, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, label, fontsize=size, color=color, weight=weight,
            ha=ha, va=va, linespacing=1.55, zorder=4)


def box(x, y, w, h, face="white", edge="#cad5e2", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.5",
                              linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


def arrow(points, color=INK, width=1.9, dashed=False):
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=17,
                                color=color, linewidth=width, zorder=3,
                                linestyle="--" if dashed else "-"))


text(6, 106, "P17 optimization", 27, weight="bold")
text(6, 100, "Model guidance proposes edits. Full-confidence scores decide which candidates stay.",
     15, MUTED)

# Shared differentiable proposal path.
box(44, 46, 67, 47, face="#eef5fc", edge="#b6cce4")
text(48, 90, "1  SHARED GRADIENT GUIDANCE", 12, BLUE, "bold")
box(48, 73, 28, 13, edge="#b6cce4")
text(62, 82, "OpenDDE", 15, BLUE, "bold", "center")
text(62, 77, "Full structure + confidence\nContacts and pose RMSD", 10.5, ha="center")
box(80, 73, 27, 13, edge="#b6cce4")
text(93.5, 82, "AbLang2", 15, BLUE, "bold", "center")
text(93.5, 77, "Antibody sequence\nplausibility", 10.5, ha="center")
box(49, 51, 57, 15, edge="#b6cce4")
text(77.5, 62, "Combine the current losses", 13, BLUE, "bold", "center")
text(77.5, 56, "Contacts + RMSD + ipTM + PAE + pTMEnergy\nAbLang2 + edit penalty → sequence gradients", 11, ha="center")
arrow([(62, 73), (62, 66)], BLUE)
arrow([(93.5, 73), (93.5, 66)], BLUE)
text(77.5, 48.4, "Model weights stay frozen", 10, MUTED, ha="center")

# Inputs and active state.
box(6, 71, 31, 20)
text(21.5, 86.5, "Active parents", 15, weight="bold", ha="center")
text(21.5, 78.5, "Initialize from WT\nFixed target and framework\nCDR edits only", 11, ha="center")
arrow([(37, 81), (41, 81), (41, 87.5), (93.5, 87.5), (93.5, 86)], BLUE)
arrow([(62, 87.5), (62, 86)], BLUE, 1.4)

box(121, 71, 50, 22, face="#eef5fc", edge="#b6cce4")
text(125, 89, "2  PROPOSE A CANDIDATE", 12, BLUE, "bold")
text(146, 80, "Gradient-biased CDR edits\nEntropy-controlled sampling\nRevert / replace / exchange · hard WT edit cap", 11, ha="center")
arrow([(106, 58), (115, 58), (115, 81), (121, 81)], BLUE)
text(119, 66, "Sequence\ngradient", 9.5, BLUE)

# Forward-only confidence path.
box(121, 37, 50, 25, face="#eaf7f3", edge="#a8d5c9")
text(125, 58, "3  FULL-CONFIDENCE EVALUATION", 11.5, GREEN, "bold")
text(146, 52, "Full OpenDDE · frozen weights", 13, GREEN, "bold", "center")
text(146, 44, "Diffusion + confidence head · forward only\nDirectional-min ipSAE for each structural seed\nSelection score = mean across fixed seeds", 10.8, ha="center")
arrow([(146, 71), (146, 62)], GREEN)
text(149, 66.5, "Discrete sequence", 10, GREEN)

# Alternative retention arms, both fed by the same scorer.
box(6, 18, 105, 23, face="#fff5e8", edge="#e1c69f")
text(10, 37.5, "4  COMPARE TWO RETENTION POLICIES", 12, AMBER, "bold")
ax.plot([57, 57], [21, 34], color="#e1c69f", linewidth=1, zorder=3)
text(10, 31.5, "Independent searches", 13, AMBER, "bold")
text(10, 25, "Offspring competes with its own parent\nSeparate trajectories; no parent exchange", 10.5)
text(61, 31.5, "Population with local competition", 12.3, AMBER, "bold")
text(61, 25, "Offspring competes with a nearby sequence\nRetain alternatives and protect the best", 10.5)
arrow([(121, 49), (116, 49), (116, 30), (111, 30)], GREEN)
text(114, 43.5, "Score", 9.5, GREEN, ha="right")
arrow([(21.5, 41), (21.5, 71)], AMBER, 2.6)
text(24, 57, "Retained candidates\nbecome the next parents", 10.5, AMBER)

box(121, 8, 50, 17)
text(146, 20, "Evaluation archive", 14, weight="bold", ha="center")
text(146, 13, "Best evaluated candidates + per-seed scores\nDecision logs, call counts, time and memory", 10.5, ha="center")
arrow([(146, 37), (146, 25)], MUTED, 1.5)
text(149, 31, "All evaluated candidates", 9.5, MUTED)

text(6, 11, "Loop until the evaluation / gradient budget is reached.", 12, weight="bold")
text(6, 5, "Shared models, proposals and scoring across both arms. ipSAE is a confidence proxy, not an affinity measurement.",
     11, MUTED)

for extension in ("svg", "pdf", "png"):
    path = OUT / f"p17_optimization_flow.{extension}"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    print(path)
