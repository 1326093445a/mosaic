#!/usr/bin/env python
"""Build a self-contained presentation deck for the P17 / JN1 gradient probe.

One HTML file with the video and every figure embedded, so it can be opened on
a machine that has none of this repo, and a 16:9 PDF printed from it. It covers
what the project is doing, how both search stages work, how the bindcraft
hand-off works, and what the two runs measured.

The long-form record stays in docs/P17_JN1.md; this is the talk version.

    python docs/make_presentation.py            # needs docs/build/mechanism.* first
    python docs/build_report.sh                 # the long-form HTML/PDF
"""
from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER, SURFACE = "#eef1f4", "#f7f9fb"
INK, INK_SOFT, INK_FAINT = "#15191e", "#4d5760", "#7d878f"
RULE, ACCENT, ACCENT_DIM = "#cfd6dd", "#2f5d82", "#8fa9bf"
NULL, PASS, BAND = "#9c4a35", "#3f6b54", "#dde6ec"

# Per-candidate mean iRMSD, from the two runs' capri.csv (WT excluded), and the
# unmodified start scored separately in each cell.
CELLS = {
    "bindcraft5": [3.38, 6.97, 7.01, 7.2, 7.33, 7.34, 7.37, 7.4, 7.64, 8.22, 8.46,
                   13.82, 13.92, 15.47, 16.02, 16.38, 19.27, 21.03, 24.04, 24.22],
    "parent5": [6.85, 6.95, 7.08, 7.23, 7.26, 7.35, 7.44, 7.5, 9.65, 14.03, 15.39,
                15.53, 15.84, 16.06, 16.07, 16.23, 20.38, 21.79, 21.87, 23.58],
    "colabdesign5": [6.91, 6.93, 7.01, 7.27, 7.35, 11.7, 11.7, 15.35, 15.35, 15.44,
                     15.59, 15.71, 15.74, 15.76, 15.95, 16.08, 16.1, 16.81, 17.21, 20.63],
    "parent7": [6.71, 6.77, 6.9, 7.08, 7.19, 7.31, 10.04, 14.3, 14.51, 15.67, 15.73,
                15.8, 15.96, 16.04, 16.07, 16.12, 16.13, 16.13, 16.33, 24.55],
    "bindcraft7": [7.38, 7.39, 7.41, 7.44, 7.49, 13.67, 13.95, 14.64, 15.11, 15.18,
                   15.24, 15.58, 15.83, 16.08, 17.24, 19.44, 21.19, 21.7, 24.21, 24.52],
}
WT = {"parent5": 14.74, "bindcraft5": 16.17, "colabdesign5": 15.99,
      "parent7": 14.03, "bindcraft7": 17.95}
ROWS = [("bindcraft5", "bindcraft", "5"), ("parent5", "parent", "5"),
        ("colabdesign5", "colabdesign", "5"), ("parent7", "parent", "7"),
        ("bindcraft7", "bindcraft", "7")]


def irmsd_figure(path: Path) -> None:
    """The acceptance bar with all 100 candidates on it."""
    fig = plt.figure(figsize=(12.0, 5.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0.13, 0.14, 0.84, 0.76])
    ax.set_facecolor(PAPER)

    ax.axvspan(6.7, 7.7, color=BAND, zorder=1)
    ax.axvline(10, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.text(10.25, len(ROWS) - 0.32, "10 Å bar", fontsize=10,
            family="DejaVu Sans Mono", color=INK, fontweight="bold")
    ax.text(7.2, len(ROWS) - 0.32, "6.7–7.7 Å", fontsize=9.5, ha="center",
            family="DejaVu Sans Mono", color=INK_FAINT)

    for r, (key, name, budget) in enumerate(ROWS):
        y = len(ROWS) - 1 - r
        vals = np.array(CELLS[key])
        ax.axhline(y, color=RULE, lw=0.8, zorder=2)
        # fan overlapping marks so the band's density is countable
        placed = []
        for v in np.sort(vals):
            dy = 0.0
            for k in range(30):
                cand = (-1) ** k * np.ceil(k / 2) * 0.13
                if not any(abs(px - v) < 0.42 and abs(py - cand) < 0.12
                           for px, py in placed):
                    dy = cand
                    break
            placed.append((v, dy))
            ax.plot(v, y + dy, "o", ms=6.5, zorder=5,
                    color=NULL if (key == "bindcraft5" and v == 3.38)
                    else (PASS if v <= 10 else ACCENT_DIM),
                    mec=PAPER, mew=0.7)
        ax.plot(WT[key], y, "x", ms=9, mew=1.8, color=INK_SOFT, zorder=6)
        hits = int((vals <= 10).sum())
        ax.text(-0.012, y, f"{name}\nbudget {budget} · {hits}/20 cleared",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, family="DejaVu Sans Mono", color=INK, linespacing=1.6)

    ax.set_xlim(0, 26)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("interface RMSD, Ångströms", fontsize=10.5, color=INK_SOFT)
    ax.tick_params(colors=INK_FAINT, labelsize=9.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def uri(path: Path, mime: str) -> str:
    return "data:%s;base64,%s" % (mime, base64.b64encode(path.read_bytes()).decode())


CSS = """
:root{--paper:#eef1f4;--surface:#f7f9fb;--ink:#15191e;--ink-soft:#4d5760;
--ink-faint:#7d878f;--rule:#cfd6dd;--accent:#2f5d82;--null:#9c4a35;--pass:#3f6b54;}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#c9ced4;color:var(--ink);
font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.deck{padding:28px 16px 60px}
.slide{position:relative;width:min(1280px,100%);aspect-ratio:16/9;margin:0 auto 26px;
background:var(--paper);border-radius:3px;box-shadow:0 2px 14px rgba(0,0,0,.14);
padding:46px 56px 40px;display:flex;flex-direction:column;overflow:hidden}
.slide.tight{padding:34px 44px 30px}
.slide.vcenter{justify-content:center}
.slide.vcenter .foot{margin-top:28px}
.n{position:absolute;right:26px;bottom:18px;font:600 11px/1 "SFMono-Regular",Menlo,monospace;
color:var(--ink-faint)}
.eyebrow{font:600 11px/1 "SFMono-Regular",Menlo,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink-faint);margin:0 0 14px}
h1{font-size:46px;line-height:1.04;letter-spacing:-.02em;margin:0 0 16px;text-wrap:balance}
h2{font-size:30px;line-height:1.12;letter-spacing:-.015em;margin:0 0 6px;text-wrap:balance}
.sub{font-size:17px;color:var(--ink-soft);margin:0 0 20px;max-width:46em}
p{font-size:17px;line-height:1.52;margin:0 0 12px;max-width:44em}
.lead{font-size:21px;line-height:1.45;color:var(--ink-soft);max-width:40em}
ul{margin:4px 0 0;padding-left:20px;font-size:16.5px;line-height:1.55}
li{margin-bottom:9px}
b,strong{font-weight:600}
code,.mono{font-family:"SFMono-Regular",Menlo,Consolas,monospace}
.fill{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
.fill img,.fill video{max-width:100%;max-height:100%;object-fit:contain;border-radius:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px;flex:1;min-height:0;align-items:center}
.cols.wide-left{grid-template-columns:1.35fr 1fr}
.cols img{width:100%;border:1px solid var(--rule);border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th{font:600 10.5px/1 "SFMono-Regular",Menlo,monospace;letter-spacing:.07em;
text-transform:uppercase;color:var(--ink-faint);text-align:right;
padding:0 10px 7px;border-bottom:1px solid var(--ink)}
th:first-child,td:first-child{text-align:left;padding-left:0}
td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--rule)}
td:first-child{font-family:"SFMono-Regular",Menlo,monospace;font-size:14px}
.hl{color:var(--null);font-weight:600}.ok{color:var(--pass);font-weight:600}
.dim{color:var(--ink-faint)}
.box{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
border-radius:2px;padding:16px 18px;margin-top:4px}
.box.warn{border-left-color:var(--null)}
.box p:last-child{margin-bottom:0}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:6px}
.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:2px;padding:14px 16px}
.kpi .v{font:700 30px/1 "SFMono-Regular",Menlo,monospace;letter-spacing:-.02em}
.kpi .k{font-size:13px;color:var(--ink-soft);margin-top:7px;line-height:1.35}
.kpi.bad .v{color:var(--null)}.kpi.good .v{color:var(--pass)}
.foot{font-size:13px;color:var(--ink-faint);margin-top:auto;padding-top:14px}
@media print{
  @page{size:338.7mm 190.5mm;margin:0}
  html,body{background:#fff}
  .deck{padding:0}
  .slide{width:100%;height:100vh;aspect-ratio:auto;margin:0;border-radius:0;
  box-shadow:none;break-after:page;page-break-after:always}
  .slide:last-child{break-after:auto;page-break-after:auto}
}
"""


def build_html(assets: dict) -> str:
    S = []

    def slide(body, cls=""):
        n = "" if not S else "%02d" % (len(S) + 1)
        S.append('<section class="slide %s">%s%s</section>' % (
            cls, body, '<div class="n">%s</div>' % n if n else ""))

    slide("""
      <p class="eyebrow">OpenDDE gradient probe &middot; P17 nanobody vs SARS-CoV-2 JN1</p>
      <h1>Can a frozen structure predictor&rsquo;s own gradient<br>move its own readouts?</h1>
      <p class="lead">A proof of concept, and a measured negative result: what the
        discrete search does, what the continuous relaxation does, and why coupling
        them the obvious way cannot work.</p>
      <div class="foot">Project record &sect;33 &middot; runs of 2026-10-15 and 2026-10-16
        &middot; 100 candidates, 300 held-out structures</div>""", "vcenter")

    slide("""
      <p class="eyebrow">What is being asked</p>
      <h2>Move a degraded complex back toward a known-good one</h2>
      <p class="sub">P17 binds the Alpha spike well and JN1 badly. The question is
        whether the predictor&rsquo;s gradient with respect to <em>sequence</em> carries
        enough information to recover the good interface by editing the binder.</p>
      <div class="kpis">
        <div class="kpi good"><div class="v">0.795</div>
          <div class="k">P17 + Alpha, measured ipSAE &middot; pose 1.98&ndash;2.96 &Aring;.
            The target.</div></div>
        <div class="kpi bad"><div class="v">0.000&ndash;0.163</div>
          <div class="k">P17 + JN1, measured ipSAE &middot; pose 22&ndash;58 &Aring;.
            The degraded start.</div></div>
        <div class="kpi"><div class="v">29</div>
          <div class="k">designable positions on the binder, edit budget 5 per
            candidate.</div></div>
      </div>
      <div class="box" style="margin-top:22px"><p><b>Acceptance bar:</b> interface RMSD
        &le; 10 &Aring;, counted <em>per candidate</em> over three held-out structural
        seeds. DockQ and CAPRI class are reported but do not gate.</p></div>""", "vcenter")

    slide("""
      <p class="eyebrow">The mechanism, end to end</p>
      <h2>Two ways to search the same 29 positions</h2>
      <div class="fill"><video controls autoplay muted loop playsinline src="__VIDEO__"></video></div>
      <div class="foot">37 s &middot; discrete search, the bindcraft relaxation, the
        projection, and what the search actually receives.</div>""", "tight")

    slide("""
      <div class="cols wide-left">
        <div><img src="__DISCRETE__" alt="Discrete search"></div>
        <div>
          <p class="eyebrow">Stage 1</p>
          <h2>Discrete search</h2>
          <ul>
            <li>Proposes <b>one whole substitution</b> at a time and scores it with the
              frozen predictor.</li>
            <li>About <b>51 of 250</b> proposals are accepted per seed.</li>
            <li>It stops on the <b>200 score-call</b> budget &mdash; never on edits.</li>
            <li>Every winner spends the <b>whole edit budget</b>: 83&ndash;99 % of all
              candidates sit at the cap.</li>
            <li>Once the budget is full, each further move is a
              <b>revert-and-add exchange</b>.</li>
          </ul>
          <div class="box" style="margin-top:14px"><p>Reaches score <b>0.5801</b>
            from a parent that scores 0.0000.</p></div>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">Stage 1, on the sequence</p>
      <h2>The discrete search, residue by residue</h2>
      <div class="fill"><video controls muted loop playsinline src="__SEQDISC__"></video></div>
      <div class="foot">20 s &middot; five recorded candidates from control search seed 2, each one substitution from the next, with the score the predictor returned.</div>""", "tight")

    slide("""
      <p class="eyebrow">Inside the discrete search</p>
      <h2>Retention: population versus independent</h2>
      <div class="fill"><video controls muted loop playsinline src="__SEQPOL__"></video></div>
      <div class="foot">19 s &middot; the four live candidates, the two retention rules,
        and what the difference is actually worth.</div>""", "tight")

    slide("""
      <p class="eyebrow">Inside the discrete search &middot; the honest ledger</p>
      <h2>Population is what we ran, not what the evidence supports</h2>
      <div class="cols">
        <div>
          <p class="sub"><b>Why it became the default.</b> Three early observations, none
            of them a matched comparison:</p>
          <ul>
            <li>&sect;17.1 pilot, 0.1351 against 0.0304 &mdash; <b>voided</b> in &sect;17.10
              on 14&ndash;20 &Aring; adjacent-C&alpha; geometry.</li>
            <li>&sect;20 Alpha 2-edit rung, 0.835/0.801 against 0.702/0.687 &mdash; stands,
              but <b>two seeds per arm</b> on a problem &sect;20 itself showed was trivial.</li>
            <li>The only candidate ever to revert the lethal substitution came from a
              population run &mdash; stands, and is a single observation.</li>
          </ul>
        </div>
        <div>
          <p class="sub"><b>What the matched test found.</b> &sect;27.2, 10 seeds,
            exploration pinned for the first time:</p>
          <table>
            <tr><th>contrast</th><th>&Delta; ipSAE</th><th>&Delta; pose</th><th></th></tr>
            <tr><td>population &rarr; independent</td><td>&minus;0.059</td>
                <td>&minus;3.0 &Aring;</td><td>inside noise</td></tr>
          </table>
          <div class="box" style="margin-top:14px"><p>Detection threshold at this seed
            count: <b>~0.11 ipSAE, ~6 &Aring;</b>. The screen <b>reversed</b> the default
            &mdash; independent is simpler and nothing measurable is lost.</p></div>
          <p class="sub" style="margin-top:14px"><b>Why it is null, mechanically.</b> Over
            5,024 proposals in the 20 control cells, the crowding rule chose a slot other
            than the parent&rsquo;s <b>9 times (0.18 %)</b>. A one- or two-position move
            sits 2 from its parent and 7&ndash;8 from every other slot, so the nearest
            incumbent is the parent by construction.</p>
        </div>
      </div>""")

    slide("""
      <div class="cols wide-left">
        <div><img src="__RELAX__" alt="Continuous relaxation"></div>
        <div>
          <p class="eyebrow">Stage 2 &middot; how the bindcraft way works</p>
          <h2>Continuous relaxation</h2>
          <p style="font-size:16px">Every position holds a <b>distribution over 20
            residues</b> instead of one residue, so all 29 move together under gradient
            descent. BindCraft&rsquo;s four-stage schedule, 50:25:45:5, rescaled to 600
            steps:</p>
          <ul style="font-size:15.5px">
            <li><b>logits, 240</b> &mdash; free optimisation in logit space</li>
            <li><b>logits, 120</b> &mdash; continues, still unconstrained</li>
            <li><b>soft, 216</b> &mdash; softmax sharpened toward one residue</li>
            <li><b>hard, 24</b> &mdash; straight-through one-hot at temperature 1.0</li>
          </ul>
          <div class="box" style="margin-top:12px"><p>Descends <b>40.50 &rarr; 12.25</b>
            on its own loss. By the end <b>15</b> positions prefer a non-parent residue
            &mdash; against a budget of <b>5</b>.</p></div>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">Stage 2, on the sequence</p>
      <h2>The relaxation, the rounding, the cliff</h2>
      <div class="fill"><video controls muted loop playsinline src="__SEQCONT__"></video></div>
      <div class="foot">20 s &middot; measured losses and the real five seed substitutions; the per-position distributions are illustrative.</div>""", "tight")

    slide("""
      <div class="cols wide-left">
        <div><img src="__CLIFF__" alt="The projection and the collapse"></div>
        <div>
          <p class="eyebrow">Stages 3 and 4 &middot; where the result dies</p>
          <h2>The hand-off</h2>
          <p style="font-size:16px">The search cannot hold a distribution, so the
            relaxation is <b>rounded</b> to whole residues inside the budget: keep the
            five highest-margin substitutions, discard the other ten.</p>
          <div class="box warn"><p>The relaxation descends <b>28.25</b> on the soft
            loss. The rounded sequence it hands over scores <b>0.0002</b> &mdash; the
            parent scores <b>0.0000</b>.</p></div>
          <p style="font-size:16px;margin-top:14px">So the seed arrives at the
            parent&rsquo;s height with <b>all five edits already spent</b>. It has
            nothing to offer and no moves left, and reverting it is the search&rsquo;s
            best move.</p>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">Result &middot; the acceptance bar</p>
      <h2>Every cell finds the same plateau</h2>
      <div class="fill"><img src="__IRMSD__" alt="iRMSD per candidate for five cells"></div>
      <div class="foot">Thirty-two of 100 candidates land in a band 1 &Aring; wide at
        6.7&ndash;7.7 &Aring;, seeded or not. One point in 100 sits below it. &times; marks
        the unmodified start scored in each cell: 14.03&ndash;17.95 &Aring; for identical
        input, which is the measurement floor.</div>""", "tight")

    slide("""
      <p class="eyebrow">Result &middot; nothing separates the arms</p>
      <h2>Seeded cells do not beat the control, at either budget</h2>
      <div class="cols" style="gap:34px">
        <div>
          <table>
            <thead><tr><th>cell</th><th>bud</th><th>&le;10 &Aring;</th><th>mean</th><th>best</th></tr></thead>
            <tbody>
              <tr><td>bindcraft</td><td>5</td><td>11/20</td><td>12.12</td><td class="hl">3.38</td></tr>
              <tr><td>parent</td><td>5</td><td>9/20</td><td>13.20</td><td>6.85</td></tr>
              <tr><td>colabdesign</td><td>5</td><td>5/20</td><td>13.53</td><td>6.91</td></tr>
              <tr><td>parent</td><td>7</td><td>6/20</td><td>13.27</td><td>6.71</td></tr>
              <tr><td>bindcraft</td><td>7</td><td>5/20</td><td>15.03</td><td>7.38</td></tr>
            </tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>comparison</th><th>bar</th><th>Fisher</th><th>p (mean)</th></tr></thead>
            <tbody>
              <tr><td>bind7 vs bind5</td><td>5 vs 11</td><td>0.105</td><td>0.121</td></tr>
              <tr><td>parent7 vs parent5</td><td>6 vs 9</td><td>0.514</td><td>0.967</td></tr>
              <tr><td>bind7 vs parent7</td><td>5 vs 6</td><td>1.000</td><td>0.284</td></tr>
              <tr><td>bind5 vs parent5</td><td>11 vs 9</td><td>0.752</td><td>0.576</td></tr>
            </tbody>
          </table>
          <div class="box" style="margin-top:16px"><p>Budget 7 gave the seeded cell two
            ordinary substitutions on top of its seed &mdash; the same move set as the
            control. It did not help, which closes the last confound.</p></div>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">Result &middot; how much of this is noise</p>
      <h2>The headline count has &plusmn;3 of slop</h2>
      <div class="cols" style="gap:34px">
        <div>
          <table>
            <thead><tr><th>control run</th><th>mean ipSAE</th><th>&le;10 &Aring;</th></tr></thead>
            <tbody>
              <tr><td>&sect;29 grad_pop</td><td>0.567</td><td>7/20</td></tr>
              <tr><td>&sect;32.6 budget 5</td><td>0.5649</td><td>9/20</td></tr>
              <tr><td>&sect;33 budget 7</td><td>0.5685</td><td class="hl">6/20</td></tr>
            </tbody>
          </table>
          <p style="margin-top:16px;font-size:16px">Same configuration, three times.
            Mean ipSAE spans <b>0.004</b>. The count spans <b>3</b>.</p>
        </div>
        <div>
          <div class="box warn"><p><b>Retracted:</b> the earlier reading that bindcraft&rsquo;s
            11/20 beat the control&rsquo;s 9/20. The count alone moves by 3 with nothing
            behind it.</p></div>
          <p style="margin-top:16px;font-size:16px">The bar is a threshold on a
            distribution whose bulk sits at 13&ndash;15 &Aring;, so a shift too small to
            measure in the mean tips several candidates across 10 &Aring;.</p>
          <p style="font-size:16px"><b>Rule:</b> report mean &plusmn; SEM beside the
            count, and treat any difference of &le;3 candidates as noise.</p>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">Result &middot; the shape of the optimum</p>
      <h2>A degenerate plateau, not a basin</h2>
      <div class="cols" style="gap:34px">
        <div>
          <table>
            <thead><tr><th></th><th>parent 7</th><th>bindcraft 7</th></tr></thead>
            <tbody>
              <tr><td>distinct edit sets</td><td class="hl">20 of 20</td><td class="hl">20 of 20</td></tr>
              <tr><td>positions touched</td><td>28</td><td>29</td></tr>
              <tr><td>distinct substitutions</td><td>107</td><td>93</td></tr>
              <tr><td>pairwise Jaccard</td><td>0.24</td><td>0.27</td></tr>
              <tr><td>used by &ge; half</td><td>5</td><td>3</td></tr>
            </tbody>
          </table>
        </div>
        <div>
          <p style="font-size:16px">Twenty search seeds give <b>twenty different</b>
            solutions, sharing about a quarter of their positions, all landing at score
            0.53&ndash;0.58 and roughly 7 &Aring;.</p>
          <div class="box"><p>There is no narrow basin for a seed to steer into. A
            different start produces a <em>different</em> solution of the same quality,
            not a better one &mdash; which is why a cell can absorb three of five seeded
            edits and still match the control.</p></div>
        </div>
      </div>""")

    slide("""
      <p class="eyebrow">What follows</p>
      <h2>The ceiling is the objective, not the search</h2>
      <p class="sub">More budget, a better start and more search seeds all reach the
        same 0.58 / 7 &Aring; ceiling. Three things follow, in the order the evidence
        supports them.</p>
      <ul>
        <li><b>Change what is optimised.</b> The objective correlates at r &asymp; &minus;0.5
          with interface RMSD <em>while holding the reference</em>. Its best attainable
          value corresponds to structures that are still incorrect. That is the only
          change here that could move the plateau&rsquo;s height.</li>
        <li><b>Couple the two stages differently, if at all.</b> The relaxation works and
          the search works; the <em>rounding between them</em> is what fails. A hand-off
          that biases which substitutions the search proposes, rather than rounding to a
          start sequence, is untested. Anything that still rounds to a seed will
          reproduce this null.</li>
        <li><b>Settle candidate 5.</b> The one CAPRI-acceptable result (iRMSD 3.19/3.42/3.52)
          did not recur. Budget 5, same seed, search seeds 20&ndash;39: one cell, 2 h 45 m.</li>
      </ul>""", "vcenter")

    slide("""
      <p class="eyebrow">What follows &middot; the untested version</p>
      <h2>Bias the proposals, do not round them</h2>
      <div class="fill"><video controls muted loop playsinline src="__SEQPROP__"></video></div>
      <div class="foot">15 s &middot; a design, not a result. No measured numbers appear in this clip.</div>""", "tight")

    slide("""
      <p class="eyebrow">Limits</p>
      <h2>What this does not establish</h2>
      <div class="box warn">
        <p><b>Nothing here is about binding.</b> All 120 structures in the budget-7 run
          are CAPRI-incorrect, best DockQ 0.175. DockQ remains a floor.</p>
      </div>
      <ul style="margin-top:18px">
        <li>Nothing says relaxation <em>cannot</em> help &mdash; only that it did not, at
          these two budgets, with this projection, under this objective.</li>
        <li>The three held-out seeds of one candidate are three predictions of one
          sequence and are correlated. They are not three measurements, which is why the
          bar counts per candidate.</li>
        <li>The composite score contains the reference structure, so it is not a blind
          selector and its correlation with quality is not a validation signal.</li>
        <li>Residue identities and proposal order in the animation are illustrative. The
          quantities on its meters are measured.</li>
      </ul>
      <div class="foot">Full record and every number: docs/P17_JN1.md &sect;33.</div>""", "vcenter")

    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>P17 / JN1 gradient probe</title><style>' + CSS + '</style></head>'
            '<body><div class="deck">' + "".join(S) + '</div></body></html>')
    for k, v in assets.items():
        html = html.replace("__%s__" % k, v)
    return html


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/build"))
    ap.add_argument("--no-pdf", dest="pdf", action="store_false", default=True)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    missing = [p for p in ("mechanism.mp4", "mechanism_discrete.png",
                           "mechanism_relax.png", "mechanism_cliff.png")
               if not (out / p).exists()]
    if missing:
        raise SystemExit("missing %s -- run docs/make_mechanism_video.py --stills first"
                         % ", ".join(missing))
    missing = [p for p in ("seq_discrete.mp4", "seq_continuous.mp4",
                           "seq_proposed.mp4", "seq_policy.mp4")
               if not (out / p).exists()]
    if missing:
        raise SystemExit("missing %s -- run docs/make_sequence_videos.py first"
                         % ", ".join(missing))

    chart = out / "irmsd_bar.png"
    irmsd_figure(chart)
    print(f"wrote {chart}")

    html = build_html({
        "VIDEO": uri(out / "mechanism.mp4", "video/mp4"),
        "DISCRETE": uri(out / "mechanism_discrete.png", "image/png"),
        "RELAX": uri(out / "mechanism_relax.png", "image/png"),
        "CLIFF": uri(out / "mechanism_cliff.png", "image/png"),
        "SEQDISC": uri(out / "seq_discrete.mp4", "video/mp4"),
        "SEQCONT": uri(out / "seq_continuous.mp4", "video/mp4"),
        "SEQPROP": uri(out / "seq_proposed.mp4", "video/mp4"),
        "SEQPOL": uri(out / "seq_policy.mp4", "video/mp4"),
        "IRMSD": uri(chart, "image/png"),
    })
    page = out / "P17_presentation.html"
    page.write_text(html, encoding="utf-8")
    print(f"wrote {page} ({page.stat().st_size / 1048576:.1f} MB, self-contained)")

    if args.pdf:
        chrome = next((c for c in ("google-chrome", "google-chrome-stable", "chromium",
                                   "chromium-browser")
                       if subprocess.run(["which", c], capture_output=True).returncode == 0),
                      None)
        if not chrome:
            print("no Chrome found; open the HTML and print it to PDF instead")
            return 0
        import tempfile
        pdf = out / "P17_presentation.pdf"
        with tempfile.TemporaryDirectory() as prof:
            subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                            "--user-data-dir=" + prof, "--no-pdf-header-footer",
                            "--print-to-pdf=" + str(pdf.resolve()),
                            "file://" + str(page.resolve())],
                           capture_output=True, check=False)
        if pdf.exists():
            print(f"wrote {pdf} ({pdf.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
