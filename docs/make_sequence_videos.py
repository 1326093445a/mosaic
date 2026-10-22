#!/usr/bin/env python
"""Render three short sequence-level animations: discrete, continuous, proposed.

Why this exists. The mechanism video (make_mechanism_video.py) draws the two
stages as abstract bars, which shows the cliff but hides what is actually being
changed: residues, at named positions, in a real nanobody. These three draw the
same story on the 29-residue designable window itself, one approach per clip,
so a clip can be dropped into a talk on its own.

    seq_discrete.mp4    the search we ran: one whole substitution at a time
    seq_continuous.mp4  the bindcraft relaxation, the rounding, the cliff
    seq_proposed.mp4    the hand-off we have not tried: bias, do not round

Everything with a number on it is measured. Provenance:

  window, parent residues   the 29 positions that move anywhere in the control
                            cell of results/p17_seeded_search_20261015_053843
  ASCENT                    five recorded candidates from control search seed 2,
                            each one substitution from the next, with the score
                            the frozen predictor returned; REJECT entries are
                            also recorded candidates, which scored no better
  SEED_BINDCRAFT            the is_wt row of that run's bindcraft cell, i.e. the
                            rounded relaxation output the search was handed
  MEAN_BEST                 mean best score over the 20 search seeds per cell
  40.50 -> 12.25, 0.0002    the relaxation's own soft loss and the score of its
                            rounded sequence (project record section 33.3)

What is illustrative, and says so on screen: the per-position distributions in
the continuous clip (the soft simplex was not kept), which ten of the fifteen
wanted substitutions are discarded, and every frame of the proposed clip, which
is a design and has no results.

    python docs/make_sequence_videos.py [--only discrete] [--stills] [--no-gif]
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
PROPOSE = "#6b4a7d"

SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]

WINDOW = list(range(26, 34)) + list(range(51, 59)) + list(range(97, 110))
PARENT = "GRTSSVYN" "ITGNGGTT" "AAGGWGKERNYAY"
BLOCKS = [(0, 8, "CDR1"), (8, 16, "CDR2"), (16, 29, "CDR3")]
assert len(WINDOW) == len(PARENT) == 29

BUDGET = 5
SCORE_MAX = 0.85

ASCENT = [
    dict(res=101, to="A", after=0.004527, call=1,   reject=("H", 0.0)),
    dict(res=54,  to="E", after=0.011049, call=2,   reject=None),
    dict(res=105, to="E", after=0.154202, call=11,  reject=("M", 0.0, 107)),
    dict(res=108, to="Q", after=0.167590, call=46,  reject=("E", 0.069616)),
    dict(res=31,  to="F", after=0.324615, call=114, reject=("W", 0.166198)),
]
CALLS = 200
MEAN_BEST = dict(parent=0.5701, bindcraft=0.5586, colabdesign=0.5294)

SEED_BINDCRAFT = [(27, "F"), (52, "S"), (54, "D"), (57, "A"), (106, "H")]
SOFT_FROM, SOFT_TO = 40.50, 12.25
HARD_SCORE = 0.0002
WANTED = 15

_rng = np.random.default_rng(17)
_POOL = "ADEFGHIKLMNQRSTVWY"
_real = {WINDOW.index(r): a for r, a in SEED_BINDCRAFT}
_others = [i for i in range(29) if i not in _real]
_extra = sorted(_rng.choice(_others, WANTED - len(_real), replace=False).tolist())

DIST = []
for i in range(29):
    par = PARENT[i]
    if i in _real:
        alt, margin = _real[i], float(_rng.uniform(0.42, 0.55))
    elif i in _extra:
        alt = _rng.choice([c for c in _POOL if c != par])
        margin = float(_rng.uniform(0.04, 0.22))
    else:
        alt = _rng.choice([c for c in _POOL if c != par])
        margin = -float(_rng.uniform(0.10, 0.45))
    third = _rng.choice([c for c in _POOL if c not in (par, alt)])
    spread = float(_rng.uniform(0.10, 0.22))
    p_alt = (1.0 - spread + margin) / 2.0
    DIST.append(dict(par=par, alt=alt, third=third,
                     p_alt=p_alt, p_par=p_alt - margin, p_third=spread,
                     margin=margin, wants=margin > 0))
KEEP = sorted(_real)
DROP = sorted(i for i in range(29) if DIST[i]["wants"] and i not in _real)
assert len(KEEP) == BUDGET and len(KEEP) + len(DROP) == WANTED
assert sorted(sorted(range(29), key=lambda i: -DIST[i]["margin"])[:BUDGET]) == KEEP


WIDTH = 4
POP = [
    ("slot 0", 0.1410, ["R27V", "S29A", "S30A", "T57Q", "K103H"]),
    ("slot 1", 0.3011, ["N54T", "T57D", "K103G", "N106P", "Y107G"]),
    ("slot 2", 0.3246, ["V31F", "N54E", "W101A", "R105E", "A108Q"]),
    ("slot 3", 0.2543, ["S29A", "N33A", "G56D", "E104N", "R105Y"]),
]
POP_HAMMING = "8 to 10"
PROPOSAL = (0.1542, ["N54E", "W101A", "R105E"], 2)
PROPOSAL_DIST = [8, 7, 2, 7]
EV_TOTAL = 5024
EV_CROWD = 9
EV_ELITE = 1140
EV_ACCEPT = 1020
POLICY_RESULT = dict(pop_ipsae=0.529, ind_ipsae=0.470, pop_pose=30.5,
                     ind_pose=27.4, sens_ipsae=0.11, sens_pose=6.0)


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def blank():
    """A frame description. Every clip fills the same fields."""
    return dict(title="", sub="", caption="", note="", tag="",
                letters=list(PARENT), state=["par"] * 29, outline=-1,
                stack=(), proposal=None, badges=False, challenged=-1,
                logo=None, logo_dim=(), row=True, schedule=None, bullets=(), meters=[], spent=0, big=None,
                budget_note="", free=None)


class Clip:
    """Base: a list of (name, seconds) cuts and a pure state(frame)."""
    cuts: list = []
    name = ""
    layout = "logo"

    def __init__(self, fps: int):
        self.fps = fps
        self.spans, t = [], 0
        for nm, sec in self.cuts:
            n = max(1, int(round(sec * fps)))
            self.spans.append((nm, t, t + n))
            t += n
        self.total = t

    def at(self, f):
        for nm, a, b in self.spans:
            if a <= f < b:
                return nm, (f - a) / max(1, b - a - 1)
        return self.spans[-1][0], 1.0


class Discrete(Clip):
    name = "discrete"
    layout = "row"
    cuts = [("open", 2.4)] + [(f"step{i}", 2.3) for i in range(5)] + \
           [("full", 3.0), ("close", 3.4)]

    def state(self, f):
        nm, u = self.at(f)
        st = blank()
        st["tag"] = "measured · control search seed 2"

        if nm == "open":
            st["title"] = "1 · Discrete search"
            st["sub"] = "what we ran: one whole substitution per proposal"
            st["caption"] = ("P17 nanobody, 29 designable residues in the three CDRs. "
                             "Every proposal is a complete sequence, scored by the frozen "
                             "predictor, then kept or thrown away.")
            st["meters"] = [("score", 0.0, "0.0000", ACCENT),
                            ("score calls used", 0.0, f"0 / {CALLS}", ACCENT_DIM)]
            return st

        if nm.startswith("step"):
            k = int(nm[4:])
            mv = ASCENT[k]
            before = 0.0 if k == 0 else ASCENT[k - 1]["after"]
            st["title"] = "1 · Discrete search"
            st["sub"] = "one whole substitution per proposal, scored and kept or thrown away"
            for j in range(k):
                i = WINDOW.index(ASCENT[j]["res"])
                st["letters"][i], st["state"][i] = ASCENT[j]["to"], "held"
            st["spent"] = k
            score, shown = before, f"{before:.4f}"
            rej = mv["reject"]
            if rej is not None and u < 0.46:
                rres = rej[2] if len(rej) > 2 else mv["res"]
                i = WINDOW.index(rres)
                st["letters"][i], st["outline"] = rej[0], i
                st["state"][i] = "probe" if u < 0.24 else "reject"
                st["note"] = (f"propose {PARENT[i]}{rres}{rej[0]}" if u < 0.24
                              else f"rejected — {rej[1]:.4f} is no better than {before:.4f}")
                if u >= 0.24:
                    shown = f"{rej[1]:.4f}"
            else:
                i = WINDOW.index(mv["res"])
                st["outline"] = i
                st["letters"][i] = mv["to"]
                early = u < (0.70 if rej is not None else 0.42)
                st["state"][i] = "probe" if early else "held"
                if early:
                    st["note"] = f"propose {PARENT[i]}{mv['res']}{mv['to']}"
                else:
                    st["spent"] = k + 1
                    score, shown = mv["after"], f"{mv['after']:.4f}"
                    st["note"] = (f"accepted — {mv['after']:.4f} "
                                  f"· {k + 1} of {BUDGET} edits spent")
            call = ASCENT[k - 1]["call"] if k else 0
            if st["spent"] == k + 1:
                call = mv["call"]
            st["meters"] = [("score", score / SCORE_MAX, shown, ACCENT),
                            ("score calls used", call / CALLS,
                             f"{call} / {CALLS}", ACCENT_DIM)]
            st["caption"] = ("200 score calls, about 51 accepted. The run stops on the "
                             "call budget, never on edits — every winner spends the "
                             "whole edit budget.")
            return st

        for mv in ASCENT:
            i = WINDOW.index(mv["res"])
            st["letters"][i], st["state"][i] = mv["to"], "held"
        st["spent"] = BUDGET
        st["meters"] = [("score", ASCENT[-1]["after"] / SCORE_MAX,
                         f"{ASCENT[-1]['after']:.4f}", ACCENT),
                        ("score calls used", 1.0, f"{CALLS} / {CALLS}", ACCENT_DIM)]

        if nm == "full":
            st["title"] = "1 · Budget full — every further move is an exchange"
            st["sub"] = "revert one substitution, add another, score the pair as one move"
            g = ease(u)
            i = WINDOW.index(ASCENT[0]["res"])
            st["state"][i] = "revert" if g > 0.35 else "held"
            st["note"] = "5 of 5 spent — no free moves left"
            st["budget_note"] = "exchange"
            st["caption"] = ("From here the search can only trade. This is the regime a "
                             "seeded run starts in, which is the whole reason seeding "
                             "costs something.")
            return st

        st["title"] = "1 · Where the discrete search lands"
        st["sub"] = "20 independent search seeds, all from the unmodified parent"
        st["note"] = (f"this seed {ASCENT[-1]['after']:.4f} · "
                      f"20-seed mean {MEAN_BEST['parent']:.4f} "
                      f"(0.3246 to 0.8037)")
        st["caption"] = ("Each seed finds a different set of five substitutions and they "
                         "all score about the same: interface RMSD near 7 Å, every "
                         "structure CAPRI-incorrect. The ceiling is the objective, "
                         "not the search.")
        return st


class Continuous(Clip):
    name = "continuous"
    cuts = [("open", 2.4), ("relax", 7.2), ("round", 3.6),
            ("cliff", 3.8), ("close", 3.2)]
    STAGES = [("logits", 240), ("logits", 120), ("soft", 216), ("hard", 24)]

    def state(self, f):
        nm, u = self.at(f)
        st = blank()
        st["tag"] = "measured, with illustrative distributions"

        if nm == "open":
            st["title"] = "2 · Continuous relaxation"
            st["sub"] = "the bindcraft route: let every position hold a distribution"
            st["caption"] = ("Instead of one residue per position, a probability over all "
                             "20. The gradient of the predictor moves all 29 at once, "
                             "600 steps on BindCraft's four-stage schedule.")
            st["logo"], st["row"] = 0.0, False
            st["meters"] = [("soft loss", 0.0, f"{SOFT_FROM:.2f}", PASS)]
            return st

        if nm == "relax":
            g = ease(u)
            st["title"] = "2 · Continuous relaxation — descending the soft loss"
            st["sub"] = "50:25:45:5 rescaled to 600 steps → 240 / 120 / 216 / 24"
            st["logo"], st["row"] = g, False
            st["schedule"] = u
            loss = SOFT_FROM - g * (SOFT_FROM - SOFT_TO)
            st["meters"] = [("soft loss", g, f"{loss:.2f}", PASS)]
            nwant = int(round(g * WANTED))
            st["note"] = (f"descended {SOFT_FROM - loss:.2f} · "
                          f"{nwant} positions now prefer a non-parent residue")
            st["caption"] = ("It works: the loss falls 28.25. By the last stage, 15 of the "
                             "29 positions want a residue the parent does not have — "
                             "against an edit budget of 5.")
            return st

        if nm == "round":
            g = ease(u)
            st["title"] = "2 · The projection back to whole residues"
            st["sub"] = "keep the five largest margins, discard the other ten"
            st["logo"] = 1.0
            for i in range(29):
                d = DIST[i]
                if i in KEEP:
                    st["letters"][i] = d["alt"]
                    st["state"][i] = "held" if g > 0.5 else "par"
                elif i in DROP:
                    st["letters"][i] = d["alt"] if g < 0.5 else d["par"]
                    st["state"][i] = "reject" if 0.25 < g < 0.75 else "par"
            st["logo_dim"] = DROP if g > 0.5 else ()
            st["spent"] = BUDGET if g > 0.5 else 0
            st["meters"] = [("soft loss", 1.0, f"{SOFT_TO:.2f}", PASS)]
            st["note"] = "10 of the 15 wanted substitutions are discarded"
            st["caption"] = ("The search cannot hold a distribution, so the relaxation has "
                             "to be rounded to whole residues inside the budget. Which ten "
                             "are dropped here is illustrative; that ten are dropped is not.")
            return st

        for i in KEEP:
            st["letters"][i], st["state"][i] = DIST[i]["alt"], "held"
        st["spent"] = BUDGET
        st["logo"], st["logo_dim"] = 1.0, DROP

        if nm == "cliff":
            st["title"] = "2 · Now score the sequence the search receives"
            st["sub"] = "same frozen predictor, on the rounded sequence"
            g = ease(min(1.0, u * 2.2))
            st["meters"] = [
                ("soft loss, descended", 1.0, f"−{SOFT_FROM - SOFT_TO:.2f}", PASS),
                ("score the search sees", HARD_SCORE / SCORE_MAX * g,
                 f"{HARD_SCORE:.4f}" if u > 0.22 else "", NULL),
            ]
            if u > 0.42:
                st["big"] = f"{HARD_SCORE:.4f}"
            st["note"] = "the unmodified parent scores 0.0000"
            st["caption"] = ("The descent does not survive the rounding. The seed arrives "
                             "at the parent's score with all five edits already committed: "
                             "nothing gained, and no free moves left.")
            return st

        st["title"] = "2 · What the search does with it"
        st["sub"] = "start in the exchange regime, revert most of the seed, land in the same place"
        st["budget_note"] = "exchange"
        g = ease(u)
        for n, i in enumerate(KEEP):
            if n >= 3 and g > 0.4:
                st["letters"][i], st["state"][i] = DIST[i]["par"], "revert"
        st["meters"] = [("20-seed mean best", MEAN_BEST["bindcraft"] / SCORE_MAX,
                         f"{MEAN_BEST['bindcraft']:.4f}", NULL),
                        ("unseeded control", MEAN_BEST["parent"] / SCORE_MAX,
                         f"{MEAN_BEST['parent']:.4f}", ACCENT)]
        st["note"] = "44 of 100 seeded edit slots reverted across the 20 winners"
        st["caption"] = ("Seeded and unseeded cells reach the same plateau. Tested at "
                         "budget 5 and again at budget 7, with three relaxation methods: "
                         "no advantage, and the exchange cost is real.")
        return st


class Proposed(Clip):
    name = "proposed"
    cuts = [("open", 2.6), ("keep", 3.2), ("bias", 6.0), ("close", 3.6)]
    TRIES = [(KEEP[0], True), (DROP[1], False), (KEEP[2], True), (DROP[4], False)]

    def state(self, f):
        nm, u = self.at(f)
        st = blank()
        st["tag"] = "proposed · not run · no measured numbers on this clip"

        if nm == "open":
            st["title"] = "3 · What we propose next"
            st["sub"] = "use the gradient to choose what to try, not where to start"
            st["caption"] = ("Both halves work on their own. What fails is the rounding "
                             "between them — so take out the rounding, and keep the "
                             "distribution as a proposal bias.")
            st["logo"] = 1.0
            return st

        if nm == "keep":
            st["title"] = "3 · Keep the distribution, do not round it"
            st["sub"] = "the search still starts at the unmodified parent"
            st["logo"] = 1.0
            st["note"] = "0 of 5 edits spent — every move still free"
            st["free"] = BUDGET
            st["bullets"] = (
                ("keep", "the per-position ranking over substitutions"),
                ("drop", "the rounded seed sequence, and the cliff with it"),
            )
            st["caption"] = ("No seed sequence, so nothing has to survive a projection and "
                             "no edits are committed before the first score call. The "
                             "relaxation's output is a ranking over substitutions.")
            return st

        if nm == "bias":
            st["title"] = "3 · Propose from the ranking, judge with the predictor"
            st["sub"] = "the gradient supplies candidates; acceptance is unchanged"
            st["logo"] = 1.0
            k = min(len(self.TRIES) - 1, int(u * len(self.TRIES)))
            frac = u * len(self.TRIES) - k
            spent = 0
            for i, ok in self.TRIES[:k]:
                if ok:
                    st["letters"][i], st["state"][i] = DIST[i]["alt"], "held"
                    spent += 1
            i, ok = self.TRIES[k]
            st["outline"] = i
            st["letters"][i] = DIST[i]["alt"]
            if frac < 0.45:
                st["state"][i] = "probe"
                st["note"] = (f"try {PARENT[i]}{WINDOW[i]}{DIST[i]['alt']} "
                              "\u2014 a substitution the relaxation ranked highly")
            elif ok:
                st["state"][i] = "held"
                spent += 1
                st["note"] = "accepted — the predictor agrees"
            else:
                st["state"][i] = "reject"
                st["note"] = "rejected — the ranking is a prior, not a verdict"
            st["spent"] = spent
            st["free"] = BUDGET - spent
            st["bullets"] = (
                ("unchanged", "200 score calls, budget 5, same acceptance rule"),
                ("changed", "which substitution gets proposed next"),
            )
            st["caption"] = ("Each proposal is still a whole sequence and still has to earn "
                             "its place. The gradient only changes the order in which "
                             "substitutions are tried, so a wrong ranking costs calls, "
                             "not edits.")
            return st

        st["title"] = "3 · Why this is the version worth running"
        st["sub"] = "and what the null already tells us about the alternatives"
        st["logo"] = 1.0
        st["note"] = "untested — this clip is a design, not a result"
        st["free"] = BUDGET
        st["bullets"] = (
            ("no rounding", "nothing is committed before the first score call"),
            ("5 free moves", "the search starts exactly where the control starts"),
        )
        st["caption"] = ("Nothing is rounded, so there is no cliff to fall off, and the "
                         "search keeps all five free moves. Anything that still hands over "
                         "a start sequence will reproduce the null we measured twice.")
        return st


def _apply(subs):
    """Turn ["V31F", ...] into a window row of letters plus its state."""
    letters, state = list(PARENT), ["par"] * 29
    for sub in subs:
        res = int(sub[1:-1])
        i = WINDOW.index(res)
        assert PARENT[i] == sub[0], sub
        letters[i], state[i] = sub[-1], "held"
    return letters, state


class Policy(Clip):
    """Independent versus population retention, on the live population."""

    name = "policy"
    layout = "stack"
    cuts = [("open", 3.2), ("indep", 4.6), ("pop", 6.0), ("verdict", 5.2)]

    def _rows(self):
        out = []
        for label, score, subs in POP:
            letters, state = _apply(subs)
            out.append(dict(label=label, score=score, letters=letters,
                            state=state, badge="", dim=False, ring=False))
        return out

    def state(self, f):
        nm, u = self.at(f)
        st = blank()
        st["tag"] = "measured \u00b7 control cell, 20 search seeds"
        rows = self._rows()
        st["stack"] = rows
        pscore, psubs, pslot = PROPOSAL
        pl, ps = _apply(psubs)
        proposal = dict(label="proposal", score=pscore, letters=pl, state=ps,
                        badge="", dim=False, ring=False)

        if nm == "open":
            st["title"] = "Retention: what the search keeps"
            st["sub"] = "the search runs four live candidates at once, not one"
            st["note"] = f"pairwise Hamming {POP_HAMMING} of a possible 10 at budget 5"
            st["bullets"] = (
                ("width 4", "every proposal is born from one slot, round-robin"),
                ("the choice", "which incumbent that proposal is allowed to replace"),
            )
            st["caption"] = ("These are the four live candidates at the end of control "
                             "search seed 2. They hold near-disjoint edit sets, so the "
                             "population is doing what it was built to do.")
            return st

        if nm == "indep":
            st["title"] = "Policy A \u00b7 independent"
            st["sub"] = "the proposal may only replace the slot it came from"
            rows[pslot]["ring"] = True
            for j, r in enumerate(rows):
                r["dim"] = j != pslot
            st["proposal"] = proposal
            st["challenged"] = pslot
            st["note"] = f"born from {POP[pslot][0]} \u2192 challenges {POP[pslot][0]}"
            st["bullets"] = (
                ("effect", "four hill-climbs that never interact"),
                ("kept", "the best candidate seen in any slot"),
            )
            st["caption"] = ("Simplest possible rule. The other three slots are "
                             "irrelevant to this proposal's fate; they are running "
                             "their own ascents in parallel.")
            return st

        if nm == "pop":
            st["title"] = "Policy B \u00b7 population"
            st["sub"] = "the proposal replaces its nearest incumbent by Hamming distance"
            st["badges"] = True
            near = int(np.argmin(PROPOSAL_DIST))
            for j, r in enumerate(rows):
                r["badge"] = str(PROPOSAL_DIST[j])
                r["dim"] = u > 0.45 and j != near
                r["ring"] = u > 0.45 and j == near
            st["proposal"] = proposal
            st["challenged"] = near if u > 0.45 else -1
            st["note"] = ("distance to each slot \u2014 nearest wins the challenge"
                          if u <= 0.45 else
                          f"nearest is {POP[near][0]}, which is its own parent")
            st["bullets"] = (
                ("crowding", "keeps distinct solutions from eating each other"),
                ("plus", "duplicates rejected, and the elite cannot be displaced"),
            )
            st["caption"] = ("A proposal changes one or two positions, so it sits 2 from "
                             "its own parent and 7 to 8 from every other slot. The "
                             "nearest incumbent is almost always the parent itself.")
            return st

        for r in rows:
            r["dim"] = True
        st["title"] = "What the difference is worth"
        st["sub"] = "measured on the proposals themselves, and on the outcome"
        R = POLICY_RESULT
        st["meters"] = [
            ("population, mean ipSAE", R["pop_ipsae"] / 0.8, f"{R['pop_ipsae']:.3f}", ACCENT),
            ("independent, mean ipSAE", R["ind_ipsae"] / 0.8, f"{R['ind_ipsae']:.3f}", INK_FAINT),
        ]
        st["note"] = (
            f"crowding fired {EV_CROWD} times in {EV_TOTAL:,} proposals "
            f"\u2014 {100 * EV_CROWD / EV_TOTAL:.2f}%\n"
            f"elite protection, the part that does fire, blocked {EV_ELITE:,} "
            f"({100 * EV_ELITE / EV_TOTAL:.0f}%)\n"
            "0.059 ipSAE apart, against a detection threshold of 0.11")
        st["caption"] = ("Section 27.2 ran the only matched comparison: 10 seeds, pinned "
                         "exploration, null on both readouts. Independent is the simpler "
                         "configuration and nothing measurable is lost by using it.")
        return st


class Painter:
    def __init__(self, fig, ax, clip):
        self.fig, self.ax, self.clip = fig, ax, clip

    X0, X1 = 0.055, 0.945
    BGAP = 0.016

    def cells(self):
        span = self.X1 - self.X0 - self.BGAP * (len(BLOCKS) - 1)
        w = span / 29
        out, x = [], self.X0
        for a, b, _ in BLOCKS:
            for i in range(a, b):
                out.append((x + (i - a) * w, w))
            x += (b - a) * w + self.BGAP
        return out

    def draw(self, st):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(self.X0, 0.905, st["title"], fontsize=22, fontweight="bold",
                color=INK, family=SANS, va="bottom")
        if st["sub"]:
            ax.text(self.X0, 0.864, st["sub"], fontsize=12.5, color=INK_SOFT,
                    family=SANS, va="bottom")

        if st["schedule"] is not None:
            self._schedule(st["schedule"])

        if self.clip.layout == "logo":
            row_y, row_h, note_y, meter_y = 0.435, 0.058, 0.395, 0.318
            if st["logo"] is not None:
                self._logo(st["logo"], st["logo_dim"])
            self._row(st, row_y, row_h)
        elif self.clip.layout == "stack":
            note_y, meter_y = 0.410, 0.250
            self._stack(st)
        else:
            row_y, row_h, note_y, meter_y = 0.640, 0.072, 0.566, 0.488
            self._row(st, row_y, row_h)

        if st["note"]:
            ax.text(self.X0, note_y, st["note"], fontsize=11.5, color=INK_FAINT,
                    family=MONO, va="top")

        y = meter_y
        for label, frac, value, colour in st["meters"]:
            self._meter(y, label, frac, value, colour)
            y -= 0.066
        if self.clip.layout != "stack":
            self._slots(st, y - 0.016)
            y -= 0.016 + 0.062
        elif st["meters"]:
            y -= 0.062
        for label, text in st["bullets"]:
            ax.text(self.X0, y, label, fontsize=11, color=INK_FAINT,
                    family=MONO, va="center")
            ax.text(0.265, y, text, fontsize=12, color=INK_SOFT,
                    family=SANS, va="center")
            y -= 0.050

        if st["big"]:
            ax.text(0.848, meter_y - 0.022, st["big"], fontsize=34,
                    fontweight="bold", color=NULL, family=MONO,
                    ha="center", va="center")

        cap = "\n".join(textwrap.wrap(st["caption"], width=118)[:3])
        ax.text(self.X0, 0.140, cap, fontsize=12.5, color=INK_SOFT,
                family=SANS, va="top", linespacing=1.5)
        ax.text(self.X0, 0.030,
                "P17 nanobody vs JN1 · frozen OpenDDE · project record §33",
                fontsize=9, color=INK_FAINT, family=MONO, va="bottom")
        if st["tag"]:
            ax.text(self.X1, 0.030, st["tag"], fontsize=9.5,
                    color=NULL if "not run" in st["tag"] else INK_FAINT,
                    family=MONO, ha="right", va="bottom")

    def _schedule(self, u):
        ax = self.ax
        y, h = 0.790, 0.024
        total = sum(n for _, n in Continuous.STAGES)
        done = u * total
        x, seen = self.X0, 0
        for name, n in Continuous.STAGES:
            w = (self.X1 - self.X0) * n / total
            filled = min(1.0, max(0.0, (done - seen) / n))
            ax.add_patch(Rectangle((x, y), w, h, facecolor=SURFACE,
                                   edgecolor=RULE, linewidth=0.8, zorder=2))
            if filled > 0:
                ax.add_patch(Rectangle((x, y), w * filled, h, facecolor=ACCENT,
                                       edgecolor="none", alpha=0.85, zorder=3))
            ax.text(x + w / 2, y + h + 0.010, f"{name} {n}", fontsize=9,
                    color=INK_FAINT, family=MONO, ha="center", va="bottom", zorder=4)
            x += w
            seen += n

    def _logo(self, g, dim=()):
        """Stacked letters per position, heights proportional to probability."""
        ax = self.ax
        y0, h = 0.522, 0.218
        ax.text(self.X0, y0 + h + 0.014,
                "probability over 20 residues, per position", fontsize=9.5,
                color=INK_FAINT, family=MONO, va="bottom")
        if dim:
            ax.text(self.X1, y0 + h + 0.014, "faded = wanted, over budget, dropped",
                    fontsize=9.5, color=INK_FAINT, family=MONO,
                    ha="right", va="bottom")
        for i, ((x, w), d) in enumerate(zip(self.cells(), DIST)):
            cx = x + w / 2
            a = 0.22 if i in dim else 1.0
            ps = [(d["par"], 1.0 - g * (1.0 - d["p_par"]), ACCENT_DIM),
                  (d["alt"], g * d["p_alt"], ACCENT),
                  (d["third"], g * d["p_third"], INK_FAINT)]
            ps.sort(key=lambda t: -t[1])
            yy = y0 + h
            for ch, p, colour in ps:
                if p < 0.07:
                    continue
                hh = h * p
                ax.text(cx, yy - hh / 2, ch, fontsize=max(6.0, 30.0 * p),
                        color=colour, family=MONO, fontweight="bold", alpha=a,
                        ha="center", va="center", zorder=4)
                yy -= hh

    def _row(self, st, y0, h):
        ax = self.ax
        if not st["row"]:
            ax.add_patch(Rectangle((self.X0, y0), self.X1 - self.X0, h,
                                   facecolor="none", edgecolor=RULE,
                                   linewidth=0.8, linestyle=(0, (3, 3)), zorder=2))
            ax.text((self.X0 + self.X1) / 2, y0 + h / 2,
                    "no whole sequence exists while the relaxation runs",
                    fontsize=11, color=INK_FAINT, family=MONO,
                    ha="center", va="center")
            return
        for (x, w), i in zip(self.cells(), range(29)):
            s = st["state"][i]
            face = {"par": SURFACE, "held": ACCENT, "probe": SURFACE,
                    "reject": SURFACE, "revert": SURFACE}[s]
            ax.add_patch(Rectangle((x, y0), w, h, facecolor=face,
                                   edgecolor=RULE, linewidth=0.7, zorder=2))
            if s == "reject":
                ax.add_patch(Rectangle((x, y0), w, h, facecolor=NULL,
                                       edgecolor="none", alpha=0.16, zorder=3))
            colour = {"par": INK_SOFT, "held": "white", "probe": INK,
                      "reject": NULL, "revert": INK_FAINT}[s]
            ax.text(x + w / 2, y0 + h / 2, st["letters"][i],
                    fontsize=15 if h < 0.065 else 17,
                    fontweight="bold" if s != "par" else "normal",
                    color=colour, family=MONO, ha="center", va="center", zorder=5)
            if s == "revert":
                ax.add_patch(Rectangle((x, y0), w, h, facecolor="none",
                                       edgecolor=NULL, linewidth=1.6,
                                       linestyle=(0, (2, 1.6)), zorder=6))
            if i == st["outline"]:
                ax.add_patch(Rectangle((x, y0), w, h, facecolor="none",
                                       edgecolor=PROPOSE if self.clip.name == "proposed" else INK,
                                       linewidth=2.0, zorder=7))
            if WINDOW[i] % 2 == 1 or i in (0, 28):
                ax.text(x + w / 2, y0 - 0.012, str(WINDOW[i]), fontsize=7.5,
                        color=INK_FAINT, family=MONO, ha="center", va="top")
        cells = self.cells()
        ly = y0 + h + 0.012 if self.clip.layout == "row" else y0 + h + 0.007
        for a, b, lab in BLOCKS:
            x = cells[a][0]
            wid = cells[b - 1][0] + cells[b - 1][1] - x
            ax.text(x + wid / 2, ly, lab, fontsize=9,
                    color=INK_FAINT, family=MONO, ha="center", va="bottom")

    GUTTER, SX0, SX1 = 0.055, 0.190, 0.900

    def _scells(self):
        span = self.SX1 - self.SX0 - self.BGAP * (len(BLOCKS) - 1)
        w = span / 29
        out, x = [], self.SX0
        for a, b, _ in BLOCKS:
            for i in range(a, b):
                out.append((x + (i - a) * w, w))
            x += (b - a) * w + self.BGAP
        return out

    def _seqrow(self, row, y, h, badges):
        ax = self.ax
        a = 0.30 if row["dim"] else 1.0
        ax.text(self.GUTTER, y + h / 2,
                "%s  %.4f" % (row["label"], row["score"]), fontsize=10,
                color=INK_FAINT, family=MONO, va="center", alpha=a)
        for (x, w), i in zip(self._scells(), range(29)):
            held = row["state"][i] == "held"
            ax.add_patch(Rectangle((x, y), w, h,
                                   facecolor=ACCENT if held else SURFACE,
                                   edgecolor=RULE, linewidth=0.6,
                                   alpha=a, zorder=2))
            ax.text(x + w / 2, y + h / 2, row["letters"][i], fontsize=11,
                    fontweight="bold" if held else "normal",
                    color="white" if held else INK_SOFT, family=MONO,
                    ha="center", va="center", alpha=a, zorder=4)
        if row["ring"]:
            ax.add_patch(Rectangle((self.SX0 - 0.006, y - 0.006),
                                   self.SX1 - self.SX0 + 0.012, h + 0.012,
                                   facecolor="none", edgecolor=INK,
                                   linewidth=1.8, zorder=6))
        if badges and row["badge"]:
            ax.text(self.SX1 + 0.030, y + h / 2, row["badge"], fontsize=14,
                    fontweight="bold",
                    color=INK if row["ring"] else INK_FAINT,
                    family=MONO, ha="center", va="center", alpha=max(a, 0.45))

    def _stack(self, st):
        ax = self.ax
        h, gap, y = 0.050, 0.016, 0.700
        if st["badges"]:
            ax.text(self.SX1 + 0.030, y + h + 0.030, "Hamming", fontsize=9,
                    color=INK_FAINT, family=MONO, ha="center", va="center")
        for row in st["stack"]:
            self._seqrow(row, y, h, st["badges"])
            y -= h + gap
        if st["proposal"] is not None:
            y -= 0.014
            ax.plot([self.SX0 - 0.010, self.SX0 - 0.010],
                    [y + h, 0.700 + h], color=RULE, linewidth=0.8, zorder=1)
            self._seqrow(dict(st["proposal"], label="proposal"), y, h, False)
            ax.add_patch(Rectangle((self.SX0 - 0.006, y - 0.006),
                                   self.SX1 - self.SX0 + 0.012, h + 0.012,
                                   facecolor="none", edgecolor=PROPOSE,
                                   linewidth=1.8, linestyle=(0, (3, 2)), zorder=6))
            if st["challenged"] >= 0:
                ty = 0.700 - st["challenged"] * (h + gap)
                ax.annotate("", xy=(self.GUTTER + 0.122, ty + h / 2),
                            xytext=(self.GUTTER + 0.122, y + h / 2),
                            arrowprops=dict(arrowstyle="-|>", color=PROPOSE,
                                            linewidth=1.6, shrinkA=2, shrinkB=2),
                            zorder=7)

    def _meter(self, y, label, frac, value, colour):
        ax = self.ax
        bx0, bx1, h = 0.265, 0.700, 0.019
        ax.text(self.X0, y + h / 2, label, fontsize=11, color=INK_FAINT,
                family=MONO, va="center")
        ax.add_patch(FancyBboxPatch((bx0, y), bx1 - bx0, h,
                                    boxstyle="round,pad=0,rounding_size=0.009",
                                    facecolor=SURFACE, edgecolor=RULE,
                                    linewidth=0.8, zorder=2))
        f = min(1.0, max(0.0, float(frac)))
        if f > 0:
            ax.add_patch(FancyBboxPatch((bx0, y), max(0.004, (bx1 - bx0) * f), h,
                                        boxstyle="round,pad=0,rounding_size=0.009",
                                        facecolor=colour, edgecolor="none", zorder=3))
        if value:
            ax.text(bx1 + 0.016, y + h / 2, value, fontsize=13, fontweight="bold",
                    color=INK, family=MONO, va="center")

    def _slots(self, st, y):
        ax = self.ax
        s, gap = 0.014, 0.006
        ax.text(self.X0, y + 0.013, "edits spent", fontsize=11, color=INK_FAINT,
                family=MONO, va="center")
        x = 0.265
        for j in range(BUDGET):
            on = j < st["spent"]
            ax.add_patch(Rectangle((x + j * (s + gap), y), s, s * (16 / 9),
                                   facecolor=ACCENT if on else SURFACE,
                                   edgecolor=ACCENT if on else RULE,
                                   linewidth=0.9, zorder=3))
        tail = f"{st['spent']} / {BUDGET}"
        if st["budget_note"]:
            tail += f"  — {st['budget_note']}"
        elif st["free"] is not None:
            tail += f"  — {st['free']} free"
        ax.text(0.265 + BUDGET * (s + gap) + 0.012, y + 0.013, tail,
                fontsize=12, fontweight="bold",
                color=NULL if st["budget_note"] else INK, family=MONO, va="center")


def render(clip, out: Path, fps: int, gif: bool, stills: bool):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(PAPER)
    painter = Painter(fig, ax, clip)

    mp4 = out / f"seq_{clip.name}.mp4"
    anim = animation.FuncAnimation(fig, lambda f: (painter.draw(clip.state(f)), [])[1],
                                   frames=clip.total, interval=1000 / fps)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart"])
    print(f"{clip.name}: {clip.total} frames at {fps} fps "
          f"({clip.total / fps:.1f} s) -> {mp4}")
    anim.save(str(mp4), writer=writer, savefig_kwargs={"facecolor": PAPER})

    if stills:
        for nm, a, b in clip.spans:
            painter.draw(clip.state(b - 1))
            p = out / f"seq_{clip.name}_{nm}.png"
            fig.savefig(p, facecolor=PAPER, dpi=150)
            print(f"  wrote {p}")
    plt.close(fig)

    if gif:
        g = out / f"seq_{clip.name}.gif"
        pal = out / f"_pal_{clip.name}.png"
        vf = "fps=12,scale=900:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-vf", vf + ",palettegen=stats_mode=diff", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                        "-i", str(pal), "-lavfi",
                        vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(g)], check=True)
        pal.unlink(missing_ok=True)
        print(f"  wrote {g} ({g.stat().st_size / 1048576:.1f} MB)")
    print(f"  wrote {mp4} ({mp4.stat().st_size / 1024:.0f} K)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("docs/build"))
    ap.add_argument("--only",
                    choices=["discrete", "continuous", "proposed", "policy"],
                    action="append")
    ap.add_argument("--gif", action="store_true", default=True)
    ap.add_argument("--no-gif", dest="gif", action="store_false")
    ap.add_argument("--stills", action="store_true",
                    help="also write one PNG per cut, for slides")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = args.only or ["discrete", "continuous", "proposed", "policy"]
    for cls in (Discrete, Continuous, Proposed, Policy):
        if cls.name in wanted:
            render(cls(args.fps), args.out, args.fps, args.gif, args.stills)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
