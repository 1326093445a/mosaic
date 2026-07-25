"""Build the VHH72 CDR-only contrast-pair table from the real AlphaSeq dataset.

Produces the empirical ground truth Path 1 (see
docs/guidance_alphaseq_testing_notes.md) compares guidance's gradient
direction against: pairs of independently-designed VHH72 variants that
differ by exactly 1 or 2 CDR positions, both with a real measured KD against
the same target (WT/Gamma/Delta RBD).

Why contrast pairs, not a per-position additive regression: the ~24k
designed VHH72 variants are a *campaign-biased* sample (some prior guided
design process chose what to try, not a random/exhaustive DMS), and are
mostly multi-mutant, not clean single-point mutants. A contrast pair (A vs
B, differ by 1-2 CDR positions, real KD difference) only claims what's
directly observable -- no additive/no-epistasis model, no extrapolation
into substitutions nobody tried. Weaker statistical power per pair than a
fitted model, but honest about what it covers.

Two real subtleties this script handles, both found and fixed during the
analysis this file reproduces:
  1. Naive whole-sequence "differs by 1-2 positions" pairs are dominated by
     FRAMEWORK mutations, not CDR ones (~78% of raw distance<=2 pairs
     involve at least one framework position) -- must restrict to the real
     CDR boundaries, not eyeball it.
  2. The 12 "WT_synonymous" AlphaSeq rows (identical WT protein sequence,
     different silent-codon DNA constructs) give a genuine assay
     reproducibility/noise-floor estimate (std ~= 0.117 in
     neg_log10_KD_nM units, range ~= 0.36) -- a meaningful fraction of
     naively-computed pair deltas sit inside that noise band and shouldn't
     be trusted as real biological signal without checking against it (or,
     better, against each measurement's own `ci_width_log10_KD_nM`).

CDR boundaries (IMGT numbering, chain_type H) were derived by running
ANARCI on the real WT VHH72 sequence (extracted from
vhh72_wt_wt_rbd.cif's chain A) -- see the module docstring's Usage section
for how to reproduce that if the sequence ever changes:
  CDR1: 1-indexed 26-33 (0-indexed 25-32)
  CDR2: 1-indexed 51-58 (0-indexed 50-57)
  CDR3: 1-indexed 97-114 (0-indexed 96-113)
Notably CDR1/CDR2 land at the same indices as a different nanobody
(P17_JN1) checked earlier in this project -- consistent with typical VHH
framework lengths, not a coincidence specific to either molecule.

Requires: read access to /home/yfeng17/SBSAb/dataset/alphaseq/ (the real
AlphaSeq dataset + Boltz-predicted CIFs this project is testing against;
not part of this repo). No GPU, no jax/mosaic import -- pure gemmi + numpy,
runs in under a minute for the full ~24k-variant set.

To rederive the CDR boundaries from scratch (only needed if the WT
reference sequence changes): run ANARCI (a separate conda env named
`anarci` exists on the machine this was developed on --
`conda activate anarci`) on the WT sequence with
`run_anarci([(name, seq)], scheme="imgt", allow={"H"})` and apply the
standard IMGT CDR ranges (CDR1 27-38, CDR2 56-65, CDR3 105-117 in IMGT
numbering, which the insertion-coded numbering above maps back to raw
sequence positions).

Usage:
    .venv/bin/python examples/alphaseq_vhh72_cdr_contrast_pairs.py
"""
import csv
import time
from collections import Counter

import gemmi
import numpy as np

ALPHASEQ_CSV = "/home/yfeng17/SBSAb/dataset/alphaseq/alphaseq_all_ptgen_v1.csv"

# 0-indexed, derived via ANARCI/IMGT on the real WT VHH72 sequence -- see
# module docstring.
CDR1 = set(range(25, 33))
CDR2 = set(range(50, 58))
CDR3 = set(range(96, 114))
CDR_ALL = CDR1 | CDR2 | CDR3

# From the 12 WT_synonymous AlphaSeq replicates (same protein, different
# silent-codon DNA, same target) -- see module docstring. A pair delta below
# this is not distinguishable from assay noise on this evidence alone.
NOISE_FLOOR_THRESHOLD = 0.3


def load_designed_vhh72_rows(csv_path: str) -> dict:
    """One row per unique antibody_group (VHH72_esm_*/VHH72_label_encoded_*
    designed-variant families only -- excludes SARS_VHH72's own antigen-DMS
    rows and the unrelated `candidate`/CR3022/m396 families)."""
    rows_by_group = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ag = row["antibody_group"]
            if ag.startswith("VHH72_esm") or ag.startswith("VHH72_label"):
                rows_by_group.setdefault(ag, []).append(row)
    return rows_by_group


def extract_sequences_and_kd(rows_by_group: dict):
    seqs, kd_by_group = {}, {}
    for ag, rows in rows_by_group.items():
        st = gemmi.read_structure(rows[0]["structure_path"])
        chain = st[0]["A"]  # VHH72 is Ligand_Chains="A" for this family
        seqs[ag] = gemmi.one_letter_code([r.name for r in chain]).upper()
        kd_by_group[ag] = {}
        for row in rows:
            try:
                kd = float(row["neg_log10_KD_nM"])
            except ValueError:
                continue
            kd_by_group[ag].setdefault(row["target_variant"], []).append(kd)
    return seqs, kd_by_group


def compute_cdr_contrast_pairs(seqs: dict, kd_by_group: dict) -> list[dict]:
    """CDR-only contrast pairs (differ at 1-2 CDR positions, same length as
    WT so raw index == IMGT-aligned position, no indels)."""
    items = [(ag, seq) for ag, seq in seqs.items() if len(seq) == 125]
    n = len(items)
    ags = [ag for ag, _ in items]
    alphabet = sorted(set("".join(s for _, s in items)))
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    C = len(alphabet)
    arr_idx = np.vectorize(char_to_idx.get)(np.array([list(seq) for _, seq in items]))

    oh = np.zeros((n, 125 * C), dtype=np.float32)
    for c_idx in range(C):
        cols = np.arange(125) * C + c_idx
        oh[:, cols] = (arr_idx == c_idx).astype(np.float32)
    dist = np.rint(125 - oh @ oh.T).astype(np.int32)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    close = np.where((d == 1) | (d == 2))[0]

    pairs = []
    for k in close:
        i, j = iu[0][k], iu[1][k]
        positions = np.where(arr_idx[i] != arr_idx[j])[0]
        if not all(p in CDR_ALL for p in positions):
            continue
        ag_a, ag_b = ags[i], ags[j]
        shared_targets = set(kd_by_group.get(ag_a, {})) & set(kd_by_group.get(ag_b, {}))
        for tv in shared_targets:
            kd_a = float(np.mean(kd_by_group[ag_a][tv]))
            kd_b = float(np.mean(kd_by_group[ag_b][tv]))
            pairs.append({
                "ag_a": ag_a, "ag_b": ag_b, "n_diff": int(len(positions)),
                "positions_0idx": positions.tolist(), "target_variant": tv,
                "kd_a": kd_a, "kd_b": kd_b, "delta": kd_a - kd_b,
            })
    return pairs


if __name__ == "__main__":
    t0 = time.time()
    print("loading AlphaSeq rows...", flush=True)
    rows_by_group = load_designed_vhh72_rows(ALPHASEQ_CSV)
    print(f"  {len(rows_by_group)} unique designed VHH72 variants", flush=True)

    print("extracting sequences + KD from CIFs/CSV...", flush=True)
    seqs, kd_by_group = extract_sequences_and_kd(rows_by_group)
    print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    lengths = Counter(len(s) for s in seqs.values())
    print(f"sequence lengths: {dict(lengths)}", flush=True)

    print("computing CDR-only contrast pairs...", flush=True)
    pairs = compute_cdr_contrast_pairs(seqs, kd_by_group)
    dist1 = [p for p in pairs if p["n_diff"] == 1]
    dist2 = [p for p in pairs if p["n_diff"] == 2]
    print(f"  usable CDR-only pairs (pair x shared-target): {len(pairs)}", flush=True)
    print(f"  distance=1 (clean, single substitution): {len(dist1)}", flush=True)
    print(f"  distance=2 (two simultaneous substitutions): {len(dist2)}", flush=True)

    deltas = np.array([abs(p["delta"]) for p in pairs])
    print(f"\n|delta neg_log10_KD_nM|: mean={deltas.mean():.3f} "
          f"median={np.median(deltas):.3f}", flush=True)
    above_noise = np.mean(deltas > NOISE_FLOOR_THRESHOLD)
    print(f"  fraction above noise-floor threshold ({NOISE_FLOOR_THRESHOLD}): "
          f"{above_noise:.1%} ({int(above_noise*len(deltas))} pairs)", flush=True)

    by_cdr = Counter()
    for p in dist1:
        pos = p["positions_0idx"][0]
        name = "CDR1" if pos in CDR1 else "CDR2" if pos in CDR2 else "CDR3"
        by_cdr[name] += 1
    print(f"\ndistance=1 pairs by CDR loop: {dict(by_cdr)}", flush=True)
