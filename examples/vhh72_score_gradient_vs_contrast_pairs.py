"""Path 1, final step: score each gradient config's p_seq shift against the
real AlphaSeq CDR contrast-pair table.

For every clean (single-substitution) CDR contrast pair A-vs-B with a real
KD difference above the noise floor, check whether a given config's
guidance-induced shift in p_seq moved probability mass toward the
empirically better residue or the empirically worse one. Aggregated over
~9,000 independent real pairs, this gives each config (raw sum / production
controller / controller + NOS-style consistency term -- see
vhh72_gradient_path_comparison.py) a real, checkable sign-agreement rate
against real binding data -- not a proxy metric, not a vibe.

Sign convention: contrast pairs record delta = kd_a - kd_b, where kd =
neg_log10_KD_nM (higher is a tighter/better binder). delta > 0 means variant
A's residue at the differing position is empirically better than variant
B's. A config's shift agrees with that pair when
sign(shift[pos, aa_b] - shift[pos, aa_a]) == sign(kd_b - kd_a) == sign(-delta)
-- i.e. probability moves toward whichever residue the real data says is
better.

Requires: examples/vhh72_gradient_path_comparison.py has been run and wrote
vhh72_gradient_path_comparison_cache.pkl (repo root). Pure numpy + csv +
gemmi otherwise -- no GPU/jax needed for this script itself.

Usage:
    .venv/bin/python examples/vhh72_score_gradient_vs_contrast_pairs.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alphaseq_vhh72_cdr_contrast_pairs import (  # noqa: E402
    ALPHASEQ_CSV,
    NOISE_FLOOR_THRESHOLD,
    compute_cdr_contrast_pairs,
    extract_sequences_and_kd,
    load_designed_vhh72_rows,
)

TOKENS = "ARNDCQEGHILKMFPSTWYV"  # mosaic.common.TOKENS -- p_seq's 20-column order
AA_TO_IDX = {c: i for i, c in enumerate(TOKENS)}

CACHE_PATH = REPO_ROOT / "vhh72_gradient_path_comparison_cache.pkl"


def score_config(shift, pairs, seqs, pos_to_token_idx, *, weighted=False):
    """Fraction of clean contrast pairs where `shift`'s implied preference at
    the differing position agrees in sign with the real KD-based preference.
    `weighted=True` weights each pair's contribution by |delta| (effect
    size) instead of counting every pair equally."""
    agree_weight = 0.0
    total_weight = 0.0
    n_pairs = 0
    n_missing_token = 0
    for p in pairs:
        if p["n_diff"] != 1:
            continue
        delta = p["delta"]
        if abs(delta) <= NOISE_FLOOR_THRESHOLD:
            continue
        pos = p["positions_0idx"][0]
        token_idx = pos_to_token_idx.get(pos)
        if token_idx is None:
            n_missing_token += 1
            continue
        aa_a = seqs[p["ag_a"]][pos]
        aa_b = seqs[p["ag_b"]][pos]
        if aa_a not in AA_TO_IDX or aa_b not in AA_TO_IDX:
            continue
        shift_diff = shift[token_idx, AA_TO_IDX[aa_b]] - shift[token_idx, AA_TO_IDX[aa_a]]
        real_favors_b = -delta  # positive if B (kd_b > kd_a) is the better residue
        agree = np.sign(shift_diff) == np.sign(real_favors_b)
        w = abs(delta) if weighted else 1.0
        if shift_diff != 0:  # exact-zero shifts are ties, excluded from both num/denom
            agree_weight += w if agree else 0.0
            total_weight += w
            n_pairs += 1
    if n_missing_token:
        print(f"    ({n_missing_token} pairs skipped: position not in this crop's CDR set)", flush=True)
    rate = agree_weight / total_weight if total_weight > 0 else None
    return rate, n_pairs


if __name__ == "__main__":
    print("=== Scoring gradient configs against real AlphaSeq CDR contrast pairs ===", flush=True)

    if not CACHE_PATH.exists():
        raise SystemExit(
            f"missing {CACHE_PATH} -- run examples/vhh72_gradient_path_comparison.py first"
        )
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    binder_token_indices = cache["binder_token_indices"]
    cdr_residue_indices_1idx = cache["cdr_residue_indices"]  # 1-indexed
    pos_to_token_idx = {
        (i - 1): int(binder_token_indices[i - 1]) for i in cdr_residue_indices_1idx
    }
    print(f"CDR positions with a token mapping: {len(pos_to_token_idx)}", flush=True)

    print("\nloading AlphaSeq VHH72 designed variants + KD...", flush=True)
    rows_by_group = load_designed_vhh72_rows(ALPHASEQ_CSV)
    seqs, kd_by_group = extract_sequences_and_kd(rows_by_group)
    pairs = compute_cdr_contrast_pairs(seqs, kd_by_group)
    dist1_pairs = [p for p in pairs if p["n_diff"] == 1]
    print(f"clean (distance=1) CDR contrast pairs: {len(dist1_pairs)}", flush=True)

    print("\n--- sign-agreement rate: does the shift favor the empirically better residue? ---", flush=True)
    for name, key in [("raw", "shift_raw"), ("controller", "shift_full"), ("controller+consistency", "shift_consistent")]:
        shift = cache[key]
        rate, n = score_config(shift, dist1_pairs, seqs, pos_to_token_idx, weighted=False)
        rate_w, n_w = score_config(shift, dist1_pairs, seqs, pos_to_token_idx, weighted=True)
        print(f"  {name:24s}: unweighted={rate:.4f} (n={n})  effect-size-weighted={rate_w:.4f} (n={n_w})"
              if rate is not None else f"  {name}: no usable pairs", flush=True)

    print(
        "\n0.5 = chance (guidance's preference is uncorrelated with real affinity). "
        ">0.5 = guidance tends to move toward empirically better residues; "
        "<0.5 = it tends to move toward empirically worse ones. This is from a "
        "SINGLE gradient evaluation (one seed, one noise level) scored against "
        "~thousands of independent real pairs -- real statistical power on the "
        "data side, but still one point on the seed/noise-level axis (see "
        "docs/guidance_alphaseq_testing_notes.md section 9).",
        flush=True,
    )
