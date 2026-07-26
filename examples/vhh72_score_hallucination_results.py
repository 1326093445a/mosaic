"""Score the real, discrete designs from vhh72_hallucination_search.py's
Pareto fronts (results/<sweep>/combined.csv) against real AlphaSeq CDR
contrast pairs -- the same real ground truth Path 1 used
(examples/alphaseq_vhh72_cdr_contrast_pairs.py), but applied to actual,
committed mutations from a real discrete search, not a p_seq marginal
shift. See docs/guidance_alphaseq_testing_notes.md section 13.

Two checks, not one:
  1. Exact/near-match: does a candidate design match (or come within a
     couple CDR edits of) a real AlphaSeq-tested sequence? If so, that's a
     direct real KD reading, not an inference -- checked first because it's
     the strongest possible evidence when it hits.
  2. Per-mutation sign agreement: for every position where a candidate
     differs from WT, look up all real, clean (single-substitution)
     contrast pairs at that position involving the candidate's chosen
     amino acid (against whatever it was actually compared to in the real
     campaign, not necessarily WT -- that's a real scope limit, reported
     explicitly, not glossed over) and compute the fraction that favor it.
     Positions with zero real single-mutation coverage are reported as
     untestable, not silently skipped or scored as neutral.

Requires read access to /home/yfeng17/SBSAb/dataset/alphaseq/ (see
alphaseq_vhh72_cdr_contrast_pairs.py). No GPU needed.

Usage:
    .venv/bin/python examples/vhh72_score_hallucination_results.py \\
        --combined-csv results/hallucination_sweep_20260725_234632/combined.csv \\
        --output results/hallucination_sweep_20260725_234632/alphaseq_scoring.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from alphaseq_vhh72_cdr_contrast_pairs import (
    ALPHASEQ_CSV,
    CDR_ALL,
    NOISE_FLOOR_THRESHOLD,
    compute_cdr_contrast_pairs,
    extract_sequences_and_kd,
    load_designed_vhh72_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# WT sequence is derived from combined.csv's own edit_count=0 row at
# runtime (main(), below) -- not hardcoded here, so there's no risk of a
# transcription mismatch against whatever the actual sweep run used.


def build_position_aa_favorability(dist1_pairs, seqs):
    """index[(pos, aa)] -> list of bool, True if this pair's evidence
    favors `aa` at `pos` (i.e. the member with `aa` has the higher KD =
    tighter/better binder), restricted to pairs above the noise floor."""
    index = {}
    for p in dist1_pairs:
        if abs(p["delta"]) <= NOISE_FLOOR_THRESHOLD:
            continue
        pos = p["positions_0idx"][0]
        aa_a = seqs[p["ag_a"]][pos]
        aa_b = seqs[p["ag_b"]][pos]
        # delta = kd_a - kd_b; delta > 0 means a is the better binder
        favors_a = p["delta"] > 0
        index.setdefault((pos, aa_a), []).append(favors_a)
        index.setdefault((pos, aa_b), []).append(not favors_a)
    return index


def score_design(seq, wt_seq, favorability_index):
    """Per-mutation sign agreement for one candidate sequence."""
    results = []
    for pos in range(len(wt_seq)):
        if pos not in CDR_ALL or seq[pos] == wt_seq[pos]:
            continue
        aa = seq[pos]
        votes = favorability_index.get((pos, aa))
        if votes is None:
            results.append({"pos": pos, "wt_aa": wt_seq[pos], "mut_aa": aa,
                             "n_pairs": 0, "agreement_rate": None})
        else:
            results.append({"pos": pos, "wt_aa": wt_seq[pos], "mut_aa": aa,
                             "n_pairs": len(votes), "agreement_rate": float(np.mean(votes))})
    return results


def cdr_hamming(a, b):
    return sum(1 for i in range(len(a)) if i in CDR_ALL and a[i] != b[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    print("loading AlphaSeq VHH72 designed variants + KD...", flush=True)
    rows_by_group = load_designed_vhh72_rows(ALPHASEQ_CSV)
    seqs, kd_by_group = extract_sequences_and_kd(rows_by_group)
    pairs = compute_cdr_contrast_pairs(seqs, kd_by_group)
    dist1_pairs = [p for p in pairs if p["n_diff"] == 1]
    print(f"  {len(seqs)} real designed variants, {len(dist1_pairs)} clean "
          f"single-substitution contrast pairs", flush=True)

    favorability_index = build_position_aa_favorability(dist1_pairs, seqs)
    n_position_aa_combos = len(favorability_index)
    print(f"  real single-substitution coverage: {n_position_aa_combos} "
          f"distinct (position, amino acid) combinations", flush=True)

    designs = []
    with open(args.combined_csv) as f:
        for row in csv.DictReader(f):
            designs.append(row)
    print(f"\nloaded {len(designs)} candidate designs from {args.combined_csv}", flush=True)

    wt_rows = [d for d in designs if int(d["edit_count"]) == 0]
    if wt_rows:
        wt_seq = wt_rows[0]["sequence"]
        print(f"WT sequence confirmed from combined.csv edit_count=0 row "
              f"(len={len(wt_seq)})", flush=True)
    else:
        raise SystemExit("no edit_count=0 row found in combined.csv -- cannot confirm WT sequence")

    print("\n=== 1. Exact / near-match check against real AlphaSeq-tested sequences ===", flush=True)
    real_seqs_125 = {ag: s for ag, s in seqs.items() if len(s) == 125}
    match_rows = []
    for d in designs:
        cand_seq = d["sequence"]
        if int(d["edit_count"]) == 0:
            continue
        best_ag, best_dist = None, 999
        for ag, real_seq in real_seqs_125.items():
            dist = cdr_hamming(cand_seq, real_seq)
            if dist < best_dist:
                best_ag, best_dist = ag, dist
        if best_dist <= 2:
            print(f"  {d['policy']}/stop_grad={d['stop_grad']}/edit_count={d['edit_count']}: "
                  f"CDR-distance {best_dist} from real variant {best_ag} "
                  f"(KD data: {kd_by_group.get(best_ag)})", flush=True)
        match_rows.append({"policy": d["policy"], "stop_grad": d["stop_grad"],
                            "edit_count": d["edit_count"], "closest_real_variant": best_ag,
                            "cdr_distance_to_closest_real_variant": best_dist})
    if not any(r["cdr_distance_to_closest_real_variant"] <= 2 for r in match_rows):
        print("  no candidate design within CDR-distance 2 of any real AlphaSeq-tested "
              "sequence -- no direct KD reading available, sign-agreement scoring below "
              "is the only real-data check for these designs.", flush=True)

    print("\n=== 2. Per-mutation sign agreement ===", flush=True)
    output_rows = []
    for d in designs:
        n_edits = int(d["edit_count"])
        if n_edits == 0:
            continue
        mutation_scores = score_design(d["sequence"], wt_seq, favorability_index)
        testable = [m for m in mutation_scores if m["agreement_rate"] is not None]
        n_testable = len(testable)
        n_total = len(mutation_scores)
        mean_agreement = float(np.mean([m["agreement_rate"] for m in testable])) if testable else None

        tag = f"{d['policy']}/stop_grad={d['stop_grad']}/edit_count={n_edits}"
        agree_str = f"{mean_agreement:.2f}" if mean_agreement is not None else "n/a"
        print(f"  {tag}: {n_testable}/{n_total} mutations have real contrast-pair "
              f"coverage, mean sign-agreement={agree_str}", flush=True)
        for m in mutation_scores:
            rate_str = f"{m['agreement_rate']:.2f}" if m["agreement_rate"] is not None else ""
            print(f"      pos {m['pos']} ({m['wt_aa']}->{m['mut_aa']}): "
                  f"n_pairs={m['n_pairs']} agreement={rate_str}", flush=True)
            output_rows.append({
                "policy": d["policy"], "stop_grad": d["stop_grad"], "edit_count": n_edits,
                "total_loss": d["total_loss"], "position_0idx": m["pos"],
                "wt_aa": m["wt_aa"], "mut_aa": m["mut_aa"], "n_real_contrast_pairs": m["n_pairs"],
                "sign_agreement_rate": m["agreement_rate"] if m["agreement_rate"] is not None else "",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = ["policy", "stop_grad", "edit_count", "total_loss", "position_0idx",
                      "wt_aa", "mut_aa", "n_real_contrast_pairs", "sign_agreement_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nwrote per-mutation scoring: {args.output}", flush=True)


if __name__ == "__main__":
    main()
