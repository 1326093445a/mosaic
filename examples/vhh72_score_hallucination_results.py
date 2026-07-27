"""Score the real, discrete designs from vhh72_hallucination_search.py's
Pareto fronts (results/<sweep>/combined.csv) against real AlphaSeq CDR
contrast pairs -- the same real ground truth Path 1 used
(examples/alphaseq_vhh72_cdr_contrast_pairs.py), but applied to actual,
committed mutations from a real discrete search, not a p_seq marginal
shift. See docs/guidance_alphaseq_testing_notes.md section 13.

Two checks, not one:
  1. Exact/near-match: does a candidate design exactly match, or come within a
     couple CDR edits of, a real AlphaSeq-tested sequence? An exact match is a
     direct real KD reading. A near match is only neighbor/context evidence and
     must not be interpreted as a direct measurement of the generated CDR mutant.
     This is checked first because exact hits would be the strongest possible
     evidence when they occur.
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
from scipy import stats

from alphaseq_vhh72_cdr_contrast_pairs import (
    ALPHASEQ_CSV,
    CDR_ALL,
    NOISE_FLOOR_THRESHOLD,
    compute_cdr_contrast_pairs,
    extract_sequences_and_kd,
    load_designed_vhh72_rows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VHH72_WT_BINDER_SEQ = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS"
)
# Prefer combined.csv's own edit_count=0 row when present. Partial
# architecture-test sweeps that start the discrete search from APGM sample/topk
# seeds may not contain an edit_count=0 row, so fall back to the fixed VHH72 WT
# binder sequence used by vhh72_hallucination_search.py.


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


def collect_config_mutations(designs, wt_seq, favorability_index):
    """Unique (position, mutant amino acid) with real coverage chosen by each
    (policy, stop_grad) config, deduplicated across seeds/edit_counts within
    that config. Necessary before any significance test: the same mutation
    reappearing across a Pareto front's edit counts (it's cumulative -- once
    picked, it usually stays) or across seeds is the same real-world claim
    about that substitution, not independent evidence, and must be counted
    once, not once per design it happens to appear in."""
    by_config = {}
    for d in designs:
        if int(d["edit_count"]) == 0:
            continue
        key = (d["policy"], d["stop_grad"])
        for pos in range(len(wt_seq)):
            if pos not in CDR_ALL or d["sequence"][pos] == wt_seq[pos]:
                continue
            mut_aa = d["sequence"][pos]
            if (pos, mut_aa) in favorability_index:
                by_config.setdefault(key, set()).add((pos, mut_aa))
    return by_config


def binomial_significance_report(by_config, favorability_index):
    """One-sided binomial test (H1: p > 0.5) on the POOLED real votes behind
    each config's unique chosen mutations: is what a config actually picked
    biased toward real favorable outcomes, or consistent with chance given
    how little coverage we have? Each real (position, amino acid) claim is
    counted once (via collect_config_mutations's dedup), not once per
    design/seed/edit_count it appears in -- pooling the same real evidence
    multiple times would make the test overconfident."""
    print("\n=== 3. Statistical significance: is a config's mutation choice "
          "biased toward real favorable outcomes, or is this noise? ===", flush=True)
    print("(one-sided binomial test, H1: p > 0.5, on pooled real contrast-pair "
          "votes behind each config's UNIQUE chosen mutations)", flush=True)

    rows = []
    all_unique = set()
    for key, muts in sorted(by_config.items()):
        votes = [v for pos_aa in muts for v in favorability_index[pos_aa]]
        n = len(votes)
        successes = sum(votes)
        if n > 0:
            p_value = stats.binomtest(successes, n, p=0.5, alternative="greater").pvalue
            print(f"  {key[0]}, stop_grad={key[1]}: {len(muts)} unique mutations, "
                  f"{n} pooled real votes, {successes}/{n} favorable ({successes/n:.1%}), "
                  f"p={p_value:.3f}", flush=True)
        else:
            p_value = float("nan")
            print(f"  {key[0]}, stop_grad={key[1]}: no real coverage", flush=True)
        rows.append({"policy": key[0], "stop_grad": key[1], "n_unique_mutations": len(muts),
                      "n_pooled_votes": n, "n_favorable": successes, "p_value": p_value})
        all_unique |= muts

    votes = [v for pos_aa in all_unique for v in favorability_index[pos_aa]]
    n = len(votes)
    successes = sum(votes)
    if n > 0:
        p_value = stats.binomtest(successes, n, p=0.5, alternative="greater").pvalue
        print(f"\n  OVERALL (all configs, deduplicated across the whole sweep): "
              f"{len(all_unique)} unique mutations, {n} pooled real votes, "
              f"{successes}/{n} favorable ({successes/n:.1%}), p={p_value:.3f}", flush=True)
        rows.append({"policy": "ALL", "stop_grad": "", "n_unique_mutations": len(all_unique),
                      "n_pooled_votes": n, "n_favorable": successes, "p_value": p_value})
    return rows


def rank_unique_mutations(designs, wt_seq, favorability_index):
    """Every unique (position, mutant amino acid) with real coverage that
    appears anywhere in the sweep, ranked by real agreement_rate (n_real_pairs
    as tiebreak) -- the concrete, inspectable list of which specific
    substitutions are well-evidenced as real positives or real negatives,
    independent of which config/seed/edit_count happened to choose them."""
    chosen_by = {}
    for d in designs:
        if int(d["edit_count"]) == 0:
            continue
        for pos in range(len(wt_seq)):
            if pos not in CDR_ALL or d["sequence"][pos] == wt_seq[pos]:
                continue
            mut_aa = d["sequence"][pos]
            if (pos, mut_aa) not in favorability_index:
                continue
            tag = f"{d['policy']}/sg{d['stop_grad']}"
            chosen_by.setdefault((pos, mut_aa), set()).add(tag)

    rows = []
    for (pos, mut_aa), configs in chosen_by.items():
        votes = favorability_index[(pos, mut_aa)]
        rows.append({
            "position_0idx": pos, "wt_aa": wt_seq[pos], "mut_aa": mut_aa,
            "n_real_pairs": len(votes), "agreement_rate": float(np.mean(votes)),
            "chosen_by_configs": ";".join(sorted(configs)),
        })
    rows.sort(key=lambda r: (-r["agreement_rate"], -r["n_real_pairs"]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wt-seq", type=str, default=None,
                   help="Optional WT binder sequence. If omitted, use combined.csv's "
                        "edit_count=0 row when present, else the built-in VHH72 WT "
                        "sequence used by the hallucination search.")
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
    if args.wt_seq is not None:
        wt_seq = args.wt_seq.strip().upper()
        print(f"WT sequence provided by --wt-seq (len={len(wt_seq)})", flush=True)
    elif wt_rows:
        wt_seq = wt_rows[0]["sequence"]
        print(f"WT sequence confirmed from combined.csv edit_count=0 row "
              f"(len={len(wt_seq)})", flush=True)
    else:
        wt_seq = VHH72_WT_BINDER_SEQ
        print("no edit_count=0 row found in combined.csv; using built-in VHH72 WT "
              f"binder sequence fallback (len={len(wt_seq)})", flush=True)
    bad_lengths = sorted({len(d["sequence"]) for d in designs if len(d["sequence"]) != len(wt_seq)})
    if bad_lengths:
        raise SystemExit(f"candidate sequence length(s) {bad_lengths} do not match WT length {len(wt_seq)}")

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
        seed = d.get("seed", "")
        seed_part = f"/seed={seed}" if seed != "" else ""
        if best_dist <= 2:
            print(f"  {d['policy']}/stop_grad={d['stop_grad']}{seed_part}/edit_count={d['edit_count']}: "
                  f"CDR-distance {best_dist} from real variant {best_ag} "
                  f"(KD data: {kd_by_group.get(best_ag)})", flush=True)
        match_rows.append({"policy": d["policy"], "stop_grad": d["stop_grad"],
                            "seed": seed, "edit_count": d["edit_count"],
                            "closest_real_variant": best_ag,
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

        seed = d.get("seed", "")
        seed_part = f"/seed={seed}" if seed != "" else ""
        tag = f"{d['policy']}/stop_grad={d['stop_grad']}{seed_part}/edit_count={n_edits}"
        agree_str = f"{mean_agreement:.2f}" if mean_agreement is not None else "n/a"
        print(f"  {tag}: {n_testable}/{n_total} mutations have real contrast-pair "
              f"coverage, mean sign-agreement={agree_str}", flush=True)
        for m in mutation_scores:
            rate_str = f"{m['agreement_rate']:.2f}" if m["agreement_rate"] is not None else ""
            print(f"      pos {m['pos']} ({m['wt_aa']}->{m['mut_aa']}): "
                  f"n_pairs={m['n_pairs']} agreement={rate_str}", flush=True)
            output_rows.append({
                "policy": d["policy"], "stop_grad": d["stop_grad"],
                "seed": seed, "edit_count": n_edits,
                "total_loss": d["total_loss"], "position_0idx": m["pos"],
                "wt_aa": m["wt_aa"], "mut_aa": m["mut_aa"], "n_real_contrast_pairs": m["n_pairs"],
                "sign_agreement_rate": m["agreement_rate"] if m["agreement_rate"] is not None else "",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = ["policy", "stop_grad", "seed", "edit_count", "total_loss", "position_0idx",
                      "wt_aa", "mut_aa", "n_real_contrast_pairs", "sign_agreement_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nwrote per-mutation scoring: {args.output}", flush=True)

    by_config = collect_config_mutations(designs, wt_seq, favorability_index)
    significance_rows = binomial_significance_report(by_config, favorability_index)
    significance_path = args.output.parent / "significance_report.csv"
    with open(significance_path, "w", newline="") as f:
        fieldnames = ["policy", "stop_grad", "n_unique_mutations", "n_pooled_votes",
                      "n_favorable", "p_value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(significance_rows)
    print(f"wrote significance report: {significance_path}", flush=True)

    ranking_rows = rank_unique_mutations(designs, wt_seq, favorability_index)
    ranking_path = args.output.parent / "mutation_ranking.csv"
    with open(ranking_path, "w", newline="") as f:
        fieldnames = ["position_0idx", "wt_aa", "mut_aa", "n_real_pairs",
                      "agreement_rate", "chosen_by_configs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranking_rows)
    print(f"wrote full mutation ranking ({len(ranking_rows)} unique substitutions): {ranking_path}", flush=True)

    print("\n=== 4. Top and bottom real-evidenced substitutions (all configs) ===", flush=True)
    print("  highest agreement-rate positives (inspect n_pairs before treating as strong):", flush=True)
    for r in ranking_rows[:5]:
        print(f"    pos {r['position_0idx']} ({r['wt_aa']}->{r['mut_aa']}): "
              f"n_pairs={r['n_real_pairs']} agreement={r['agreement_rate']:.2f} "
              f"chosen_by={r['chosen_by_configs']}", flush=True)
    print("  lowest agreement-rate negatives (inspect n_pairs before treating as strong):", flush=True)
    for r in ranking_rows[-5:][::-1]:
        print(f"    pos {r['position_0idx']} ({r['wt_aa']}->{r['mut_aa']}): "
              f"n_pairs={r['n_real_pairs']} agreement={r['agreement_rate']:.2f} "
              f"chosen_by={r['chosen_by_configs']}", flush=True)


if __name__ == "__main__":
    main()
