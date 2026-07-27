"""Curate a candidate shortlist from a hallucination sweep's Pareto fronts,
using real per-mutation AlphaSeq evidence (mutation_ranking.csv, from
vhh72_score_hallucination_results.py) as a red/green filter -- not as a
generation-time bias. See docs/guidance_alphaseq_testing_notes.md section
13.9's decision (mcmc, stop_grad=0) and the follow-up discussion: this is
explicitly the "candidate track," kept separate from the "methodology
track" (testing whether APGM-seeded MCMC adds value) so the two questions
("what do we have now" vs. "does the search architecture work") don't get
tangled. Using real AlphaSeq data to bias the search itself would leak the
evaluation signal into generation -- fine for engineering final candidates
(this script), not clean for judging the search mechanism.

Per-mutation classification, using each unique (position, mutant amino
acid)'s real agreement_rate from mutation_ranking.csv:
  - "supported":  agreement_rate >= --support-threshold (default 0.6)
                  AND n_real_pairs >= --min-support-pairs (default 3)
  - "avoid":      agreement_rate <= --avoid-threshold (default 0.2),
                  n_real_pairs >= 1 (a real, evidenced negative)
  - "neutral":    real coverage exists but is ambiguous (between the two
                  thresholds)
  - "untested":   no real AlphaSeq coverage at this (position, amino acid)
                  at all -- most of the space (~61% by the last count),
                  neither penalized nor rewarded here; this is exactly the
                  part of the space post-hoc AlphaSeq scoring can't help
                  with, and where the model's own signal is all we have.

By default, candidates with ANY "avoid" mutation are excluded (real,
evidenced negative -- no reason to carry it forward when the edit budget
allows dropping it); remaining candidates are ranked by (n_supported
descending, total_loss ascending). Everything is inspectable in the full
output CSV, so nothing here is a hidden cut.

Usage:
    .venv/bin/python examples/vhh72_select_candidates.py \\
        --combined-csv results/hallucination_sweep_discrete_3seed/combined.csv \\
        --mutation-ranking-csv results/hallucination_sweep_discrete_3seed/mutation_ranking.csv \\
        --policy mcmc --stop-grad 0 --min-edit-count 3 --max-edit-count 5 \\
        --output results/hallucination_sweep_discrete_3seed/candidate_shortlist.csv
"""
import argparse
import csv
from pathlib import Path


VHH72_WT_BINDER_SEQ = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS"
)


def load_ranking(path: Path):
    index = {}
    for row in csv.DictReader(open(path)):
        key = (int(row["position_0idx"]), row["mut_aa"])
        index[key] = {"n_real_pairs": int(row["n_real_pairs"]),
                      "agreement_rate": float(row["agreement_rate"])}
    return index


def classify_mutation(pos, mut_aa, ranking, support_threshold, min_support_pairs, avoid_threshold):
    info = ranking.get((pos, mut_aa))
    if info is None:
        return "untested", None, 0
    rate = info["agreement_rate"]
    n = info["n_real_pairs"]
    if rate >= support_threshold and n >= min_support_pairs:
        return "supported", rate, n
    if rate <= avoid_threshold and n >= 1:
        return "avoid", rate, n
    return "neutral", rate, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=Path, required=True)
    p.add_argument("--mutation-ranking-csv", type=Path, required=True)
    p.add_argument("--policy", type=str, default="mcmc")
    p.add_argument("--stop-grad", type=int, default=0)
    p.add_argument("--min-edit-count", type=int, default=3)
    p.add_argument("--max-edit-count", type=int, default=5)
    p.add_argument("--support-threshold", type=float, default=0.6)
    p.add_argument("--min-support-pairs", type=int, default=3,
                    help="Minimum AlphaSeq contrast-pair count required before "
                         "a high agreement_rate is labelled supported. Prevents "
                         "one-pair 100%% hits from being treated as strong support.")
    p.add_argument("--avoid-threshold", type=float, default=0.2)
    p.add_argument("--keep-avoid", action="store_true",
                    help="Don't exclude candidates with an evidenced-negative "
                         "mutation -- annotate them instead of dropping them.")
    p.add_argument("--wt-seq", type=str, default=None,
                    help="Optional WT binder sequence. If omitted, use combined.csv's "
                         "edit_count=0 row when present, else the built-in VHH72 WT "
                         "sequence used by the hallucination search.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    designs = list(csv.DictReader(open(args.combined_csv)))
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

    ranking = load_ranking(args.mutation_ranking_csv)

    candidates = [
        d for d in designs
        if d["policy"] == args.policy and int(d["stop_grad"]) == args.stop_grad
        and args.min_edit_count <= int(d["edit_count"]) <= args.max_edit_count
    ]
    print(f"loaded {len(candidates)} candidates matching policy={args.policy} "
          f"stop_grad={args.stop_grad} edit_count in [{args.min_edit_count}, "
          f"{args.max_edit_count}]", flush=True)

    rows = []
    for d in candidates:
        seq = d["sequence"]
        mutations = [(pos, wt_seq[pos], seq[pos]) for pos in range(len(wt_seq)) if seq[pos] != wt_seq[pos]]
        annotated = []
        tiers = {"supported": 0, "avoid": 0, "neutral": 0, "untested": 0}
        for pos, wt_aa, mut_aa in mutations:
            tier, rate, n = classify_mutation(
                pos, mut_aa, ranking, args.support_threshold,
                args.min_support_pairs, args.avoid_threshold,
            )
            tiers[tier] += 1
            rate_str = f"{rate:.2f}" if rate is not None else "n/a"
            annotated.append({
                "pos0": f"{wt_aa}{pos}{mut_aa}[{tier}:{rate_str}/n={n}]",
                "pos1": f"{wt_aa}{pos + 1}{mut_aa}[{tier}:{rate_str}/n={n}]",
            })

        rows.append({
            "policy": d["policy"], "stop_grad": d["stop_grad"], "seed": d.get("seed", ""),
            "edit_count": int(d["edit_count"]), "total_loss": float(d["total_loss"]),
            "sequence": seq, "n_supported": tiers["supported"], "n_avoid": tiers["avoid"],
            "n_neutral": tiers["neutral"], "n_untested": tiers["untested"],
            "mutations_0idx": ";".join(a["pos0"] for a in annotated),
            "mutations_1idx": ";".join(a["pos1"] for a in annotated),
        })

    excluded = [r for r in rows if r["n_avoid"] > 0]
    kept = rows if args.keep_avoid else [r for r in rows if r["n_avoid"] == 0]
    kept.sort(key=lambda r: (-r["n_supported"], r["total_loss"]))

    print(f"\n{len(excluded)}/{len(rows)} candidates have at least one evidenced-negative "
          f"mutation (agreement <= {args.avoid_threshold}, n_real_pairs >= 1)"
          + (" -- kept anyway (--keep-avoid)" if args.keep_avoid else " -- excluded by default"), flush=True)
    print(f"{len(kept)} candidates in the final shortlist, ranked by "
          f"(n_supported desc, total_loss asc)\n", flush=True)

    for r in kept[:10]:
        print(f"  seed={r['seed']} edit_count={r['edit_count']} loss={r['total_loss']:.4f} "
              f"supported={r['n_supported']} avoid={r['n_avoid']} neutral={r['n_neutral']} "
              f"untested={r['n_untested']}", flush=True)
        print(f"    0idx: {r['mutations_0idx']}", flush=True)
        print(f"    1idx: {r['mutations_1idx']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        fieldnames = ["policy", "stop_grad", "seed", "edit_count", "total_loss", "sequence",
                      "n_supported", "n_avoid", "n_neutral", "n_untested",
                      "mutations_0idx", "mutations_1idx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nwrote shortlist ({len(kept)} candidates): {args.output}", flush=True)


if __name__ == "__main__":
    main()
