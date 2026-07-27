"""Combine the per-seed CSVs from p17_hallucination_dispatch.py into one
combined CSV. Simpler than examples/vhh72_hallucination_aggregate.py --
one fixed config, only a seed axis, so no policy/stop_grad grouping or
win-rate tally is needed here.

Usage:
    .venv/bin/python examples/p17_hallucination_aggregate.py \\
        --results-dir results/p17_sweep --num-seeds 250 \\
        --output-csv results/p17_sweep/combined.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--num-seeds", type=int, default=250)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    seeds = range(args.seed_offset, args.seed_offset + args.num_seeds)
    csv_paths = [
        args.results_dir / f"p17_mcmc_stopgrad0_seed{seed}.csv" for seed in seeds
    ]
    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        raise SystemExit(f"no p17_mcmc_stopgrad0_seed{{N}}.csv files found in {args.results_dir}")

    rows = []
    fieldnames = None
    for path in csv_paths:
        with open(path) as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                row["edit_count"] = int(row["edit_count"])
                row["total_loss"] = float(row["total_loss"])
                row["seed"] = int(row["seed"])
                row["num_mutations_from_wt"] = int(row["num_mutations_from_wt"])
                rows.append(row)

    print(f"loaded {len(rows)} rows from {len(csv_paths)}/{args.num_seeds} expected seed files", flush=True)
    missing = args.num_seeds - len(csv_paths)
    if missing:
        print(f"WARNING: {missing} seed file(s) missing -- check dispatch log for failures", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["seed"], r["edit_count"])):
            writer.writerow(row)
    print(f"wrote combined CSV: {args.output_csv}", flush=True)

    # Quick summary: best (lowest) total_loss per seed at the max edit_count.
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    max_edit_count = max(r["edit_count"] for r in rows)
    losses_at_max = [
        min(r["total_loss"] for r in seed_rows if r["edit_count"] == max_edit_count)
        for seed_rows in by_seed.values()
        if any(r["edit_count"] == max_edit_count for r in seed_rows)
    ]
    if losses_at_max:
        print(f"\nedit_count={max_edit_count}: n={len(losses_at_max)} seeds, "
              f"mean={sum(losses_at_max)/len(losses_at_max):.4f}, "
              f"min={min(losses_at_max):.4f}, max={max(losses_at_max):.4f}", flush=True)


if __name__ == "__main__":
    main()
