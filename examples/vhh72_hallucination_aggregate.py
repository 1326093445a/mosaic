"""Aggregate the 4 CSVs from vhh72_hallucination_dispatch.py into one
combined CSV and a comparison figure: total_loss vs. edit_count, one line
per (policy, stop_grad) combination, across the full Pareto front each job
already records (edit_count 0..budget).

Usage:
    .venv/bin/python examples/vhh72_hallucination_aggregate.py \\
        --results-dir results/hallucination_sweep \\
        --output-csv results/hallucination_sweep/combined.csv \\
        --output-figure results/hallucination_sweep/pareto_comparison.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ["greedy", "mcmc"]
STOP_GRAD_SETTINGS = [0, 1]


def load_all(results_dir: Path):
    # Matches vhh72_hallucination_dispatch.py's exact naming convention, not
    # a bare *.csv glob -- a results directory could plausibly have other
    # CSVs in it (this bit a first test run directly, reusing a scratch dir
    # that still had an unrelated CSV from earlier work).
    csv_paths = sorted(
        results_dir / f"{policy}_stopgrad{stop_grad}.csv"
        for policy in POLICIES for stop_grad in STOP_GRAD_SETTINGS
    )
    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        raise SystemExit(
            f"no {{policy}}_stopgrad{{0,1}}.csv files found in {results_dir}"
        )
    rows = []
    for path in csv_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                row["edit_count"] = int(row["edit_count"])
                row["total_loss"] = float(row["total_loss"])
                row["stop_grad"] = int(row["stop_grad"])
                row["num_mutations_from_wt"] = int(row["num_mutations_from_wt"])
                rows.append(row)
    print(f"loaded {len(rows)} rows from {len(csv_paths)} files", flush=True)
    return rows


def write_combined_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "stop_grad", "edit_count", "total_loss", "num_mutations_from_wt", "sequence"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["policy"], r["stop_grad"], r["edit_count"])):
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"wrote combined CSV: {output_csv}", flush=True)


def plot_pareto_comparison(rows, output_figure: Path):
    groups = {}
    for row in rows:
        key = (row["policy"], row["stop_grad"])
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    for (policy, stop_grad), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda r: r["edit_count"])
        x = [r["edit_count"] for r in group_rows]
        y = [r["total_loss"] for r in group_rows]
        label = f"{policy}, stop_grad={stop_grad}"
        style = "-" if stop_grad == 1 else "--"
        marker = "o" if policy == "greedy" else "^"
        ax.plot(x, y, style, marker=marker, label=label)

    ax.set_xlabel("edit count (mutations from WT)")
    ax.set_ylabel("total loss (lower is better)")
    ax.set_title("VHH72 hallucination search: Pareto front by policy / gradient-flow setting")
    ax.legend()
    ax.grid(alpha=0.3)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=150, bbox_inches="tight")
    print(f"wrote figure: {output_figure}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-figure", type=Path, required=True)
    args = p.parse_args()

    rows = load_all(args.results_dir)
    write_combined_csv(rows, args.output_csv)
    plot_pareto_comparison(rows, args.output_figure)


if __name__ == "__main__":
    main()
