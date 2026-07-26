"""Aggregate the CSVs from vhh72_hallucination_dispatch.py into one combined
CSV, a per-seed variance/win-rate summary, and a comparison figure showing
mean +/- std across seeds: total_loss vs. edit_count, one line per
(policy, stop_grad) combination, across the full Pareto front each job
already records (edit_count 0..budget).

Usage:
    .venv/bin/python examples/vhh72_hallucination_aggregate.py \\
        --results-dir results/hallucination_sweep --seeds 0,1,2,3,4 \\
        --output-csv results/hallucination_sweep/combined.csv \\
        --output-summary results/hallucination_sweep/seed_variance_summary.csv \\
        --output-figure results/hallucination_sweep/pareto_comparison.png

    # Partial architecture-test sweep, matching dispatch --policies/--stop-grads:
    .venv/bin/python examples/vhh72_hallucination_aggregate.py \\
        --results-dir results/hallucination_sweep_apgm_sample_mcmc_sg0 \\
        --seeds 0,1,2 --policies mcmc --stop-grads 0 \\
        --output-csv results/hallucination_sweep_apgm_sample_mcmc_sg0/combined.csv \\
        --output-summary results/hallucination_sweep_apgm_sample_mcmc_sg0/seed_variance_summary.csv \\
        --output-figure results/hallucination_sweep_apgm_sample_mcmc_sg0/pareto_comparison.png
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ["greedy", "mcmc"]
STOP_GRAD_SETTINGS = [0, 1]


def _parse_csv_strs(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one comma-separated value")
    return out


def _parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in _parse_csv_strs(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def load_all(results_dir: Path, seeds: list[int],
             policies: list[str], stop_grad_settings: list[int]):
    # Matches vhh72_hallucination_dispatch.py's exact naming convention, not
    # a bare *.csv glob -- a results directory could plausibly have other
    # CSVs in it (this bit a first test run directly, reusing a scratch dir
    # that still had an unrelated CSV from earlier work).
    csv_paths = []
    for policy in policies:
        for stop_grad in stop_grad_settings:
            for seed in seeds:
                path = results_dir / f"{policy}_stopgrad{stop_grad}_seed{seed}.csv"
                if not path.exists() and seeds == [0]:
                    path = results_dir / f"{policy}_stopgrad{stop_grad}.csv"
                if path.exists():
                    csv_paths.append((path, seed))
    if not csv_paths:
        raise SystemExit(
            f"no requested policy/stop_grad/seed CSV files found in {results_dir}"
        )
    rows = []
    for path, fallback_seed in sorted(csv_paths):
        with open(path) as f:
            for row in csv.DictReader(f):
                row["edit_count"] = int(row["edit_count"])
                row["total_loss"] = float(row["total_loss"])
                row["stop_grad"] = int(row["stop_grad"])
                row["num_mutations_from_wt"] = int(row["num_mutations_from_wt"])
                row["seed"] = int(row.get("seed", fallback_seed))
                rows.append(row)
    print(f"loaded {len(rows)} rows from {len(csv_paths)} files "
          f"({len(csv_paths)} of {len(policies) * len(stop_grad_settings) * len(seeds)} expected)", flush=True)
    return rows


def write_combined_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "stop_grad", "seed", "edit_count", "total_loss",
                  "num_mutations_from_wt", "sequence"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["policy"], r["stop_grad"], r["seed"], r["edit_count"])):
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"wrote combined CSV: {output_csv}", flush=True)


def write_seed_variance_summary(rows, output_summary: Path,
                                policies: list[str],
                                stop_grad_settings: list[int]):
    """Two things a single combined CSV can't answer directly: (1) per
    (policy, stop_grad, edit_count), how much does total_loss actually vary
    across seeds (mean/std/min/max) -- a config that "wins" by a margin
    smaller than its own seed-to-seed noise isn't a real win; (2) per seed,
    which config had the lowest total_loss at the max edit_count (the
    Pareto front's most-mutated point, where "best overall" comparisons in
    docs/guidance_alphaseq_testing_notes.md section 13.1 were made) --
    tallied into a win-rate per config, the direct answer to "did
    mcmc,stop_grad=0's win replicate, or was it a one-off."
    """
    groups = {}
    for row in rows:
        key = (row["policy"], row["stop_grad"], row["edit_count"])
        groups.setdefault(key, []).append(row["total_loss"])

    summary_rows = []
    for (policy, stop_grad, edit_count), losses in sorted(groups.items()):
        summary_rows.append({
            "policy": policy, "stop_grad": stop_grad, "edit_count": edit_count,
            "n_seeds": len(losses), "mean_total_loss": statistics.fmean(losses),
            "std_total_loss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
            "min_total_loss": min(losses), "max_total_loss": max(losses),
        })

    max_edit_count = max(r["edit_count"] for r in rows)
    by_seed_at_max = {}
    for row in rows:
        if row["edit_count"] != max_edit_count:
            continue
        by_seed_at_max.setdefault(row["seed"], []).append(row)

    expected_configs = {(policy, stop_grad) for policy in policies for stop_grad in stop_grad_settings}
    win_counts = {cfg: 0 for cfg in expected_configs}
    n_seeds_compared = 0
    for seed, seed_rows in sorted(by_seed_at_max.items()):
        configs_present = {(r["policy"], r["stop_grad"]) for r in seed_rows}
        if configs_present != expected_configs:
            print(f"  seed {seed}: skipping win-tally, missing configs "
                  f"({expected_configs - configs_present})", flush=True)
            continue
        best = min(seed_rows, key=lambda r: r["total_loss"])
        win_counts[(best["policy"], best["stop_grad"])] += 1
        n_seeds_compared += 1

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", newline="") as f:
        fieldnames = ["policy", "stop_grad", "edit_count", "n_seeds",
                      "mean_total_loss", "std_total_loss", "min_total_loss", "max_total_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote seed variance summary: {output_summary}", flush=True)

    print(f"\nwin-rate at edit_count={max_edit_count} (lowest total_loss per seed, "
          f"{n_seeds_compared} seeds with all requested configs present):", flush=True)
    for (policy, stop_grad), count in sorted(win_counts.items(), key=lambda kv: -kv[1]):
        rate = count / n_seeds_compared if n_seeds_compared else 0.0
        print(f"  {policy}, stop_grad={stop_grad}: {count}/{n_seeds_compared} ({rate:.0%})", flush=True)


def plot_pareto_comparison(rows, output_figure: Path):
    groups = {}
    for row in rows:
        key = (row["policy"], row["stop_grad"])
        groups.setdefault(key, {}).setdefault(row["edit_count"], []).append(row["total_loss"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for (policy, stop_grad), by_edit_count in sorted(groups.items()):
        edit_counts = sorted(by_edit_count.keys())
        means = [statistics.fmean(by_edit_count[e]) for e in edit_counts]
        stds = [statistics.pstdev(by_edit_count[e]) if len(by_edit_count[e]) > 1 else 0.0
                for e in edit_counts]
        label = f"{policy}, stop_grad={stop_grad} (n={len(by_edit_count[edit_counts[0]])})"
        style = "-" if stop_grad == 1 else "--"
        marker = "o" if policy == "greedy" else "^"
        ax.errorbar(edit_counts, means, yerr=stds, fmt=style + marker, label=label, capsize=3)

    ax.set_xlabel("edit count (mutations from WT)")
    ax.set_ylabel("total loss (lower is better, mean +/- std across seeds)")
    ax.set_title("VHH72 hallucination search: Pareto front by policy / gradient-flow setting")
    ax.legend()
    ax.grid(alpha=0.3)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=150, bbox_inches="tight")
    print(f"wrote figure: {output_figure}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--seeds", type=str, default="0",
                    help="Comma-separated seeds to look for, must match the "
                         "--seeds passed to vhh72_hallucination_dispatch.py.")
    p.add_argument("--policies", type=str, default=",".join(POLICIES),
                    help="Comma-separated policies to aggregate. Default: greedy,mcmc.")
    p.add_argument("--stop-grads", type=str, default=",".join(map(str, STOP_GRAD_SETTINGS)),
                    help="Comma-separated stop_grad settings to aggregate. Default: 0,1.")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, required=True)
    p.add_argument("--output-figure", type=Path, required=True)
    args = p.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    policies = _parse_csv_strs(args.policies)
    stop_grad_settings = _parse_csv_ints(args.stop_grads)
    unknown_policies = sorted(set(policies) - set(POLICIES))
    unknown_stop_grads = sorted(set(stop_grad_settings) - set(STOP_GRAD_SETTINGS))
    if unknown_policies:
        raise SystemExit(f"unknown policies: {unknown_policies}; allowed: {POLICIES}")
    if unknown_stop_grads:
        raise SystemExit(f"unknown stop_grad settings: {unknown_stop_grads}; allowed: {STOP_GRAD_SETTINGS}")

    rows = load_all(args.results_dir, seeds, policies, stop_grad_settings)
    write_combined_csv(rows, args.output_csv)
    write_seed_variance_summary(rows, args.output_summary, policies, stop_grad_settings)
    plot_pareto_comparison(rows, args.output_figure)


if __name__ == "__main__":
    main()
