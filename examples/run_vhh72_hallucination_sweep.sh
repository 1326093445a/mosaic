#!/usr/bin/env bash
# Runs the full VHH72 hallucination search pipeline end to end:
#   1. vhh72_hallucination_dispatch.py -- launches N jobs (2 search
#      policies, greedy/mcmc, x 2 AbLang2 gradient-flow settings x
#      len(SEEDS) seeds) across the given GPUs; each job internally runs
#      vhh72_hallucination_search.py once, real edit-budget-constrained
#      search, writes a Pareto-front CSV.
#   2. vhh72_hallucination_aggregate.py -- combines all result CSVs into
#      one combined CSV, a per-seed variance/win-rate summary CSV, and a
#      Pareto-front comparison figure (mean +/- std across seeds).
#
# `set -e` means step 2 does NOT run if step 1 reports any job failed
# (vhh72_hallucination_dispatch.py exits nonzero in that case specifically
# so this doesn't silently aggregate an incomplete result set).
#
# Usage:
#   examples/run_vhh72_hallucination_sweep.sh [DEVICES] [EDIT_BUDGET] [SEEDS] [OUTPUT_DIR]
#
# All arguments optional:
#   DEVICES     comma-separated physical GPU ids, e.g. 0,1,2,3 (default: 0,1,2,3)
#   EDIT_BUDGET max mutations from WT (default: 5)
#   SEEDS       comma-separated seeds, e.g. 0,1,2,3,4 (default: 0 -- single seed)
#   OUTPUT_DIR  where results land (default: results/hallucination_sweep_<timestamp>)
#
# Examples:
#   examples/run_vhh72_hallucination_sweep.sh
#   examples/run_vhh72_hallucination_sweep.sh 0,1,2,3
#   examples/run_vhh72_hallucination_sweep.sh 0,1,2,3 5 0,1,2,3,4 results/my_run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICES="${1:-0,1,2,3}"
EDIT_BUDGET="${2:-5}"
SEEDS="${3:-0}"
OUTPUT_DIR="${4:-results/hallucination_sweep_$(date +%Y%m%d_%H%M%S)}"

echo "=== VHH72 hallucination sweep ==="
echo "repo root:    $REPO_ROOT"
echo "devices:      $DEVICES"
echo "edit budget:  $EDIT_BUDGET"
echo "seeds:        $SEEDS"
echo "output dir:   $OUTPUT_DIR"
echo

mkdir -p "$OUTPUT_DIR"

echo "[1/2] dispatching jobs (2 policies x 2 stop_grad settings x seeds=$SEEDS) across devices $DEVICES..."
.venv/bin/python examples/vhh72_hallucination_dispatch.py \
    --devices "$DEVICES" \
    --edit-budget "$EDIT_BUDGET" \
    --seeds "$SEEDS" \
    --output-dir "$OUTPUT_DIR"

echo
echo "[2/2] aggregating results..."
.venv/bin/python examples/vhh72_hallucination_aggregate.py \
    --results-dir "$OUTPUT_DIR" \
    --seeds "$SEEDS" \
    --output-csv "$OUTPUT_DIR/combined.csv" \
    --output-summary "$OUTPUT_DIR/seed_variance_summary.csv" \
    --output-figure "$OUTPUT_DIR/pareto_comparison.png"

echo
echo "=== done ==="
echo "per-job CSVs + logs: $OUTPUT_DIR/{policy}_stopgrad{0,1}_seed{N}.{csv,log}"
echo "combined CSV:        $OUTPUT_DIR/combined.csv"
echo "seed variance/win-rate summary: $OUTPUT_DIR/seed_variance_summary.csv"
echo "figure:               $OUTPUT_DIR/pareto_comparison.png"
