#!/usr/bin/env bash
# Runs the full P17 hallucination search pipeline end to end:
#   1. p17_hallucination_dispatch.py -- launches N_SEEDS jobs (one fixed,
#      validated config: mcmc, stop_grad=0, argmax/WT-seeded, distogram
#      in-loop, real hotspot restriction -- see
#      examples/p17_hallucination_search.py's own defaults) across the
#      given GPUs; each job writes a Pareto-front CSV.
#   2. p17_hallucination_aggregate.py -- combines all per-seed CSVs into
#      one combined CSV.
#
# `set -e` means step 2 does NOT run if step 1 reports any job failed
# (p17_hallucination_dispatch.py exits nonzero in that case specifically
# so this doesn't silently aggregate an incomplete result set).
#
# Usage:
#   examples/run_p17_hallucination_sweep.sh [DEVICES] [OUTPUT_DIR] [NUM_SEEDS] [EDIT_BUDGET]
#
# All arguments optional:
#   DEVICES     comma-separated physical GPU ids, e.g. 0,1,2,3 (default: 0,1,2,3)
#   OUTPUT_DIR  where results land (default: results/p17_sweep_<timestamp>)
#   NUM_SEEDS   number of seeds to run, 0..NUM_SEEDS-1 (default: 250)
#   EDIT_BUDGET max mutations from WT (default: 5)
#
# Examples:
#   examples/run_p17_hallucination_sweep.sh
#   examples/run_p17_hallucination_sweep.sh 0,1,2,3 results/p17_sweep_250 250
#   examples/run_p17_hallucination_sweep.sh 0,1,2,3 results/p17_smoke 4
#
# NOTE: p17_hallucination_search.py has not yet been run for real on the
# cluster. Strongly recommend running a small NUM_SEEDS (e.g. 2-4) first as
# a smoke test before committing to the full 250 -- each job is expensive
# (real OpenDDE + AbLang2 + 200-step APGM + discrete search on the full,
# uncropped complex; comparable VHH72 jobs took on the order of an hour
# each), so an undiscovered bug at seed 0 would otherwise waste the whole
# batch's compute.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICES="${1:-0,1,2,3}"
OUTPUT_DIR="${2:-results/p17_sweep_$(date +%Y%m%d_%H%M%S)}"
NUM_SEEDS="${3:-250}"
EDIT_BUDGET="${4:-5}"

echo "=== P17 hallucination sweep ==="
echo "repo root:    $REPO_ROOT"
echo "devices:      $DEVICES"
echo "num seeds:    $NUM_SEEDS"
echo "edit budget:  $EDIT_BUDGET"
echo "output dir:   $OUTPUT_DIR"
echo

mkdir -p "$OUTPUT_DIR"

echo "[1/2] dispatching $NUM_SEEDS jobs across devices $DEVICES..."
.venv/bin/python examples/p17_hallucination_dispatch.py \
    --devices "$DEVICES" \
    --num-seeds "$NUM_SEEDS" \
    --edit-budget "$EDIT_BUDGET" \
    --output-dir "$OUTPUT_DIR"

echo
echo "[2/2] aggregating results..."
.venv/bin/python examples/p17_hallucination_aggregate.py \
    --results-dir "$OUTPUT_DIR" \
    --num-seeds "$NUM_SEEDS" \
    --output-csv "$OUTPUT_DIR/combined.csv"

echo
echo "=== done ==="
echo "per-seed CSVs + logs: $OUTPUT_DIR/p17_mcmc_stopgrad0_seed{N}.{csv,log}"
echo "combined CSV:         $OUTPUT_DIR/combined.csv"
