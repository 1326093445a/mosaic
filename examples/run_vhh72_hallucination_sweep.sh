#!/usr/bin/env bash
# Runs the full VHH72 hallucination search pipeline end to end:
#   1. vhh72_hallucination_dispatch.py -- launches N jobs (selected search
#      policies x selected AbLang2 gradient-flow settings x len(SEEDS))
#      across the given GPUs; each job internally runs
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
#   examples/run_vhh72_hallucination_sweep.sh [DEVICES] [EDIT_BUDGET] [SEEDS] [OUTPUT_DIR] [APGM_SEED_MODE] [APGM_TOPK_THRESHOLD] [POLICIES] [STOP_GRADS] [APGM_INIT_WT_PROB] [APGM_SCALE] [OPENDDE_PATH] [APGM_STEPS] [DISCRETE_STEPS] [OPENDDE_SAMPLING_STEPS] [OPENDDE_NUM_SAMPLES]
#
# All arguments optional:
#   DEVICES     comma-separated physical GPU ids, e.g. 0,1,2,3 (default: 0,1,2,3)
#   EDIT_BUDGET max mutations from WT (default: 5)
#   SEEDS       comma-separated seeds, e.g. 0,1,2,3,4 (default: 0 -- single seed)
#   OUTPUT_DIR  where results land (default: results/hallucination_sweep_<timestamp>)
#   APGM_SEED_MODE argmax|sample|topk (default: argmax)
#   APGM_TOPK_THRESHOLD threshold for topk mode (default: 0.15)
#   POLICIES    comma-separated policies to run (default: greedy,mcmc)
#   STOP_GRADS  comma-separated stop_grad settings to run (default: 1,0)
#   APGM_INIT_WT_PROB WT amino-acid probability for APGM init (default: 1.0)
#   APGM_SCALE  simplex_APGM scale (default: 1.2)
#   OPENDDE_PATH distogram|full for the in-loop hallucination OpenDDE loss
#                (default: distogram)
#   APGM_STEPS   number of APGM continuous-relaxation steps (default: 200)
#   DISCRETE_STEPS max greedy/MCMC steps (default: 200 for greedy, 100 for mcmc;
#                  one value is passed to both, so use this mainly for mcmc-only
#                  smoke/full-path runs)
#   OPENDDE_SAMPLING_STEPS full-path diffusion steps (default: model default)
#   OPENDDE_NUM_SAMPLES full-path coordinate samples per loss call (default: 1)
#
# Examples:
#   examples/run_vhh72_hallucination_sweep.sh
#   examples/run_vhh72_hallucination_sweep.sh 0,1,2,3
#   examples/run_vhh72_hallucination_sweep.sh 0,1,2,3 5 0,1,2,3,4 results/my_run
#   examples/run_vhh72_hallucination_sweep.sh 0,1,2,3 5 0,1,2 results/apgm_sample_mcmc_sg0 sample 0.15 mcmc 0 0.80 1.0
#   examples/run_vhh72_hallucination_sweep.sh 4,5,6,7 5 0,1,2 results/full_mcmc_sg0_smoke argmax 0.15 mcmc 0 1.0 1.2 full 0 10 8 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICES="${1:-0,1,2,3}"
EDIT_BUDGET="${2:-5}"
SEEDS="${3:-0}"
OUTPUT_DIR="${4:-results/hallucination_sweep_$(date +%Y%m%d_%H%M%S)}"
APGM_SEED_MODE="${5:-argmax}"
APGM_TOPK_THRESHOLD="${6:-0.15}"
POLICIES="${7:-greedy,mcmc}"
STOP_GRADS="${8:-1,0}"
APGM_INIT_WT_PROB="${9:-1.0}"
APGM_SCALE="${10:-1.2}"
OPENDDE_PATH="${11:-distogram}"
APGM_STEPS_RUN="${12:-200}"
DISCRETE_STEPS="${13:-}"
OPENDDE_SAMPLING_STEPS="${14:-}"
OPENDDE_NUM_SAMPLES="${15:-1}"

GREEDY_STEPS="200"
MCMC_STEPS="100"
if [[ -n "$DISCRETE_STEPS" ]]; then
    GREEDY_STEPS="$DISCRETE_STEPS"
    MCMC_STEPS="$DISCRETE_STEPS"
fi

echo "=== VHH72 hallucination sweep ==="
echo "repo root:    $REPO_ROOT"
echo "devices:      $DEVICES"
echo "edit budget:  $EDIT_BUDGET"
echo "seeds:        $SEEDS"
echo "policies:     $POLICIES"
echo "stop_grads:   $STOP_GRADS"
echo "apgm seed:    $APGM_SEED_MODE"
echo "topk thresh:  $APGM_TOPK_THRESHOLD"
echo "apgm init:    $APGM_INIT_WT_PROB"
echo "apgm scale:   $APGM_SCALE"
echo "apgm steps:   $APGM_STEPS_RUN"
echo "opendde path: $OPENDDE_PATH"
echo "full samples: $OPENDDE_NUM_SAMPLES"
if [[ -n "$OPENDDE_SAMPLING_STEPS" ]]; then
    echo "full steps:   $OPENDDE_SAMPLING_STEPS"
else
    echo "full steps:   model default"
fi
echo "greedy steps: $GREEDY_STEPS"
echo "mcmc steps:   $MCMC_STEPS"
echo "output dir:   $OUTPUT_DIR"
echo

mkdir -p "$OUTPUT_DIR"

echo "[1/2] dispatching jobs (policies=$POLICIES x stop_grads=$STOP_GRADS x seeds=$SEEDS) across devices $DEVICES..."
DISPATCH_CMD=(
    .venv/bin/python examples/vhh72_hallucination_dispatch.py
    --devices "$DEVICES" \
    --edit-budget "$EDIT_BUDGET" \
    --seeds "$SEEDS" \
    --policies "$POLICIES" \
    --stop-grads "$STOP_GRADS" \
    --apgm-seed-mode "$APGM_SEED_MODE" \
    --apgm-topk-threshold "$APGM_TOPK_THRESHOLD" \
    --apgm-init-wt-prob "$APGM_INIT_WT_PROB" \
    --apgm-scale "$APGM_SCALE" \
    --apgm-steps "$APGM_STEPS_RUN" \
    --greedy-steps "$GREEDY_STEPS" \
    --mcmc-steps "$MCMC_STEPS" \
    --opendde-path "$OPENDDE_PATH" \
    --opendde-num-samples "$OPENDDE_NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR"
)
if [[ -n "$OPENDDE_SAMPLING_STEPS" ]]; then
    DISPATCH_CMD+=(--opendde-sampling-steps "$OPENDDE_SAMPLING_STEPS")
fi
"${DISPATCH_CMD[@]}"

echo
echo "[2/2] aggregating results..."
.venv/bin/python examples/vhh72_hallucination_aggregate.py \
    --results-dir "$OUTPUT_DIR" \
    --seeds "$SEEDS" \
    --policies "$POLICIES" \
    --stop-grads "$STOP_GRADS" \
    --output-csv "$OUTPUT_DIR/combined.csv" \
    --output-summary "$OUTPUT_DIR/seed_variance_summary.csv" \
    --output-figure "$OUTPUT_DIR/pareto_comparison.png"

echo
echo "=== done ==="
echo "per-job CSVs + logs: $OUTPUT_DIR/{policy}_stopgrad{setting}_seed{N}.{csv,log}"
echo "combined CSV:        $OUTPUT_DIR/combined.csv"
echo "seed variance/win-rate summary: $OUTPUT_DIR/seed_variance_summary.csv"
echo "figure:               $OUTPUT_DIR/pareto_comparison.png"
