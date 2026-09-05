#!/usr/bin/env bash
# Runs the REAL P17 discrete MCMC search (edit_budgeted_gradient_mcmc,
# unmodified) for one seed, interleaved with real confidence-aware
# full-OpenDDE rescoring of the current best candidate between chunks of
# MCMC steps. Answers: what does this actually cost on a real seed, and
# does the real signal track the cheap search's own trajectory? See
# examples/p17_hallucination_mcmc_with_full_opendde_rescoring.py's own
# docstring for the full design and its one known limitation (the
# already-tried-mutation set resets each chunk).
#
# Applies the same jopendde patches as the smoke test / bisection (both
# confirmed necessary and correct -- see patches/*.py) before running.
#
# Usage:
#   examples/run_p17_hallucination_mcmc_with_full_opendde_rescoring.sh [DEVICE] [SEED] [EDIT_BUDGET] [MCMC_STEPS] [RESCORE_EVERY] [OUTPUT_DIR]
#
# Args (all optional):
#   DEVICE         single physical GPU id, e.g. 0. "inherit" keeps the
#                  caller's CUDA_VISIBLE_DEVICES. Default: 0
#   SEED           MCMC seed. Default: 0
#   EDIT_BUDGET    max mutations from WT. Default: 5 (matches production)
#   MCMC_STEPS     total discrete search steps. Default: 100 (matches
#                  production p17_hallucination_search.py's default)
#   RESCORE_EVERY  MCMC steps per chunk between rescoring calls. Default: 20
#                  (5 rescoring calls across a 100-step run)
#   OUTPUT_DIR     where the rescoring CSV + log go. Default:
#                  results/p17_mcmc_rescoring_<timestamp>
#
# Example:
#   examples/run_p17_hallucination_mcmc_with_full_opendde_rescoring.sh 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-0}"
SEED="${2:-0}"
EDIT_BUDGET="${3:-5}"
MCMC_STEPS="${4:-100}"
RESCORE_EVERY="${5:-20}"
OUTPUT_DIR="${6:-results/p17_mcmc_rescoring_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/run.log"
CSV_PATH="$OUTPUT_DIR/rescoring_seed${SEED}.csv"

echo "=== P17 MCMC search + real full-OpenDDE confidence rescoring ==="
echo "repo root:      $REPO_ROOT"
echo "device:         $DEVICE"
echo "seed:           $SEED"
echo "edit budget:    $EDIT_BUDGET"
echo "mcmc steps:     $MCMC_STEPS"
echo "rescore every:  $RESCORE_EVERY steps"
echo "output dir:     $OUTPUT_DIR"
echo "log:            $LOG_PATH"
echo "csv:            $CSV_PATH"
echo

echo "[0/1] applying jopendde patches (idempotent)..."
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
echo

cmd=(
    .venv/bin/python examples/p17_hallucination_mcmc_with_full_opendde_rescoring.py
    --seed "$SEED"
    --edit-budget "$EDIT_BUDGET"
    --mcmc-steps "$MCMC_STEPS"
    --rescore-every "$RESCORE_EVERY"
    --output "$CSV_PATH"
)

if [[ "$DEVICE" == "inherit" ]]; then
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
else
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
fi

echo
echo "=== done ==="
echo "log: $LOG_PATH"
echo "csv: $CSV_PATH"
echo "Check the 'totals' line near the end of the log for real cheap-search"
echo "vs. rescoring wall-clock time, and the CSV for whether real_value/"
echo "real_iptm/etc. track cheap_best_val across chunks."
