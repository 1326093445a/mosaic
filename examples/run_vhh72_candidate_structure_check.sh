#!/usr/bin/env bash
# Run full OpenDDE diffusion/coordinate structural sanity check for a selected
# VHH72 hallucination candidate.
#
# Default target is the current curated candidate from
# results/hallucination_sweep_discrete_3seed/candidate_shortlist.csv:
#   mcmc, stop_grad=0, seed=1, edit_count=4
#
# Usage:
#   examples/run_vhh72_candidate_structure_check.sh [DEVICE] [COMBINED_CSV] [SELECT] [OUTPUT_DIR] [SEED]
#
# Args:
#   DEVICE       single physical GPU id, e.g. 4. Use "inherit" to keep the
#                caller's CUDA_VISIBLE_DEVICES. Default: 4
#   COMBINED_CSV combined hallucination Pareto CSV. Default:
#                results/hallucination_sweep_discrete_3seed/combined.csv
#   SELECT       policy:stop_grad:seed:edit_count. Default: mcmc:0:1:4
#   OUTPUT_DIR   where CIFs/summary/log go. Default:
#                results/hallucination_sweep_discrete_3seed/structures_seed1_edit4
#   SEED         OpenDDE diffusion RNG seed. Default: 0
#
# Example:
#   examples/run_vhh72_candidate_structure_check.sh 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-4}"
COMBINED_CSV="${2:-results/hallucination_sweep_discrete_3seed/combined.csv}"
SELECT="${3:-mcmc:0:1:4}"
OUTPUT_DIR="${4:-results/hallucination_sweep_discrete_3seed/structures_seed1_edit4}"
SEED="${5:-0}"

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/run.log"

echo "=== VHH72 full-OpenDDE structure check ==="
echo "repo root:    $REPO_ROOT"
echo "device:       $DEVICE"
echo "combined csv: $COMBINED_CSV"
echo "select:       $SELECT"
echo "output dir:   $OUTPUT_DIR"
echo "seed:         $SEED"
echo "log:          $LOG_PATH"
echo

cmd=(
    .venv/bin/python examples/vhh72_predict_hallucination_structures.py
    --combined-csv "$COMBINED_CSV"
    --select "$SELECT"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
)

if [[ "$DEVICE" == "inherit" ]]; then
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
else
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
fi

echo
echo "=== done ==="
echo "summary: $OUTPUT_DIR/real_structure_summary.csv"
echo "CIFs:    $OUTPUT_DIR/*.cif"
