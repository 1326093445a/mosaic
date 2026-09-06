#!/usr/bin/env bash
# End-to-end check: does OpenDDE's confidence-aware full path actually
# converge to a sane structure/confidence reading as --diffusion-steps
# increases, or were the p17_hallucination_mcmc_with_full_opendde_rescoring.py
# run's suspiciously bad numbers (ipTM 0.17-0.21, binder_pose_rmsd 19-51A)
# just an artifact of that run's N_DIFFUSION_STEPS=8 being too few? Scores
# the SAME fixed candidate sequence at several diffusion-step counts in
# parallel, one per GPU. See
# examples/p17_opendde_diffusion_steps_convergence_check.py's docstring for
# the real reference point this compares against (ipTM 0.87-0.93, RMSD ~6A
# at 200 steps, docs/guidance_alphaseq_testing_notes.md section 13.3).
#
# Usage:
#   examples/run_p17_opendde_diffusion_steps_convergence_check.sh [DEVICES] [RESCORING_CSV] [OUTPUT_DIR] [STEPS_LIST]
#
# Args (all optional):
#   DEVICES        comma-separated physical GPU ids. Default: 0,1,2,3,4,5,6,7
#                   (all 8)
#   RESCORING_CSV  path to an existing rescoring_seed*.csv (from
#                  run_p17_hallucination_mcmc_with_full_opendde_rescoring.sh)
#                  to pull the candidate sequence from (its LAST row).
#                  Default: the most recently modified
#                  results/p17_mcmc_rescoring_*/rescoring_seed*.csv
#   OUTPUT_DIR     Default: results/p17_opendde_diffusion_convergence_<timestamp>
#   STEPS_LIST     comma-separated diffusion step counts to test, one per
#                  device (extra values queue after the first batch).
#                  Default: 8,16,24,32,48,64,96,128 (8 values for 8 GPUs)
#
# Example:
#   examples/run_p17_opendde_diffusion_steps_convergence_check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICES="${1:-0,1,2,3,4,5,6,7}"
RESCORING_CSV="${2:-}"
OUTPUT_DIR="${3:-results/p17_opendde_diffusion_convergence_$(date +%Y%m%d_%H%M%S)}"
STEPS_LIST="${4:-8,16,24,32,48,64,96,128}"

if [[ -z "$RESCORING_CSV" ]]; then
    RESCORING_CSV="$(ls -t results/p17_mcmc_rescoring_*/rescoring_seed*.csv 2>/dev/null | head -1)"
    if [[ -z "$RESCORING_CSV" ]]; then
        echo "ERROR: no RESCORING_CSV given and no results/p17_mcmc_rescoring_*/rescoring_seed*.csv found" >&2
        exit 1
    fi
fi

SEQUENCE="$(.venv/bin/python -c "
import csv
with open('$RESCORING_CSV') as f:
    rows = list(csv.DictReader(f))
print(rows[-1]['sequence'])
")"

mkdir -p "$OUTPUT_DIR"

echo "=== P17 OpenDDE diffusion-steps convergence check ==="
echo "repo root:      $REPO_ROOT"
echo "devices:        $DEVICES"
echo "rescoring csv:  $RESCORING_CSV"
echo "sequence:       $SEQUENCE"
echo "output dir:     $OUTPUT_DIR"
echo "steps list:     $STEPS_LIST"
echo

echo "[0/1] applying jopendde patches (idempotent)..."
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
echo

.venv/bin/python examples/p17_opendde_diffusion_steps_convergence_dispatch.py \
    --devices "$DEVICES" \
    --diffusion-steps-list "$STEPS_LIST" \
    --sequence "$SEQUENCE" \
    --output-dir "$OUTPUT_DIR"

echo
echo "=== done ==="
echo "summary: $OUTPUT_DIR/convergence_summary.csv"
