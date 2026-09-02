#!/usr/bin/env bash
# Single-candidate smoke test: does one OpenDDE FULL-path forward+backward
# (real gradient, confidence-metric-aware loss -- IPTMLoss/BinderTargetPAE/
# TargetBinderPAE/pTMEnergy alongside contact/pose/naturalness/edit-budget)
# fit in GPU memory at P17's real complex size (307 residues)? See
# examples/p17_opendde_full_gradient_smoke_test.py's own docstring for why
# this is a different, narrower, previously-untested question than the
# already-fixed 234GiB eager-execution bug (docs/guidance_alphaseq_testing_notes.md
# section 13.3) -- that was forward-only; this measures forward+backward.
#
# Needs a large-memory GPU (H200-class, ~141GB) -- this WILL OOM on a 24GB
# consumer GPU (already confirmed insufficient even for forward-only at this
# complex size), so don't bother running it there; that's not new information.
#
# Usage:
#   examples/run_p17_opendde_full_gradient_smoke_test.sh [DEVICE] [OUTPUT_DIR]
#
# Args:
#   DEVICE     single physical GPU id, e.g. 0. Use "inherit" to keep the
#              caller's CUDA_VISIBLE_DEVICES. Default: 0
#   OUTPUT_DIR where the log goes. Default:
#              results/p17_opendde_full_gradient_smoke_<timestamp>
#
# Example:
#   examples/run_p17_opendde_full_gradient_smoke_test.sh 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-0}"
OUTPUT_DIR="${2:-results/p17_opendde_full_gradient_smoke_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/run.log"

echo "=== P17 full-OpenDDE confidence-aware gradient smoke test ==="
echo "repo root:  $REPO_ROOT"
echo "device:     $DEVICE"
echo "output dir: $OUTPUT_DIR"
echo "log:        $LOG_PATH"
echo

cmd=(.venv/bin/python examples/p17_opendde_full_gradient_smoke_test.py)

if [[ "$DEVICE" == "inherit" ]]; then
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
else
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
fi

echo
echo "=== done ==="
echo "log: $LOG_PATH"
echo
echo "Read the two 'peak_bytes_in_use' lines in the log:"
echo "  - if both calls completed: a single confidence-aware gradient candidate fits here."
echo "  - if it OOM'd: whichever peak_bytes_in_use line printed last is the real number"
echo "    to reason about gradient checkpointing against."
