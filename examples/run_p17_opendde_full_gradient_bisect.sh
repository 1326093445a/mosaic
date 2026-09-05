#!/usr/bin/env bash
# Bisects which loss term's gradient through OpenDDE's full path causes the
# ~199GiB blowup seen in run_p17_opendde_full_gradient_smoke_test.sh -- runs
# contact/pose/confidence/ablang2 each ALONE, in its own subprocess, so one
# OOM doesn't corrupt the GPU allocator state for the next. See
# examples/p17_opendde_full_gradient_bisect.py's own docstring for why this
# replaced guessing from HLO shapes (that guess -- OuterProductMean -- was
# patched, verified numerically correct, and STILL reproduced the exact same
# crash, so the shape match was evidently coincidental).
#
# Usage:
#   examples/run_p17_opendde_full_gradient_bisect.sh [DEVICE] [OUTPUT_DIR]
#
# Args:
#   DEVICE     single physical GPU id, e.g. 0. Use "inherit" to keep the
#              caller's CUDA_VISIBLE_DEVICES. Default: 0
#   OUTPUT_DIR where the log goes. Default:
#              results/p17_opendde_full_gradient_bisect_<timestamp>
#
# Example:
#   examples/run_p17_opendde_full_gradient_bisect.sh 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-0}"
OUTPUT_DIR="${2:-results/p17_opendde_full_gradient_bisect_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/run.log"

echo "=== P17 full-OpenDDE gradient bisection (contact / pose / confidence / ablang2) ==="
echo "repo root:  $REPO_ROOT"
echo "device:     $DEVICE"
echo "output dir: $OUTPUT_DIR"
echo "log:        $LOG_PATH"
echo

cmd=(.venv/bin/python examples/p17_opendde_full_gradient_bisect.py)

if [[ "$DEVICE" == "inherit" ]]; then
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
else
    CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
fi

echo
echo "=== done ==="
echo "log: $LOG_PATH"
echo "Look at the BISECTION SUMMARY block at the end of the log for which"
echo "component(s) actually failed."
