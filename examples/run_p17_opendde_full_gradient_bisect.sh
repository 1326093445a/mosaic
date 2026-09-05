#!/usr/bin/env bash
# Bisects which loss term's gradient through OpenDDE's full path causes the
# ~199GiB blowup seen in run_p17_opendde_full_gradient_smoke_test.sh -- runs
# contact/pose/confidence/ablang2 each ALONE, in its own subprocess, so one
# OOM doesn't corrupt the GPU allocator state for the next. See
# examples/p17_opendde_full_gradient_bisect.py's own docstring for the
# results this already produced: contact/ablang2 pass, pose/confidence fail
# identically -- localizing the real culprit to
# StructuralTokenExpander._pair_project_by_role_full (fixed by
# patches/patch_jopendde_structural_token_expander.py), not the trunk's
# OuterProductMean (patches/patch_jopendde_outer_product_mean.py -- a real,
# separate, also-necessary fix, just not the one causing THIS crash).
# Applies both patches before running so a fresh run reflects the current
# best understanding rather than re-discovering the same failure.
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

echo "[0/1] applying jopendde patches (idempotent)..."
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
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
