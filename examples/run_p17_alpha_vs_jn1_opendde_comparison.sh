#!/usr/bin/env bash
# Real, forward-only OpenDDE comparison: P17-vs-Alpha (binding) vs.
# P17-vs-JN.1 (non-binding), same settings for both (no template, no MSA,
# recycling_steps=3 by default) -- see
# examples/p17_alpha_vs_jn1_opendde_comparison.py's docstring.
#
# Usage:
#   examples/run_p17_alpha_vs_jn1_opendde_comparison.sh [OUTPUT_DIR] [DIFFUSION_STEPS] [RECYCLING_STEPS] [NUM_SEEDS] [DEVICE]
#
# Args (all optional):
#   OUTPUT_DIR        Default: results/p17_alpha_vs_jn1_<timestamp>
#   DIFFUSION_STEPS   Default: 64
#   RECYCLING_STEPS   Default: 3 (user-specified value for this check)
#   NUM_SEEDS         Default: 3 (seeds 0..NUM_SEEDS-1)
#   DEVICE            physical GPU id, single device -- this script is not
#                     parallelized across GPUs (only 2 complexes x few
#                     seeds; runs one after another on one GPU). Default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-results/p17_alpha_vs_jn1_$(date +%Y%m%d_%H%M%S)}"
DIFFUSION_STEPS="${2:-64}"
RECYCLING_STEPS="${3:-3}"
NUM_SEEDS="${4:-3}"
DEVICE="${5:-0}"

mkdir -p "$OUTPUT_DIR"

echo "=== P17-vs-Alpha vs. P17-vs-JN.1 OpenDDE comparison ==="
echo "repo root:        $REPO_ROOT"
echo "output dir:       $OUTPUT_DIR"
echo "diffusion steps:  $DIFFUSION_STEPS"
echo "recycling steps:  $RECYCLING_STEPS"
echo "num seeds:        $NUM_SEEDS"
echo "device:           $DEVICE"
echo

echo "[0/1] applying jopendde patches (idempotent)..."
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
echo

CUDA_VISIBLE_DEVICES="$DEVICE" .venv/bin/python examples/p17_alpha_vs_jn1_opendde_comparison.py \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --recycling-steps "$RECYCLING_STEPS" \
    --num-seeds "$NUM_SEEDS" \
    --output "$OUTPUT_DIR/comparison.csv" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo
echo "=== done ==="
echo "csv: $OUTPUT_DIR/comparison.csv"
echo "log: $OUTPUT_DIR/run.log"
