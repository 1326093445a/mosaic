#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-vhh/VHH72_boltz2_caat_sensitivity}"

AA_PANEL="${AA_PANEL:-FWYRMSTV}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
SAMPLING_STEPS="${SAMPLING_STEPS:-200}"
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"
MAX_COMBO_EDITS="${MAX_COMBO_EDITS:-7}"
TOP_POSITIONS="${TOP_POSITIONS:-12}"
MIN_DELTA_IPSAE="${MIN_DELTA_IPSAE:-0.05}"
MIN_ABS_Z="${MIN_ABS_Z:-2.0}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

echo "========================================"
echo "VHH72 Boltz2 CAAT-style sensitivity scan"
echo "========================================"
echo "Output:          ${OUT_DIR}"
echo "AA_PANEL:        ${AA_PANEL}"
echo "baseline sample: ${BASELINE_SAMPLES}"
echo "variant samples: ${NUM_SAMPLES}"
echo "sampling steps:  ${SAMPLING_STEPS}"
echo "recycling:       ${RECYCLING_STEPS}"
echo "combo edits:     ${MAX_COMBO_EDITS}"
echo "top positions:   ${TOP_POSITIONS}"
echo

uv run python optimization/vhh/boltz2_caat_sensitivity.py \
  --preset vhh72 \
  --aa-panel "${AA_PANEL}" \
  --baseline-samples "${BASELINE_SAMPLES}" \
  --num-samples "${NUM_SAMPLES}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --recycling-steps "${RECYCLING_STEPS}" \
  --max-combo-edits "${MAX_COMBO_EDITS}" \
  --top-positions "${TOP_POSITIONS}" \
  --min-delta-ipsae "${MIN_DELTA_IPSAE}" \
  --min-abs-z "${MIN_ABS_Z}" \
  --output-dir "${OUT_DIR}"
