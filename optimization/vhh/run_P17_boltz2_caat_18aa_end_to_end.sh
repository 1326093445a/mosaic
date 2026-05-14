#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-vhh/P17_boltz2_caat_18aa}"

# Standard amino acids excluding Cys and Met.
AA_PANEL="${AA_PANEL:-ARNDQEGHILKFPSTWYV}"
BASELINE_SAMPLES="${BASELINE_SAMPLES:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
SAMPLING_STEPS="${SAMPLING_STEPS:-200}"
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"
MAX_COMBO_EDITS="${MAX_COMBO_EDITS:-7}"
TOP_POSITIONS="${TOP_POSITIONS:-12}"
MIN_DELTA_IPSAE="${MIN_DELTA_IPSAE:-0.10}"
MIN_ABS_Z="${MIN_ABS_Z:-2.0}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

run_one() {
  local target="$1"
  local preset
  case "${target}" in
    alpha) preset="p17-alpha" ;;
    jn1) preset="p17-jn1" ;;
    *)
      echo "Unknown target: ${target}" >&2
      exit 2
      ;;
  esac

  echo "=============================================="
  echo "P17 Boltz2 CAAT-style 18-AA scan: ${target}"
  echo "=============================================="
  echo "Preset:          ${preset}"
  echo "Output:          ${OUT_ROOT}/${target}"
  echo "AA_PANEL:        ${AA_PANEL}"
  echo "baseline sample: ${BASELINE_SAMPLES}"
  echo "variant samples: ${NUM_SAMPLES}"
  echo "sampling steps:  ${SAMPLING_STEPS}"
  echo "recycling:       ${RECYCLING_STEPS}"
  echo

  uv run python optimization/vhh/boltz2_caat_sensitivity.py \
    --preset "${preset}" \
    --aa-panel "${AA_PANEL}" \
    --baseline-samples "${BASELINE_SAMPLES}" \
    --num-samples "${NUM_SAMPLES}" \
    --sampling-steps "${SAMPLING_STEPS}" \
    --recycling-steps "${RECYCLING_STEPS}" \
    --max-combo-edits "${MAX_COMBO_EDITS}" \
    --top-positions "${TOP_POSITIONS}" \
    --min-delta-ipsae "${MIN_DELTA_IPSAE}" \
    --min-abs-z "${MIN_ABS_Z}" \
    --output-dir "${OUT_ROOT}/${target}"
}

run_one alpha
run_one jn1

echo "=============================================="
echo "Plotting Alpha vs JN.1 CAAT summaries"
echo "=============================================="
uv run python optimization/vhh/plot_P17_boltz2_caat.py \
  --root "${OUT_ROOT}" \
  --aa-order "${AA_PANEL}" \
  --min-delta-ipsae "${MIN_DELTA_IPSAE}" \
  --min-abs-z "${MIN_ABS_Z}"

echo
echo "Done. Main figure:"
echo "${OUT_ROOT}/P17_alpha_vs_jn1_caat_summary.png"
