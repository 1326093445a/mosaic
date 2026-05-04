#!/usr/bin/env bash
# VHH72 CDR redesign benchmark against SARS-CoV-2 WT RBD.
#
# Defaults are benchmark-ready and intentionally mirror the real VHH workflow:
#   MODE=v4, BUDGET=7, refold 5 samples in one batch, polish batch size 3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

YAML_FILE="${SCRIPT_DIR}/VHH72_WT_SARS-CoV-2_RBD.yaml"
PDB_FILE="${SCRIPT_DIR}/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb"

MODE="${MODE:-v4}"
SEED="${SEED:-0}"
START_SEED="${START_SEED:-${SEED}}"
NUM_DESIGNS="${NUM_DESIGNS:-1}"
DEVICES="${DEVICES:-}"
START_SIGMA_FRAC="${START_SIGMA_FRAC:-0.25}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-200}"
STEP_SCALE="${STEP_SCALE:-2.0}"
NOISE_SCALE="${NOISE_SCALE:-0.88}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-sigma_squared}"
N_OUTER_ITERATIONS="${N_OUTER_ITERATIONS:-3}"
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"
REFOLD_SAMPLING_STEPS="${REFOLD_SAMPLING_STEPS:-200}"
REFOLD_NUM_SAMPLES="${REFOLD_NUM_SAMPLES:-5}"
REFOLD_BATCH_SIZE="${REFOLD_BATCH_SIZE:-5}"
IPSAE_PAE_CUTOFF="${IPSAE_PAE_CUTOFF:-12.0}"
REFOLD_RMSD_THRESHOLD="${REFOLD_RMSD_THRESHOLD:-2.5}"

BUDGET="${BUDGET:-7}"
WEIGHT_EDIT_BUDGET="${WEIGHT_EDIT_BUDGET:-10.0}"
WEIGHT_ESM2="${WEIGHT_ESM2:-${WEIGHT_ESMC:-0.10}}"
WEIGHT_ABLANG2="${WEIGHT_ABLANG2:-${WEIGHT_ABLANG:-0.10}}"
ESM2_MODEL="${ESM2_MODEL:-esm2_t33_650M_UR50D}"
CLIP_GRADIENT_NORM="${CLIP_GRADIENT_NORM:-1.0}"

POLISH_STEPS="${POLISH_STEPS:-200}"
POLISH_BATCH_SIZE="${POLISH_BATCH_SIZE:-3}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

OUTPUT_DIR="${1:-${SCRIPT_DIR}/VHH72_WT_mosaic_${MODE}_b${BUDGET}_seed${SEED}_sigma${START_SIGMA_FRAC}}"

# ANARCI Kabat paper design set mapped to BoltzGen/Mosaic 1-based chain-order
# positions. See VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv.
CDR_INDICES=(
  30 31
  52 53 54 55 56 57 58 59 60 61 62 63 64 65
  100 101 102 103 104 105 106 107 108 109 110 111
)

cd "${ROOT_DIR}"

echo "========================================"
echo "VHH72 WT - SARS-CoV-2 WT RBD benchmark"
echo "========================================"
echo "Mode:              ${MODE}"
echo "YAML:              ${YAML_FILE}"
echo "Structure:         ${PDB_FILE}"
echo "Output:            ${OUTPUT_DIR}"
echo "Budget:            ${BUDGET}"
echo "Seed:              ${SEED}"
echo "num_designs:       ${NUM_DESIGNS}"
if [[ -n "${DEVICES}" ]]; then
  echo "devices:           ${DEVICES}"
fi
echo "start_sigma_frac:  ${START_SIGMA_FRAC}"
echo "num_steps:         ${NUM_SAMPLING_STEPS}"
echo "noise_scale:       ${NOISE_SCALE}"
echo "step_scale:        ${STEP_SCALE}"
echo "lambda:            ${LAMBDA_MAX} (${LAMBDA_SCHEDULE})"
echo "ESM2 model/weight: ${ESM2_MODEL} / ${WEIGHT_ESM2}"
echo "AbLang2 weight:    ${WEIGHT_ABLANG2}"
echo "polish:            ${POLISH_STEPS} step(s), batch ${POLISH_BATCH_SIZE}"
echo "refold:            ${REFOLD_NUM_SAMPLES} sample(s), ${REFOLD_SAMPLING_STEPS} steps, batch ${REFOLD_BATCH_SIZE}"
echo "ipSAE PAE cutoff:  ${IPSAE_PAE_CUTOFF}"
echo "RMSD filter:       binder CA <= ${REFOLD_RMSD_THRESHOLD} A"
echo "XLA preallocate:   ${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "GPU allocator:     ${TF_GPU_ALLOCATOR}"
echo "XLA flags:         ${XLA_FLAGS}"
echo ""

EXTRA_ARGS=()
if [[ "${NUM_DESIGNS}" != "1" ]]; then
  EXTRA_ARGS+=(--num-designs "${NUM_DESIGNS}" --start-seed "${START_SEED}")
fi
if [[ -n "${DEVICES}" ]]; then
  EXTRA_ARGS+=(--devices "${DEVICES}")
fi

uv run python examples/boltzgen_vhh_guided.py \
  --mode "${MODE}" \
  --boltzgen-yaml "${YAML_FILE}" \
  --complex-cif "${PDB_FILE}" \
  --binder-chain A \
  --target-chains E \
  --cdr-indices "${CDR_INDICES[@]}" \
  --budget "${BUDGET}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --start-sigma-frac "${START_SIGMA_FRAC}" \
  --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
  --step-scale "${STEP_SCALE}" \
  --noise-scale "${NOISE_SCALE}" \
  --lambda-max "${LAMBDA_MAX}" \
  --lambda-schedule "${LAMBDA_SCHEDULE}" \
  --n-outer-iterations "${N_OUTER_ITERATIONS}" \
  --recycling-steps "${RECYCLING_STEPS}" \
  --refold-sampling-steps "${REFOLD_SAMPLING_STEPS}" \
  --refold-num-samples "${REFOLD_NUM_SAMPLES}" \
  --refold-batch-size "${REFOLD_BATCH_SIZE}" \
  --ipsae-pae-cutoff "${IPSAE_PAE_CUTOFF}" \
  --refold-rmsd-threshold "${REFOLD_RMSD_THRESHOLD}" \
  --weight-edit-budget "${WEIGHT_EDIT_BUDGET}" \
  --esm2-model "${ESM2_MODEL}" \
  --weight-esm2 "${WEIGHT_ESM2}" \
  --weight-ablang2 "${WEIGHT_ABLANG2}" \
  --clip-gradient-norm "${CLIP_GRADIENT_NORM}" \
  --polish-steps "${POLISH_STEPS}" \
  --polish-batch-size "${POLISH_BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results: ${OUTPUT_DIR}"
