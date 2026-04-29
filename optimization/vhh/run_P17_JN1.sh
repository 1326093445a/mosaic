#!/usr/bin/env bash
# P17 nanobody CDR redesign against JN.1 RBD using mosaic-guided BoltzGen.
#
# Defaults are chosen for a first smoke run. For the full end-to-end pipeline:
#   MODE=v4 NUM_SAMPLING_STEPS=200 RECYCLING_STEPS=3 N_OUTER_ITERATIONS=3 \
#     REFOLD_SAMPLING_STEPS=200 REFOLD_NUM_SAMPLES=5 REFOLD_BATCH_SIZE=5 \
#     IPSAE_PAE_CUTOFF=12 \
#     bash optimization/vhh/run_P17_JN1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

YAML_FILE="${SCRIPT_DIR}/P17_JN1.yaml"
PDB_FILE="${SCRIPT_DIR}/P17_JN1.pdb"

MODE="${MODE:-v1}"
SEED="${SEED:-0}"
START_SEED="${START_SEED:-${SEED}}"
NUM_DESIGNS="${NUM_DESIGNS:-1}"
DEVICES="${DEVICES:-}"
START_SIGMA_FRAC="${START_SIGMA_FRAC:-0.35}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-80}"
STEP_SCALE="${STEP_SCALE:-2.0}"
NOISE_SCALE="${NOISE_SCALE:-0.88}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-sigma_squared}"
N_OUTER_ITERATIONS="${N_OUTER_ITERATIONS:-1}"
RECYCLING_STEPS="${RECYCLING_STEPS:-1}"
REFOLD_SAMPLING_STEPS="${REFOLD_SAMPLING_STEPS:-25}"
REFOLD_NUM_SAMPLES="${REFOLD_NUM_SAMPLES:-1}"
REFOLD_BATCH_SIZE="${REFOLD_BATCH_SIZE:-${REFOLD_NUM_SAMPLES}}"
IPSAE_PAE_CUTOFF="${IPSAE_PAE_CUTOFF:-12.0}"
REFOLD_RMSD_THRESHOLD="${REFOLD_RMSD_THRESHOLD:-2.5}"

BUDGET="${BUDGET:-7}"
WEIGHT_EDIT_BUDGET="${WEIGHT_EDIT_BUDGET:-10.0}"
WEIGHT_ESMC="${WEIGHT_ESMC:-0.10}"
WEIGHT_ABLANG="${WEIGHT_ABLANG:-0.10}"
CLIP_GRADIENT_NORM="${CLIP_GRADIENT_NORM:-1.0}"

POLISH_STEPS="${POLISH_STEPS:-80}"
POLISH_BATCH_SIZE="${POLISH_BATCH_SIZE:-8}"

OUTPUT_DIR="${1:-${SCRIPT_DIR}/P17_JN1_mosaic_${MODE}_b${BUDGET}_seed${SEED}_sigma${START_SIGMA_FRAC}}"

CDR_INDICES=(
  26 27 28 29 30 31 32 33
  51 52 53 54 55 56 57 58
  97 98 99 100 101 102 103 104 105 106 107 108 109
)

cd "${ROOT_DIR}"

echo "========================================"
echo "P17 nanobody - JN.1 RBD guided redesign"
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
echo "refold:            ${REFOLD_NUM_SAMPLES} sample(s), ${REFOLD_SAMPLING_STEPS} steps, batch ${REFOLD_BATCH_SIZE}"
echo "ipSAE PAE cutoff:  ${IPSAE_PAE_CUTOFF}"
echo "RMSD filter:       binder CA <= ${REFOLD_RMSD_THRESHOLD} A"
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
  --binder-chain B \
  --target-chains T \
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
  --weight-esmc "${WEIGHT_ESMC}" \
  --weight-ablang "${WEIGHT_ABLANG}" \
  --clip-gradient-norm "${CLIP_GRADIENT_NORM}" \
  --polish-steps "${POLISH_STEPS}" \
  --polish-batch-size "${POLISH_BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results: ${OUTPUT_DIR}"
