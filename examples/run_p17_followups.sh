#!/usr/bin/env bash
# The four follow-up arms from docs/P17_JN1.md section 20.11, in priority order.
#
#   1. decoy     NEGATIVE CONTROL. The identical search against a target P17
#                should fail on. Every other control asks whether a solution
#                exists; none asks whether this pipeline reports success
#                regardless of the target. If the decoy reaches the confidence
#                section 20.5 reports, that result says nothing about JN.1.
#                Cheapest thing that can overturn the headline, so it runs first.
#                The decoy keeps the real target everywhere except its contact
#                epitope, which is permuted. Measured mean target pLDDT on
#                2026-10-05: epitope-scrambled 0.838, real JN.1 0.890, whole-
#                chain shuffle 0.405, IL7RA trimmed to 184 aa 0.425. Only the
#                first is usable -- a decoy the predictor cannot fold scores
#                low for reasons that have nothing to do with binding.
#   2. budget    All eight JN.1 runs stopped at their score-call ceiling with
#                two still climbing (section 20.8), so the result is cut off,
#                not converged. Raises the ceiling and warms the chain.
#   3. posezero  --weight-pose 0. Pose still measured, no longer guiding
#                proposals. Separates a real pose gain from the objective
#                reporting on itself (section 19.6 item 1).
#   4. alpha5    The 5-edit Alpha rung, where both axes have room, unlike the
#                2-edit rung that section 20.3 shows was a one-substitution
#                problem.
#
# Arms are independent; each can be skipped. All record the per-(position,
# residue) delta matrix, which cannot be recovered after a run.
#
# None of these establishes binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SEEDS="0 1 2 3"
EDIT_BUDGET=5
STEPS=64
DTYPE=bf16
AGGREGATION=stable
# Budget arm: the measured ceiling was 32 scored sequences per worker.
BUDGET_SCORE_CALLS=300
BUDGET_GRADIENT_CALLS=150
BUDGET_PROPOSALS=3000
BUDGET_ENTROPY=0.8
BUDGET_ACCEPT=0.05
ARMS="decoy budget posezero alpha5"
ALLOW_BUSY=false
# Empty means the default epitope scramble, which is what should be used. A
# structure here switches to a real unrelated protein instead; IL7RA does not
# fold in this predictor (0.425 pLDDT), so expect that to fail its own gate.
DECOY_STRUCTURE=""
DECOY_CHAIN=A
ALPHA_RUN=""
STAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --devices LIST     GPU indices (default: inherited CUDA_VISIBLE_DEVICES or 0-7)
  --arms "A B"       subset of: decoy budget posezero alpha5 (default: all four)
  --seeds "N N"      search seeds per arm (default: 0 1 2 3)
  --alpha-run PATH   run holding the damage ladder for alpha5 (default: newest)
  --decoy-structure PATH  use a real unrelated protein instead of the default
                     epitope scramble. Its own pLDDT is checked first, and no
                     local candidate passes that check.
  --decoy-chain ID   chain to take it from (default: A)
  --allow-busy-gpus  launch even though a requested device holds a process
  --dry-run          print the plan; create nothing, load nothing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --alpha-run) ALPHA_RUN="$2"; shift 2 ;;
        --decoy-structure) DECOY_STRUCTURE="$2"; shift 2 ;;
        --decoy-chain) DECOY_CHAIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO_DIR"
if [[ -n "$DECOY_STRUCTURE" && ! -f "$DECOY_STRUCTURE" ]]; then
    echo "Decoy structure not found: $DECOY_STRUCTURE" >&2; exit 2
fi
if [[ -n "$DECOY_STRUCTURE" ]]; then
    DECOY_ARGS="--decoy-structure $DECOY_STRUCTURE --decoy-structure-chain $DECOY_CHAIN"
    DECOY_LABEL="$DECOY_STRUCTURE chain $DECOY_CHAIN, trimmed to the target length"
else
    DECOY_ARGS="--decoy-mode epitope"
    DECOY_LABEL="real JN.1 target with its contact epitope permuted"
fi
if [[ -z "$ALPHA_RUN" ]]; then
    ALPHA_RUN="$(ls -d "$REPO_DIR"/results/p17_alpha_recovery_* 2>/dev/null | tail -1 || true)"
fi
OUT_ROOT="$REPO_DIR/results/p17_followups_$STAMP"

echo "P17 follow-up arms"
echo "  repo:     $REPO_DIR"
echo "  devices:  $DEVICES"
echo "  arms:     $ARMS"
echo "  seeds:    $SEEDS   budget $EDIT_BUDGET edits, $STEPS steps, $DTYPE"
echo "  output:   $OUT_ROOT/<arm>"
echo "  saliency: recorded for every arm (tables/saliency.csv)"
echo "  decoy:    $DECOY_LABEL"

COMMON=(--devices "$DEVICES" --search-seeds $SEEDS --steps "$STEPS"
        --opendde-dtype "$DTYPE" --save-saliency)

plan_decoy()    { echo "$PYTHON_BIN $DRIVER decoy --output-dir $OUT_ROOT/decoy --edit-budget $EDIT_BUDGET $DECOY_ARGS ${COMMON[*]}"; }
plan_budget()   { echo "$PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/budget --edit-budget $EDIT_BUDGET --max-score-calls $BUDGET_SCORE_CALLS --max-gradient-calls $BUDGET_GRADIENT_CALLS --max-proposals $BUDGET_PROPOSALS --target-entropy $BUDGET_ENTROPY --acceptance-temperature $BUDGET_ACCEPT ${COMMON[*]}"; }
plan_posezero() { echo "$PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/posezero --edit-budget $EDIT_BUDGET --weight-pose 0 ${COMMON[*]}"; }
# alpha5 reads the damage ladder and calibration from the completed 2-edit run
# but writes to a fresh directory: that run's search/ already holds
# population_seed0 and the search refuses a pre-existing output directory.
plan_alpha5()   { echo "$PYTHON_BIN $DRIVER search --output-dir $OUT_ROOT/alpha5 --ladder $ALPHA_RUN/ladder/ladder.json --calibration $ALPHA_RUN/calibrate/calibration.json --select-rung 5 --edit-budget 5 ${COMMON[*]}"; }

# Occupancy. Each worker preallocates most of its device, so another process
# on a requested GPU makes that worker die on allocation rather than erroring
# usefully. This launcher had no such check until 2026-10-05.
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES// /}"
if command -v nvidia-smi >/dev/null 2>&1; then
    BUSY=""
    MISSING=""
    for index in "${DEVICE_ARRAY[@]}"; do
        [[ -n "$index" ]] || continue
        # An absent device prints "No devices were found" on *stdout* and exits
        # nonzero, so the exit code is what distinguishes "does not exist" from
        # "exists and is idle". Counting lines alone called every absent device
        # busy, which is the bug this check had in run_p17_recovery_all.sh.
        if ! PIDS="$(nvidia-smi --id="$index" --query-compute-apps=pid \
                     --format=csv,noheader 2>&1)"; then
            MISSING="$MISSING $index"
            continue
        fi
        COUNT="$(printf '%s' "$PIDS" | grep -c . || true)"
        [[ "${COUNT:-0}" -gt 0 ]] && BUSY="$BUSY $index"
    done
    if [[ -n "$MISSING" ]]; then
        # Not an occupancy problem, so --allow-busy-gpus does not cover it.
        echo "  ERROR: requested device(s)$MISSING do not exist on this host." >&2
        "$DRY_RUN" || exit 2
    fi
    if [[ -n "$BUSY" ]]; then
        echo "  WARNING: device(s)$BUSY already hold a process." >&2
        if ! "$ALLOW_BUSY" && ! "$DRY_RUN"; then
            echo "Workers preallocate most of each device, so they would die on" >&2
            echo "allocation. Wait for the current run, request a different" >&2
            echo "--devices, or pass --allow-busy-gpus." >&2
            exit 2
        fi
    elif [[ -z "$MISSING" ]]; then
        echo "  gpus:     all ${#DEVICE_ARRAY[@]} requested devices exist and are free"
    fi
else
    echo "  WARNING: nvidia-smi unavailable, so occupancy was NOT checked." >&2
    "$ALLOW_BUSY" || "$DRY_RUN" || {
        echo "Pass --allow-busy-gpus to proceed anyway." >&2; exit 2; }
fi

if "$DRY_RUN"; then
    echo
    echo "Dry run. Commands, in order:"
    for arm in $ARMS; do
        if [[ "$arm" == "alpha5" && -z "$ALPHA_RUN" ]]; then
            echo "  alpha5: SKIPPED, no results/p17_alpha_recovery_* found"; continue
        fi
        echo "  [$arm]"; echo "    $(plan_$arm)"
    done
    echo
    echo "Note: alpha5 reuses the completed run's ladder and calibration but"
    echo "writes to a fresh directory, because that run's search/ already"
    echo "holds the 2-edit arms and the search refuses an existing output dir."
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

mkdir -p "$OUT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"

for patch in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
    "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$patch.py" \
        >> "$OUT_ROOT/patches.log" 2>&1
done
echo "Applied five OpenDDE patches."

SUMMARY=()
for arm in $ARMS; do
    if [[ "$arm" == "alpha5" && -z "$ALPHA_RUN" ]]; then
        SUMMARY+=("$arm: skipped, no Alpha run found"); continue
    fi
    echo
    echo "################ Arm: $arm ################"
    # Arms are independent, so one failure must not cancel the rest.
    if eval "$(plan_$arm)"; then
        SUMMARY+=("$arm: PASSED")
    else
        SUMMARY+=("$arm: FAILED, see its logs/")
        echo "Arm $arm failed; continuing with the remaining arms." >&2
    fi
done

echo
echo "################ Done ################"
printf '%s\n' "${SUMMARY[@]/#/  }"
echo
echo "Read, in order:"
echo "  1. $OUT_ROOT/decoy/fold_check.json  -- can the predictor fold the decoy"
echo "     target on its own (mean_target_plddt)? If not, its low interface score"
echo "     proves nothing and the arm is void. Ignore target_aligned_rmsd_A here:"
echo "     it measures shape difference from JN.1, not whether the decoy folded."
echo "  2. $OUT_ROOT/decoy/tables/jn1_recovery.csv vs the real JN.1 run's."
echo "     Comparable confidence means that run's gains are not about its target."
echo "  3. $OUT_ROOT/budget/   -- where the search actually plateaus."
echo "  4. $OUT_ROOT/posezero/ -- pose gain with pose out of the gradient."
echo "  5. tables/saliency.csv in each -- per-position first-order deltas."
echo "     These show where the gradient points, not why a residue is good."
