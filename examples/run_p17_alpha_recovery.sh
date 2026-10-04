#!/usr/bin/env bash
# P17+Alpha gradient recovery control -- docs/P17_JN1.md section 19.3B.
#
# Why this control: P17 -> JN.1 is not known to have a solution, so a null
# result there cannot separate a bad gradient from an empty feasible set.
# Damaging P17+Alpha and measuring recovery has a known answer by
# construction -- the sequence you started from -- so both outcomes inform.
#
# Stages, each gated on the last:
#   ladder/      damage ladder, fixed and recorded before any scoring (CPU)
#   calibrate/   forward score the reference and every rung; which rungs
#                actually reached the non-binding regime
#   search/      recovery searches from the selected rung
#   heldout/     unseen structural seeds for the archived winners
#
# A passing run establishes that following OpenDDE's sequence gradient does or
# does not restore its own confidence and pose metrics. It establishes nothing
# about binding, affinity or biology.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DEVICES_FROM_ENV=false
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && DEVICES_FROM_ENV=true
EDITS="2 5 8 12"
DAMAGE_SEED=0
SEARCH_SEEDS="0 1"
SELECT_RUNG=""
SAMPLING_STEPS=64
AGGREGATION=stable
DTYPE=bf16
EDIT_BUDGET=""
MAX_SCORE_CALLS=32
MAX_GRADIENT_CALLS=32
MAX_PROPOSALS=320
OUTPUT_DIR=""
DRY_RUN=false
SKIP_CALIBRATION=false

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --devices LIST        GPU indices (default: inherited CUDA_VISIBLE_DEVICES or 0-7)
  --edits "N N N"       damage ladder edit counts (default: 2 5 8 12)
  --damage-seed N       damage RNG (default: 0)
  --select-rung N       skip calibration's choice and search from this edit count
  --search-seeds "N N"  search seeds per arm (default: 0 1)
  --edit-budget N       search edit cap (default: the selected rung's edit count)
  --steps N             sampling steps (default: 64; 8 fails backbone checks)
  --skip-calibration    reuse an existing calibrate/ stage in --output-dir
  --output-dir PATH     default: results/p17_alpha_recovery_<timestamp>_<pid>
  --dry-run             print the plan; create nothing, load nothing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; DEVICES_FROM_ENV=false; shift 2 ;;
        --edits) EDITS="$2"; shift 2 ;;
        --damage-seed) DAMAGE_SEED="$2"; shift 2 ;;
        --select-rung) SELECT_RUNG="$2"; shift 2 ;;
        --search-seeds) SEARCH_SEEDS="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --steps) SAMPLING_STEPS="$2"; shift 2 ;;
        --aggregation) AGGREGATION="$2"; shift 2 ;;
        --opendde-dtype) DTYPE="$2"; shift 2 ;;
        --max-score-calls) MAX_SCORE_CALLS="$2"; shift 2 ;;
        --max-gradient-calls) MAX_GRADIENT_CALLS="$2"; shift 2 ;;
        --max-proposals) MAX_PROPOSALS="$2"; shift 2 ;;
        --skip-calibration) SKIP_CALIBRATION=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_DIR/results/p17_alpha_recovery_$(date +%Y%m%d_%H%M%S)_$$"
fi
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES//[[:space:]]/}"
N_DEVICES="${#DEVICE_ARRAY[@]}"

echo "P17+Alpha gradient recovery control"
echo "  repo:        $REPO_DIR"
echo "  reference:   $REPO_DIR/P17_Alpha.pdb (experimental 8GZ5; chains A target / B binder)"
echo "  devices:     $DEVICES ($N_DEVICES)$([[ "$DEVICES_FROM_ENV" == true ]] && echo ' [inherited]')"
echo "  damage:      edits [$EDITS], seed $DAMAGE_SEED"
echo "  search:      seeds [$SEARCH_SEEDS], steps $SAMPLING_STEPS, $DTYPE, $AGGREGATION aggregation"
echo "  ceilings:    $MAX_SCORE_CALLS score / $MAX_GRADIENT_CALLS gradient / $MAX_PROPOSALS proposals"
echo "  output:      $OUTPUT_DIR"

if [[ ! -f "$REPO_DIR/P17_Alpha.pdb" ]]; then
    echo "P17_Alpha.pdb is absent from this checkout." >&2
    exit 2
fi
if [[ "$SAMPLING_STEPS" -lt 64 ]]; then
    echo "WARNING: 8-step predictions failed backbone checks on every path and" >&2
    echo "have not been retested since the numerical fixes (status doc)." >&2
fi

if "$DRY_RUN"; then
    echo
    echo "Dry run. Stages that would execute:"
    echo "  1. ladder/      $PYTHON_BIN examples/p17_alpha_reference.py --edits $EDITS --damage-seed $DAMAGE_SEED"
    echo "  2. calibrate/   1 reference + $(echo $EDITS | wc -w) damaged forward scorings, sharded over $N_DEVICES GPUs"
    echo "  3. search/      $(echo $SEARCH_SEEDS | wc -w) seed(s) x {independent, population}, one worker per GPU"
    echo "  4. heldout/     archived winners on structural seeds 101/102/103"
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"/{ladder,calibrate,search,heldout,logs}
cd "$REPO_DIR"

# Allocator settings that the H200 forward controls ran successfully with.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"

echo
echo "=== Stage 1/4: damage ladder (CPU, no model) ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_reference.py" \
    --edits $EDITS --damage-seed "$DAMAGE_SEED" \
    --out "$OUTPUT_DIR/ladder/ladder.json" | tee "$OUTPUT_DIR/logs/ladder.log"

echo
echo "=== Stage 2/4: dependency patches ==="
for patch in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
    "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$patch.py" \
        >> "$OUTPUT_DIR/logs/patches.log" 2>&1
done
echo "Applied five OpenDDE patches (idempotent; see logs/patches.log)."

echo
echo "=== Stage 3/4: damage calibration (forward scoring) ==="
if "$SKIP_CALIBRATION" && [[ -f "$OUTPUT_DIR/calibrate/calibration.json" ]]; then
    echo "Reusing $OUTPUT_DIR/calibrate/calibration.json"
else
    "$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" calibrate \
        --ladder "$OUTPUT_DIR/ladder/ladder.json" \
        --output-dir "$OUTPUT_DIR/calibrate" \
        --devices "$DEVICES" --steps "$SAMPLING_STEPS" --opendde-dtype "$DTYPE" \
        2>&1 | tee "$OUTPUT_DIR/logs/calibrate.log"
fi

echo
echo "=== Stage 4/4: recovery search and held-out rescoring ==="
SELECT_ARGS=()
[[ -n "$SELECT_RUNG" ]] && SELECT_ARGS+=(--select-rung "$SELECT_RUNG")
[[ -n "$EDIT_BUDGET" ]] && SELECT_ARGS+=(--edit-budget "$EDIT_BUDGET")
"$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" search \
    --ladder "$OUTPUT_DIR/ladder/ladder.json" \
    --calibration "$OUTPUT_DIR/calibrate/calibration.json" \
    --output-dir "$OUTPUT_DIR" \
    --devices "$DEVICES" --search-seeds $SEARCH_SEEDS \
    --steps "$SAMPLING_STEPS" --opendde-dtype "$DTYPE" \
    --max-score-calls "$MAX_SCORE_CALLS" \
    --max-gradient-calls "$MAX_GRADIENT_CALLS" \
    --max-proposals "$MAX_PROPOSALS" \
    "${SELECT_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/search.log"

echo
echo "Completed. Results: $OUTPUT_DIR"
echo "Read tables/recovery.csv against the section 19.1 scale bar."
