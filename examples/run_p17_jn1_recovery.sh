#!/usr/bin/env bash
# P17 -> JN.1 gradient recovery -- the question, not the control.
#
# OpenDDE already places WT P17 against JN.1 in the non-binding regime
# (ipSAE 0.000-0.163, pose 22-58 A), so there is no damage stage: the starting
# point exists without being constructed. The search asks whether following
# OpenDDE's own sequence gradient moves those two readouts toward the binding
# regime that the Alpha control measures in this same path.
#
# Read this alongside run_p17_alpha_recovery.sh, not instead of it. No
# solution is known to exist in this feasible set, so a null result here is
# only interpretable when the Alpha control passed: without it, "no recovery"
# cannot be separated from "no reachable solution".
#
# Population policy only, by default. Section 17 keeps population as the
# provisional policy and section 19.4 deprioritizes the population-versus-
# independent comparison -- it varies search memory, which is second-order to
# whether the gradient moves the metrics at all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DEVICES_FROM_ENV=false
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && DEVICES_FROM_ENV=true
SEARCH_SEEDS="0 1 2 3"
POLICIES="population"
EDIT_BUDGET=5
SAMPLING_STEPS=64
AGGREGATION=stable
DTYPE=bf16
MAX_SCORE_CALLS=32
MAX_GRADIENT_CALLS=32
MAX_PROPOSALS=320
OUTPUT_DIR=""
DRY_RUN=false
SKIP_PREP=false
SMOKE=false
SELECTION_SEEDS="0 1"
HELDOUT_SEEDS="101 102 103"

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --devices LIST        GPU indices (default: inherited CUDA_VISIBLE_DEVICES or 0-7)
  --search-seeds "N N"  search seeds (default: 0 1 2 3)
  --policies "P [P]"    population and/or independent (default: population)
  --edit-budget N       WT-relative cap (default: 5)
  --steps N             sampling steps (default: 64; 8 fails backbone checks)
  --skip-prep           skip patch application and cache warm-up
  --smoke               run the whole chain at minimum cost: one worker, one
                        selection seed, one held-out seed, ceilings of 1/1/2.
                        Exercises search -> rescore -> merge -> table with the
                        real models in minutes. Run this once before a real
                        launch: both failures on 2026-10-04 were contract bugs
                        between stages that a smoke run would have surfaced
                        before the search, not after it.
  --output-dir PATH     default: results/p17_jn1_recovery_<timestamp>_<pid>
  --dry-run             print the plan; create nothing, load nothing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; DEVICES_FROM_ENV=false; shift 2 ;;
        --search-seeds) SEARCH_SEEDS="$2"; shift 2 ;;
        --policies) POLICIES="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --steps) SAMPLING_STEPS="$2"; shift 2 ;;
        --aggregation) AGGREGATION="$2"; shift 2 ;;
        --opendde-dtype) DTYPE="$2"; shift 2 ;;
        --max-score-calls) MAX_SCORE_CALLS="$2"; shift 2 ;;
        --max-gradient-calls) MAX_GRADIENT_CALLS="$2"; shift 2 ;;
        --max-proposals) MAX_PROPOSALS="$2"; shift 2 ;;
        --skip-prep) SKIP_PREP=true; shift ;;
        --smoke) SMOKE=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if "$SMOKE"; then
    # Smallest configuration that still traverses every stage boundary. The
    # point is coverage of the handoffs, not a scientific result.
    SEARCH_SEEDS="0"
    POLICIES="population"
    MAX_SCORE_CALLS=1
    MAX_GRADIENT_CALLS=1
    MAX_PROPOSALS=2
    SELECTION_SEEDS="0"
    HELDOUT_SEEDS="101"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    SUFFIX=""
    "$SMOKE" && SUFFIX="_smoke"
    OUTPUT_DIR="$REPO_DIR/results/p17_jn1_recovery${SUFFIX}_$(date +%Y%m%d_%H%M%S)_$$"
fi
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES//[[:space:]]/}"
N_DEVICES="${#DEVICE_ARRAY[@]}"
N_WORKERS=$(( $(echo $POLICIES | wc -w) * $(echo $SEARCH_SEEDS | wc -w) ))

echo "P17 -> JN.1 gradient recovery"
echo "  repo:        $REPO_DIR"
echo "  reference:   $REPO_DIR/P17_JN1.pdb (modeled; chains T target / B binder)"
echo "  start:       WT P17, already non-binding -- no damage stage"
echo "  devices:     $DEVICES ($N_DEVICES)$([[ "$DEVICES_FROM_ENV" == true ]] && echo ' [inherited]')"
echo "  search:      [$POLICIES] x seeds [$SEARCH_SEEDS] = $N_WORKERS workers"
echo "  budget:      $EDIT_BUDGET edits, CDR-only mask (29 of 123 positions)"
echo "  model:       $SAMPLING_STEPS steps, $DTYPE, $AGGREGATION aggregation"
echo "  seeds:       selection [$SELECTION_SEEDS], held-out [$HELDOUT_SEEDS]"
echo "  ceilings:    $MAX_SCORE_CALLS score / $MAX_GRADIENT_CALLS gradient / $MAX_PROPOSALS proposals"
if "$SMOKE"; then
    echo "  MODE:        SMOKE -- exercises every stage at minimum cost."
    echo "               Not a result. Verify it completes, then launch for real."
fi
echo "  output:      $OUTPUT_DIR"

if [[ ! -f "$REPO_DIR/P17_JN1.pdb" ]]; then
    echo "P17_JN1.pdb is absent from this checkout." >&2
    exit 2
fi
if [[ "$SAMPLING_STEPS" -lt 64 ]]; then
    echo "WARNING: 8-step predictions failed backbone checks on every path and" >&2
    echo "have not been retested since the numerical fixes (status doc)." >&2
fi
if (( N_WORKERS > N_DEVICES )); then
    echo "NOTE: $N_WORKERS workers over $N_DEVICES devices runs in batches." >&2
fi

if "$DRY_RUN"; then
    echo
    echo "Dry run. Stages that would execute:"
    echo "  1. patches + cache warm-up (CPU)"
    echo "  2. search/     $N_WORKERS workers, one per GPU"
    echo "  3. heldout/    archived winners on structural seeds [$HELDOUT_SEEDS]"
    echo "  4. tables/jn1_recovery.csv, against this run's own WT baseline"
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"/{search,heldout,logs}
cd "$REPO_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"

if ! "$SKIP_PREP"; then
    echo
    echo "=== Stage 1/3: patches and shared cache warm-up ==="
    for patch in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
        "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$patch.py" \
            >> "$OUTPUT_DIR/logs/patches.log" 2>&1
    done
    echo "Applied five OpenDDE patches (idempotent; see logs/patches.log)."
    # The atom-template cache key embeds the JOpenDDE build id, so it must be
    # built after the patches. Workers within a stage would otherwise race to
    # create it and to download the AbLang2 paired checkpoint, both first-use
    # artifacts. Building once here, on CPU, is what the pose launcher does.
    echo "Warming the atom-template cache and AbLang2 checkpoint on CPU..."
    JAX_PLATFORMS=cpu "$PYTHON_BIN" - <<'PYWARM' >> "$OUTPUT_DIR/logs/cache_warm.log" 2>&1
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "examples"))
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.losses.ablang2 import load_ablang2
from mosaic.structure_prediction import TargetChain
from p17_hallucination_search import load_structure

_, binder_seq, target_seq = load_structure()
model = OpenDDEModelAbag(compute_precision="bf16")
model.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])
load_ablang2()
print("caches ready")
PYWARM
    echo "Caches ready (see logs/cache_warm.log; first build can take minutes)."
fi

echo
echo "=== Stage 2/3 and 3/3: search, then held-out rescoring ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" jn1 \
    --output-dir "$OUTPUT_DIR" \
    --devices "$DEVICES" \
    --search-seeds $SEARCH_SEEDS \
    --policies $POLICIES \
    --selection-seeds $SELECTION_SEEDS \
    --heldout-seeds $HELDOUT_SEEDS \
    --edit-budget "$EDIT_BUDGET" \
    --steps "$SAMPLING_STEPS" --opendde-dtype "$DTYPE" \
    --max-score-calls "$MAX_SCORE_CALLS" \
    --max-gradient-calls "$MAX_GRADIENT_CALLS" \
    --max-proposals "$MAX_PROPOSALS" 2>&1 | tee "$OUTPUT_DIR/logs/search.log"

echo
echo "Completed. Results: $OUTPUT_DIR"
if "$SMOKE"; then
    echo "Smoke run completed every stage. tables/jn1_recovery.csv exists but is"
    echo "not a result: one worker, one seed, two proposals. Launch for real now."
else
    echo "Read tables/jn1_recovery.csv against the Alpha control's measured reference."
fi
