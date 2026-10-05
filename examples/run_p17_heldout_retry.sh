#!/usr/bin/env bash
# Re-run only the held-out rescoring stage for a finished search directory.
#
# The searches are the expensive part -- hours on eight GPUs -- and held-out
# rescoring is minutes. Every failure of this project so far has been a
# contract bug between stages, which means the searches were already correct
# and only the stage after them needed fixing. Re-running the whole arm would
# discard that work.
#
# This happened three times:
#   2026-10-04  p17_rescore_winners.py loaded the JN.1 reference for an Alpha
#               archive, 195 aa against 184.
#   2026-10-04  the archived `binder_sequence` is the search's *start*, so
#               comparing it to the reference refused every recovery run.
#   2026-10-05  the decoy arm's archived target is the decoy by design, and
#               the gate demanded it match the reference exactly.
#
# The retry writes into a `_retry`-suffixed stage directory, so the failed
# attempt stays on disk for comparison rather than being overwritten.
#
# `--target` is inferred from the directory name where it can be: an `alpha5`
# or `alpha`-named arm needs the Alpha reference and a calibration file, and
# everything else is scored against JN.1. Pass it explicitly to override.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"

RUN=""
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TARGET=""
SUFFIX="_retry"
DTYPE=bf16
AGGREGATION=stable
CALIBRATION=""
SELECT_RUNG=""
EDIT_BUDGET=""
ALLOW_BUSY=false
SKIP_PREP=false
DRY_RUN=false

usage() {
    sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Usage: bash examples/run_p17_heldout_retry.sh --run PATH [options]

  --run PATH         the arm directory holding search/, e.g.
                     results/p17_followups_20261005_061410/decoy

Options:
  --target jn1|alpha which reference to score against (default: inferred)
  --devices LIST     GPU indices (default: inherited CUDA_VISIBLE_DEVICES or 0-7)
  --suffix STR       stage directory suffix (default: _retry)
  --calibration PATH required for --target alpha
  --select-rung N    passed through for --target alpha
  --edit-budget N    passed through for --target alpha
  --opendde-dtype    fp32|bf16 (default: bf16)
  --skip-prep        skip patch application and cache warm-up
  --allow-busy-gpus  run even though a requested device holds a process
  --dry-run          print the plan; create nothing, load nothing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --calibration) CALIBRATION="$2"; shift 2 ;;
        --select-rung) SELECT_RUNG="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --opendde-dtype) DTYPE="$2"; shift 2 ;;
        --skip-prep) SKIP_PREP=true; shift ;;
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$RUN" ]]; then
    echo "--run is required." >&2; usage >&2; exit 2
fi
cd "$REPO_DIR"
RUN="$(cd "$RUN" 2>/dev/null && pwd || true)"
if [[ -z "$RUN" ]]; then
    echo "No such run directory." >&2; exit 2
fi

# The searches are the precondition: this stage reads their summary.json, so a
# directory with none of them cannot be retried and the driver would say so
# only after loading a model.
mapfile -t COMPLETED < <(find "$RUN/search" -mindepth 2 -maxdepth 2 \
    -name summary.json 2>/dev/null | sort)
if [[ ${#COMPLETED[@]} -eq 0 ]]; then
    echo "No completed search runs under $RUN/search." >&2
    echo "Held-out rescoring needs finished searches; nothing to retry." >&2
    exit 2
fi

if [[ -z "$TARGET" ]]; then
    case "$(basename "$RUN")" in
        alpha*) TARGET=alpha ;;
        *)      TARGET=jn1 ;;
    esac
    INFERRED=" (inferred from the directory name)"
else
    INFERRED=""
fi
if [[ "$TARGET" == alpha && -z "$CALIBRATION" ]]; then
    # The Alpha table is written against the measured reference and damaged
    # start from that run's calibration, so it cannot be reconstructed here.
    GUESS="$RUN/calibrate/calibration.json"
    if [[ -f "$GUESS" ]]; then
        CALIBRATION="$GUESS"
    else
        echo "--target alpha needs --calibration; none found at $GUESS." >&2
        exit 2
    fi
fi
if [[ "$TARGET" != alpha && "$TARGET" != jn1 ]]; then
    echo "--target must be jn1 or alpha." >&2; exit 2
fi

echo "P17 held-out rescoring retry"
echo "  run:      $RUN"
echo "  searches: ${#COMPLETED[@]} completed"
echo "  target:   $TARGET$INFERRED"
echo "  stage:    heldout$SUFFIX/  (the failed attempt is preserved)"
echo "  devices:  $DEVICES"
[[ -n "$CALIBRATION" ]] && echo "  calib:    $CALIBRATION"

# Occupancy, with the same absent-versus-busy distinction as the other
# launchers: each worker preallocates most of its device.
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES// /}"
if command -v nvidia-smi >/dev/null 2>&1; then
    BUSY=""; MISSING=""
    for index in "${DEVICE_ARRAY[@]}"; do
        [[ -n "$index" ]] || continue
        if ! PIDS="$(nvidia-smi --id="$index" --query-compute-apps=pid \
                     --format=csv,noheader 2>&1)"; then
            MISSING="$MISSING $index"; continue
        fi
        COUNT="$(printf '%s' "$PIDS" | grep -c . || true)"
        [[ "${COUNT:-0}" -gt 0 ]] && BUSY="$BUSY $index"
    done
    if [[ -n "$MISSING" ]]; then
        echo "  ERROR: requested device(s)$MISSING do not exist on this host." >&2
        "$DRY_RUN" || exit 2
    fi
    if [[ -n "$BUSY" ]]; then
        echo "  WARNING: device(s)$BUSY already hold a process." >&2
        if ! "$ALLOW_BUSY" && ! "$DRY_RUN"; then
            echo "Workers preallocate most of each device. Free them, request" >&2
            echo "different --devices, or pass --allow-busy-gpus." >&2
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

COMMAND=("$PYTHON_BIN" "$DRIVER" heldout
         --output-dir "$RUN" --target "$TARGET"
         --devices "$DEVICES" --suffix "$SUFFIX" --opendde-dtype "$DTYPE")
[[ -n "$CALIBRATION" ]] && COMMAND+=(--calibration "$CALIBRATION")
[[ -n "$SELECT_RUNG" ]] && COMMAND+=(--select-rung "$SELECT_RUNG")
[[ -n "$EDIT_BUDGET" ]] && COMMAND+=(--edit-budget "$EDIT_BUDGET")

if "$DRY_RUN"; then
    echo
    echo "Dry run. Would run:"
    printf '  %q' "${COMMAND[@]}"; echo
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

mkdir -p "$RUN/logs"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"

if ! "$SKIP_PREP"; then
    # The workers run in fresh processes, so they need the patches applied in
    # this checkout regardless of what the original launcher did.
    for patch in outer_product_mean structural_token_expander bf16_dtype \
                 aggregation padding; do
        "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$patch.py" \
            >> "$RUN/logs/patches_retry.log" 2>&1
    done
    echo "Applied five OpenDDE patches (idempotent; see logs/patches_retry.log)."
    # The atom-template cache key embeds the JOpenDDE build id, so it is built
    # after the patches. Eight workers would otherwise race to create it and
    # to download the AbLang2 paired checkpoint.
    echo "Warming the atom-template cache and AbLang2 checkpoint on CPU..."
    JAX_PLATFORMS=cpu "$PYTHON_BIN" - \
        >> "$RUN/logs/cache_warm_retry.log" 2>&1 <<'PYWARM'
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
    echo "Caches ready (see logs/cache_warm_retry.log)."
fi

echo
"${COMMAND[@]}"
STATUS=$?

echo
if [[ $STATUS -eq 0 ]]; then
    echo "Read, in order:"
    if [[ "$TARGET" == jn1 ]]; then
        echo "  1. $RUN/tables/jn1_recovery.csv -- gains over this run's own WT"
        echo "     baseline, on held-out seeds. For the decoy arm, compare"
        echo "     mean_ipsae against the real JN.1 arm's: comparable confidence"
        echo "     means that arm's gains were not about its target."
        echo "  2. $RUN/tables/jn1_context.json -- what the numbers do not show."
    else
        echo "  1. $RUN/tables/recovery.csv -- recovered fraction against this"
        echo "     run's own measured reference, not a literature number."
    fi
    echo "  3. $RUN/heldout$SUFFIX/ -- the per-shard predictions behind them."
    echo "Nothing here establishes binding."
fi
exit $STATUS
