#!/usr/bin/env bash
# Run the whole recovery sequence in order, with each stage gating the next.
#
#   1. smoke     JN.1 chain at minimum cost, real models, minutes. Proves every
#                stage boundary before any real compute. Fatal if it fails:
#                both cluster failures on 2026-10-04 were contract bugs between
#                stages, surfaced after a search rather than before it.
#   2. alpha     Salvage the held-out stage of an existing Alpha control run,
#                reusing its completed searches. Non-fatal and skippable: the
#                JN.1 question does not depend on it, but a passing Alpha
#                control is what makes a JN.1 null result interpretable.
#   3. jn1       The real JN.1 run: population policy, several search seeds.
#
# Stages run sequentially so they never contend for a GPU. Nothing here pulls,
# commits or installs; it only launches work from the current checkout.
#
# None of this establishes binding. Both readouts are OpenDDE's own opinion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ALPHA_RUN=""
SEARCH_SEEDS=""
EDIT_BUDGET=5
SKIP_SMOKE=false
SKIP_ALPHA=false
SKIP_JN1=false
ALLOW_BUSY=false
DRY_RUN=false

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --devices LIST        GPU indices (default: inherited CUDA_VISIBLE_DEVICES or 0-7)
  --alpha-run PATH      Alpha run to salvage (default: newest results/p17_alpha_recovery_*)
  --search-seeds "N N"  JN.1 search seeds (default: one per allocated GPU)
  --edit-budget N       JN.1 WT-relative cap (default: 5)
  --skip-smoke          skip stage 1 (not advised; it is what catches wiring bugs)
  --skip-alpha          skip stage 2
  --skip-jn1            skip stage 3, e.g. to only salvage Alpha
  --allow-busy-gpus     proceed even if a requested GPU already holds a process
  --dry-run             print the plan; create nothing, load nothing
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --alpha-run) ALPHA_RUN="$2"; shift 2 ;;
        --search-seeds) SEARCH_SEEDS="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --skip-smoke) SKIP_SMOKE=true; shift ;;
        --skip-alpha) SKIP_ALPHA=true; shift ;;
        --skip-jn1) SKIP_JN1=true; shift ;;
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO_DIR"
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES//[[:space:]]/}"
N_DEVICES="${#DEVICE_ARRAY[@]}"
SMOKE_DEVICE="${DEVICE_ARRAY[0]}"

# One search seed per allocated GPU unless asked otherwise: with a single
# policy the workers are better spent on replication than on a policy contrast
# the proof of concept does not need (docs/P17_JN1.md section 19.4).
if [[ -z "$SEARCH_SEEDS" ]]; then
    SEARCH_SEEDS="$(seq -s' ' 0 $((N_DEVICES - 1)))"
fi

# Newest Alpha run, unless one was named. `ls -d` ordering is lexicographic,
# which is chronological for these timestamped names.
if [[ -z "$ALPHA_RUN" ]] && ! "$SKIP_ALPHA"; then
    ALPHA_RUN="$(ls -d "$REPO_DIR"/results/p17_alpha_recovery_* 2>/dev/null | tail -1 || true)"
fi

echo "P17 recovery sequence"
echo "  repo:     $REPO_DIR"
echo "  devices:  $DEVICES ($N_DEVICES)"
echo "  stage 1:  $("$SKIP_SMOKE" && echo skipped || echo "JN.1 smoke on device $SMOKE_DEVICE")"
if "$SKIP_ALPHA"; then
    echo "  stage 2:  skipped"
elif [[ -z "$ALPHA_RUN" ]]; then
    echo "  stage 2:  no results/p17_alpha_recovery_* found -- will skip"
else
    echo "  stage 2:  Alpha held-out salvage of $(basename "$ALPHA_RUN")"
fi
echo "  stage 3:  $("$SKIP_JN1" && echo skipped || echo "JN.1 search, seeds [$SEARCH_SEEDS], budget $EDIT_BUDGET")"

# The occupancy check the individual launchers lack. Each worker preallocates
# 90% of its device, so another process on a requested GPU means that worker
# dies rather than erroring cleanly.
if command -v nvidia-smi >/dev/null 2>&1; then
    BUSY=""
    MISSING=""
    for index in "${DEVICE_ARRAY[@]}"; do
        # An absent device prints "No devices were found" on *stdout* and exits
        # nonzero, so the exit code is what distinguishes "does not exist" from
        # "exists and is idle". Counting lines alone reported every absent
        # device as busy.
        if ! PIDS="$(nvidia-smi --id="$index" --query-compute-apps=pid \
                     --format=csv,noheader 2>&1)"; then
            MISSING="$MISSING $index"
            continue
        fi
        COUNT="$(printf '%s' "$PIDS" | grep -c . || true)"
        [[ "${COUNT:-0}" -gt 0 ]] && BUSY="$BUSY $index"
    done
    if [[ -n "$MISSING" ]]; then
        # Not an occupancy problem, so --allow-busy-gpus does not cover it:
        # work cannot be placed on a device that is not there.
        echo "  ERROR: requested device(s)$MISSING do not exist on this host." >&2
        "$DRY_RUN" || exit 2
    fi
    if [[ -n "$BUSY" ]]; then
        echo "  WARNING: device(s)$BUSY already hold a process." >&2
        if ! "$ALLOW_BUSY" && ! "$DRY_RUN"; then
            echo "Workers preallocate 90% of each device. Free them, request a" >&2
            echo "different --devices, or pass --allow-busy-gpus." >&2
            exit 2
        fi
    elif [[ -z "$MISSING" ]]; then
        echo "  gpus:     all $N_DEVICES requested devices exist and are free"
    fi
else
    echo "  WARNING: nvidia-smi unavailable, so occupancy was NOT checked." >&2
    "$ALLOW_BUSY" || "$DRY_RUN" || { echo "Pass --allow-busy-gpus to proceed anyway." >&2; exit 2; }
fi

if "$DRY_RUN"; then
    echo
    echo "Dry run. Commands that would run, in order:"
    "$SKIP_SMOKE" || echo "  bash examples/run_p17_jn1_recovery.sh --smoke --devices $SMOKE_DEVICE"
    if ! "$SKIP_ALPHA" && [[ -n "$ALPHA_RUN" ]]; then
        echo "  $PYTHON_BIN examples/p17_alpha_recovery.py heldout --target alpha \\"
        echo "      --output-dir $ALPHA_RUN \\"
        echo "      --calibration $ALPHA_RUN/calibrate/calibration.json --devices $DEVICES"
    fi
    "$SKIP_JN1" || echo "  bash examples/run_p17_jn1_recovery.sh --devices $DEVICES --search-seeds $SEARCH_SEEDS --edit-budget $EDIT_BUDGET"
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

SUMMARY=()

if ! "$SKIP_SMOKE"; then
    echo
    echo "################ Stage 1/3: JN.1 smoke ################"
    if bash "$SCRIPT_DIR/run_p17_jn1_recovery.sh" --smoke --devices "$SMOKE_DEVICE"; then
        SUMMARY+=("stage 1 smoke: PASSED")
    else
        echo "Smoke run failed. The chain is broken upstream of any real" >&2
        echo "compute; fix that before spending GPU days. Later stages were" >&2
        echo "not launched." >&2
        exit 1
    fi
else
    SUMMARY+=("stage 1 smoke: skipped")
fi

if ! "$SKIP_ALPHA" && [[ -n "$ALPHA_RUN" ]]; then
    echo
    echo "################ Stage 2/3: Alpha held-out salvage ################"
    CALIBRATION="$ALPHA_RUN/calibrate/calibration.json"
    if [[ ! -f "$CALIBRATION" ]]; then
        echo "No $CALIBRATION; cannot build the recovery table. Skipping." >&2
        SUMMARY+=("stage 2 alpha: skipped, no calibration.json")
    elif "$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" heldout \
            --target alpha --output-dir "$ALPHA_RUN" \
            --calibration "$CALIBRATION" --devices "$DEVICES"; then
        SUMMARY+=("stage 2 alpha: PASSED -> $ALPHA_RUN/tables/recovery.csv")
    else
        # Not fatal: the JN.1 question stands on its own, and the completed
        # Alpha searches are untouched and can be salvaged again later.
        echo "Alpha held-out salvage failed. Its searches are intact; retry" >&2
        echo "the heldout stage alone. Continuing to JN.1." >&2
        SUMMARY+=("stage 2 alpha: FAILED, searches intact, see its logs/")
    fi
else
    SUMMARY+=("stage 2 alpha: skipped")
fi

if ! "$SKIP_JN1"; then
    echo
    echo "################ Stage 3/3: JN.1 recovery ################"
    if bash "$SCRIPT_DIR/run_p17_jn1_recovery.sh" --devices "$DEVICES" \
            --search-seeds "$SEARCH_SEEDS" --edit-budget "$EDIT_BUDGET"; then
        SUMMARY+=("stage 3 jn1: PASSED")
    else
        SUMMARY+=("stage 3 jn1: FAILED, see its logs/")
        printf '%s\n' "" "Summary:" "${SUMMARY[@]/#/  }" >&2
        exit 1
    fi
else
    SUMMARY+=("stage 3 jn1: skipped")
fi

echo
echo "################ Done ################"
printf '%s\n' "${SUMMARY[@]/#/  }"
echo
echo "What to read, in order:"
echo "  1. <alpha run>/calibrate/calibration.json  the reference's own measured"
echo "     confidence and pose -- the recovery target, and the section 19.1"
echo "     scale bar reproduced in the gradient path."
echo "  2. <alpha run>/tables/recovery.csv         ipsae_recovery_fraction on"
echo "     held-out seeds. A solution provably existed here."
echo "  3. <jn1 run>/tables/jn1_recovery.csv       ipsae_gain_over_wt and"
echo "     pose_change_vs_wt_A. No solution is known to exist, so read this"
echo "     only alongside a passing Alpha control."
echo
echo "Pose sits in the proposal gradient, so pose improvement is partly"
echo "circular; separating that needs an arm where pose is measured but not"
echo "optimized (docs/P17_JN1.md section 19.6)."
