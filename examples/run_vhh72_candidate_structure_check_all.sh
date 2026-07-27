#!/usr/bin/env bash
# Run full OpenDDE diffusion/coordinate structural sanity checks for every row
# in a VHH72 hallucination candidate shortlist, parallelized across GPUs.
#
# Default target is the current curated shortlist:
#   results/hallucination_sweep_discrete_3seed/candidate_shortlist.csv
#
# Each candidate is selected by policy:stop_grad:seed:edit_count and run in its
# own subprocess/output directory. WT is included only in the first job to avoid
# repeated WT refolds in every parallel worker.
#
# Usage:
#   examples/run_vhh72_candidate_structure_check_all.sh [DEVICES] [COMBINED_CSV] [SHORTLIST_CSV] [OUTPUT_ROOT] [SEED] [DRY_RUN]
#
# Args:
#   DEVICES       comma-separated physical GPU ids, e.g. 4,5,6,7.
#                 Use "inherit" to keep caller's CUDA_VISIBLE_DEVICES and run
#                 jobs one at a time. Default: 4,5,6,7
#   COMBINED_CSV  combined hallucination Pareto CSV. Default:
#                 results/hallucination_sweep_discrete_3seed/combined.csv
#   SHORTLIST_CSV candidate shortlist CSV with policy/stop_grad/seed/edit_count.
#                 Default: results/hallucination_sweep_discrete_3seed/candidate_shortlist.csv
#   OUTPUT_ROOT   root output directory. Default:
#                 results/hallucination_sweep_discrete_3seed/structures_shortlist
#   SEED          OpenDDE diffusion RNG seed. Default: 0
#   DRY_RUN       1 prints planned commands without running OpenDDE. Default: 0
#
# Example:
#   examples/run_vhh72_candidate_structure_check_all.sh 4,5,6,7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICES="${1:-4,5,6,7}"
COMBINED_CSV="${2:-results/hallucination_sweep_discrete_3seed/combined.csv}"
SHORTLIST_CSV="${3:-results/hallucination_sweep_discrete_3seed/candidate_shortlist.csv}"
OUTPUT_ROOT="${4:-results/hallucination_sweep_discrete_3seed/structures_shortlist}"
SEED="${5:-0}"
DRY_RUN="${6:-0}"

if [[ ! -f "$COMBINED_CSV" ]]; then
    echo "ERROR: combined CSV not found: $COMBINED_CSV" >&2
    exit 1
fi
if [[ ! -f "$SHORTLIST_CSV" ]]; then
    echo "ERROR: shortlist CSV not found: $SHORTLIST_CSV" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mapfile -t JOB_LINES < <(
    .venv/bin/python - "$SHORTLIST_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with open(path) as f:
    rows = list(csv.DictReader(f))
for row in rows:
    policy = row["policy"]
    stop_grad = row["stop_grad"]
    seed = row.get("seed", "")
    edit_count = row["edit_count"]
    if seed == "":
        raise SystemExit(
            f"{path} lacks seed for row {row}; run seed-aware scoring/selection first"
        )
    select = f"{policy}:{stop_grad}:{seed}:{edit_count}"
    tag = f"{policy}_stopgrad{stop_grad}_seed{seed}_edit{edit_count}"
    print(f"{select}\t{tag}")
PY
)

if [[ "${#JOB_LINES[@]}" -eq 0 ]]; then
    echo "ERROR: no candidates found in $SHORTLIST_CSV" >&2
    exit 1
fi

if [[ "$DEVICES" == "inherit" ]]; then
    DEVICE_LIST=("inherit")
else
    IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
fi
MAX_PARALLEL="${#DEVICE_LIST[@]}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "ERROR: no devices parsed from '$DEVICES'" >&2
    exit 1
fi

echo "=== VHH72 full-OpenDDE shortlist structure check ==="
echo "repo root:      $REPO_ROOT"
echo "devices:        $DEVICES"
echo "combined csv:   $COMBINED_CSV"
echo "shortlist csv:  $SHORTLIST_CSV"
echo "output root:    $OUTPUT_ROOT"
echo "seed:           $SEED"
echo "candidates:     ${#JOB_LINES[@]}"
echo "max parallel:   $MAX_PARALLEL"
echo "dry run:        $DRY_RUN"
echo

PIDS=()
TAGS=()
LOGS=()
FAILURES=0

launch_job() {
    local job_index="$1"
    local device="$2"
    local select="$3"
    local tag="$4"
    local out_dir="$OUTPUT_ROOT/$tag"
    local log_path="$out_dir/run.log"
    mkdir -p "$out_dir"

    local include_wt_flag="--no-include-wt"
    if [[ "$job_index" == "0" ]]; then
        include_wt_flag="--include-wt"
    fi

    local cmd=(
        .venv/bin/python examples/vhh72_predict_hallucination_structures.py
        --combined-csv "$COMBINED_CSV"
        --select "$select"
        --output-dir "$out_dir"
        --seed "$SEED"
        "$include_wt_flag"
    )

    echo "[launch] $tag device=$device select=$select include_wt=$include_wt_flag"
    echo "         log=$log_path"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '         cmd='
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi

    if [[ "$device" == "inherit" ]]; then
        PYTHONUNBUFFERED=1 "${cmd[@]}" >"$log_path" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 "${cmd[@]}" >"$log_path" 2>&1 &
    fi
    PIDS+=("$!")
    TAGS+=("$tag")
    LOGS+=("$log_path")
}

wait_batch() {
    local i
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[done] ${TAGS[$i]} ok log=${LOGS[$i]}"
        else
            echo "[done] ${TAGS[$i]} FAILED log=${LOGS[$i]}" >&2
            FAILURES=$((FAILURES + 1))
        fi
    done
    PIDS=()
    TAGS=()
    LOGS=()
}

for job_index in "${!JOB_LINES[@]}"; do
    line="${JOB_LINES[$job_index]}"
    select="${line%%$'\t'*}"
    tag="${line#*$'\t'}"
    device="${DEVICE_LIST[$((job_index % MAX_PARALLEL))]}"
    launch_job "$job_index" "$device" "$select" "$tag"

    if [[ "$DRY_RUN" != "1" && "${#PIDS[@]}" -eq "$MAX_PARALLEL" ]]; then
        wait_batch
    fi
done

if [[ "$DRY_RUN" != "1" && "${#PIDS[@]}" -gt 0 ]]; then
    wait_batch
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "dry run complete; no OpenDDE jobs launched."
    exit 0
fi

if [[ "$FAILURES" -gt 0 ]]; then
    echo "ERROR: $FAILURES structure job(s) failed; inspect run.log files under $OUTPUT_ROOT" >&2
    exit 1
fi

.venv/bin/python - "$OUTPUT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/real_structure_summary.csv")):
    with open(path) as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["job_dir"] = str(path.parent)
            rows.append(row)

if not rows:
    raise SystemExit(f"no per-job real_structure_summary.csv files found under {root}")

fieldnames = [
    "job_dir", "tag", "policy", "stop_grad", "edit_count",
    "real_binder_pose_rmsd", "mean_plddt", "cif",
]
out = root / "real_structure_summary_all.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
print(f"wrote combined summary: {out}")
PY

echo
echo "=== done ==="
echo "combined summary: $OUTPUT_ROOT/real_structure_summary_all.csv"
echo "per-candidate outputs: $OUTPUT_ROOT/*/"
