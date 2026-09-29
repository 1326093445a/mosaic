#!/usr/bin/env bash
# One search process per GPU; no distributed gradients within a search.
# Paths follow this checkout, including /storage/frank/mosaic on the H200 node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash examples/run_p17_confidence_search_multi_gpu.sh [OPTIONS]

  --mode pilot|smoke   pilot (default): 4 seeds/policy, width 4, 32 score calls,
                      32 gradient calls, 320 proposals per run.
                      smoke: 1 seed/policy, width 2, 2 score calls,
                      1 gradient call, 2 proposals per run.
  --devices CSV       GPU IDs/UUIDs; defaults to CUDA_VISIBLE_DEVICES if set,
                      otherwise 0,1,2,3,4,5,6,7. Use allocated GPUs only.
  --num-seeds N       Override search seeds per policy (0 through N-1).
  --output-dir PATH   Fresh batch directory; relative paths use the repo root.
  --dry-run           Print commands without patches, files, or model execution.
  -h, --help          Show this help.

Both policies use full gradients, edit cap 5, 8 sampling steps, proposal-model
seed 0 and selection seed 0. Loss weights use the Python runner's defaults.
Jobs run in batches with at most one process per listed GPU. The launcher
waits for all workers and exits nonzero if a worker fails; it does not resume.
Requires the checkout's .venv and model assets available on the cluster.

Examples on the H200 node:
  cd /storage/frank/mosaic
  bash examples/run_p17_confidence_search_multi_gpu.sh --dry-run
  bash examples/run_p17_confidence_search_multi_gpu.sh --mode smoke --devices 0,1
  bash examples/run_p17_confidence_search_multi_gpu.sh
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MODE=pilot
DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
NUM_SEEDS=""
OUTPUT_DIR=""
DRY_RUN=0
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --mode|--devices|--num-seeds|--output-dir)
            (( $# >= 2 )) || die "$1 requires a value"
            case "$1" in
                --mode) MODE="$2" ;;
                --devices) DEVICES="$2" ;;
                --num-seeds) NUM_SEEDS="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

case "$MODE" in
    pilot) DEFAULT_SEEDS=4; WIDTH=4; SCORE_CALLS=32; GRADIENT_CALLS=32; PROPOSALS=320 ;;
    smoke) DEFAULT_SEEDS=1; WIDTH=2; SCORE_CALLS=2; GRADIENT_CALLS=1; PROPOSALS=2 ;;
    *) die "--mode must be pilot or smoke" ;;
esac
NUM_SEEDS="${NUM_SEEDS:-$DEFAULT_SEEDS}"
[[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]] || die "--num-seeds must be a positive integer"
[[ -n "$DEVICES" && "$DEVICES" != ,* && "$DEVICES" != *, && "$DEVICES" != *,,* ]] \
    || die "--devices must be a nonempty comma-separated list"
IFS=',' read -r -a GPU_IDS <<< "$DEVICES"
declare -A SEEN_GPUS=()
for device in "${GPU_IDS[@]}"; do
    [[ "$device" =~ ^[0-9]+$ || "$device" =~ ^(GPU-|MIG-)[A-Za-z0-9/-]+$ ]] \
        || die "Invalid GPU ID/UUID: $device"
    [[ ! ${SEEN_GPUS[$device]+present} ]] || die "Duplicate GPU: $device"
    SEEN_GPUS[$device]=1
done
OUTPUT_DIR="${OUTPUT_DIR:-results/p17_confidence_${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || die "Output directory already exists: $OUTPUT_DIR"
PYTHON="$REPO_ROOT/.venv/bin/python"
COMMON_ARGS=(--proposal-path full --proposal-model-seed 0 --selection-seeds 0
    --width "$WIDTH" --edit-budget 5 --sampling-steps 8
    --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
    --max-proposals "$PROPOSALS")
PATCHES=(patches/patch_jopendde_outer_product_mean.py
    patches/patch_jopendde_structural_token_expander.py
    patches/patch_jopendde_bf16_dtype.py)

echo "Repo: $REPO_ROOT"
echo "Mode: $MODE; seeds per policy: $NUM_SEEDS; GPUs: $DEVICES"
echo "Output: $OUTPUT_DIR"
if (( ! DRY_RUN )); then
    [[ -x "$PYTHON" ]] || die "Missing executable: $PYTHON"
    [[ -f "$REPO_ROOT/P17_JN1.pdb" ]] || die "Missing P17_JN1.pdb"
    mkdir -p "$(dirname "$OUTPUT_DIR")"
    mkdir "$OUTPUT_DIR"
    # Apply shared dependency patches serially, before any workers start.
    for patch in "${PATCHES[@]}"; do
        "$PYTHON" "$patch" >> "$OUTPUT_DIR/patches.log" 2>&1 \
            || die "Patch failed: $patch; see $OUTPUT_DIR/patches.log"
    done
    printf 'run\tgpu\tpid\texit_code\n' > "$OUTPUT_DIR/status.tsv"
else
    echo "Dry run: patches would run once before workers: ${PATCHES[*]}"
fi

PIDS=(); RUN_NAMES=(); RUN_GPUS=()
FAILED=0
cleanup() {
    if (( ${#PIDS[@]} )); then
        kill "${PIDS[@]}" 2>/dev/null || true
        for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_batch() {
    local index code
    for index in "${!PIDS[@]}"; do
        code=0
        wait "${PIDS[$index]}" || code=$?
        printf '%s\t%s\t%s\t%s\n' "${RUN_NAMES[$index]}" "${RUN_GPUS[$index]}" \
            "${PIDS[$index]}" "$code" >> "$OUTPUT_DIR/status.tsv"
        echo "Finished ${RUN_NAMES[$index]}: exit $code"
        if (( code != 0 )); then FAILED=1; fi
    done
    PIDS=(); RUN_NAMES=(); RUN_GPUS=()
}

JOB=0
for policy in independent population; do
    for (( seed=0; seed<NUM_SEEDS; seed++ )); do
        device="${GPU_IDS[$((JOB % ${#GPU_IDS[@]}))]}"
        name="${policy}_seed${seed}"
        cmd=(env "CUDA_VISIBLE_DEVICES=$device" PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
            "$PYTHON" examples/p17_confidence_search.py
            --policy "$policy" --seed "$seed" "${COMMON_ARGS[@]}"
            --output-dir "$OUTPUT_DIR/$name")
        printf -v command_line '%q ' "${cmd[@]}"
        printf -v log_path '%q' "$OUTPUT_DIR/$name.log"
        if (( DRY_RUN )); then
            echo "$command_line > $log_path 2>&1 &"
        else
            echo "$command_line > $log_path 2>&1 &" >> "$OUTPUT_DIR/commands.sh"
            "${cmd[@]}" > "$OUTPUT_DIR/$name.log" 2>&1 &
            PIDS+=("$!"); RUN_NAMES+=("$name"); RUN_GPUS+=("$device")
            echo "Started $name on GPU $device (PID $!)"
        fi
        JOB=$((JOB + 1))
        if (( JOB % ${#GPU_IDS[@]} == 0 )); then
            if (( DRY_RUN )); then
                echo 'wait # finish this batch before reusing GPUs'
            else
                echo 'wait' >> "$OUTPUT_DIR/commands.sh"
                wait_batch
            fi
        fi
    done
done
if (( ! DRY_RUN )); then
    echo 'wait' >> "$OUTPUT_DIR/commands.sh"
    wait_batch
    echo "Batch complete. Worker exit codes: $OUTPUT_DIR/status.tsv"
fi
exit "$FAILED"
