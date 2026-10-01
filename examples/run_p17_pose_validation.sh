#!/usr/bin/env bash
# Distribute candidates across GPUs, then repeat on the same GPU assignments.
# Paths follow the checkout, including /storage/frank/mosaic on the cluster.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash examples/run_p17_pose_validation.sh [OPTIONS]

  --input PATH       Completed pilot directory or tar.gz archive.
                     Default: results/p17_confidence_pilot_20260929_194857_1770469
  --devices CSV      Allocated GPU IDs/UUIDs. Default: CUDA_VISIBLE_DEVICES,
                     or 0,1,2,3,4,5,6,7 if unset. --device remains an alias.
  --output-dir PATH  Fresh parent output directory. Default: timestamped path
                     under results/p17_pose_validation_...
  --dry-run          Validate inputs and show both stages without patches,
                     model loading, output files, or GPU computation.
  -h, --help         Show this help.

Stages (parallel workers within each stage; fresh processes for the repeat):
  validation/        WT + unique saved winners, prediction seeds 0, 1, 2.
  repeat_seed0/      Same sequences, seed 0 again for a repeatability check.
Nine unique sequences are split across eight GPUs (one handles two sequences).
For the completed pilot this means 27 + 9 = 36 forward predictions total.
Each candidate uses the same GPU in both stages; GPU memory is not pooled.
No optimization, gradients, or AbLang2 inference. Original sampling settings
are reused. Outputs include CIF/PDB, confidence arrays, per-seed RMSD/score
CSVs, source candidate mappings, and logs. Stage 2 runs only if stage 1 succeeds.

Example:
  cd /storage/frank/mosaic
  bash examples/run_p17_pose_validation.sh --devices 0,1,2,3,4,5,6,7
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
INPUT="results/p17_confidence_pilot_20260929_194857_1770469"
DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
OUTPUT_DIR=""
DRY_RUN=0
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --input|--device|--devices|--output-dir)
            (( $# >= 2 )) || die "$1 requires a value"
            case "$1" in
                --input) INPUT="$2" ;;
                --device|--devices) DEVICES="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done
[[ -n "$DEVICES" && "$DEVICES" != ,* && "$DEVICES" != *, && "$DEVICES" != *,,* ]] \
    || die "--devices requires a comma-separated GPU list"
IFS=',' read -r -a GPU_IDS <<< "$DEVICES"
declare -A SEEN_GPUS=()
for device in "${GPU_IDS[@]}"; do
    [[ "$device" =~ ^[0-9]+$ || "$device" =~ ^(GPU-|MIG-)[A-Za-z0-9/-]+$ ]] \
        || die "Invalid GPU ID/UUID: $device"
    [[ ! ${SEEN_GPUS[$device]+present} ]] || die "Duplicate GPU: $device"
    SEEN_GPUS[$device]=1
done
NUM_SHARDS=${#GPU_IDS[@]}
[[ "$INPUT" = /* ]] || INPUT="$REPO_ROOT/$INPUT"
[[ -e "$INPUT" ]] || die "Input not found: $INPUT"
OUTPUT_DIR="${OUTPUT_DIR:-results/p17_pose_validation_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || die "Output directory already exists: $OUTPUT_DIR"
PYTHON="$REPO_ROOT/.venv/bin/python"
RUNNER="$REPO_ROOT/examples/p17_rescore_winners.py"
[[ -x "$PYTHON" ]] || die "Missing Python environment: $PYTHON"
[[ -f "$RUNNER" && -f "$REPO_ROOT/examples/p17_search_outputs.py" ]] \
    || die "Sync the updated rescoring and output scripts to this checkout first"

# Preflight both stage plans before touching installed dependencies or results.
echo "Repo: $REPO_ROOT"
echo "GPUs: $DEVICES; output: $OUTPUT_DIR"
for stage in validation repeat_seed0; do
    seeds=(0)
    if [[ "$stage" == validation ]]; then seeds=(0 1 2); fi
    echo "Plan: $stage"
    for index in "${!GPU_IDS[@]}"; do
        "$PYTHON" "$RUNNER" --input "$INPUT" --output-dir "$OUTPUT_DIR/$stage/shard_$index" \
            --seeds "${seeds[@]}" --num-shards "$NUM_SHARDS" --shard-index "$index" --dry-run
    done
done
if (( DRY_RUN )); then exit 0; fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR/logs"
printf 'stage\tshard\tgpu\tpid\texit_code\n' > "$OUTPUT_DIR/status.tsv"
printf '#!/usr/bin/env bash\ncd %q\n' "$REPO_ROOT" > "$OUTPUT_DIR/commands.sh"
cat > "$OUTPUT_DIR/README.md" <<'EOF'
# P17 pose validation and repeatability

- `validation/`: WT + archived winners, prediction seeds 0, 1, 2.
- `repeat_seed0/`: same sequences, seed 0 in a fresh process.
- Each stage has `shard_N/` folders with CIF/PDB structures, confidence arrays,
  reference PDB, source-run mappings and local score/RMSD tables.
- Stage-level `tables/*.csv` combine all workers; prediction file paths are
  relative to the stage directory, e.g. `shard_0/structures/...`.
- Assignment uses global candidate ID modulo GPU count. Each candidate stays
  on the same GPU for both stages, with a fresh process for the repeat.
- `logs/`: dependency patch log and console logs for both stages.
- `status.tsv`: GPU, PID and exit code per worker. A failed first stage prevents
  stage 2 from starting; an absent stage row means it has not completed.
- `commands.sh`: exact stage commands for provenance.

Compare matching sequences/seed 0 between stages for repeatability. Compare
winners with WT on seeds 1 and 2 for an initial held-out check (the original
September 29 pilot selected only on seed 0). Inspect target-aligned binder pose
RMSD alongside the CIF structures. These are new predictions, not the unsaved
original structures. Two process executions are a diagnostic repeatability check,
not a complete reproducibility study or evidence of binding affinity.
EOF

for patch in patches/patch_jopendde_outer_product_mean.py \
             patches/patch_jopendde_structural_token_expander.py \
             patches/patch_jopendde_bf16_dtype.py \
             patches/patch_jopendde_aggregation.py \
             patches/patch_jopendde_padding.py; do
    "$PYTHON" "$patch" >> "$OUTPUT_DIR/logs/patches.log" 2>&1 \
        || die "Patch failed; see $OUTPUT_DIR/logs/patches.log"
done

PIDS=()
cleanup() {
    if (( ${#PIDS[@]} )); then
        kill "${PIDS[@]}" 2>/dev/null || true
        for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for stage in validation repeat_seed0; do
    seeds=(0)
    if [[ "$stage" == validation ]]; then seeds=(0 1 2); fi
    mkdir "$OUTPUT_DIR/$stage"
    for index in "${!GPU_IDS[@]}"; do
        device="${GPU_IDS[$index]}"
        cmd=(env "CUDA_VISIBLE_DEVICES=$device" PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
            "$PYTHON" "$RUNNER" --input "$INPUT" --output-dir "$OUTPUT_DIR/$stage/shard_$index"
            --seeds "${seeds[@]}" --num-shards "$NUM_SHARDS" --shard-index "$index")
        printf '%q ' "${cmd[@]}" >> "$OUTPUT_DIR/commands.sh"
        printf '> %q 2>&1 &\n' "$OUTPUT_DIR/logs/${stage}_shard${index}.log" >> "$OUTPUT_DIR/commands.sh"
        "${cmd[@]}" > "$OUTPUT_DIR/logs/${stage}_shard${index}.log" 2>&1 &
        PIDS+=("$!")
        echo "Started $stage shard $index on GPU $device (PID $!)"
    done
    failed=0
    for index in "${!PIDS[@]}"; do
        code=0
        wait "${PIDS[$index]}" || code=$?
        printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$index" "${GPU_IDS[$index]}" "${PIDS[$index]}" "$code" \
            >> "$OUTPUT_DIR/status.tsv"
        echo "Finished $stage shard $index: exit $code"
        if (( code != 0 )); then failed=1; fi
    done
    PIDS=()
    echo 'wait' >> "$OUTPUT_DIR/commands.sh"
    if (( failed )); then
        die "Stage $stage failed; see $OUTPUT_DIR/logs/ and status.tsv. Later stages were not started."
    fi
    merge_cmd=("$PYTHON" -c
        'import sys; from pathlib import Path; sys.path.insert(0, "examples"); from p17_rescore_winners import merge_shard_tables; merge_shard_tables(Path(sys.argv[1]), int(sys.argv[2]))'
        "$OUTPUT_DIR/$stage" "$NUM_SHARDS")
    printf '%q ' "${merge_cmd[@]}" >> "$OUTPUT_DIR/commands.sh"
    printf '\n' >> "$OUTPUT_DIR/commands.sh"
    "${merge_cmd[@]}"
done
echo "Both stages completed. Combined CSVs: $OUTPUT_DIR/{validation,repeat_seed0}/tables/"
