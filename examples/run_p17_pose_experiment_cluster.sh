#!/usr/bin/env bash
# Single entry point for the real gated pose run on an eight-GPU H200 node.
#
# Does everything in one script: environment preflight, dependency patches,
# shared cache warm-up, then the gated diagnostic -> four-arm population
# ablation -> held-out rescoring workflow at the settings the reviewed forward
# controls support. Paths follow this checkout, including /storage/frank/mosaic.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_pose_experiment_cluster.sh --dry-run
#   bash examples/run_p17_pose_experiment_cluster.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
# Default only when the environment has not already allocated devices: a
# scheduler-managed subset must not be silently replaced by 0-7.
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DEVICES_FROM_ENV=false
DEVICES_EXPLICIT=false
INHERITED_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$INHERITED_DEVICES" ]]; then
    DEVICES_FROM_ENV=true
fi
SAMPLING_STEPS=64
AGGREGATION=stable
WEIGHT_POSE=1.0
POSE_MARGIN=3.0
SEARCH_SEEDS="0 1"
MAX_SCORE_CALLS=32
MAX_GRADIENT_CALLS=32
MAX_PROPOSALS=320
OUTPUT_DIR=""
DRY_RUN=false
ALLOW_BUSY=false
SKIP_PREP=false
SKIP_FORWARD_CHECK=false

usage() {
    cat <<'EOF'
Usage: bash examples/run_p17_pose_experiment_cluster.sh [options]

Runs the whole thing: preflight, patches, caches, then the gated experiment.

Stages (from examples/p17_pose_experiment.py, unchanged):
  diagnostic/  2 workers, proposal-model seeds 0 and 1, two GPUs
  search/      4 arms (A neither, B guidance, C retention, D both) x search
               seeds, one worker per GPU
  heldout/     WT plus archived winners on structural seeds 101/102/103,
               sharded across the allocated GPUs

Both diagnostic reports must pass every required check before search launches,
and every search worker must succeed before held-out rescoring. The gate
requires proposal influence to be positively detected, so the search stage runs
only if the pose gradient was shown to move proposals above the repeat
threshold. The 3 A target-fit limit, the influence rule and the geometry
thresholds are not relaxed here.

Two settings differ from p17_pose_experiment.py's own defaults, deliberately:

  * 64 sampling steps, not 8. Every path failed the backbone checks at 8 steps
    in the WT control batch, and the reviewed post-fix forward batch ran at 64
    only. The schema-2 gate requires predicted-backbone plausibility, so 8
    steps would risk failing the gate for reasons unrelated to pose.
  * Stable aggregation pinned via MOSAIC_OPENDDE_AGGREGATION, which the
    dependency patch reads at JAX trace time. It was the only arm whose two
    identical-input forward repeats were bitwise identical in every tested
    worker. The variable is recorded in each worker's config.json.

Before any GPU worker starts:

  1. Preflight. Confirms JAX sees every requested device, reports each device
     kind and memory limit, records jax/jaxlib/numpy/jopendde/jablang/torch
     versions, and checks the ABAG checkpoint, the CCD components file and the
     reference PDB. It refuses to launch onto requested GPUs that already hold
     another process, because this workflow preallocates 90% of each device.
     The occupancy check is scoped to the requested devices and excludes this
     preflight's own CUDA context.
  2. Dependency patches. All five, idempotent and source-verifying. They are
     applied here because the atom-template cache key embeds the JOpenDDE build
     id, so the cache below must be built against patched sources.
     p17_pose_experiment.py applies the same five again when it starts, and
     logs them into the run directory; that repetition is harmless but it does
     mean patching is not skippable from here.
  3. Shared caches, on CPU with seeded host RNG: the schema-2 atom-template
     cache and the AbLang2 paired checkpoint. Both are first-use artifacts and
     the workers would otherwise race to create or download them. The template
     build can take a few minutes on a new machine.

Prep logs and preflight.json go to results/p17_pose_prep_<timestamp>/; the run
creates its own separate results directory.

  --devices CSV           Allocated GPU indices (default: 0,1,2,3,4,5,6,7).
  --search-seeds "A B"    Search seeds per arm (default: "0 1").
  --sampling-steps N      Diffusion steps for gradient and selection calls
                          (default: 64).
  --weight-pose W         Pose RMSD coefficient in the proposal gradient for
                          arms B and D (default: 1.0). Arms A and C use 0.
  --pose-margin A         Retention ceiling margin in angstroms for arms C and
                          D: worst selection-seed WT pose RMSD + margin
                          (default: 3.0). Also the common reporting ceiling.
  --max-score-calls N     Unique scored sequences per search worker (default: 32).
  --max-gradient-calls N  Unique parent gradients per search worker (default: 32).
  --max-proposals N       Proposal attempts per search worker (default: 320).
  --aggregation MODE      stable or original (default: stable). Use original
                          only as a deliberate control.
  --output-dir PATH       Fresh result directory; existing ones are rejected.
  --allow-busy-gpus       Launch even if a requested GPU has other processes.
  --skip-prep             Skip this script's patch application and cache
                          warm-up; preflight still runs, and the Python
                          launcher still applies the five patches itself. Use
                          only when the caches already exist for this exact
                          JOpenDDE build.
  --skip-forward-check    Launch without completed forward-control evidence.
  --dry-run               Preflight, then print the plan. Creates no run
                          directory, applies no patches, loads no model. Still
                          writes preflight evidence, and reports preflight
                          problems without aborting so the plan can be
                          previewed on any machine.
  -h, --help              Show this help.

Raising --weight-pose does not guarantee proportionally stronger guidance: the
composite gradient is clipped and the proposal temperature is recalibrated to a
target entropy, so a larger coefficient can be partly absorbed. Read the
diagnostic's measured influence rather than assuming the weight transferred.
Retention still ranks on raw mean ipSAE, with pose entering arms C and D as a
feasibility ordering only.

Exit 0 requires all three stages to complete. A passing gate establishes
interpretable measurement and detectable proposal influence, not improved pose,
binding or affinity. PYTHON_BIN selects an existing environment.
EOF
}

while (($#)); do
    case "$1" in
        --devices|--search-seeds|--sampling-steps|--weight-pose|--pose-margin|\
--max-score-calls|--max-gradient-calls|--max-proposals|--aggregation|--output-dir)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --devices) DEVICES="$2"; DEVICES_EXPLICIT=true ;;
                --search-seeds) SEARCH_SEEDS="$2" ;;
                --sampling-steps) SAMPLING_STEPS="$2" ;;
                --weight-pose) WEIGHT_POSE="$2" ;;
                --pose-margin) POSE_MARGIN="$2" ;;
                --max-score-calls) MAX_SCORE_CALLS="$2" ;;
                --max-gradient-calls) MAX_GRADIENT_CALLS="$2" ;;
                --max-proposals) MAX_PROPOSALS="$2" ;;
                --aggregation) AGGREGATION="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
            esac
            shift 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --skip-prep) SKIP_PREP=true; shift ;;
        --skip-forward-check) SKIP_FORWARD_CHECK=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python unavailable: $PYTHON_BIN; set PYTHON_BIN to an existing environment." >&2
    exit 2
fi
if [[ ! "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--devices expects comma-separated GPU indices: $DEVICES" >&2
    exit 2
fi
if [[ "$AGGREGATION" != "stable" && "$AGGREGATION" != "original" ]]; then
    echo "--aggregation must be stable or original" >&2
    exit 2
fi
for name in SAMPLING_STEPS MAX_SCORE_CALLS MAX_GRADIENT_CALLS MAX_PROPOSALS; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer for ${name}: ${!name}" >&2
        exit 2
    fi
done
for name in WEIGHT_POSE POSE_MARGIN; do
    if [[ ! "${!name}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Expected a nonnegative number for ${name}: ${!name}" >&2
        exit 2
    fi
done
read -r -a SEED_LIST <<<"$SEARCH_SEEDS"
if ((${#SEED_LIST[@]} == 0)); then
    echo "--search-seeds needs at least one seed" >&2
    exit 2
fi
for seed in "${SEED_LIST[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Search seeds must be nonnegative integers: $seed" >&2
        exit 2
    fi
done

cd "$REPO_DIR"

# Refuse to reach outside a scheduler-provided allocation. Overwriting an
# inherited CUDA_VISIBLE_DEVICES with different indices either escapes the
# allocation or fails against a cgroup restriction; neither is worth guessing.
if "$DEVICES_FROM_ENV" && "$DEVICES_EXPLICIT" && [[ "$DEVICES" != "$INHERITED_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES is already set to '$INHERITED_DEVICES' but" >&2
    echo "--devices requested '$DEVICES'. Request a matching allocation, or" >&2
    echo "unset CUDA_VISIBLE_DEVICES to address physical devices directly." >&2
    exit 2
fi
if "$DEVICES_FROM_ENV"; then
    echo "Devices: $DEVICES (inherited from CUDA_VISIBLE_DEVICES)"
else
    echo "Devices: $DEVICES"
fi

# The project handoff keeps pose diagnostics and search comparisons pending
# until forward controls have completed and passed. Check that evidence exists
# in this checkout rather than assuming it.
if ! "$DRY_RUN" && ! "$SKIP_FORWARD_CHECK"; then
    "$PYTHON_BIN" "$SCRIPT_DIR/p17_forward_evidence.py" \
        --steps "$SAMPLING_STEPS" --dtype bf16 --aggregation "$AGGREGATION" \
        --reference "$REPO_DIR/P17_JN1.pdb"
fi

PREP_DIR="$REPO_DIR/results/p17_pose_prep_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PREP_DIR"
echo "Prep directory: $PREP_DIR"

# Preflight runs without preallocation so it cannot reserve pools the workers
# need moments later. A dry run reports problems without aborting.
PREFLIGHT_STATUS=0
STRICT=true
if "$DRY_RUN"; then STRICT=false; fi
env -u XLA_PYTHON_CLIENT_MEM_FRACTION -u XLA_CLIENT_MEM_FRACTION \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    MOSAIC_PREFLIGHT_INHERITED_CVD="${CUDA_VISIBLE_DEVICES:-}" \
    MOSAIC_PREFLIGHT_DEVICES="$DEVICES" \
    MOSAIC_PREFLIGHT_ALLOW_BUSY="$ALLOW_BUSY" \
    MOSAIC_PREFLIGHT_STRICT="$STRICT" \
    MOSAIC_PREFLIGHT_OUT="$PREP_DIR/preflight.json" \
    "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$PREP_DIR/preflight.log" || PREFLIGHT_STATUS=$?
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

requested = os.environ["MOSAIC_PREFLIGHT_DEVICES"].split(",")
allow_busy = os.environ["MOSAIC_PREFLIGHT_ALLOW_BUSY"] == "true"
report = {"requested_devices": requested}
problems, warnings = [], []

import jax
from mosaic.cache import cache_dir

report["versions"] = {}
for name in ("jax", "jaxlib", "numpy", "jopendde", "jablang", "equinox", "torch"):
    try:
        module = __import__(name)
        report["versions"][name] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 - record, do not abort preflight
        report["versions"][name] = f"unavailable: {type(exc).__name__}"
if str(report["versions"].get("jax", "")).startswith(("0.11", "0.12")):
    warnings.append(
        f"jax {report['versions']['jax']} is outside the declared <0.11 range; "
        "recorded as an environment caveat, not resolved here"
    )
try:
    import torch

    report["torch_cuda_available"] = bool(torch.cuda.is_available())
except Exception:  # noqa: BLE001
    report["torch_cuda_available"] = None

# CUDA_VISIBLE_DEVICES is set to the requested list, so JAX should expose
# exactly those devices, renumbered from zero.
try:
    devices = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
except Exception as exc:  # noqa: BLE001 - no CUDA backend is a reportable problem
    devices = []
    problems.append(f"JAX exposes no CUDA device: {type(exc).__name__}: {exc}")
report["visible_gpu_count"] = len(devices)
report["devices"] = []
for device in devices:
    entry = {"id": device.id, "kind": device.device_kind}
    try:
        limit = device.memory_stats().get("bytes_limit")
        entry["bytes_limit_GiB"] = round(limit / 2**30, 2) if limit else None
    except Exception:  # noqa: BLE001
        entry["bytes_limit_GiB"] = None
    report["devices"].append(entry)
if len(devices) != len(requested):
    problems.append(
        f"requested {len(requested)} GPUs but JAX exposes {len(devices)}; "
        "check the allocation and CUDA_VISIBLE_DEVICES"
    )
kinds = {entry["kind"] for entry in report["devices"]}
if kinds and not all("H200" in kind for kind in kinds):
    warnings.append(
        f"device kinds {sorted(kinds)} are not all H200; memory defaults and "
        "the recorded peaks were measured on H200"
    )
if len(kinds) > 1:
    warnings.append(f"mixed device kinds in one allocation: {sorted(kinds)}")

# Preallocation reserves ~90% of each device, so a resident process elsewhere
# makes the launch fail. The requested indices are always physical here: the
# launcher refuses to request devices differing from an inherited allocation,
# and it exports the requested set as CUDA_VISIBLE_DEVICES for the children.
# An unavailable query leaves occupancy unknown, which is a problem under
# strict mode rather than a warning: a passing preflight must not imply a
# check that never ran.
report["occupancy_checked"] = False
report["inherited_cuda_visible_devices"] = (
    os.environ.get("MOSAIC_PREFLIGHT_INHERITED_CVD") or None
)
unchecked = warnings if allow_busy else problems
if shutil.which("nvidia-smi") is None:
    unchecked.append(
        "nvidia-smi not found, so GPU occupancy could not be checked; pass "
        "--allow-busy-gpus to launch without that check"
    )
else:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,pci.bus_id",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if smi.returncode != 0 or apps.returncode != 0:
        # Both queries must succeed: an empty process list from a failed query
        # is indistinguishable from an idle GPU.
        failure = (smi if smi.returncode != 0 else apps).stderr.strip()[:200]
        unchecked.append(
            f"nvidia-smi failed, so occupancy could not be checked: {failure}"
        )
    else:
        report["occupancy_checked"] = True
        rows = {}
        for line in smi.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                rows[parts[0]] = {
                    "name": parts[1],
                    "memory_used_MiB": int(parts[2]),
                    "memory_total_MiB": int(parts[3]),
                    "bus_id": parts[4].lower(),
                }
        report["nvidia_smi"] = {k: v for k, v in rows.items() if k in requested}
        # Count only processes on the requested GPUs: another user's job on an
        # unrequested device must not block this launch. Enumerating devices
        # above opened this preflight's own CUDA context, which nvidia-smi
        # reports like any other process, so exclude it too.
        requested_bus_ids = {rows[d]["bus_id"] for d in requested if d in rows}
        own_pid = str(os.getpid())
        # Track this preflight's own usage PER DEVICE. It holds one context on
        # each requested GPU, so a single summed total subtracted from every
        # device would discount other users' memory many times over.
        resident, own_by_bus = [], {}
        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            bus_id, pid, used = parts
            if pid == own_pid:
                own_by_bus[bus_id.lower()] = own_by_bus.get(bus_id.lower(), 0) + int(
                    float(used)
                )
            elif bus_id.lower() in requested_bus_ids:
                resident.append({"bus_id": bus_id, "pid": pid, "used_MiB": used})
        report["preflight_own_MiB_by_bus_id"] = own_by_bus
        report["compute_processes_on_requested_gpus"] = resident
        report["compute_process_count"] = len(resident)
        missing = [d for d in requested if d not in rows]
        if missing:
            problems.append(f"nvidia-smi does not report requested GPUs {missing}")
        busy = [
            d for d in requested
            if d in rows
            and rows[d]["memory_used_MiB"] - own_by_bus.get(rows[d]["bus_id"], 0) > 2048
        ]
        if busy or resident:
            message = (
                f"GPUs in use: {busy or 'none by memory'}; "
                f"{len(resident)} other compute process(es) present. "
                "This workflow preallocates 90% of each device."
            )
            (warnings if allow_busy else problems).append(message)

root = cache_dir()
report["mosaic_cache_dir"] = str(root)
assets = {
    "abag_checkpoint": root / "opendde" / "checkpoint" / "opendde_abag.pt",
    "ccd_components": root / "opendde" / "common" / "components.cif",
    "reference_pdb": Path("P17_JN1.pdb").resolve(),
}
report["assets"] = {}
for name, path in assets.items():
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    report["assets"][name] = {"path": str(path), "exists": exists, "bytes": size}
    if not exists:
        problems.append(f"missing {name}: {path}")
    elif size == 0:
        problems.append(f"empty {name}: {path}")
try:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".preflight_write_probe"
    probe.write_text("ok")
    probe.unlink()
    report["cache_writable"] = True
except OSError as exc:
    report["cache_writable"] = False
    problems.append(f"cache directory is not writable: {root} ({exc})")

strict = os.environ["MOSAIC_PREFLIGHT_STRICT"] == "true"
report["warnings"] = warnings
report["problems"] = problems
report["strict"] = strict
report["passed"] = not problems
Path(os.environ["MOSAIC_PREFLIGHT_OUT"]).write_text(json.dumps(report, indent=2) + "\n")

for entry in report["devices"]:
    print(f"GPU {entry['id']}: {entry['kind']}, limit {entry['bytes_limit_GiB']} GiB")
print(f"Mosaic cache: {report['mosaic_cache_dir']}")
print("Versions: " + ", ".join(f"{k} {v}" for k, v in report["versions"].items()))
for warning in warnings:
    print(f"WARNING: {warning}")
for problem in problems:
    print(f"PROBLEM: {problem}", file=sys.stderr)
if problems and not strict:
    print(
        f"{len(problems)} preflight problem(s); continuing because this is a dry run.",
        file=sys.stderr,
    )
sys.exit(0 if report["passed"] or not strict else 1)
PY

if ((PREFLIGHT_STATUS != 0)); then
    echo "Preflight failed; see $PREP_DIR/preflight.json. Nothing was launched." >&2
    exit 1
fi
if "$DRY_RUN" && grep -q '"passed": false' "$PREP_DIR/preflight.json"; then
    echo "Preflight reported problems above; a real launch would stop here."
    echo "Evidence: $PREP_DIR/preflight.json"
else
    echo "Preflight passed: $PREP_DIR/preflight.json"
fi

if "$DRY_RUN"; then
    echo "Dry run: skipping patches and cache warm-up."
elif "$SKIP_PREP"; then
    echo "Skipping this script's patch application and cache warm-up on request;"
    echo "p17_pose_experiment.py still applies the five patches itself."
else
    echo "Applying dependency patches (the launcher reapplies them; idempotent)..."
    for name in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
        "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$name.py"
    done 2>&1 | tee "$PREP_DIR/patches.log"

    # Build both first-use artifacts once, on CPU, with seeded host RNG. The
    # workers would otherwise race to create them with different RNG states.
    echo "Preparing shared atom-template cache on CPU (this can take minutes)..."
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
from mosaic.models.opendde import _get_atom_templates
_get_atom_templates()
print('atom-template cache ready')
" 2>&1 | tee "$PREP_DIR/template_cache.log"

    echo "Fetching the AbLang2 paired checkpoint on CPU..."
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
from ablang2.load_model import load_model
load_model('ablang2-paired')
print('ablang2-paired checkpoint ready')
" 2>&1 | tee "$PREP_DIR/ablang2_cache.log"
fi

# Aggregation is selected at JAX trace time, per fresh worker process.
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"
# One worker per allocated H200: reserve the pool up front to reduce
# fragmentation. Explicit caller settings win. Never set both memory-fraction
# aliases; newer JAX rejects that combination.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
fi
export PYTHONUNBUFFERED=1

echo "Aggregation: $MOSAIC_OPENDDE_AGGREGATION; sampling steps: $SAMPLING_STEPS; OpenDDE compute: bf16"
echo "Pose weight (arms B/D): $WEIGHT_POSE; retention margin (arms C/D): $POSE_MARGIN A"

ARGS=(--devices "$DEVICES"
      --sampling-steps "$SAMPLING_STEPS"
      --opendde-dtype bf16
      --weight-pose "$WEIGHT_POSE"
      --pose-margin "$POSE_MARGIN"
      --search-seeds "${SEED_LIST[@]}"
      --max-score-calls "$MAX_SCORE_CALLS"
      --max-gradient-calls "$MAX_GRADIENT_CALLS"
      --max-proposals "$MAX_PROPOSALS")
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_DIR")"
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if "$DRY_RUN"; then
    ARGS+=(--dry-run)
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/p17_pose_experiment.py" "${ARGS[@]}"
