#!/usr/bin/env bash
# Single entry point for the P17+Alpha gradient recovery control.
#
# Does everything in one script: environment preflight, dependency patches,
# shared cache warm-up, then damage ladder -> damage calibration -> recovery
# search -> held-out rescoring. Paths follow this checkout, including
# /storage/frank/mosaic.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_alpha_recovery.sh --dry-run
#   bash examples/run_p17_alpha_recovery.sh
#
# WHY THIS CONTROL (docs/P17_JN1.md section 19.3B). P17 -> JN.1 is not known
# to have a solution, so a null result there cannot separate a bad gradient
# from an empty feasible set. Damaging P17+Alpha and measuring recovery has a
# known answer by construction -- the sequence you started from -- so both
# outcomes are informative. P17_Alpha.pdb is also experimental (PDB 8GZ5,
# X-ray, 1.70 A) rather than modeled, so pose RMSD is measured against a real
# arrangement, and the Alpha case is the one the predictor reproduces to
# +-0.001 ipTM instead of JN.1's 0.22/0.30/0.58 across seeds.
#
# A completed run establishes whether following OpenDDE's own sequence
# gradient restores its own confidence and pose metrics from a degraded start.
# It establishes nothing about binding, affinity or biology.
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
[[ -n "$INHERITED_DEVICES" ]] && DEVICES_FROM_ENV=true

EDITS="2 5 8 12"
DAMAGE_SEED=0
SELECT_RUNG=""
SEARCH_SEEDS="0 1"
EDIT_BUDGET=""
SAMPLING_STEPS=64
AGGREGATION=stable
DTYPE=bf16
MAX_SCORE_CALLS=32
MAX_GRADIENT_CALLS=32
MAX_PROPOSALS=320
OUTPUT_DIR=""
DRY_RUN=false
ALLOW_BUSY=false
SKIP_PREP=false
SKIP_CALIBRATION=false

usage() {
    cat <<'EOF'
Usage: bash examples/run_p17_alpha_recovery.sh [options]

Runs the whole thing: preflight, patches, caches, then all four stages.

Stages, each gated on the one before:
  ladder/     damage ladder, fixed and written to disk before any scoring (CPU)
  calibrate/  forward-scores the undamaged reference and every damage rung,
              then reports which rungs actually reached the non-binding regime
  search/     recovery searches from the selected rung: independent and
              population policies x search seeds, one worker per GPU
  heldout/    archived winners on structural seeds 101/102/103, sharded

Calibration refuses to continue if no rung reached the non-binding regime, or
if the undamaged reference itself failed to score -- without it there is no
recovery target to measure against. Every search worker must succeed before
held-out rescoring runs.

Settings pinned deliberately, matching the reviewed forward controls:
  * 64 sampling steps, not 8. Every path failed the backbone checks at 8 steps
    in the WT control batch; the post-fix forward batch ran at 64 only.
  * Stable aggregation via MOSAIC_OPENDDE_AGGREGATION, read by the dependency
    patch at JAX trace time. It was the only arm whose identical-input forward
    repeats were bitwise identical in every tested worker.
  * The epitope is derived from P17_Alpha.pdb's own CA contacts. The five JN.1
    hotspot constants are that target's positions and do not transfer: Alpha
    is 195 residues numbered 334-528 against JN.1's 184.

Options:
  --devices LIST         GPU indices (default: inherited CUDA_VISIBLE_DEVICES, else 0-7)
  --edits "N N N"        damage ladder edit counts (default: 2 5 8 12)
  --damage-seed N        damage RNG (default: 0)
  --select-rung N        search from this edit count, overriding calibration
  --search-seeds "N N"   search seeds per policy (default: 0 1)
  --edit-budget N        search edit cap (default: the selected rung's count)
  --steps N              sampling steps (default: 64)
  --aggregation MODE     stable | original (default: stable)
  --opendde-dtype D      bf16 | fp32 (default: bf16)
  --max-score-calls N    unique scored sequences per worker (default: 32)
  --max-gradient-calls N unique parent gradients per worker (default: 32)
  --max-proposals N      proposal attempts per worker (default: 320)
  --allow-busy-gpus      downgrade the GPU occupancy check to a warning
  --skip-prep            skip this script's patches and cache warm-up
  --skip-calibration     reuse an existing calibrate/ stage in --output-dir
  --output-dir PATH      default: results/p17_alpha_recovery_<timestamp>_<pid>
  --dry-run              report the plan and preflight; create no run directory
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; DEVICES_EXPLICIT=true; shift 2 ;;
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
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --skip-prep) SKIP_PREP=true; shift ;;
        --skip-calibration) SKIP_CALIBRATION=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Refuse to reach outside a scheduler-provided allocation. Overwriting an
# inherited CUDA_VISIBLE_DEVICES with different indices either escapes the
# allocation or fails against a cgroup restriction; neither is worth guessing.
if "$DEVICES_FROM_ENV" && "$DEVICES_EXPLICIT" && [[ "$DEVICES" != "$INHERITED_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES is already set to '$INHERITED_DEVICES' but" >&2
    echo "--devices requested '$DEVICES'. Request a matching allocation, or" >&2
    echo "unset CUDA_VISIBLE_DEVICES to address physical devices directly." >&2
    exit 2
fi

DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"
N_DEVICES="${#DEVICE_ARRAY[@]}"
N_RUNGS=$(echo "$EDITS" | wc -w)
N_SEEDS=$(echo "$SEARCH_SEEDS" | wc -w)

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_DIR/results/p17_alpha_recovery_$(date +%Y%m%d_%H%M%S)_$$"
fi

cd "$REPO_DIR"

echo "P17+Alpha gradient recovery control"
echo "  repo:       $REPO_DIR"
echo "  python:     $PYTHON_BIN"
echo "  reference:  P17_Alpha.pdb (experimental 8GZ5; chain A target, chain B binder)"
echo "  devices:    $DEVICES ($N_DEVICES)$("$DEVICES_FROM_ENV" && echo ' [inherited]' || true)"
echo "  damage:     edits [$EDITS], seed $DAMAGE_SEED"
echo "  search:     $N_SEEDS seed(s) x {independent, population} = $((N_SEEDS * 2)) workers"
echo "  settings:   $SAMPLING_STEPS steps, $DTYPE, $AGGREGATION aggregation"
echo "  ceilings:   $MAX_SCORE_CALLS score / $MAX_GRADIENT_CALLS gradient / $MAX_PROPOSALS proposals"
echo "  output:     $OUTPUT_DIR"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "No interpreter at $PYTHON_BIN. Create the venv first, or set PYTHON_BIN." >&2
    exit 2
fi
if [[ "$SAMPLING_STEPS" -lt 64 ]]; then
    echo "WARNING: 8-step predictions failed backbone checks on every path and" >&2
    echo "have not been retested since the numerical fixes." >&2
fi
if [[ "$AGGREGATION" != "stable" ]]; then
    echo "WARNING: aggregation '$AGGREGATION' is a deliberate control; only the" >&2
    echo "stable kernel gave bitwise-identical forward repeats." >&2
fi

PREP_DIR="$REPO_DIR/results/p17_alpha_prep_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PREP_DIR"
echo "Prep directory: $PREP_DIR"

# ---------------------------------------------------------------- preflight --
# Runs without preallocation so it cannot reserve pools the workers need
# moments later. A dry run reports problems without aborting.
PREFLIGHT_STATUS=0
STRICT=true
"$DRY_RUN" && STRICT=false
env -u XLA_PYTHON_CLIENT_MEM_FRACTION -u XLA_CLIENT_MEM_FRACTION \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    MOSAIC_PREFLIGHT_DEVICES="$DEVICES" \
    MOSAIC_PREFLIGHT_ALLOW_BUSY="$ALLOW_BUSY" \
    MOSAIC_PREFLIGHT_STRICT="$STRICT" \
    MOSAIC_PREFLIGHT_OUT="$PREP_DIR/preflight.json" \
    MOSAIC_PREFLIGHT_REPO="$REPO_DIR" \
    "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$PREP_DIR/preflight.log" || PREFLIGHT_STATUS=$?
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path(os.environ["MOSAIC_PREFLIGHT_REPO"])
requested = os.environ["MOSAIC_PREFLIGHT_DEVICES"].split(",")
allow_busy = os.environ["MOSAIC_PREFLIGHT_ALLOW_BUSY"] == "true"
strict = os.environ["MOSAIC_PREFLIGHT_STRICT"] == "true"
report = {"requested_devices": requested}
problems, warnings = [], []

import jax
from mosaic.cache import cache_dir

report["versions"] = {}
for name in ("jax", "jaxlib", "numpy", "jopendde", "jablang", "equinox", "torch", "gemmi"):
    try:
        report["versions"][name] = getattr(__import__(name), "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 - record, do not abort preflight
        report["versions"][name] = f"unavailable: {type(exc).__name__}"

# Device visibility. jax sees exactly what CUDA_VISIBLE_DEVICES exposes, so
# the count must match the request before eight workers are planned onto it.
report["devices"] = []
try:
    devices = jax.devices()
except Exception as exc:  # noqa: BLE001
    devices = []
    problems.append(f"jax could not enumerate devices: {type(exc).__name__}: {exc}")
for device in devices:
    stats = {}
    try:
        stats = device.memory_stats() or {}
    except Exception:  # noqa: BLE001
        pass
    limit = stats.get("bytes_limit")
    report["devices"].append(
        {
            "id": device.id,
            "kind": device.device_kind,
            "bytes_limit_GiB": round(limit / 1024**3, 2) if limit else None,
        }
    )
if devices and len(devices) != len(requested):
    problems.append(
        f"requested {len(requested)} device(s) but jax sees {len(devices)}"
    )
if any(d.platform != "gpu" for d in devices):
    problems.append(f"non-GPU backend: {sorted({d.platform for d in devices})}")

# Occupancy. This workflow preallocates 90% of each device, so another
# process on a requested GPU means an OOM minutes later rather than now. An
# unavailable query leaves occupancy unknown, which is a problem under strict
# launch rather than a warning: a passing preflight must not imply a check
# that never ran.
report["occupancy_checked"] = False
if shutil.which("nvidia-smi") is None:
    (warnings if allow_busy else problems).append(
        "nvidia-smi not found, so GPU occupancy could not be checked; pass "
        "--allow-busy-gpus to proceed anyway"
    )
else:
    def smi(query, extra=()):
        return subprocess.run(
            ["nvidia-smi", f"--query-{query}", *extra,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()

    try:
        # Scope by PCI bus id, not by index. nvidia-smi's index numbering is
        # not reliably the same as the physical indices passed in, and a
        # process on an unrequested GPU must not block a valid launch.
        bus_of = {}
        for line in smi("gpu=index,pci.bus_id").splitlines():
            if line.strip():
                index, bus = (part.strip() for part in line.split(",", 1))
                bus_of[index] = bus
        apps = smi("compute-apps=gpu_bus_id,pid,used_memory")
    except Exception as exc:  # noqa: BLE001
        (warnings if allow_busy else problems).append(
            f"nvidia-smi failed, so occupancy could not be checked: {exc}"
        )
    else:
        report["occupancy_checked"] = True
        missing = [d for d in requested if d not in bus_of]
        if missing:
            problems.append(f"nvidia-smi does not report requested GPUs {missing}")
        wanted = {bus_of[d] for d in requested if d in bus_of}
        # `jax.devices()` above opened this preflight's own CUDA context on
        # the requested GPUs, so exclude our own pid rather than counting it
        # as a foreign tenant.
        mine = os.getpid()
        foreign = []
        for line in apps.splitlines():
            if not line.strip():
                continue
            bus, pid, used = (part.strip() for part in line.split(",", 2))
            if bus in wanted and int(pid) != mine:
                foreign.append({"gpu_bus_id": bus, "pid": int(pid), "used_MiB": used})
        report["foreign_processes"] = foreign
        if foreign:
            (warnings if allow_busy else problems).append(
                f"requested GPUs already hold {len(foreign)} other process(es): "
                f"{foreign}"
            )

# Assets. Missing any of these fails minutes into a worker instead of now.
root = Path(cache_dir("."))
report["mosaic_cache_dir"] = str(root)
assets = {
    "abag_checkpoint": root / "opendde" / "checkpoint" / "opendde_abag.pt",
    "ccd_components": root / "opendde" / "common" / "components.cif",
    "alpha_reference": repo / "P17_Alpha.pdb",
}
report["assets"] = {}
for name, path in assets.items():
    present = path.is_file()
    report["assets"][name] = {"path": str(path), "present": present}
    if not present:
        problems.append(f"{name} is absent: {path}")

try:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".preflight_write_probe"
    probe.write_text("ok")
    probe.unlink()
    report["cache_writable"] = True
except OSError as exc:
    report["cache_writable"] = False
    problems.append(f"cache directory is not writable: {root} ({exc})")

# The reference must load and pass its own correspondence checks before any
# GPU time: a dropped or misnumbered residue shifts every position the pose
# loss compares.
sys.path.insert(0, str(repo / "examples"))
try:
    from p17_alpha_reference import (
        ALPHA_BINDER_CHAIN,
        ALPHA_TARGET_CHAIN,
        ca_spacing_report,
        contact_epitope,
        load_complex,
    )

    reference = load_complex(
        assets["alpha_reference"], ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN
    )
    spacing = [
        ca_spacing_report(reference["binder_ca"], "binder"),
        ca_spacing_report(reference["target_ca"], "target"),
    ]
    epitope = contact_epitope(reference["binder_ca"], reference["target_ca"], 8.0)
    report["reference"] = {
        "sha256": reference["source_sha256"],
        "binder_residues": len(reference["binder_seq"]),
        "target_residues": len(reference["target_seq"]),
        "binder_numbering": reference["binder_numbering"],
        "target_numbering": reference["target_numbering"],
        "altlocs_resolved": len(reference["binder_altlocs_resolved"])
        + len(reference["target_altlocs_resolved"]),
        "dropped_residues": {
            "binder": reference["binder_dropped_residues"],
            "target": reference["target_dropped_residues"],
        },
        "contact_epitope_residues": int(epitope.size),
        "spacing": spacing,
    }
    for entry in spacing:
        if not entry["passed"]:
            problems.append(f"reference backbone spacing failed: {entry}")
    if epitope.size == 0:
        problems.append("reference has no CA contacts at 8 A; no epitope to anchor")
except Exception as exc:  # noqa: BLE001
    problems.append(f"reference audit failed: {type(exc).__name__}: {exc}")

report["warnings"] = warnings
report["problems"] = problems
report["strict"] = strict
report["passed"] = not problems
Path(os.environ["MOSAIC_PREFLIGHT_OUT"]).write_text(json.dumps(report, indent=2) + "\n")

for entry in report["devices"]:
    print(f"GPU {entry['id']}: {entry['kind']}, limit {entry['bytes_limit_GiB']} GiB")
print(f"Mosaic cache: {report['mosaic_cache_dir']}")
print("Versions: " + ", ".join(f"{k} {v}" for k, v in report["versions"].items()))
if "reference" in report:
    ref = report["reference"]
    print(
        f"Reference: binder {ref['binder_residues']} aa, target "
        f"{ref['target_residues']} aa, {ref['altlocs_resolved']} altlocs resolved, "
        f"{ref['contact_epitope_residues']}-residue contact epitope"
    )
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
else
    echo "Preflight passed: $PREP_DIR/preflight.json"
fi

# ------------------------------------------------------------------ dry run --
if "$DRY_RUN"; then
    cat <<EOF

Dry run. Stages a real launch would execute, in order:

  1. patches       five OpenDDE patches, idempotent and source-verifying
  2. caches        atom-template and AbLang2 checkpoints, once, on CPU
  3. ladder/       $N_RUNGS damage rungs at [$EDITS], seed $DAMAGE_SEED
  4. calibrate/    $((N_RUNGS + 1)) forward scorings (reference + rungs) x 2
                   selection seeds, at most $N_DEVICES at a time
                   GATE: at least one rung must reach the non-binding regime
  5. search/       $((N_SEEDS * 2)) recovery searches from the selected rung
                   GATE: every worker must exit 0
  6. heldout/      $N_DEVICES shards on structural seeds 101/102/103

No run directory was created and no model was loaded.
Evidence from this dry run: $PREP_DIR/preflight.json
EOF
    exit 0
fi

mkdir -p "$OUTPUT_DIR"/{ladder,calibrate,search,heldout,logs,tables}
cp "$PREP_DIR/preflight.json" "$OUTPUT_DIR/preflight.json"

# --------------------------------------------------------- patches & caches --
if "$SKIP_PREP"; then
    echo "Skipping patches and cache warm-up on request."
else
    echo
    echo "=== Dependency patches ==="
    for name in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
        "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$name.py"
    done 2>&1 | tee "$OUTPUT_DIR/logs/patches.log"

    # Build both first-use artifacts once, on CPU, with seeded host RNG. The
    # workers would otherwise race to create them with different RNG states.
    echo
    echo "=== Shared caches (CPU; the template cache can take minutes) ==="
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
from mosaic.models.opendde import _get_atom_templates
_get_atom_templates()
print('atom-template cache ready')
" 2>&1 | tee "$OUTPUT_DIR/logs/template_cache.log"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
from ablang2.load_model import load_model
load_model('ablang2-paired')
print('ablang2-paired checkpoint ready')
" 2>&1 | tee "$OUTPUT_DIR/logs/ablang2_cache.log"
fi

# Aggregation is selected at JAX trace time, per fresh worker process.
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"
# One worker per allocated GPU: reserve the pool up front to reduce
# fragmentation, which section 17.9 identified behind the earlier OOMs.
# Explicit caller settings win. Never set both memory-fraction aliases;
# newer JAX rejects that combination.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
fi
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------- stages --
echo
echo "=== Stage 1/3: damage ladder (CPU, no model) ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_reference.py" \
    --edits $EDITS --damage-seed "$DAMAGE_SEED" \
    --out "$OUTPUT_DIR/ladder/ladder.json" 2>&1 | tee "$OUTPUT_DIR/logs/ladder.log"

echo
echo "=== Stage 2/3: damage calibration (forward scoring) ==="
if "$SKIP_CALIBRATION" && [[ -f "$OUTPUT_DIR/calibrate/calibration.json" ]]; then
    echo "Reusing $OUTPUT_DIR/calibrate/calibration.json"
else
    "$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" calibrate \
        --ladder "$OUTPUT_DIR/ladder/ladder.json" \
        --output-dir "$OUTPUT_DIR/calibrate" \
        --devices "$DEVICES" --steps "$SAMPLING_STEPS" \
        --opendde-dtype "$DTYPE" 2>&1 | tee "$OUTPUT_DIR/logs/calibrate.log"
fi

echo
echo "=== Stage 3/3: recovery search and held-out rescoring ==="
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

cat <<EOF

Completed. Results: $OUTPUT_DIR

  tables/recovery.csv            every scored sequence against this run's own
                                 reference and damaged confidence, on the
                                 held-out seeds the search never selected on
  tables/recovery_context.json   what the recovery fraction is relative to
  calibrate/calibration.json     which rungs reached the non-binding regime
  ladder/ladder.json             the damaged starting points, fixed in advance
  preflight.json                 devices, versions, assets, reference audit

Read recovery.csv against the section 19.1 scale bar: P17+Alpha measured
ipSAE 0.795 and pose 1.98-2.96 A, P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A.
Recovery of the predictor's own metrics is not evidence of binding.
EOF
