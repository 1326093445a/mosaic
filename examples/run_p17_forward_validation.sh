#!/usr/bin/env bash
# Fixed-input forward validation; uses the existing WT diagnostic runner.
# Paths follow this checkout: /home/yfeng17/mosaic locally or
# /storage/frank/mosaic on the cluster. No path edits or arguments are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
REFERENCE="$REPO_DIR/P17_JN1.pdb"
DEVICES="0,1,2,3,4,5,6,7"
OUTPUT_DIR="$REPO_DIR/results/p17_forward_validation_$(date -u +%Y%m%d_%H%M%S)_$$"
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: bash examples/run_p17_forward_validation.sh [options]

Cluster (all defaults):
  bash /storage/frank/mosaic/examples/run_p17_forward_validation.sh

Uses this checkout's .venv, P17_JN1.pdb, GPUs 0-7, and a new results directory.
All options below are optional overrides.

  --reference PATH   Existing PDB or mmCIF with protein chains B and T.
                     Default: checked-in P17_JN1.pdb (not a predicted CIF).
  --devices CSV      Allocated GPU indices (default: 0,1,2,3,4,5,6,7).
  --output-dir PATH  Fresh result directory; existing directories are rejected.
  --dry-run          Check the input and print the plan without model execution.
  -h, --help         Show this help.

Uses the existing forward-control preset: 10 workers, 20 predictions,
two seeds, two identical-input evaluations per worker, and three model paths.
The reference supplies sequences and geometry checks; its coordinates are not
used as a fixed-coordinate constraint or structural template.

Results include raw/mapped structures, arrays, geometry, repeats, and memory.
Exit 0 requires completed controls and passing existing geometry/mapping checks.
Repeat variation is reported separately, not rejected by a new threshold.
No gradients, sequence search, or subsequent optimization are launched.
PYTHON_BIN selects an existing environment; model assets may be fetched by the
underlying runner if absent. The underlying runner applies dependency patches.
EOF
}

while (($#)); do
    case "$1" in
        --reference|--devices|--output-dir)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --reference) REFERENCE="$2" ;;
                --devices) DEVICES="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
            esac
            shift 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python unavailable: $PYTHON_BIN; set PYTHON_BIN to an existing environment." >&2
    exit 2
fi

# Resolve caller-relative paths before moving to the checkout.
REFERENCE="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$REFERENCE")"
OUTPUT_DIR="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_DIR")"
PYTHON_BIN="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).absolute())' "$PYTHON_BIN")"
cd "$REPO_DIR"

"$PYTHON_BIN" - "$REFERENCE" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
import gemmi

reference, output = map(Path, sys.argv[1:])
if not reference.is_file():
    raise SystemExit(f"Reference not found: {reference}")
if output.exists():
    raise SystemExit(f"Output already exists: {output}")
structure = gemmi.read_structure(str(reference))
if len(structure) != 1:
    raise SystemExit("Reference must contain exactly one model")
model = structure[0]
for name in ("B", "T"):
    chains = [chain for chain in model if chain.name == name]
    if len(chains) != 1 or len(chains[0]) == 0:
        raise SystemExit(f"Reference needs one nonempty protein chain {name}")
    chain = chains[0]
    sequence = gemmi.one_letter_code([r.name for r in chain]).upper()
    if set(sequence) - set("ARNDCQEGHILKMFPSTWYV"):
        raise SystemExit(f"Chain {name} contains unsupported residues")
    for residue in chain:
        for atom in ("N", "CA", "C", "O"):
            if len(residue[atom]) != 1:
                raise SystemExit(f"Chain {name}, residue {residue.seqid}: expected one {atom}")
print(f"Reference: {reference}\nResults: {output}")
PY

# Match the existing validation launcher's allocator policy; caller overrides win.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
fi

# Pin the existing forward-control preset so runner-default changes cannot
# silently alter a no-argument launch.
export PYTHONUNBUFFERED=1
ARGS=(--reference "$REFERENCE" --devices "$DEVICES" --output-dir "$OUTPUT_DIR"
      --paths mosaic direct native --steps 64 --seeds 0 1
      --recycles 4 --opendde-dtype bf16 --aggregation-modes original stable)
if "$DRY_RUN"; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/p17_wt_validation.py" "${ARGS[@]}" --dry-run
fi

"$PYTHON_BIN" "$SCRIPT_DIR/p17_wt_validation.py" "${ARGS[@]}"
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
rows = summary.get("rows", [])
passed = (
    summary.get("completed") is True
    and summary.get("geometry_passed") is True
    and bool(rows)
    and all(row.get("completed") is True and row.get("geometry_passed") is True
            and row.get("mapping_agrees") is not False for row in rows)
)
print(f"Summary: {root / 'summary.json'}\nGeometry and repeats: {root / 'tables/geometry.csv'}")
if not passed:
    raise SystemExit("Forward validation failed; inspect saved results. No subsequent stage was launched.")
print("Forward controls completed and geometry checks passed. Review repeat differences separately.")
PY
