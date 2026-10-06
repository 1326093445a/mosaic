#!/usr/bin/env bash
# 2x2 screen: registry restraint x edit budget, JN.1, population, 20 workers/factor.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_registry_screen.sh --dry-run
#   bash examples/run_p17_registry_screen.sh
#
# WHAT THIS TESTS. Two factors, both live questions as of section 26:
#
#   registry off/on  The restraint added in section 26 scores named reference
#                    contact pairs rather than aggregate proximity, so a binder
#                    flipped at the correct epitope breaks it where the existing
#                    contact term is satisfied. It is 3-47x more sensitive to a
#                    paratope-preserving flip on synthetic distograms and has
#                    never touched a real model.
#   budget 5/7       Section 22.1 showed 5 edits sufficed for Alpha sequence
#                    recovery and were not found, so budget is not the limit
#                    there. Whether 5 CDR substitutions can RE-ORIENT a 123
#                    residue domain is a different and untested question
#                    (section 26.5), and 7 is the cheapest probe of it. Run as
#                    an arm, not a replacement: 5 -> 7 enlarges the feasible set
#                    ~4800x and section 24.6's objection still applies.
#
# POWER, stated so the result is not over-read. At 10 search seeds per cell and
# the variances measured in section 23.2, main effects resolve at about
# 0.11 ipSAE and 6 A of pose. This is a screening design: it says which
# quadrant deserves 20 seeds, not whether a 0.05 difference is real.
#
# EVERY OTHER SETTING IS PINNED ACROSS ALL FOUR CELLS. Section 21.7 records why:
# the earlier `budget` arm moved score calls, target entropy and acceptance
# temperature together, which confounded both the pose comparison and the
# budget attribution. Entropy and acceptance temperature are therefore fixed
# here at the documented defaults and are not exposed as options.
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SEEDS_PER_CELL=10
SCORE_CALLS=200
GRADIENT_CALLS=200
PROPOSALS=2000
STEPS=64
DTYPE=bf16
AGGREGATION=stable
WEIGHT_REGISTRY=0.5
REGISTRY_CUTOFF=8.0
# Pinned: see the header. Not exposed.
TARGET_ENTROPY=0.6
ACCEPT_TEMP=0.02
OUT_ROOT=""
DRY_RUN=false
ARMS="off5 on5 off7 on7"

usage() {
    sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --devices LIST          GPUs (default: inherited CUDA_VISIBLE_DEVICES else 0-7)
  --seeds-per-cell N      search seeds per cell (default: 10, so 40 workers)
  --score-calls N         unique scored sequences per worker (default: 200)
  --weight-registry W     registry weight in the "on" cells (default: 0.5)
  --registry-cutoff A     reference contact cutoff for the pair set (default: 8.0)
  --arms "a b"            subset of: off5 on5 off7 on7
  --output-dir PATH       default: results/p17_registry_screen_<timestamp>
  --dry-run               print the plan; create nothing, load nothing
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --seeds-per-cell) SEEDS_PER_CELL="$2"; shift 2 ;;
        --score-calls) SCORE_CALLS="$2"; shift 2 ;;
        --weight-registry) WEIGHT_REGISTRY="$2"; shift 2 ;;
        --registry-cutoff) REGISTRY_CUTOFF="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_registry_screen_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
N_CELLS=$(echo "$ARMS" | wc -w)

cell_registry() { case "$1" in on5|on7) echo "$WEIGHT_REGISTRY" ;; *) echo 0.0 ;; esac; }
cell_budget()   { case "$1" in off7|on7) echo 7 ;; *) echo 5 ;; esac; }

echo "P17 registry x budget screen (JN.1, population)"
echo "  devices:        $DEVICES (${#DEV[@]})"
echo "  cells:          $ARMS"
echo "  seeds per cell: $SEEDS_PER_CELL  ->  $((N_CELLS * SEEDS_PER_CELL)) workers"
echo "  per worker:     $SCORE_CALLS score calls, $STEPS sampling steps, $DTYPE, $AGGREGATION"
echo "  pinned:         target_entropy $TARGET_ENTROPY, acceptance_temperature $ACCEPT_TEMP"
echo "  registry:       weight $WEIGHT_REGISTRY at a $REGISTRY_CUTOFF A reference cutoff"
echo "  output:         $OUT_ROOT"
echo
echo "  Power at $SEEDS_PER_CELL seeds/cell: main effects resolve at ~0.11 ipSAE"
echo "  and ~6 A pose (section 23.2 variances). A screening design, not a settled answer."
echo

COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
        --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
        --max-proposals "$PROPOSALS" --target-entropy "$TARGET_ENTROPY"
        --acceptance-temperature "$ACCEPT_TEMP" --save-saliency)

if "$DRY_RUN"; then
    echo "Dry run. Commands that would execute, in order:"
    for arm in $ARMS; do
        echo
        echo "  [$arm]  registry=$(cell_registry "$arm")  budget=$(cell_budget "$arm")"
        echo "    $PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/$arm \\"
        echo "      --edit-budget $(cell_budget "$arm") --weight-registry $(cell_registry "$arm") \\"
        echo "      --registry-contact-distance $REGISTRY_CUTOFF \\"
        echo "      --search-seeds $SEEDS ${COMMON[*]}"
    done
    echo
    echo "Then, for analysis:"
    echo "  python examples/p17_search_trajectory.py $OUT_ROOT/<cell>/search/*"
    echo "  python examples/p17_site_gate_audit.py $OUT_ROOT/<cell>"
    echo
    echo "No directory was created and no model was loaded."
    exit 0
fi

mkdir -p "$OUT_ROOT"
cd "$REPO_DIR"
export MOSAIC_OPENDDE_AGGREGATION="$AGGREGATION"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
[[ -z "${XLA_CLIENT_MEM_FRACTION:-}" ]] && \
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONUNBUFFERED=1

echo "Applying dependency patches..."
for name in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
    "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$name.py"
done >"$OUT_ROOT/patches.log" 2>&1
echo "Applied five OpenDDE patches."

# Record the plan before anything runs, so the cell definitions are auditable
# even if a cell fails.
"$PYTHON_BIN" - <<PY > "$OUT_ROOT/plan.json"
import json
cells = {}
for arm in "$ARMS".split():
    cells[arm] = dict(
        weight_registry=float("$WEIGHT_REGISTRY") if arm.startswith("on") else 0.0,
        edit_budget=7 if arm.endswith("7") else 5,
    )
print(json.dumps(dict(
    cells=cells, seeds_per_cell=$SEEDS_PER_CELL, score_calls=$SCORE_CALLS,
    sampling_steps=$STEPS, dtype="$DTYPE", aggregation="$AGGREGATION",
    target_entropy=$TARGET_ENTROPY, acceptance_temperature=$ACCEPT_TEMP,
    registry_contact_distance=$REGISTRY_CUTOFF, devices="$DEVICES",
    pinned_because="section 21.7: the earlier budget arm moved score calls, "
                   "entropy and acceptance temperature together, confounding "
                   "both the pose comparison and the budget attribution",
    power="~0.11 ipSAE and ~6 A pose at this seed count (section 23.2)",
), indent=2))
PY
echo "Plan: $OUT_ROOT/plan.json"

SUMMARY=()
for arm in $ARMS; do
    echo
    echo "################ cell: $arm (registry=$(cell_registry "$arm"), budget=$(cell_budget "$arm")) ################"
    # Cells are independent, so one failure must not cancel the rest.
    if "$PYTHON_BIN" "$DRIVER" jn1 \
        --output-dir "$OUT_ROOT/$arm" \
        --edit-budget "$(cell_budget "$arm")" \
        --weight-registry "$(cell_registry "$arm")" \
        --registry-contact-distance "$REGISTRY_CUTOFF" \
        --search-seeds $SEEDS "${COMMON[@]}" 2>&1 | tee "$OUT_ROOT/$arm.log"; then
        SUMMARY+=("$arm: PASSED")
    else
        SUMMARY+=("$arm: FAILED, see $arm.log")
        echo "Cell $arm failed; continuing with the remaining cells." >&2
    fi
done

echo
echo "################ Done ################"
printf '%s\n' "${SUMMARY[@]/#/  }"
cat <<EOF

Analysis, both post-hoc and GPU-free:

  python examples/p17_search_trajectory.py $OUT_ROOT/<cell>/search/*
      move trajectory: predicted_loss_delta against realized_score_delta,
      sign agreement, restores_reference, per-position history

  python examples/p17_site_gate_audit.py $OUT_ROOT/<cell>
      site recall, paratope versus framework engagement, and the rotation /
      centroid decomposition of the pose error -- which is what distinguishes
      a displaced binder from one flipped at the correct site

Read against the section 19.1 scale bar: P17+Alpha measured ipSAE 0.795 and
pose 1.98-2.96 A; P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. The registry
value is ~0 at the reference by construction, so it reads directly as distance
from reference registry. Confidence recovery without pose recovery is the
outcome every previous run produced; the question is whether this one differs.
EOF
