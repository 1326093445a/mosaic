#!/usr/bin/env bash
# One script: the whole overnight screen on JN.1. Run this on the cluster.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_overnight_screen.sh --dry-run
#   bash examples/run_p17_overnight_screen.sh
#
# WHAT IT RUNS. Eight cells, ten search seeds each, eighty workers, JN.1 as
# the only input, population policy unless a cell says otherwise:
#
#   pop_off5   baseline: registry off, budget 5          <- what every prior run was
#   pop_on5    registry on, budget 5
#   pop_off7   registry off, budget 7
#   pop_on7    registry on, budget 7
#   indep_on5  independent policy instead of population  <- section 10.7, never run
#   apgm_on5   simplex_APGM continuous seeding, then population
#   bc_on5     BindCraft-STYLE four-stage seeding, then population
#   bc_on7     BindCraft-style seeding at budget 7
#
# TO BE UNAMBIGUOUS ABOUT THE bc_ CELLS: no BindCraft code, model or pipeline
# stage runs. `bindcraft_design` in src/mosaic/optimizers.py reimplements only
# their four-stage parameterisation anneal -- logits (50 then 25 steps), soft
# (45), hard (5), per default_4stage_multimer.json. The structure predictor is
# OpenDDE throughout, never AlphaFold2, and the objective is this project's own
# composite loss. BindCraft's ProteinMPNN redesign, PyRosetta filters, held-out
# AF2 re-prediction, hotspot handling and loss weights are all absent. What is
# being tested is the SCHEDULE, which section 24 identified as the one
# structural difference between this search and theirs.
#
# The first four are a 2x2 over the two live questions: does the registry
# restraint from section 26 do anything against a real model, and can 7 edits
# re-orient where 5 could not. The last four ask whether the SEARCH POLICY is
# the limit: BindCraft and Germinal both anneal the sequence parameterisation
# rather than taking discrete steps from a fixed start, and this pipeline has
# never had a continuous phase. `bindcraft_design` and `colabdesign_stage` have
# been sitting in src/mosaic/optimizers.py unused since before this project.
#
# WHAT IS PINNED, AND WHY. target_entropy 0.6, acceptance_temperature 0.02,
# 200 score calls, 64 sampling steps, stable aggregation -- identical in every
# cell, and deliberately not exposed as options. Section 21.7 records the
# reason: the earlier `budget` arm moved score calls, entropy and acceptance
# temperature together, which confounded both the pose comparison and the
# budget attribution. One factor per cell or the run says nothing.
#
# POWER. Ten seeds per cell resolves main effects at roughly 0.11 ipSAE and
# 6 A of pose, from the variances in section 23.2. This is a screening design:
# it says which cell deserves twenty seeds, not whether a 0.05 difference is
# real. Do not read a 0.03 gap as a result.
#
# UNTESTED AGAINST A REAL MODEL: the registry term (section 26.4) and all three
# seeding methods. Cells are independent and a failure in one does not cancel
# the rest, so the baseline still completes if the new paths break.
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"
SEEDER="$SCRIPT_DIR/p17_continuous_seed.py"

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
TARGET_ENTROPY=0.6      # pinned, see header
ACCEPT_TEMP=0.02        # pinned, see header
OUT_ROOT=""
DRY_RUN=false
ALL_CELLS="pop_off5 pop_on5 pop_off7 pop_on7 indep_on5 apgm_on5 bc_on5 bc_on7"
CELLS="$ALL_CELLS"

usage() {
    sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --devices LIST        GPUs (default: inherited CUDA_VISIBLE_DEVICES else 0-7)
  --seeds-per-cell N    search seeds per cell (default: $SEEDS_PER_CELL)
  --score-calls N       unique scored sequences per worker (default: $SCORE_CALLS)
  --weight-registry W   registry weight in the "on" cells (default: $WEIGHT_REGISTRY)
  --cells "a b"         subset of: $ALL_CELLS
  --output-dir PATH     default: results/p17_overnight_screen_<timestamp>
  --dry-run             print the plan; create nothing, load nothing
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --seeds-per-cell) SEEDS_PER_CELL="$2"; shift 2 ;;
        --score-calls) SCORE_CALLS="$2"; shift 2 ;;
        --weight-registry) WEIGHT_REGISTRY="$2"; shift 2 ;;
        --cells) CELLS="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_overnight_screen_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
N_CELLS=$(echo "$CELLS" | wc -w)

cell_registry() { case "$1" in *_on*) echo "$WEIGHT_REGISTRY" ;; *) echo 0.0 ;; esac; }
cell_budget()   { case "$1" in *7) echo 7 ;; *) echo 5 ;; esac; }
cell_policy()   { case "$1" in indep_*) echo independent ;; *) echo population ;; esac; }
cell_seeder()   { case "$1" in apgm_*) echo apgm ;; bc_*) echo bindcraft ;; *) echo "" ;; esac; }

echo "P17 overnight screen -- JN.1 only"
echo "  devices:        $DEVICES (${#DEV[@]})"
echo "  cells:          $CELLS"
echo "  seeds per cell: $SEEDS_PER_CELL  ->  $((N_CELLS * SEEDS_PER_CELL)) workers"
echo "  per worker:     $SCORE_CALLS score calls, $STEPS steps, $DTYPE, $AGGREGATION"
echo "  pinned:         target_entropy $TARGET_ENTROPY, acceptance_temperature $ACCEPT_TEMP"
echo "  registry:       weight $WEIGHT_REGISTRY at $REGISTRY_CUTOFF A"
echo "  output:         $OUT_ROOT"
echo
printf "  %-11s %-9s %-7s %-12s %s\n" CELL REGISTRY BUDGET POLICY SEEDING
for c in $CELLS; do
    printf "  %-11s %-9s %-7s %-12s %s\n" "$c" "$(cell_registry "$c")" \
        "$(cell_budget "$c")" "$(cell_policy "$c")" "$(cell_seeder "$c" || echo WT)"
done
echo
echo "  Power: main effects resolve at ~0.11 ipSAE and ~6 A pose at this seed"
echo "  count. Screening design -- do not read a 0.03 gap as a result."
echo

if "$DRY_RUN"; then
    echo "Dry run. Per cell, in order:"
    for c in $CELLS; do
        s=$(cell_seeder "$c" || true)
        echo
        echo "  [$c]"
        [[ -n "$s" ]] && echo "    seed: $PYTHON_BIN $SEEDER --method $s --edit-budget $(cell_budget "$c") --weight-registry $(cell_registry "$c") --out $OUT_ROOT/$c/seed.json"
        echo "    run:  $PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/$c --policies $(cell_policy "$c") --edit-budget $(cell_budget "$c") --weight-registry $(cell_registry "$c") --search-seeds $SEEDS ..."
    done
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

"$PYTHON_BIN" - <<PY > "$OUT_ROOT/plan.json"
import json
cells = {}
for c in "$CELLS".split():
    cells[c] = dict(
        weight_registry=float("$WEIGHT_REGISTRY") if "_on" in c else 0.0,
        edit_budget=7 if c.endswith("7") else 5,
        policy="independent" if c.startswith("indep_") else "population",
        seeding=("simplex_APGM" if c.startswith("apgm_") else
                 "bindcraft_style_4stage_anneal" if c.startswith("bc_")
                 else None),
    )
print(json.dumps(dict(
    input="P17_JN1.pdb", cells=cells, seeds_per_cell=$SEEDS_PER_CELL,
    score_calls=$SCORE_CALLS, sampling_steps=$STEPS, dtype="$DTYPE",
    aggregation="$AGGREGATION", target_entropy=$TARGET_ENTROPY,
    acceptance_temperature=$ACCEPT_TEMP,
    registry_contact_distance=$REGISTRY_CUTOFF, devices="$DEVICES",
    pinned_because="section 21.7: the earlier budget arm moved score calls, "
                   "entropy and acceptance temperature together",
    power="~0.11 ipSAE and ~6 A pose at this seed count (section 23.2)",
    untested_against_a_real_model=["registry term", "all three seeding methods"],
    seeding_note="bc_ cells borrow BindCraft's four-stage parameterisation "
                 "anneal only, reimplemented in mosaic.optimizers. No "
                 "BindCraft code, model or pipeline stage is involved; the "
                 "predictor is OpenDDE and the objective is this project's own.",
    registry_note="'registry' is the pairwise correspondence -- which binder "
                  "residue contacts which target residue -- as opposed to "
                  "aggregate proximity. A 180-degree flip preserves the "
                  "contact set while reversing the correspondence, which is "
                  "why named-pair restraints are used and proximity ones are "
                  "not (section 26.1).",
), indent=2))
PY
echo "Plan: $OUT_ROOT/plan.json"

COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
        --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
        --max-proposals "$PROPOSALS" --target-entropy "$TARGET_ENTROPY"
        --acceptance-temperature "$ACCEPT_TEMP"
        --registry-contact-distance "$REGISTRY_CUTOFF" --save-saliency)

SUMMARY=()
for c in $CELLS; do
    echo
    echo "################ cell $c ################"
    reg=$(cell_registry "$c"); bud=$(cell_budget "$c")
    pol=$(cell_policy "$c");   sdr=$(cell_seeder "$c" || true)
    mkdir -p "$OUT_ROOT/$c"

    # A seeding stage failure must not take the cell's baseline with it, nor
    # the remaining cells: skip the cell and say so.
    START=()
    if [[ -n "$sdr" ]]; then
        echo "  continuous seeding: $sdr"
        if CUDA_VISIBLE_DEVICES="${DEV[0]}" JAX_PLATFORMS=cuda \
           "$PYTHON_BIN" "$SEEDER" --method "$sdr" --edit-budget "$bud" \
             --weight-registry "$reg" \
             --registry-contact-distance "$REGISTRY_CUTOFF" \
             --sampling-steps "$STEPS" --opendde-dtype "$DTYPE" \
             --out "$OUT_ROOT/$c/seed.json" \
             >"$OUT_ROOT/$c/seed.log" 2>&1; then
            START=(--start-sequence "$("$PYTHON_BIN" -c "
import json,sys; print(json.load(open('$OUT_ROOT/$c/seed.json'))['seed_sequence'])")")
            echo "  seeded: $(basename "$OUT_ROOT/$c")/seed.json"
        else
            SUMMARY+=("$c: SKIPPED, seeding failed (see $c/seed.log)")
            echo "  seeding failed; skipping this cell." >&2
            continue
        fi
    fi

    if "$PYTHON_BIN" "$DRIVER" jn1 \
        --output-dir "$OUT_ROOT/$c/run" --policies "$pol" \
        --edit-budget "$bud" --weight-registry "$reg" \
        --search-seeds $SEEDS "${START[@]}" "${COMMON[@]}" \
        2>&1 | tee "$OUT_ROOT/$c/run.log"; then
        SUMMARY+=("$c: PASSED")
    else
        SUMMARY+=("$c: FAILED (see $c/run.log)")
        echo "  cell $c failed; continuing." >&2
    fi
done

echo
echo "################ Done ################"
printf '%s\n' "${SUMMARY[@]/#/  }"
cat <<EOF

Analysis tomorrow, both post-hoc and GPU-free:

  python examples/p17_search_trajectory.py $OUT_ROOT/<cell>/run/search/*
      per-move: predicted_loss_delta against realized_score_delta, sign
      agreement, restores_reference, and a per-position history

  python examples/p17_site_gate_audit.py $OUT_ROOT/<cell>/run
      site recall, paratope versus framework engagement, and the rotation /
      centroid decomposition of the pose error -- which is what separates a
      displaced binder from one flipped at the correct site

Scale bar (section 19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. Every prior run recovered
confidence without pose; whether any cell here differs is the question.
EOF
