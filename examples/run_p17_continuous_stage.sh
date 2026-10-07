#!/usr/bin/env bash
# One script: the continuous seeding stage on JN.1, then the discrete handoff.
# Run this on the cluster.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_continuous_stage.sh --dry-run
#   bash examples/run_p17_continuous_stage.sh --stages seed      # ~25 min, cheap
#   bash examples/run_p17_continuous_stage.sh                    # seed + search
#
# WHY A CLUSTER SCRIPT. The local box is a 24 GB RTX 4090. The seeding stage is
# the same forward-plus-backward through OpenDDE at 64 sampling steps that the
# search already pays for, and section 28.1 records the three cells that
# completed 125 steps doing it on one H200. 24 GB has not been shown to hold
# it, and a continuous optimizer carries state the discrete path does not --
# momentum in simplex_APGM, stage buffers in the annealed variants -- so this
# runs where the memory is.
#
# WHAT IT RUNS, IN TWO STAGES.
#
#   stage `seed`    p17_continuous_seed.py, three methods x three seeds, nine
#                   runs over the available GPUs. Writes one JSON per run and
#                   prints the truncation verdict below. ~25 min.
#
#   stage `search`  the discrete harness from each seed sequence, one cell per
#                   method, plus a parent-start control. 20 search seeds per
#                   cell, 80 workers, ~4-5 h.
#
# THE GATE BETWEEN THEM, WHICH IS THE POINT OF SPLITTING THEM. Stage 0's
# synthetic screen (section 30.3) found no soft-to-hard ROUNDING gap -- the
# relaxed optimum put 0.977-0.999 of its mass on one residue at every setting
# tried -- and located the whole failure in TRUNCATION: the relaxed optimum
# wants more edits than the budget allows, and `project_to_budget` throws the
# rest away. Whether that happens here is exactly what
# `candidate_substitutions_considered` in the stage-`seed` JSON reports. If it
# comes back at or near 29, every seed is an argmax of a budget-truncated
# optimum and stage `search` is measuring the projection, not the relaxation.
# The script prints that number per run and refuses to guess what it means.
#
# WHAT THE ANCHORED INIT IS FOR. `--init parent` (the default since 2026-10-07)
# adds `--init-logit-scale` to the parent column, so the softmax opens at
# p_parent = 0.886 and the EditBudget expectation opens at 29 x 0.114 = 3.3,
# inside a budget of 5. The old noise init opened at roughly 27.6 against the
# same budget -- a hinge of 5.0 * relu(27.6 - 5) = 113, dwarfing the ~34
# carried by every other term combined, so the first many steps spent their
# gradient walking back toward the parent. Covered by tests/test_continuous_seed.py.
#
# ⚠️ THE RISK THE SWEEP FLAGGED, AND WHAT MEASURES IT. The anchored init puts
# the hinge at zero at step 0; it does not keep it there. In the synthetic
# sweep the hinge held at coupling <= 0.3 (optimum stayed at 5 edits) and lost
# above it (7 edits at coupling 1.0, 10.5 at 3.0). Which regime the real
# composite objective sits in is set by the hinge's gradient against the other
# terms', and THAT QUANTITY HAS NEVER BEEN MEASURED -- it is section 26.5's
# per-term gradient norms, still outstanding as section 27.7 item 1. It costs
# one forward-plus-backward pass and it should be read before this script's
# stage `search`, not after. The same number also gates the interpretation of
# section 27's registry null and of section 30.2's pose result, which is why
# it is now the first thing in the queue.
#
# HOW THE BUDGET IS SHARED, EXACTLY. `--budget-anchor reference` on the
# discrete stage. Without it the seed's edits and the search's edits are
# counted from different origins and the pair spends `--edit-budget` EACH --
# 10 edits from the parent on a budget of 5, which is section 27.7 item 5's
# third defect. With it, `wt` inside the search becomes the reference while the
# search still STARTS at the seed, so the hard cap in `_moves` and the soft
# `EditBudget` hinge both count drift from the reference, a reverted position
# returns to the reference, and p17_confidence_search.py refuses outright if
# the seed is already over budget. Swapping the loss anchor is safe because the
# start sequence is validated to differ from the reference only inside the
# designable mask, so the scaffold `SetPositions` pins is identical either way.
# Verified against the real `run_gradient_search` in tests/test_continuous_seed.py.
#
# WHAT THE CONTROL IS. `parent` -- the same harness, same budget, same entropy,
# same policy, started at the parent instead of at a seed. It is deliberately
# re-run here rather than borrowed from the section 29 screen's `grad_pop`
# cell, which is nominally the same configuration, because that cell predates
# today's `--budget-anchor` change and a control should share a code version
# with its treatment. Use `--cells` to drop it if the compute is needed
# elsewhere; section 30.2's grad_pop numbers are the fallback comparison.
#
# WHAT IS PINNED. edit_budget 5, weight_registry 0.0, target_entropy 0.6,
# population policy, 200 score calls, 64 sampling steps, bf16, stable
# aggregation, acceptance_temperature 0.02, held-out structural seeds
# 101/102/103 -- identical in every cell and identical to the section 29
# `grad_pop` cell. Section 21.7 records why: the earlier budget arm moved score
# calls, entropy and acceptance temperature together and confounded two
# comparisons at once. The ONLY thing that differs between cells is where the
# search starts.
#
# STEPS ARE PINNED ACROSS METHODS, which the defaults do not do. Left alone,
# bindcraft runs 125 steps (50+25+45+5), colabdesign 120 and apgm 200, so apgm
# would get 1.6x the gradient calls and any difference between methods would be
# partly a compute difference. $SEED_STEPS applies to all three.
#
# WHAT THIS CAN AND CANNOT SHOW. The decisive readout is an advantage at the
# FINAL, FEASIBLE, DISCRETE candidate under the original constraints -- not a
# lower relaxed loss, not successful execution, not a more sophisticated
# search. Section 30.3's synthetic result is a prediction of a null here: where
# discrete search already saturates its budget, continuous seeding had no
# headroom to add, and every hand-off there came in marginally WORSE than
# discrete search alone (99.8% and 99.5% of the exhaustive optimum against
# 99.9%, all far inside the 0.855 SEM). This script is the real-model test of
# that prediction. A null confirms a synthetic finding; it does not establish
# that relaxation cannot help, only that it did not help at this budget with
# this projection.
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"
SEEDER="$SCRIPT_DIR/p17_continuous_seed.py"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
METHODS="apgm bindcraft colabdesign"
SEED_REPLICATES=3       # continuous-stage seeds per method
SEED_STEPS=125          # pinned across methods, see header
SEED_LR=0.1
INIT=parent             # see header; `noise` reproduces the pre-2026-10-07 start
INIT_LOGIT_SCALE=5.0
SEEDS_PER_CELL=20
SCORE_CALLS=200
GRADIENT_CALLS=200
PROPOSALS=2000
STEPS=64
DTYPE=bf16
AGGREGATION=stable
EDIT_BUDGET=5           # pinned, shared between the stages, see header
WEIGHT_REGISTRY=0.0     # pinned off, see header
REGISTRY_CUTOFF=8.0
ENTROPY=0.6             # pinned, matches the section 29 grad_ cells
ACCEPT_TEMP=0.02        # pinned, see header
POLICY=population
OUT_ROOT=""
DRY_RUN=false
ALL_STAGES="seed search"
STAGES="$ALL_STAGES"
ALL_CELLS="parent apgm bindcraft colabdesign"
CELLS="$ALL_CELLS"

usage() {
    sed -n '2,112p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --devices LIST        GPUs (default: inherited CUDA_VISIBLE_DEVICES else 0-7)
  --stages "a b"        subset of: $ALL_STAGES  (default: both)
  --cells "a b"         stage-search cells, subset of: $ALL_CELLS
  --methods "a b"       stage-seed methods, subset of: $METHODS
  --seed-replicates N   continuous seeds per method (default: $SEED_REPLICATES)
  --seed-steps N        continuous steps, all methods (default: $SEED_STEPS)
  --init {parent,noise} relaxed-variable start (default: $INIT)
  --seeds-per-cell N    search seeds per stage-search cell (default: $SEEDS_PER_CELL)
  --score-calls N       unique scored sequences per worker (default: $SCORE_CALLS)
  --seed-dir PATH       reuse an earlier stage-seed output instead of rerunning
  --output-dir PATH     default: results/p17_continuous_stage_<timestamp>
  --dry-run             print the plan; create nothing, load nothing
EOF
}

SEED_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --cells) CELLS="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --seed-replicates) SEED_REPLICATES="$2"; shift 2 ;;
        --seed-steps) SEED_STEPS="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --seeds-per-cell) SEEDS_PER_CELL="$2"; shift 2 ;;
        --score-calls) SCORE_CALLS="$2"; shift 2 ;;
        --seed-dir) SEED_DIR="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_continuous_stage_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
REPLICATES=$(seq 0 $((SEED_REPLICATES - 1)) | tr '\n' ' ')
[[ -z "$SEED_DIR" ]] && SEED_DIR="$OUT_ROOT/seeds"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# The seed each stage-search cell starts from. `parent` is the control and
# supplies none, so the harness starts at the reference and the budget anchor
# is left at its own default.
cell_seed_json() { echo "$SEED_DIR/${1}_r0.json"; }

echo "P17 continuous stage -- JN.1 only"
echo "  devices:        $DEVICES (${#DEV[@]})"
echo "  stages:         $STAGES"
echo "  output:         $OUT_ROOT"
if has_stage seed; then
echo
echo "  [seed]  methods:      $METHODS"
echo "          replicates:   $SEED_REPLICATES per method"
echo "          steps:        $SEED_STEPS (pinned across methods, see header)"
echo "          init:         $INIT (logit scale $INIT_LOGIT_SCALE)"
echo "          budget:       $EDIT_BUDGET, shared with the search stage"
fi
if has_stage search; then
echo
echo "  [search] cells:       $CELLS"
echo "           seeds/cell:  $SEEDS_PER_CELL  ->  $(echo "$CELLS" | wc -w) x $SEEDS_PER_CELL workers"
echo "           per worker:  $SCORE_CALLS score calls, $STEPS steps, $DTYPE, $AGGREGATION"
echo "           pinned:      edit_budget $EDIT_BUDGET, registry $WEIGHT_REGISTRY,"
echo "                        target_entropy $ENTROPY, policy $POLICY,"
echo "                        acceptance_temperature $ACCEPT_TEMP"
echo "           varied:      where the search starts, and nothing else"
echo "           anchor:      --budget-anchor reference on every seeded cell,"
echo "                        so the seed's edits are already on the meter"
echo "           seeds from:  $SEED_DIR"
fi
echo
echo "  Section 30.3 predicts a null: where discrete search saturates its"
echo "  budget, continuous seeding had no headroom, and every synthetic"
echo "  hand-off came in marginally worse than discrete search alone."
echo "  Section 26.5's per-term gradient norms are still unmeasured and gate"
echo "  the interpretation of this run. They cost one backward pass."
echo

if "$DRY_RUN"; then
    echo "Dry run."
    if has_stage seed; then
        echo
        echo "  [seed] $( echo "$METHODS" | wc -w ) x $SEED_REPLICATES runs:"
        for m in $METHODS; do
            for r in $REPLICATES; do
                echo "    $PYTHON_BIN $SEEDER --method $m --seed $r --steps $SEED_STEPS --init $INIT --init-logit-scale $INIT_LOGIT_SCALE --edit-budget $EDIT_BUDGET --weight-registry $WEIGHT_REGISTRY --sampling-steps $STEPS --opendde-dtype $DTYPE --lr $SEED_LR --out $SEED_DIR/${m}_r${r}.json"
            done
        done
    fi
    if has_stage search; then
        echo
        echo "  [search] per cell, in order:"
        for c in $CELLS; do
            echo
            echo "    [$c]"
            if [[ "$c" == parent ]]; then
                echo "      run:  $PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/$c/run --policies $POLICY --edit-budget $EDIT_BUDGET --weight-registry $WEIGHT_REGISTRY --target-entropy $ENTROPY --search-seeds $SEEDS ...   (control: no --start-sequence)"
            else
                echo "      seed: $(cell_seed_json "$c")"
                echo "      run:  $PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/$c/run --policies $POLICY --edit-budget $EDIT_BUDGET --weight-registry $WEIGHT_REGISTRY --target-entropy $ENTROPY --start-sequence <from that JSON> --budget-anchor reference --search-seeds $SEEDS ..."
            fi
        done
    fi
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
print(json.dumps(dict(
    input="P17_JN1.pdb", stages="$STAGES".split(),
    seed_stage=dict(
        methods="$METHODS".split(), replicates=$SEED_REPLICATES,
        steps=$SEED_STEPS, init="$INIT", init_logit_scale=$INIT_LOGIT_SCALE,
        lr=$SEED_LR,
        steps_pinned_because="left alone bindcraft runs 125 steps, colabdesign "
                             "120 and apgm 200, so apgm would get 1.6x the "
                             "gradient calls and a method difference would be "
                             "partly a compute difference",
        init_because="noise logits open the EditBudget expectation at ~27.6 "
                     "against a budget of 5, a hinge of ~113 against the ~34 "
                     "carried by every other term combined; the anchored start "
                     "opens at 3.3 and the hinge is inactive at step 0",
    ),
    search_stage=dict(
        cells="$CELLS".split(), seeds_per_cell=$SEEDS_PER_CELL,
        score_calls=$SCORE_CALLS, gradient_calls=$GRADIENT_CALLS,
        edit_budget=$EDIT_BUDGET, weight_registry=$WEIGHT_REGISTRY,
        target_entropy=$ENTROPY, policy="$POLICY",
        acceptance_temperature=$ACCEPT_TEMP,
        budget_anchor="reference on every seeded cell, default on the control",
        control="parent -- same harness, same budget, same entropy, same "
                "policy, started at the parent. Re-run rather than borrowed "
                "from section 29's grad_pop cell because that cell predates "
                "the --budget-anchor change and a control should share a code "
                "version with its treatment.",
        varied="where the search starts, and nothing else",
    ),
    sampling_steps=$STEPS, dtype="$DTYPE", aggregation="$AGGREGATION",
    devices="$DEVICES",
    question="whether a continuous relaxation, projected onto the shared edit "
             "budget, gives the discrete harness a better start than the "
             "parent does, measured at the final feasible discrete candidate",
    budget_sharing="--budget-anchor reference makes wt inside the search the "
                   "reference while the search still starts at the seed, so "
                   "the hard cap in _moves and the soft EditBudget hinge both "
                   "count drift from the reference and the two stages spend "
                   "one budget rather than one each",
    gate="candidate_substitutions_considered in each seed JSON. Stage 0 found "
         "no rounding gap (relaxed optima put 0.977-0.999 of their mass on one "
         "residue) and located the failure in truncation. If this comes back "
         "near 29, every seed is an argmax of a budget-truncated optimum and "
         "the search stage measures the projection, not the relaxation.",
    prediction="null. Section 30.3: where discrete search saturates its "
               "budget, continuous seeding had no headroom, and every "
               "synthetic hand-off came in marginally worse than discrete "
               "search alone -- 99.8% and 99.5% of the exhaustive optimum "
               "against 99.9%, all far inside the 0.855 SEM.",
    unmeasured_and_gating="section 26.5's per-term gradient norms and pairwise "
                          "cosines (section 27.7 item 1). They set whether the "
                          "EditBudget hinge holds against the other terms at "
                          "the real objective's gradient scale, which the "
                          "synthetic sweep showed decides whether the relaxed "
                          "optimum stays inside the budget. One backward pass.",
    does_not_establish="that relaxation cannot help -- only whether it helped "
                       "at this budget with this projection",
), indent=2))
PY
echo "Plan: $OUT_ROOT/plan.json"

SUMMARY=()

############################ stage: seed ############################
if has_stage seed; then
    echo
    echo "################ stage seed ################"
    mkdir -p "$SEED_DIR"
    i=0
    pids=()
    for m in $METHODS; do
        for r in $REPLICATES; do
            dev="${DEV[$((i % ${#DEV[@]}))]}"
            out="$SEED_DIR/${m}_r${r}.json"
            echo "  [$m r$r] device $dev -> $out"
            CUDA_VISIBLE_DEVICES="$dev" "$PYTHON_BIN" "$SEEDER" \
                --method "$m" --seed "$r" --steps "$SEED_STEPS" \
                --init "$INIT" --init-logit-scale "$INIT_LOGIT_SCALE" \
                --edit-budget "$EDIT_BUDGET" \
                --weight-registry "$WEIGHT_REGISTRY" \
                --registry-contact-distance "$REGISTRY_CUTOFF" \
                --sampling-steps "$STEPS" --opendde-dtype "$DTYPE" \
                --lr "$SEED_LR" --out "$out" \
                >"$SEED_DIR/${m}_r${r}.log" 2>&1 &
            pids+=($!)
            i=$((i + 1))
        done
    done
    failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
    if ((failed)); then
        SUMMARY+=("seed: $failed of $i runs FAILED (see $SEED_DIR/*.log)")
        echo "  $failed of $i seeding runs failed; see $SEED_DIR/*.log" >&2
    else
        SUMMARY+=("seed: $i/$i PASSED")
    fi

    echo
    echo "---- the gate: what the relaxation actually wanted ----"
    "$PYTHON_BIN" - "$SEED_DIR" <<'PY' | tee "$SEED_DIR/gate.txt"
import json, sys
from pathlib import Path

rows = []
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    if p.name == "gate.json":
        continue
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        print(f"  {p.name}: unreadable ({exc})")
        continue
    rows.append(d)
    margins = [e["margin"] for e in d["edits"]]
    head = (f"  {d['method']:<12} seed {d['seed']}  "
            f"{d['hamming_from_parent']} edits kept / "
            f"{d['candidate_substitutions_considered']} wanted")
    if margins:
        print(f"{head}   margins {min(margins):.3f}-{max(margins):.3f}")
    else:
        print(f"{head}   (the relaxation asked for no substitution at all)")

if rows:
    wanted = [d["candidate_substitutions_considered"] for d in rows]
    budget = rows[0]["edit_budget"]
    print()
    print(f"  substitutions wanted: min {min(wanted)}, max {max(wanted)}, "
          f"budget {budget}, designable 29")
    if max(wanted) >= 25:
        print("  VERDICT: truncation regime. The relaxation wants nearly every")
        print("  designable position changed, so each seed is an argmax of a")
        print("  budget-truncated optimum and the search stage would measure")
        print("  the projection rather than the relaxation. This is the")
        print("  failure mode section 30.3 isolated. Consider the masked k=budget")
        print("  variant (section 28.5 mechanism 2) before spending the night.")
    elif max(wanted) <= budget:
        print("  VERDICT: no truncation. Every wanted substitution fits the")
        print("  budget, so the projection is an identity and the hand-off")
        print("  carries the relaxed optimum intact. This is the regime in")
        print("  which the stage-search comparison is clean.")
    else:
        print("  VERDICT: partial truncation. The projection is discarding")
        print("  some substitutions the relaxation asked for but is not")
        print("  swamped. The stage-search result is interpretable with the")
        print("  margin spread above reported alongside it.")
    print()
    print("  Unique seed sequences: "
          f"{len({d['seed_sequence'] for d in rows})} of {len(rows)} runs")
PY
    echo
    echo "  Seeds in $SEED_DIR"
fi

############################ stage: search ############################
if has_stage search; then
    echo
    echo "################ stage search ################"
    COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
            --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
            --max-proposals "$PROPOSALS"
            --target-entropy "$ENTROPY"
            --acceptance-temperature "$ACCEPT_TEMP"
            --registry-contact-distance "$REGISTRY_CUTOFF" --save-saliency)

    for c in $CELLS; do
        echo
        echo "################ cell $c ################"
        mkdir -p "$OUT_ROOT/$c"
        extra=()
        if [[ "$c" != parent ]]; then
            seed_json="$(cell_seed_json "$c")"
            if [[ ! -f "$seed_json" ]]; then
                SUMMARY+=("$c: SKIPPED (no seed at $seed_json)")
                echo "  no seed JSON at $seed_json; skipping cell $c" >&2
                continue
            fi
            seq=$("$PYTHON_BIN" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["seed_sequence"])' \
                "$seed_json")
            spent=$("$PYTHON_BIN" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["hamming_from_parent"])' \
                "$seed_json")
            echo "  start: $seed_json ($spent of $EDIT_BUDGET edits spent)"
            extra=(--start-sequence "$seq" --budget-anchor reference)
        else
            echo "  start: the parent (control; budget anchor left at its default)"
        fi

        if "$PYTHON_BIN" "$DRIVER" jn1 \
            --output-dir "$OUT_ROOT/$c/run" --policies "$POLICY" \
            --edit-budget "$EDIT_BUDGET" --weight-registry "$WEIGHT_REGISTRY" \
            --search-seeds $SEEDS ${extra[@]+"${extra[@]}"} "${COMMON[@]}" \
            2>&1 | tee "$OUT_ROOT/$c/run.log"; then
            SUMMARY+=("$c: PASSED")
        else
            SUMMARY+=("$c: FAILED (see $c/run.log)")
            echo "  cell $c failed; continuing." >&2
        fi
    done
fi

echo
echo "################ Done ################"
if ((${#SUMMARY[@]})); then printf '%s\n' "${SUMMARY[@]/#/  }"; fi
cat <<EOF

Read the gate first: $SEED_DIR/gate.txt

Then, from each stage-search cell's tables/jn1_recovery.csv, the comparison is
mean and best ipSAE and mean pose over the non-WT winners, per cell, with the
per-cell spread -- each seeded cell against the \`parent\` control. Aggregate per
candidate, not per structure: the three held-out seeds of one candidate are
correlated and are not three independent measurements (section 30.2).

Post-hoc and GPU-free:

  python examples/p17_search_trajectory.py $OUT_ROOT/<cell>/run/search/*
      per-move predicted against realized delta, and a per-position history.
      On a seeded cell this also shows whether the search spent its REMAINING
      budget or sat on the seed, which a mean ipSAE cannot distinguish.

  python examples/p17_site_gate_audit.py $OUT_ROOT/<cell>/run
      site recall, contact-face versus fixed-region engagement, and the
      rotation / centroid decomposition (section 25.1)

  python examples/p17_capri_audit.py $OUT_ROOT/<cell>/run --out capri.csv
      fnat / iRMSD / LRMSD / DockQ on the held-out structures, via the DockQ
      CLI. Section 30.2's audit of the section 29 screen graded all 240 non-WT
      structures CAPRI-incorrect at best DockQ 0.174, so read iRMSD and LRMSD
      as the sensitive columns and DockQ as a floor, not a grade.

Scale bar (section 19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. The section 29 screen's grad_pop
cell -- the nominal configuration of the \`parent\` control here -- reached mean
ipSAE 0.567 and mean pose 29.3 A, iRMSD 12.5 A, with every structure
CAPRI-incorrect.

A null here confirms section 30.3's synthetic prediction. It does not establish
that relaxation cannot help, only that it did not help at this budget with this
projection.

Establishes nothing about binding.
EOF
