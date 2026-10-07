#!/usr/bin/env bash
# One script: the attribution screen on JN.1. Run this on the cluster.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_attribution_screen.sh --dry-run
#   bash examples/run_p17_attribution_screen.sh
#
# WHAT IT RUNS. A 2x2 over proposal source and search policy, JN.1 as the only
# input, twenty search seeds per cell, eighty workers:
#
#   grad_pop     gradient-weighted proposals, population    <- what every prior run was
#   grad_indep   gradient-weighted proposals, independent
#   unif_pop     uniform proposals, population
#   unif_indep   uniform proposals, independent
#
# THE QUESTION. Section 21.5 measured P(ipSAE improves | the first-order delta
# predicted an improvement) and found it equal to the base rate in all four
# seeds. But that statistic is computed WITHIN the pool the gradient already
# selected, because softmax(-delta/temp) picks moves on exactly that quantity,
# so it cannot say whether that pool beats a uniformly drawn one. Section 21.6
# records the same gap and section 24.4 lists this arm as never run. It is the
# only experiment that attributes any of sections 20-21's movement to the
# gradient rather than to the expensive ipSAE filter plus the feasible-set
# constraints, and it is open in BOTH directions.
#
# HOW THE UNIFORM ARM WORKS, EXACTLY. No new code path. `_proposal_distribution`
# in src/mosaic/search.py returns np.full(count, 1/count) when target_entropy
# == 1, and SearchConfig validates 0 < target_entropy <= 1, so --target-entropy
# 1.0 draws uniformly from the feasible move set that `_moves` already builds.
# Same designable mask, same edit cap, same revert-and-exchange rules, same
# ipSAE retention, same acceptance temperature. The ONLY difference between the
# arms is which move gets sampled.
#
# WHAT THIS MATCHES, AND WHAT IT THEREFORE CANNOT ANSWER. Both arms still
# compute the gradient; the uniform arm just does not sample from it. So the
# cells are matched on score calls, gradient calls and wall-clock, which makes
# this a clean one-variable comparison of the proposal DISTRIBUTION. It does
# NOT test the economics: gradient_seconds is 1196 of 2664 s per worker
# (section 21.5), so skipping the gradient entirely would buy ~1.8x more
# candidates scored in the same time, and whether that trade is favourable is a
# DIFFERENT question needing a bypass at the single call site search.py:320.
# That bypass is not in this script and has not been tested. Do not report this
# screen as having settled whether the gradient earns its 45%.
#
# WHY POLICY IS THE SECOND FACTOR. Section 27.2 measured independent against
# population at ten seeds and found it null -- 0.470 against 0.529 ipSAE, 27.4
# against 30.5 A pose, both inside that screen's own ~0.11 / ~6 A sensitivity --
# which reversed section 24.9's default. But every arm there used gradient
# proposals. If all parents climb the same gradient, population has little
# diversity to preserve, which would explain a null. Crossing policy with
# proposal source tests that INTERACTION at no extra cost, since both proposal
# arms have to run regardless.
#
# WHAT IS PINNED, AND WHY. edit_budget 5 (section 27.3 closed 5-vs-7: budget 7
# gained no confidence and raised the flipped-pose rate to 26-27 of 33 against
# 18-23), weight_registry 0.0, 200 score calls, 64 sampling steps, stable
# aggregation, acceptance_temperature 0.02 -- identical in every cell. Section
# 21.7 records why: the earlier budget arm moved score calls, entropy and
# acceptance temperature together and confounded two comparisons at once.
#
# REGISTRY IS OFF ON PURPOSE. Section 27.2 found the term null at weight 0.5
# and section 27.5 explains that the null is uninterpretable until the per-term
# gradient-norm measurement of section 26.5 is made. Putting an uninterpreted
# term into an attribution experiment would confound the thing being attributed.
#
# THE ONE ASYMMETRY THAT CANNOT BE PINNED. target_entropy is 0.6 in the grad_
# cells and 1.0 in the unif_ cells, because that IS the manipulation. Section
# 21.7's rule is that every future comparison pins entropy across cells; this is
# the single case where that is impossible, and it is recorded here rather than
# passed over. Nothing else differs between a grad_ and a unif_ cell.
#
# POWER. Twenty seeds per cell resolves main effects at roughly 0.078 ipSAE,
# from section 23.2's variances scaled by sqrt(10/20). The proposal factor is a
# whole-distribution change and is the one this design can resolve. THE POLICY
# MAIN EFFECT PROBABLY CANNOT BE: section 27.2 measured it at 0.059, which sits
# below that threshold. Treat policy here as a screen for an interaction, not as
# a settled comparison, and do not read a 0.03 gap as a result.
#
# NOT RUN HERE, AND STILL OUTSTANDING: the per-term gradient norms and pairwise
# cosines (sections 26.5, 27.7 item 1), which currently block the interpretation
# of the registry null and which `p17_pose_diagnostics.py` does not yet report
# per term; Test B, the saliency rank of the correct residue on the 5-edit Alpha
# rung, which is free and needs no GPU; and the 5-edit Alpha anchor of section
# 23.1, which needs its ladder and calibration stages regenerated after the
# cluster's results/ was cleared (section 21.1).
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SEEDS_PER_CELL=20
SCORE_CALLS=200
GRADIENT_CALLS=200
PROPOSALS=2000
STEPS=64
DTYPE=bf16
AGGREGATION=stable
EDIT_BUDGET=5           # pinned, see header
WEIGHT_REGISTRY=0.0     # pinned off, see header
REGISTRY_CUTOFF=8.0
GRAD_ENTROPY=0.6        # the manipulation
UNIF_ENTROPY=1.0        # the manipulation: exact uniform, see header
ACCEPT_TEMP=0.02        # pinned, see header
OUT_ROOT=""
DRY_RUN=false
ALL_CELLS="grad_pop grad_indep unif_pop unif_indep"
CELLS="$ALL_CELLS"

usage() {
    sed -n '2,78p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --devices LIST        GPUs (default: inherited CUDA_VISIBLE_DEVICES else 0-7)
  --seeds-per-cell N    search seeds per cell (default: $SEEDS_PER_CELL)
  --score-calls N       unique scored sequences per worker (default: $SCORE_CALLS)
  --cells "a b"         subset of: $ALL_CELLS
  --output-dir PATH     default: results/p17_attribution_screen_<timestamp>
  --dry-run             print the plan; create nothing, load nothing
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --seeds-per-cell) SEEDS_PER_CELL="$2"; shift 2 ;;
        --score-calls) SCORE_CALLS="$2"; shift 2 ;;
        --cells) CELLS="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_attribution_screen_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
N_CELLS=$(echo "$CELLS" | wc -w)

cell_entropy() { case "$1" in unif_*) echo "$UNIF_ENTROPY" ;; *) echo "$GRAD_ENTROPY" ;; esac; }
cell_policy()  { case "$1" in *_indep) echo independent ;; *) echo population ;; esac; }
cell_source()  { case "$1" in unif_*) echo uniform ;; *) echo gradient ;; esac; }

echo "P17 attribution screen -- JN.1 only"
echo "  devices:        $DEVICES (${#DEV[@]})"
echo "  cells:          $CELLS"
echo "  seeds per cell: $SEEDS_PER_CELL  ->  $((N_CELLS * SEEDS_PER_CELL)) workers"
echo "  per worker:     $SCORE_CALLS score calls, $STEPS steps, $DTYPE, $AGGREGATION"
echo "  pinned:         edit_budget $EDIT_BUDGET, registry $WEIGHT_REGISTRY,"
echo "                  acceptance_temperature $ACCEPT_TEMP"
echo "  manipulated:    target_entropy $GRAD_ENTROPY (gradient) vs $UNIF_ENTROPY (uniform)"
echo "  output:         $OUT_ROOT"
echo
printf "  %-12s %-10s %-12s %s\n" CELL PROPOSALS POLICY ENTROPY
for c in $CELLS; do
    printf "  %-12s %-10s %-12s %s\n" "$c" "$(cell_source "$c")" \
        "$(cell_policy "$c")" "$(cell_entropy "$c")"
done
echo
echo "  Both arms compute the gradient; the uniform arm does not sample from"
echo "  it. Matched on score calls, gradient calls and wall-clock -- so this"
echo "  tests the proposal distribution, NOT whether the gradient earns its"
echo "  45% of wall-clock. See the header."
echo
echo "  Power: main effects resolve at ~0.078 ipSAE at this seed count. The"
echo "  policy main effect was measured at 0.059 and probably will not resolve."
echo

if "$DRY_RUN"; then
    echo "Dry run. Per cell, in order:"
    for c in $CELLS; do
        echo
        echo "  [$c]"
        echo "    run:  $PYTHON_BIN $DRIVER jn1 --output-dir $OUT_ROOT/$c/run --policies $(cell_policy "$c") --edit-budget $EDIT_BUDGET --weight-registry $WEIGHT_REGISTRY --target-entropy $(cell_entropy "$c") --search-seeds $SEEDS ..."
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
        proposal_source="uniform" if c.startswith("unif_") else "gradient",
        target_entropy=$UNIF_ENTROPY if c.startswith("unif_") else $GRAD_ENTROPY,
        policy="independent" if c.endswith("_indep") else "population",
        edit_budget=$EDIT_BUDGET,
        weight_registry=$WEIGHT_REGISTRY,
    )
print(json.dumps(dict(
    input="P17_JN1.pdb", cells=cells, seeds_per_cell=$SEEDS_PER_CELL,
    score_calls=$SCORE_CALLS, gradient_calls=$GRADIENT_CALLS,
    sampling_steps=$STEPS, dtype="$DTYPE", aggregation="$AGGREGATION",
    acceptance_temperature=$ACCEPT_TEMP, devices="$DEVICES",
    question="whether the gradient-weighted proposal pool beats a uniformly "
             "drawn one from the same feasible move set; section 21.5's "
             "conditional-versus-base-rate statistic is computed within the "
             "pool the gradient already selected and cannot reach this",
    uniform_mechanism="target_entropy == 1 makes _proposal_distribution "
                      "return np.full(count, 1/count) over the move set "
                      "_moves already builds. No new code path.",
    matched_on=["score_calls", "gradient_calls", "wall_clock"],
    does_not_test="whether the gradient earns its 45% of wall-clock; both "
                  "arms still pay for it. That needs a bypass at the single "
                  "call site search.py:320, which is not in this script and "
                  "has not been tested.",
    pinned_because="section 21.7: the earlier budget arm moved score calls, "
                   "entropy and acceptance temperature together",
    unpinnable_asymmetry="target_entropy differs between arms because that is "
                         "the manipulation; nothing else differs",
    registry_off_because="section 27.2 found the term null at 0.5 and section "
                         "27.5 shows that null is uninterpretable until the "
                         "per-term gradient-norm measurement is made",
    budget_pinned_because="section 27.3: budget 7 gained no confidence and "
                          "raised the flipped-pose rate to 26-27 of 33",
    power="~0.078 ipSAE main effects at this seed count (section 23.2 scaled); "
          "the policy main effect was measured at 0.059 and probably will not "
          "resolve",
    not_run_here=["per-term gradient norms and cosines (sections 26.5, 27.7)",
                  "Test B, saliency rank of the correct residue (free, no GPU)",
                  "the 5-edit Alpha anchor (section 23.1; needs ladder and "
                  "calibration regenerated per section 21.1)"],
), indent=2))
PY
echo "Plan: $OUT_ROOT/plan.json"

COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
        --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
        --max-proposals "$PROPOSALS"
        --acceptance-temperature "$ACCEPT_TEMP"
        --registry-contact-distance "$REGISTRY_CUTOFF" --save-saliency)

SUMMARY=()
for c in $CELLS; do
    echo
    echo "################ cell $c ################"
    ent=$(cell_entropy "$c"); pol=$(cell_policy "$c")
    echo "  proposals: $(cell_source "$c") (target_entropy $ent), policy: $pol"
    mkdir -p "$OUT_ROOT/$c"

    if "$PYTHON_BIN" "$DRIVER" jn1 \
        --output-dir "$OUT_ROOT/$c/run" --policies "$pol" \
        --edit-budget "$EDIT_BUDGET" --weight-registry "$WEIGHT_REGISTRY" \
        --target-entropy "$ent" \
        --search-seeds $SEEDS "${COMMON[@]}" \
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
      per-move: predicted_loss_delta against realized_score_delta and a
      per-position history. In the unif_ cells the delta is still recorded but
      was not used to sample, which makes those cells the control for section
      21.5's statistic rather than another instance of it.

  python examples/p17_site_gate_audit.py $OUT_ROOT/<cell>/run
      site recall, contact-face versus fixed-region engagement, and the
      rotation / centroid decomposition -- which is what separates a displaced
      designed chain from one flipped at the correct site (section 25.1)

The comparison to make, from each cell's tables/jn1_recovery.csv: mean and best
ipSAE and mean pose over the non-WT winners, per cell, with the per-cell spread.
Read the proposal factor first and the policy factor only as a screen.

Scale bar (section 19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. Section 27 reached mean ipSAE
0.457-0.529 and mean pose 26-37 A with every structure CAPRI-incorrect, best
DockQ 0.185 against an acceptable threshold of 0.23.

Establishes nothing about binding.
EOF
