#!/usr/bin/env bash
# One script: the continuous seeding stage on JN.1, then the discrete handoff.
# Run this on the cluster.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_continuous_stage.sh --dry-run
#   bash examples/run_p17_continuous_stage.sh --stages seed      # ~2 h at 600 steps
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
#                   prints the truncation verdict below. ~2 h at the 600 steps
#                   this now defaults to; the ~25 min figure in an earlier
#                   version of this line was the old 125-step cost.
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
# WHY THE SEEDING DEFAULTS CHANGED ON 2026-10-13, AND HOW MUCH OF IT IS PROVEN.
# The question above was never reached, because the relaxation never moved.
# Across seven replicates of three methods the seeds came back 0, 0, 0, 0, 1, 1
# and 2 edits from the parent on a budget of 5, and `hamming_from_parent = 0`
# means the argmax was still the parent at all 29 designable positions.
#
# ONE cause has direct evidence, measured on the CPU toy loss through the same
# `run_relaxation` that main() runs (tests/test_continuous_seed.py):
#
#   THE EDITBUDGET HINGE. `w * relu(E(s) - b)` with `E(s) = sum_pos (1 - p_parent)`
#   is the expected Hamming distance, which at a one-hot equals Hamming exactly
#   -- inert for the discrete search. On a RELAXED sequence its only route to
#   satisfaction is driving p_parent toward 1. At scale 5.0 and 125 steps the
#   optimizers wanted 4, 6 and 0 substitutions (bindcraft, colabdesign, apgm)
#   with it on and 27, 29 and 29 with it off, and it decides the outcome at
#   every step count and scale tried. `project_to_budget` already enforces the
#   budget exactly, so `SEED_EDIT_WEIGHT=0.0` loses nothing. The DISCRETE search
#   keeps 5.0.
#
#   TESTED ON THE REAL LOSS, 2026-10-14 (section 32): six runs, 600 steps,
#   scale 1.0, hinge 0.0, on OpenDDE at bf16 and 64 sampling steps.
#
#     bindcraft    15 wanted / 29 designable, both seeds (budget 5)
#     colabdesign  15 wanted, both seeds
#     apgm          0 wanted, both seeds
#
#   One cell, no hinge-on arm, so this does NOT attribute the movement to the
#   hinge rather than to the scale, the step count or the hard temperature --
#   all four changed together. What it says is that the toy's PREDICTION for
#   this cell holds for the two logit-space methods and fails for `apgm`,
#   which the toy grid put at 21 here. The real loss is
#   also far less swamped than the toy predicted for the other two -- 15 of
#   29, against 26-28 on the toy -- so the hand-off is a partially truncated
#   argmax, not a wholly truncated one. The kept five carry probability
#   margins of 0.707-0.985 across the four moving runs.
#
# Three more changes were made on principle and their effect is UNMEASURED:
#
#   THE INIT SCALE, 5.0 -> 1.0, to match ColabDesign's own wildtype start. An
#   earlier version of this header attributed the null to a 148:1 gradient
#   weight from the softmax chain rule; that was withdrawn the same day. It was
#   drawn from biohub_optimizer, which this stage does not use, and the toy shows
#   no such effect: with the hinge off the scale only changes how many steps a
#   logit-space method needs (scale 1.0 has moved by 24 steps, 5.0 has not; by
#   125 both have). A cluster probe (old code, hinge on, 125 steps, n=1 per
#   cell) gave colabdesign 5/2/1/3 edits at scale 1..4 against 0 at 5, and
#   bindcraft 0/1/2/2 against 1: mixed, and one seed.
#   THE STRAIGHT-THROUGH STAGE. For colabdesign it never ran at all (`hard=False`
#   for the whole stage); for bindcraft it ran 5 of 125 steps at temp 1e-2, where
#   the estimator's gradient is a saturated softmax's Jacobian and ColabDesign's
#   own `lr_scale = soft*temp` shrinks the step another 100x. `SEED_HARD_TEMP=1.0`
#   fixes that, and it matters for a reason the toy loss cannot test: OpenDDE was
#   trained on one-hot sequences, so this is the only branch that shows it an
#   in-distribution input (Fast SeqProp, arXiv:2005.11275: "the predictive power
#   of P may be poor on a continuous relaxation").
#   THE STEP COUNT, 125 -> 600, and `--steps` now actually REACHES bindcraft,
#   whose four stage lengths were hardcoded so every bindcraft run was 125 steps
#   whatever was asked for. Fast SeqProp reports un-normalised softmax
#   relaxation failing to converge in 20,000 updates and its normalised variant
#   converging in 200.
#
# `SEED_LOGIT_NORM` and `SEED_STOCHASTIC_HARD` are OFF. Both were shipped ON
# earlier on 2026-10-13 and both were wrong to ship that way. One real defect:
# on a UNIFORM start the auto-calibrated gamma is 0, and normalisation then
# multiplies every update back to exactly zero with a live gradient --
# ColabDesign's own un-seeded start is `np.zeros` and `--init noise` has spread
# ~0.01, so both were affected, and `project_to_budget` cannot show it because
# it only takes an argmax. The calibration now refuses a degenerate spread
# instead of freezing on it.
#
# A second argument given here at the time -- that gamma caps a 20-way softmax
# at about p=0.125 per residue, so the relaxation could never become confident
# -- was WRONG and is withdrawn. Normalisation is over the whole (29, 20) block,
# so gamma fixes the GLOBAL deviation: concentrating the spread in one row
# reaches p=0.91 at the same gamma. What gamma bounds is the aggregate, i.e.
# every row cannot be peaked at once.
#
# They stay off because they are unvalidated on the real loss and because gamma
# couples the initialisation to a standing constraint on every later iterate.
# Only the hinge has evidence behind it, so only the hinge is on: isolate it on
# the real loss before the full 600-step stage is spent. See section 31.10 of
# docs/P17_JN1.md and `bash examples/run_p17_hinge_probe.sh`.
#
# WHAT THE ANCHORED INIT IS FOR, AND WHAT IT NO LONGER DOES. `--init parent`
# (the default since 2026-10-07) adds `--init-logit-scale` to the parent column.
# The old noise init opened the EditBudget expectation at roughly 27.6 against a
# budget of 5 -- a hinge of 5.0 * relu(27.6 - 5) = 113, dwarfing the ~34 carried
# by every other term combined, so the first many steps spent their gradient
# walking back toward the parent.
#
# The arithmetic depends on the SCALE, and the numbers quoted here until
# 2026-10-14 were scale 5.0's:
#
#   scale 5.0   p_parent 0.886   expectation 29 x 0.114 =  3.3   hinge   0
#   scale 1.0   p_parent 0.125   expectation 29 x 0.875 = 25.4   hinge ~102
#
# So at the scale this script now configures, the anchored init does NOT open
# inside the budget -- it opens in the same regime the noise init did. That is
# harmless here only because SEED_EDIT_WEIGHT is 0.0. Re-enable the hinge at
# scale 1.0 and the problem the anchored init was introduced to solve is back.
# Covered by tests/test_continuous_seed.py.
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
# WHAT IS PINNED. edit_budget 5 -- the DEFAULT, not a constant: `--edit-budget`
# overrides it, and `plan.json` records both the value used and the default so
# a reader can see which runs differ. Raising it on a `--stages search` run
# that reuses an earlier `--seed-dir` is the one case that matters: the seed
# was projected onto the SEED stage's budget, so a larger search budget leaves
# the difference free for the search to spend on top of the seed, instead of
# the exchange-only moves a saturated start is restricted to. Every cell in
# one invocation still shares one budget. Also pinned:
# weight_registry 0.0, target_entropy 0.6,
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
ALL_METHODS="apgm bindcraft colabdesign"
METHODS="$ALL_METHODS"
SEED_REPLICATES=3       # continuous-stage seeds per method
SEED_STEPS=600          # see "WHY THE SEEDING DEFAULTS CHANGED" in the header
SEED_LR=0.1
INIT=parent             # see header; `noise` reproduces the pre-2026-10-07 start
INIT_LOGIT_SCALE=1.0    # was 5.0; matches ColabDesign's own wildtype start
SEED_EDIT_WEIGHT=0.0    # EditBudget hinge OFF during the relaxation only
SEED_HARD_TEMP=1.0      # was 1e-2, where the straight-through gradient is ~0
SEED_LOGIT_NORM=0       # OFF: unvalidated on the real loss, see the header.
                        # NOT because gamma "caps the softmax" -- that claim is
                        # withdrawn; normalisation is over the whole (29, 20)
                        # block, so gamma bounds the aggregate, not each row.
SEED_STOCHASTIC_HARD=0  # OFF: argmax is what ColabDesign does
# Resolved here, not inside the stage: `set -u` is on and the --dry-run listing
# below reads these too, so resolving them in the stage body would abort the
# dry run on an unbound variable without ever reaching the real command.
SEED_NORM_FLAG=$([[ "$SEED_LOGIT_NORM" == "1" ]] \
    && echo --logit-norm || echo --no-logit-norm)
SEED_HARD_FLAG=$([[ "$SEED_STOCHASTIC_HARD" == "1" ]] \
    && echo --stochastic-hard || echo --no-stochastic-hard)
SEEDS_PER_CELL=20
SCORE_CALLS=200
GRADIENT_CALLS=200
PROPOSALS=2000
STEPS=64
DTYPE=bf16
AGGREGATION=stable
EDIT_BUDGET=5           # default, shared between the stages, see header
EDIT_BUDGET_DEFAULT=5   # what --edit-budget overrode, for the echo and the plan
WEIGHT_REGISTRY=0.0     # pinned off, see header
REGISTRY_CUTOFF=8.0
ENTROPY=0.6             # pinned, matches the section 29 grad_ cells
ACCEPT_TEMP=0.02        # pinned, see header
POLICY=population
OUT_ROOT=""
DRY_RUN=false
ALLOW_BUSY_GPUS=false
FAILURES=0
SEED_WARNINGS=0   # seeds that reach the search with no gain over the parent
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
  --edit-budget N       edits allowed from the budget anchor (default:
                        $EDIT_BUDGET). Both stages use it. Raising it for a
                        search that reuses an EARLIER --seed-dir is how a
                        seeded cell gets free moves on top of its seed.
  --seed-dir PATH       reuse an earlier stage-seed output instead of rerunning
  --output-dir PATH     default: results/p17_continuous_stage_<timestamp>
  --allow-busy-gpus     start even though a device cannot hold the
                        preallocation alongside what is already resident
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
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --seed-dir) SEED_DIR="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --allow-busy-gpus) ALLOW_BUSY_GPUS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$EDIT_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
    echo "--edit-budget must be a positive integer, got '$EDIT_BUDGET'" >&2
    exit 2
fi
# Said out loud wherever the budget is printed. A run at a budget the rest
# of this header was written against 5 is a different experiment, and the
# word "pinned" would otherwise still be on the line.
BUDGET_NOTE=""
[[ "$EDIT_BUDGET" != "$EDIT_BUDGET_DEFAULT" ]] && \
    BUDGET_NOTE=" (OVERRIDDEN, default $EDIT_BUDGET_DEFAULT)"

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_continuous_stage_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
# Distinctness, not just count: eight entries all reading `0` passed the
# capacity check and put every job on one GPU, which is the failure this
# script's own refusal exists to prevent.
for d in "${DEV[@]}"; do
    if [[ -z "$d" || ! "$d" =~ ^[0-9]+$ ]]; then
        echo "Refusing to start: --devices has an empty or non-numeric entry" >&2
        echo "('$d' in '$DEVICES')." >&2
        exit 2
    fi
done
if (( $(printf '%s\n' "${DEV[@]}" | sort -u | wc -l) != ${#DEV[@]} )); then
    echo "Refusing to start: --devices repeats a device ('$DEVICES'). Counting" >&2
    echo "entries does not establish that many distinct GPUs, and two OpenDDE" >&2
    echo "processes on one device is what killed apgm_r0 on 2026-10-07." >&2
    exit 2
fi
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
REPLICATES=$(seq 0 $((SEED_REPLICATES - 1)) | tr '\n' ' ')
[[ -z "$SEED_DIR" ]] && SEED_DIR="$OUT_ROOT/seeds"

# Absolute, BEFORE the `cd "$REPO_DIR"` further down. A relative --output-dir
# was created in the invocation directory and then read from the repo root, so
# launching from anywhere but the repo put the results in one place and looked
# for them in another.
_abs() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
OUT_ROOT="$(_abs "$OUT_ROOT")"
SEED_DIR="$(_abs "$SEED_DIR")"

# Validate the selections. An unrecognised value used to select NOTHING and
# exit 0: `--stages sead` ran no stage at all and reported success, which is
# the worst possible response to a typo in a 6-hour job.
_check_selection() {  # flag, allowed, selected
    local flag="$1" allowed="$2" selected="$3" item
    if [[ -z "${selected// /}" ]]; then
        echo "Refusing to start: $flag is empty, which would run nothing." >&2
        exit 2
    fi
    for item in $selected; do
        if [[ " $allowed " != *" $item "* ]]; then
            echo "Refusing to start: $flag got '$item', which is not one of:" >&2
            echo "    $allowed" >&2
            echo "An unrecognised value selects nothing and would exit 0." >&2
            exit 2
        fi
    done
    if (( $(printf '%s\n' $selected | sort -u | wc -l) != $(echo $selected | wc -w) )); then
        echo "Refusing to start: $flag repeats a value ('$selected'). Duplicates" >&2
        echo "write to the same output paths, so the runs would overwrite each" >&2
        echo "other while the count suggested more data." >&2
        exit 2
    fi
}
_check_selection --stages  "$ALL_STAGES"  "$STAGES"
_check_selection --cells   "$ALL_CELLS"   "$CELLS"
_check_selection --methods "$ALL_METHODS" "$METHODS"

# Re-join on single spaces, because validation word-splits on any whitespace
# while `has_stage` does a literal `*" $1 "*` match. A tab-separated
# "seed<TAB>search" validated as two good tokens and then matched NO stage, so
# the script ran zero jobs and exited 0. The two must see the same tokens.
STAGES="$(echo $STAGES)"
CELLS="$(echo $CELLS)"
METHODS="$(echo $METHODS)"

source "$SCRIPT_DIR/lib_gpu_capacity.sh"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# The seed each stage-search cell starts from: the lowest-numbered replicate
# that actually exists. `parent` is the control and supplies none, so the
# harness starts at the reference and the budget anchor is left at its default.
#
# It reads any replicate rather than `_r0` specifically because on 2026-10-08
# nine seeding runs over eight GPUs put two on device 0, both of those died,
# and one of them was `apgm_r0` -- which skipped the whole apgm cell while
# `apgm_r1` and `apgm_r2` sat finished and unused.
cell_seed_json() {
    local r
    for r in $REPLICATES; do
        [[ -f "$SEED_DIR/${1}_r${r}.json" ]] && { echo "$SEED_DIR/${1}_r${r}.json"; return 0; }
    done
    echo "$SEED_DIR/${1}_r0.json"   # nothing exists; the caller reports it missing
    return 1
}

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
echo "          budget:       $EDIT_BUDGET$BUDGET_NOTE, shared with the search stage"
fi
if has_stage search; then
echo
echo "  [search] cells:       $CELLS"
echo "           seeds/cell:  $SEEDS_PER_CELL  ->  $(echo "$CELLS" | wc -w) x $SEEDS_PER_CELL workers"
echo "           per worker:  $SCORE_CALLS score calls, $STEPS steps, $DTYPE, $AGGREGATION"
echo "           pinned:      edit_budget $EDIT_BUDGET$BUDGET_NOTE, registry $WEIGHT_REGISTRY,"
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
echo "  Section 26.5's per-term gradient norms WERE measured on 2026-10-13"
echo "  (section 31.6): pose 0.511 of the norm at the reference against"
echo "  ablang2 0.085, and ablang2 0.356 at a 5-edit winner, where the"
echo "  structure terms have collapsed 4-7x. contact|registry is +0.942 at"
echo "  the reference. The hinge's own gradient was 0.000 at both points,"
echo "  which is the relu'(0) = 0 result, so the synthetic sweep's coupling"
echo "  question is NOT answered by them: it needs the hinge active."
echo

if "$DRY_RUN"; then
    echo "Dry run."
    if has_stage seed; then
        echo
        echo "  [seed] $( echo "$METHODS" | wc -w ) x $SEED_REPLICATES runs:"
        # The real run refuses runs > devices (see stage `seed`). Say so HERE,
        # or the plan looks fine and the launch dies two seconds in.
        _planned=$(( $(echo "$METHODS" | wc -w) * $(echo "$REPLICATES" | wc -w) ))
        if (( _planned > ${#DEV[@]} )); then
            echo
            echo "    ⚠️  THIS PLAN WILL BE REFUSED: $_planned runs over" >&2
            echo "        ${#DEV[@]} devices. Pass --seed-replicates" >&2
            echo "        $(( ${#DEV[@]} / $(echo "$METHODS" | wc -w) )), or --devices with at" >&2
            echo "        least $_planned entries." >&2
            echo >&2
        fi
        for m in $METHODS; do
            for r in $REPLICATES; do
                echo "    $PYTHON_BIN $SEEDER --method $m --seed $r --steps $SEED_STEPS --init $INIT --init-logit-scale $INIT_LOGIT_SCALE --edit-budget $EDIT_BUDGET --weight-edit-budget $SEED_EDIT_WEIGHT --hard-temp $SEED_HARD_TEMP $SEED_NORM_FLAG $SEED_HARD_FLAG --weight-registry $WEIGHT_REGISTRY --sampling-steps $STEPS --opendde-dtype $DTYPE --lr $SEED_LR --out $SEED_DIR/${m}_r${r}.json"
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
# The guard used to test `XLA_CLIENT_MEM_FRACTION` -- a name nothing sets and
# nothing reads -- while exporting `XLA_PYTHON_CLIENT_MEM_FRACTION`. Harmless,
# because the `:-` default preserves a caller's value either way, but it was a
# condition on a variable that does not exist.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
gpu_capacity_check "$DEVICES" "$XLA_PYTHON_CLIENT_MEM_FRACTION" "$ALLOW_BUSY_GPUS"
export PYTHONUNBUFFERED=1

# A FRESH output directory. Reusing one and then failing every seed job made
# the gate report the PREVIOUS run's JSONs as this run's result -- the worst
# possible failure, because it is silent and it reads as data.
if [[ -e "$OUT_ROOT" ]] && [[ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]]; then
    echo "Refusing to start: $OUT_ROOT exists and is not empty. Results left by" >&2
    echo "an earlier run would be read as this run's. Pass --output-dir with a" >&2
    echo "fresh path. Found:" >&2
    ls -A "$OUT_ROOT" | sed 's/^/        /' >&2
    exit 2
fi

# `--ignore-submodules=untracked`: plain porcelain reports ` M OpenDDE` whenever
# the nested checkout has untracked content, which it permanently does, so with
# no flag this is the constant `true` on a clean tree. `untracked` rather than
# `all`, which would also hide a tracked edit inside OpenDDE and a changed
# OpenDDE revision. Neither records the INSTALLED dependency, which patches/
# rewrites in site-packages at runtime.
# `--seed-dir` exists to REUSE an earlier seed output with `--stages search`.
# If stage `seed` is also running it writes into that directory, and the
# OUT_ROOT guard above does not cover it -- so an external seed directory let
# a run overwrite an old result and exit 0.
if has_stage seed && [[ -n "$(ls -A "$SEED_DIR" 2>/dev/null)" ]]; then
    echo "Refusing to start stage 'seed': $SEED_DIR is not empty, and stage" >&2
    echo "'seed' writes into it -- this would overwrite an earlier run's seeds." >&2
    echo "Use --stages search to reuse them, or point --seed-dir somewhere new." >&2
    echo "Found:" >&2
    ls -A "$SEED_DIR" | sed 's/^/        /' >&2
    exit 2
fi

GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$REPO_DIR" status --porcelain --ignore-submodules=untracked 2>/dev/null)"
if [[ -n "$GIT_DIRTY" ]]; then
    echo "NOTE: the working tree is dirty, so this run is not reproducible from a" >&2
    echo "      commit alone. Commit first if the result is to be cited:" >&2
    echo "$GIT_DIRTY" | sed 's/^/        /' >&2
fi

echo "Applying dependency patches..."
for name in outer_product_mean structural_token_expander bf16_dtype aggregation padding; do
    "$PYTHON_BIN" "$REPO_DIR/patches/patch_jopendde_$name.py"
done >"$OUT_ROOT/patches.log" 2>&1
echo "Applied five OpenDDE patches."

"$PYTHON_BIN" - <<PY > "$OUT_ROOT/plan.json"
import json
print(json.dumps(dict(
    input="P17_JN1.pdb", stages="$STAGES".split(),
    git_commit="$GIT_COMMIT",
    git_dirty=bool("""$GIT_DIRTY""".strip()),
    git_dirty_entries=len([l for l in """$GIT_DIRTY""".splitlines() if l.strip()]),
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
                     "carried by every other term combined. The anchored start "
                     "opens at 3.3 AT INIT SCALE 5.0 (p_parent 0.886), where "
                     "the hinge is inactive at step 0. At the configured scale "
                     "1.0 p_parent is only 0.125, so the expectation opens at "
                     "29 x 0.875 = 25.4 and the hinge would open at ~102 -- "
                     "the very regime the anchored init was added to avoid. "
                     "That is harmless only because SEED_EDIT_WEIGHT is 0.0; "
                     "re-enabling the hinge at scale 1.0 reinstates it.",
    ),
    search_stage=dict(
        cells="$CELLS".split(), seeds_per_cell=$SEEDS_PER_CELL,
        score_calls=$SCORE_CALLS, gradient_calls=$GRADIENT_CALLS,
        edit_budget=$EDIT_BUDGET, edit_budget_default=$EDIT_BUDGET_DEFAULT,
        weight_registry=$WEIGHT_REGISTRY,
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
         "residue) and located the failure in truncation. Near 29 means each "
         "seed is an argmax of a heavily budget-truncated optimum: projection "
         "ranks by margins the optimisation produced, so the two jointly "
         "decide the hand-off, but most of what was wanted is discarded. At or "
         "under the budget means no FURTHER truncation after the argmax -- not "
         "that the argmax preserved the relaxed solution, which only section "
         "30.3's screen speaks to. It does not gate execution: with both "
         "stages enabled, search starts after this prints. Use --stages seed "
         "to read it first.",
    prediction="null. Section 30.3: where discrete search saturates its "
               "budget, continuous seeding had no headroom, and every "
               "synthetic hand-off came in marginally worse than discrete "
               "search alone -- 99.8% and 99.5% of the exhaustive optimum "
               "against 99.9%, all far inside the 0.855 SEM.",
    gradient_norms="measured 2026-10-13, section 31.6. pose 0.511 of the norm "
                   "at the reference, ablang2 0.085; at a 5-edit winner the "
                   "structure terms have collapsed 4-7x and ablang2 holds flat "
                   "at 0.356. contact|registry +0.942 at the reference, +0.641 "
                   "at the winner. The EditBudget term measured 0.000 at both "
                   "points, which is relu'(0) = 0 at a feasible one-hot -- so "
                   "these norms do NOT settle the synthetic sweep's coupling "
                   "question, which needs the hinge active on a relaxed input.",
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
    n_runs=$(( $(echo "$METHODS" | wc -w) * SEED_REPLICATES ))
    if (( n_runs > ${#DEV[@]} )); then
        echo "  ⚠️  $n_runs runs over ${#DEV[@]} GPUs: $(( n_runs - ${#DEV[@]} )) device(s)"
        echo "      will host two OpenDDE instances at once. On 2026-10-08 both"
        echo "      runs on the doubled-up device died. --seed-replicates"
        echo "      $(( ${#DEV[@]} / $(echo "$METHODS" | wc -w) )) keeps one run per GPU."
    fi
    # Refuse rather than double up. `i % ${#DEV[@]}` silently wraps, which is
    # how 9 seeding runs landed on 8 GPUs on 2026-10-07 with two processes on
    # device 0; both died and took the whole apgm cell with them (see the
    # header). The advice printed above is not enough on its own.
    n_runs=$(( $(echo "$METHODS" | wc -w) * $(echo "$REPLICATES" | wc -w) ))
    if (( n_runs > ${#DEV[@]} )); then
        echo "Refusing to start stage 'seed': $n_runs runs over ${#DEV[@]} devices" >&2
        echo "would put two processes on one GPU, which is what killed apgm_r0 on" >&2
        echo "2026-10-07. Reduce SEED_REPLICATES to $(( ${#DEV[@]} / $(echo "$METHODS" | wc -w) ))," >&2
        echo "or pass --devices with at least $n_runs entries." >&2
        exit 2
    fi
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
                --weight-edit-budget "$SEED_EDIT_WEIGHT" \
                --hard-temp "$SEED_HARD_TEMP" \
                $SEED_NORM_FLAG $SEED_HARD_FLAG \
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
        FAILURES=$((FAILURES + failed))
        echo "  $failed of $i seeding runs failed; see $SEED_DIR/*.log" >&2
    else
        SUMMARY+=("seed: $i/$i PASSED")
    fi

    echo
    echo "---- the gate: what the relaxation actually wanted ----"
    # Named files, not a glob. A glob reads whatever JSON is in the directory,
    # so a reused output path plus a run in which every job died reported the
    # PREVIOUS run's seeds as this run's result.
    # name:method:seed -- presence and readability are not enough. Six
    # correctly NAMED files reporting the wrong method and seed were accepted,
    # because nothing compared a result's contents with the job that was
    # supposed to produce it.
    EXPECTED_SEEDS=()
    for m in $METHODS; do
        for r in $REPLICATES; do EXPECTED_SEEDS+=("${m}_r${r}.json:${m}:${r}"); done
    done
    # No pipe: `pipefail` plus the gate's own nonzero exit would abort the
    # whole script inside the pipeline, so the summary and exit code below
    # would never run.
    gate_status=0
    "$PYTHON_BIN" - "$SEED_DIR" "${EXPECTED_SEEDS[@]}" > "$SEED_DIR/gate.txt" 2>&1 <<'PY' || gate_status=$?
import json, sys
from pathlib import Path

root, specs = Path(sys.argv[1]), sys.argv[2:]
names = [s.split(":")[0] for s in specs]
rows, missing, broken, mismatched = [], [], [], []
for spec in specs:
    name, want_method, want_seed = spec.split(":")
    p = root / name
    if not p.exists():
        missing.append(name)
        continue
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        print(f"  {p.name}: unreadable ({exc})")
        broken.append(name)
        continue
    wrong = []
    if str(d.get("method")) != want_method:
        wrong.append(f"method {d.get('method')!r} != {want_method!r}")
    if str(d.get("seed")) != want_seed:
        wrong.append(f"seed {d.get('seed')!r} != {want_seed}")
    if wrong:
        print(f"  {p.name}: IDENTITY MISMATCH " + "; ".join(wrong))
        mismatched.append(name)
        continue
    rows.append(d)
    margins = [e["margin"] for e in d["edits"]]
    # The loss at the two endpoints, when the run recorded it. `0 wanted` and
    # `26 wanted` are both claims about WHERE the relaxation ended; this is
    # the only field that says whether it ended anywhere good, and it makes
    # two runs' basins comparable (section 32.5).
    if d.get("loss_at_relaxed") is not None:
        print(f"  {'':12} {'':6}   loss {d['loss_at_init']} -> "
              f"{d['loss_at_relaxed']}  (descent {d['loss_descent']})")
    head = (f"  {d['method']:<12} seed {d['seed']}  "
            f"{d['hamming_from_parent']} edits kept / "
            f"{d['candidate_substitutions_considered']} wanted")
    if margins:
        print(f"{head}   margins {min(margins):.3f}-{max(margins):.3f}")
        # A tie across the cut means the kept set was chosen by the sort's
        # fall-through -- position index, descending -- and not by the
        # relaxation. A saturated run ties completely, and then every
        # replicate hands over the same five positions (section 32.5).
        discarded = d.get("projection_tied_discarded")
        if discarded:
            print(f"      TIED AT THE CUT: {d['projection_tied_group']} "
                  f"candidates share the margin of the last kept one and "
                  f"{discarded} of them were discarded by position index, not "
                  f"by the optimisation. These five edits are an arbitrary "
                  f"member of that group, and every replicate picks the SAME "
                  f"one, so they are not independent draws.")
        elif d.get("projection_margin_gap_at_cut") == 0.0:
            print("      the cut sits on a zero margin gap; see "
                  "projection_tied_group in the JSON")
    else:
        # 0 wanted has two causes and the argmax cannot separate them: the
        # anchored start already has the parent on top everywhere, so a
        # relaxation that converged on the parent and one that never moved
        # look identical here. `relaxed_peak_probability_mean` separates them
        # -- at `--init-logit-scale 1.0` the start's own peak IS 0.125.
        peak = d.get("relaxed_peak_probability_mean")
        if peak is None:
            print(f"{head}   (the relaxation asked for no substitution at all;"
                  f" pre-2026-10-14 seed JSON, so whether it MOVED is not"
                  f" recorded)")
        else:
            moved = "converged on the parent" if peak > 0.3 else \
                    "may never have left the start"
            print(f"{head}   (no substitution asked for; peak probability "
                  f"{peak} mean, {d.get('relaxed_parent_probability_mean')} "
                  f"on the parent -- {moved})")

if rows:
    wanted = [d["candidate_substitutions_considered"] for d in rows]
    budget = rows[0]["edit_budget"]
    print()
    # `designable 29` was a hardcoded constant here. It is correct today and
    # it is still the denominator the `>= 25` threshold below was chosen
    # against, but nothing made it track `CDR_RESIDUE_INDICES_1IDX`: changing
    # the mask would have left the gate printing a stale denominator under a
    # threshold that no longer meant what it says. Read it from the runs.
    recorded = {d["designable_positions"] for d in rows
                if d.get("designable_positions") is not None}
    if len(recorded) > 1:
        sys.exit(f"  the seed runs disagree on how many positions are "
                 f"designable ({sorted(recorded)}); they are not one "
                 f"comparison and the gate below would mean nothing")
    # A MISSING count is not a disagreeing one, and saying so would be the
    # wrong refusal with the right exit code. But a directory holding both
    # kinds holds output from two different builds of the seeder, which is
    # equally not one comparison.
    unrecorded = [d for d in rows if d.get("designable_positions") is None]
    if recorded and unrecorded:
        sys.exit(f"  {len(unrecorded)} of {len(rows)} seed runs predate "
                 f"2026-10-14 and do not record how many positions were "
                 f"designable, while the rest do: this directory holds output "
                 f"from two builds of the seeder, so it is not one comparison")
    designable = recorded.pop() if recorded else None
    shown = "29 (assumed; not recorded before 2026-10-14)" \
        if designable is None else str(designable)
    print(f"  substitutions wanted: min {min(wanted)}, max {max(wanted)}, "
          f"budget {budget}, designable {shown}")
    if designable is not None and designable != 29:
        print(f"  NOTE: the >= 25 threshold below was chosen against 29 "
              f"designable positions, not {designable}. Read the per-seed "
              f"counts, not the verdict.")
    n = len(wanted)
    n_heavy = sum(1 for w in wanted if w >= 25)
    n_fits = sum(1 for w in wanted if w <= budget)
    print(f"  per-seed regimes: {n_heavy}/{n} want >= 25, "
          f"{n_fits}/{n} fit the budget, {n - n_heavy - n_fits}/{n} between")
    if max(wanted) >= 25:
        print(f"  VERDICT: truncation regime for {n_heavy} of {n} seeds. Those")
        print("  seeds want nearly every designable position changed, so each")
        print("  of THEM is an argmax of a heavily budget-truncated optimum.")
        print("  The verdict is on the maximum, so it does not describe the")
        print("  other seeds -- read the per-seed lines above before treating")
        print("  this as a property of the stage. Projection ranks by margins")
        print("  the optimisation produced, so the two jointly decide the")
        print("  hand-off; what is discarded is most of what was wanted. This")
        print("  is the failure mode section 30.3 isolated. Consider the masked")
        print("  k=budget variant (section 28.5 mechanism 2) before spending")
        print("  the night.")
    elif max(wanted) <= budget:
        print("  VERDICT: no truncation. Every wanted substitution fits the")
        print("  budget, so the projection discards nothing. That is NOT the")
        print("  same as the hand-off carrying the relaxed optimum intact:")
        print("  the argmax itself may have lost something, and only section")
        print("  30.3's synthetic screen speaks to that. What it does mean is")
        print("  that no FURTHER truncation happens after the argmax.")
    else:
        print("  VERDICT: partial truncation. The projection is discarding")
        print(f"  some substitutions the relaxation asked for -- for {n - n_fits}")
        print(f"  of {n} seeds -- but is not swamped. The stage-search result is")
        print("  interpretable with the margin spread above reported alongside")
        print("  it, per seed rather than in aggregate.")
    print()
    print("  Unique seed sequences: "
          f"{len({d['seed_sequence'] for d in rows})} of {len(rows)} runs")

if missing or broken or mismatched:
    print()
    if mismatched:
        print(f"  WRONG RESULTS: {len(mismatched)} of {len(names)} seed JSONs do "
              f"not carry the method and seed of the job that should have "
              f"written them: {', '.join(mismatched)}")
    if missing:
        print(f"  INCOMPLETE: {len(missing)} of {len(names)} seed runs produced "
              f"no JSON: {', '.join(missing)}")
    if broken:
        print(f"  UNREADABLE: {len(broken)} of {len(names)} seed JSONs could not "
              f"be parsed: {', '.join(broken)}")
    print("  The verdict above, if any, covers only the runs that finished, so")
    print("  it is NOT the per-method comparison this stage was asked for. A")
    print("  job can exit 0 and write nothing, which is why this is checked")
    print("  against the expected file list rather than against exit codes.")
if not rows:
    print()
    print("  NO RESULTS AT ALL. There is no gate to read.")
if missing or broken or mismatched or not rows:
    sys.exit(1)
PY
    cat "$SEED_DIR/gate.txt"
    if (( gate_status != 0 )); then
        FAILURES=$((FAILURES + 1))
        SUMMARY+=("seed: the gate found no usable results")
    fi
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
            seed_json="$(cell_seed_json "$c" || true)"
            if [[ ! -f "$seed_json" ]]; then
                SUMMARY+=("$c: SKIPPED (no seed at $seed_json)")
                # A skipped cell is a missing arm of the comparison, so it
                # counts. Without this, `--stages search` with no seeds on disk
                # skips every seeded cell, runs the parent control alone, and
                # exits 0.
                FAILURES=$((FAILURES + 1))
                echo "  no seed JSON at $seed_json; skipping cell $c" >&2
                continue
            fi
            # The gate only runs in stage `seed`, so a seed directory reused
            # with `--stages search` is never identity-checked there. Check it
            # here: the cell name IS the method that should have produced the
            # file, and a mislabelled seed would send the search off from the
            # wrong starting sequence under the right cell's name.
            # Method AND replicate. Checking only the method accepted
            # `apgm_r0.json` carrying `"seed": 99` -- the right method from the
            # wrong run, which is exactly the file a half-finished earlier
            # directory leaves behind. The replicate is taken from the
            # filename, which is what `cell_seed_json` selected on.
            if ! "$PYTHON_BIN" -c '
import json, re, sys
path = sys.argv[1]
want_method = sys.argv[2]
d = json.load(open(path))
hit = re.search(r"_r(\d+)\.json$", path)
want_seed = hit.group(1) if hit else None
got_method = str(d.get("method"))
got_seed = str(d.get("seed"))
wrong = []
if got_method != want_method:
    wrong.append("method " + got_method + " != " + want_method)
if want_seed is not None and got_seed != want_seed:
    wrong.append("seed " + got_seed + " != " + want_seed)
if wrong:
    sys.exit("  " + path + ": IDENTITY MISMATCH " + "; ".join(wrong))
' "$seed_json" "$c"; then
                SUMMARY+=("$c: FAILED (seed JSON identity does not match)")
                FAILURES=$((FAILURES + 1))
                echo "  cell $c: the seed JSON does not carry the method and" >&2
                echo "  replicate of the run that should have written it, so it" >&2
                echo "  cannot be the seed for this cell; skipping." >&2
                continue
            fi
            seq=$("$PYTHON_BIN" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["seed_sequence"])' \
                "$seed_json")
            spent=$("$PYTHON_BIN" -c \
                'import json,sys; print(json.load(open(sys.argv[1]))["hamming_from_parent"])' \
                "$seed_json")
            echo "  start: $seed_json ($spent of $EDIT_BUDGET edits spent)"
            # Is the seed worth the edits it spends? A hand-off that does not
            # beat the parent under the seeder's own loss cannot help: it
            # arrives with its budget committed and no advantage, so the
            # search's best move is to revert it and climb from the parent.
            # That is what happened on 2026-10-15, where a seed that descended
            # 28.25 in SOFT space entered the search scoring 0.0002 against the
            # parent's 0.0000. `rounded_loss` in the seed stage records the
            # hard-sequence number; seeds written before 2026-10-16 lack it.
            gate=$("$PYTHON_BIN" -c '
import json, sys
d = json.load(open(sys.argv[1]))
if "rounded_gain_over_parent" not in d:
    print("UNRECORDED seed predates the check; rerun --stages seed to record it")
elif d["rounded_gain_over_parent"] is None:
    print("UNRECORDED the seed stage could not evaluate it: "
          + str(d.get("rounded_loss_eval_error") or "no reason recorded"))
elif d["rounded_gain_over_parent"] > 0:
    print("OK beats the parent by " + str(d["rounded_gain_over_parent"]))
else:
    print("USELESS gain " + str(d["rounded_gain_over_parent"]) + " over the parent")
' "$seed_json")
            case "$gate" in
                OK*)
                    echo "  seed value: ${gate#OK }" ;;
                UNRECORDED*)
                    echo "  seed value: UNKNOWN -- ${gate#UNRECORDED }" ;;
                *)
                    echo "  seed value: NO GAIN -- ${gate#USELESS }." >&2
                    echo "  This seed is at or below the parent under the" >&2
                    echo "  objective, and spends $spent of $EDIT_BUDGET edits to get" >&2
                    echo "  there, leaving $((EDIT_BUDGET - spent)) free moves against the" >&2
                    echo "  control's $EDIT_BUDGET. Expect it to revert and climb from" >&2
                    echo "  the parent; the cell measures initialisation, not seeding." >&2
                    SEED_WARNINGS=$((SEED_WARNINGS + 1)) ;;
            esac
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
            FAILURES=$((FAILURES + 1))
            echo "  cell $c failed; continuing." >&2
        fi
    done
fi

echo
echo "################ Done ################"
if ((${#SUMMARY[@]})); then printf '%s\n' "${SUMMARY[@]/#/  }"; fi
# A seed with no gain over the parent is not a failure -- the cell ran and its
# numbers are real -- but it is not the experiment the cell was meant to be, so
# say so once at the end where it cannot be missed.
if ((SEED_WARNINGS)); then
    echo
    echo "NOTE: $SEED_WARNINGS seeded cell(s) started from a seed that does not beat the"
    echo "parent under the objective. Those cells measure where the search starts,"
    echo "not whether seeding helps, because the search can revert the seed and"
    echo "climb from the parent for free. Read them accordingly."
fi
# Exit nonzero on any failure. These used to go into SUMMARY and no further, so
# a run in which every job died still exited 0 -- and an exit code is what a
# scheduler, a wrapper script or a tired reader actually checks.
if ((FAILURES)); then
    echo >&2
    echo "FAILED: $FAILURES stage failure(s) above, so this is NOT the comparison" >&2
    echo "the script was asked for. Fix them and re-run before reading anything" >&2
    echo "into the numbers." >&2
    exit 1
fi
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
      CLI. It reports this project's own bar first: candidates at or under
      10 A iRMSD, counted per candidate rather than per structure. On the
      section 29 screen that was gradient 16/40 against uniform 4/40, best
      6.66 A. Every structure is CAPRI-incorrect at best DockQ 0.174, which is
      a stricter standard than this project's, so read DockQ as a floor.

Scale bar (section 19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. The section 29 screen's grad_pop
cell -- the nominal configuration of the \`parent\` control here -- reached mean
ipSAE 0.567 and put 7 of 20 candidates inside the 10 A iRMSD bar, best 6.66 A.

A null here confirms section 30.3's synthetic prediction. It does not establish
that relaxation cannot help, only that it did not help at this budget with this
projection.

Establishes nothing about binding.
EOF
