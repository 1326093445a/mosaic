#!/usr/bin/env bash
# One script: everything outstanding as of 2026-10-08, cheapest and most
# diagnostic first, including the salvage of the night that failed.
#
#   cd /storage/frank/mosaic
#   bash examples/run_p17_consolidated.sh --dry-run
#   bash examples/run_p17_consolidated.sh --stages "free norms salvage"   # minutes
#   bash examples/run_p17_consolidated.sh                                 # all of it
#
# ORDERING PRINCIPLE. Thirty sections of this project have produced nulls that
# cannot be told apart from each other, and the reason is that the expensive
# runs went first. This script inverts that. Each stage is cheaper than the next
# and narrows what the next one can mean, so a stop after any stage leaves
# something interpretable behind.
#
#   free      seconds, no GPU    Test B, and the accidental reproducibility read
#   norms     ~2 min, 1 GPU      §26.5's per-term gradient norms and cosines
#   salvage   ~10 min, N GPUs    re-run the held-out stage the bindcraft cell lost
#   alpha     ~2-4 h, N GPUs     THE CONTROL: 5-edit Alpha recovery
#   explore   ~6 h, N GPUs       is the ceiling search-limited, and do pair moves help
#   seeded    ~4-5 h, N GPUs     the continuous cells that never ran
#
# THE BAR IS COMPUTED, NOT LEFT AS AN INSTRUCTION. alpha, explore and seeded
# each produce held-out structures and each ends with p17_capri_audit.py over
# them, reporting candidates inside this project's iRMSD bar (--irmsd-bar,
# default 10 A) per candidate rather than per structure. That step used to be a
# line of prose in the closing text, which meant `explore` -- whose entire
# purpose is to compare the BEST iRMSD across cells -- handed back the inputs to
# its own headline question and stopped one command short of answering it.
# jn1_recovery.csv carries ipSAE and pose RMSD; neither is iRMSD.
#
# DockQ pins numpy<2 and cannot share this environment, so it is an isolated
# tool install (`uv tool install dockq`) that a fresh checkout will not have.
# A missing DockQ is a SKIP naming the cause, never a stage failure, and
# `--stages audit --output-dir <that run>` finishes the job afterwards without
# re-running anything. `--skip-audit` turns it off.
#
# WHAT `explore` IS FOR, AND THE CLAIM IT CORRECTS. Section 30.3 concluded that
# discrete search saturates its budget, and that conclusion was over-reached.
# It is solid at budget 2, where all 147,117 feasible sequences were enumerated
# and the search hit the exact optimum 30/30. At budget 5 the space is
# C(29,5) x 19^5 ~= 2.9e11, so no exhaustive reference exists and the "99.9% of
# reference" figure was measured against the best any arm found -- the same
# circularity caught once at budget 2 and missed one budget up. Worse, 200
# score calls over 2.9e11 sequences is about 7e-10 of the space, so "the search
# has saturated" cannot be asserted of the real problem at all.
#
# Two readings are live and section 30's own numbers do not separate them:
#
#   structural ceiling   Both arms reach nearly the same best (6.66 A against
#                        7.09 A) while their means differ by 5-6 A, and §21.4
#                        found more evaluations buying confidence rather than
#                        placement. Then ~6.6 A is the typical value of a broad
#                        basin and no amount of exploration beats it.
#   search-limited       We have sampled 1e-9 of the space. The apparent
#                        ceiling is just where 200 draws land.
#
# THREE CELLS SEPARATE THEM, and they share a code version so the comparison is
# not against an older archive:
#
#   base      arity 1, 200 score calls   the matched baseline
#   calls600  arity 1, 600 score calls   3x the depth, same operator. If the
#                                        BEST drops below 6.6 A the ceiling is
#                                        search-limited; if it sits there, it is
#                                        structural.
#   arity2    arity 2, 200 score calls   every feasible position-disjoint pair,
#                                        at matched compute against `base`
#
# WHY THE PAIR OPERATOR, AND WHAT IT DOES NOT DO. A pair move changes two
# positions in one proposal, so the accept test never sees the intermediate
# state: where edit A alone would be rejected and A+B would be kept, a
# single-move walk cannot reach A+B because A is rejected first. That is the
# mechanism, and tests/test_multi_position_moves.py demonstrates it in
# isolation -- arity 1 never reaches the target pair, arity 2 always does,
# across ten seeds.
#
# It does NOT let the gradient find combinations. The gradient is linear, so a
# pair's predicted delta is exactly the sum of its two single deltas with no
# interaction term, and two edits that look bad alone look bad together. Any
# synergy is discovered by the realised score, never by the proposal ranking.
# A gain here must not be reported as the gradient having seen an interaction.
#
# The operator is a strict generalisation of what already existed: the single
# moves it offers are the same set `_moves` offers at every budget level, and
# the exchanges `_moves` builds at the cap -- edit here, revert there -- are
# the subset of pairs whose second edit returns a position to wt. Overhead is
# 1.1% of model time at worst, and the entropy target is normalised by
# log(move count), so the exploration pressure is comparable across arities
# rather than drifting with the 551 -> 147,117 change in set size.
#
# WHY `alpha` IS THE ONE THAT MATTERS. Every JN.1 null so far is ambiguous
# between two readings: the machinery does not work, or P17 cannot reach an
# Alpha-like registry against JN.1 at all. The reference for JN.1 is Alpha
# geometry transplanted onto a target it was never measured against, so the
# second reading is live, and no JN.1 experiment can distinguish them.
#
# The Alpha rung can. On P17+Alpha the predictor scores the right answer
# correctly -- ipSAE 0.795, pose 1.98-2.96 A (§19.1) -- so damage that binder by
# 5 edits and the task has a solution the predictor demonstrably sees. Recovery
# means the JN.1 failure is a statement about JN.1 and every null we hold is
# informative. Failure to recover means no JN.1 result means anything. This has
# been listed as outstanding since §23.1 and deferred every time, including by
# §29.6, and that deferral is what left thirty sections uninterpretable.
#
# WHY `free` AND `norms` COME FIRST. §26.5's per-term gradient norms gate two
# questions: §27.2's registry null (a term two orders of magnitude below its
# neighbours was present, not tried) and which term produced §30.2's 5.17 A
# iRMSD gain. One forward-plus-backward pass per term. Running the expensive
# stages before this number is known is how §27 happened.
#
# WHAT `norms` CANNOT ANSWER, AND WHY. Not §30.4's EditBudget hinge. An earlier
# version of this header claimed it could; that was wrong. E(s) = sum_pos
# (1 - p_parent) equals Hamming EXACTLY at a one-hot, so 5*relu(E - budget) has
# gradient 0 everywhere below its kink and, because relu'(0) = 0 in JAX, exactly
# 0 AT it. The term therefore reads 0.000 at the reference (Hamming 0) and 0.000
# at a winner five edits out (Hamming 5, sitting on the kink) -- both confirmed
# on 2026-10-12. The hinge is structurally invisible at every one-hot with
# Hamming <= budget, and can only be measured at an interior point of the
# relaxed parameterisation, which is the one regime this instrument never
# evaluates. §30.4 stays open and this stage cannot close it.
#
# WHAT `salvage` RECOVERS, AND WHY IT IS NOT A RE-RUN. On 2026-10-08 all eight
# held-out shards of the continuous stage's bindcraft cell failed. The searches
# themselves completed: p17_rescore_winners.py counted the winner's edits from
# the archived START sequence while the search counted them from the REFERENCE
# under --budget-anchor reference, so a winner 5 edits from the reference
# measured 6 from the seed and was rejected for breaking a budget it never
# broke. Fixed, with three regression tests. The hours of search on disk are
# untouched and held-out rescoring is minutes, so this re-runs only the stage
# that failed -- the fourth time this project has had a contract bug between
# stages and the fourth time the searches survived it.
#
# NOT IN THIS SCRIPT, AND WHY. Four things remain outside it. The first two are
# code that does not exist yet; the last two are small but unwritten. Nothing
# here is forgotten, and a launcher is not where any of it belongs:
#
#   * PUTTING REGISTRY INTO RETENTION. Every arm in §§20-30 varied the proposal
#     side; the referee has been mean directional-min ipSAE throughout. §30.2 is
#     the evidence that this is the binding constraint -- the uniform arm raised
#     ipSAE from 0.000 to 0.533 while leaving pose at or worse than its start,
#     so the referee accepts candidates that get confident without getting
#     placed. §28.2's argument was that retention must stay DISCRETE, which is
#     true and untouched; it said nothing about retention having to stay ipSAE.
#     To avoid §19.6's circularity, score on registry and keep pose RMSD as the
#     untouched held-out readout, or the reverse. Not both.
#   * THE MASKED k = BUDGET RELAXATION (§28.5 mechanism 2). Relaxing exactly
#     `budget` rows makes feasibility a property of the parameterisation instead
#     of a penalty, so the hinge is deleted rather than tuned. §30.4's bind is
#     the argument: feasibility needs an init scale >= 4.51, the softmax
#     Jacobian peaks at 3.0, so every feasible start is already past the
#     gradient peak and there is no usable operating point.
#   * THE GRADIENT BYPASS at src/mosaic/search.py's single gradient call site.
#     Substituting a constant array makes every delta zero, hits the
#     `scale == 0` branch and yields the same uniform distribution while
#     leaving `gradient_calls` at zero. That is the only way to ask whether the
#     gradient earns its 45% of wall-clock, which §29.3 is explicit that the
#     attribution screen did NOT test, because both of its arms paid for the
#     gradient. About five lines, still unwritten, still untested.
#   * SPLITTING POSITION FROM IDENTITY SELECTION (§30.4 item 2). The operator
#     samples a (position, residue) pair in one draw, so nothing separates "the
#     gradient knows where" from "the gradient knows what". A cell that takes
#     the position from the gradient and the residue uniformly would, and it is
#     the cheapest remaining attribution split -- but it needs a proposal
#     change, not a flag.
#
# ONE LEVER THAT IS A FLAG, AND IS OFF BY DEFAULT. `--edit-budget` is now
# exposed, and `budget15` is a recognised explore cell that is deliberately not
# in the default set. 5 substitutions across 29 positions is a tight corner of
# sequence space and the search exhausts it; §27.3 tried 7 and found nothing,
# but 7 is barely more than 5 and the honest test is 15. That CHANGES THE
# PROBLEM rather than improving the method, so its result is not comparable to
# the other cells on the 10 A bar, and whether to spend the compute is a scope
# decision rather than a run. Opt in with
# --explore-cells "base calls600 arity2 budget15".
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DRIVER="$SCRIPT_DIR/p17_alpha_recovery.py"
POLICY=population

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DTYPE=bf16
AGGREGATION=stable
STEPS=64
SCORE_CALLS=200
GRADIENT_CALLS=200
PROPOSALS=2000
EDIT_BUDGET=5
ACCEPT_TEMP=0.02
ENTROPY=0.6
WEIGHT_REGISTRY=0.0
REGISTRY_CUTOFF=8.0
NORMS_REGISTRY_WEIGHT=0.5   # the weight §27.2 found null; see header
NORMS_REPEATS=3
ALPHA_SEEDS="0 1 2 3"
ALPHA_RUNG=5
SEEDED_CELLS="apgm bindcraft colabdesign"
SEEDS_PER_CELL=20

EXPLORE_CELLS="base calls600 arity2"
EXPLORE_SEEDS=20
EXPLORE_DEEP_CALLS=600   # the depth cell, 3x `base`
EXPLORE_WIDE_BUDGET=15   # the budget15 cell; off by default, see header
IRMSD_BAR=10.0           # this project's acceptance bar, not CAPRI's
DOCKQ_BIN="${DOCKQ_BIN:-}"
ALL_STAGES="free norms salvage alpha explore seeded"
# `audit` is recognised but not in the default run, because the three structure
# stages already audit themselves. Ask for it when a first pass reported
# `capri: SKIPPED (no DockQ ...)`, or to re-read an existing run at a different
# bar: it needs only --output-dir and touches no GPU.
VALID_STAGES="$ALL_STAGES audit"
STAGES="$ALL_STAGES"
PRIOR_RUN=""
SEED_DIR=""
OUT_ROOT=""
DRY_RUN=false
SKIP_AUDIT=false

usage() {
    # The whole header, not a fixed line range: it has grown twice and a hard
    # range truncated --help mid-sentence, dropping the section that says what
    # this script deliberately leaves out.
    local end
    end=$(( $(grep -n '^set -euo' "${BASH_SOURCE[0]}" | head -1 | cut -d: -f1) - 1 ))
    sed -n "2,${end}p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --stages "a b"        subset of: $ALL_STAGES  (default: all, in that order)
                        plus 'audit', which is not in the default run: it
                        re-reads an existing --output-dir against the iRMSD
                        bar and needs no GPU.
  --prior-run PATH      the 2026-10-07 continuous-stage run directory. Required
                        by \`salvage\` (its bindcraft cell) and by \`seeded\`
                        (its seeds/), and used by \`free\` for the
                        reproducibility read.
  --seed-dir PATH       override the seeds/ directory \`seeded\` reads
  --devices LIST        GPUs (default: inherited CUDA_VISIBLE_DEVICES else 0-7)
  --alpha-rung N        damage rung the control recovers from (default: $ALPHA_RUNG)
  --alpha-seeds "N N"   search seeds for the control (default: $ALPHA_SEEDS)
  --seeds-per-cell N    search seeds per seeded cell (default: $SEEDS_PER_CELL)
  --cells "a b"         seeded cells, subset of: $SEEDED_CELLS
  --norms-repeats N     re-evaluations per term, to separate a real difference
                        from sampling noise (default: $NORMS_REPEATS)
  --explore-cells "a b" subset of: $EXPLORE_CELLS
  --explore-seeds N     search seeds per explore cell (default: $EXPLORE_SEEDS)
  --deep-calls N        score calls for the depth cell (default: $EXPLORE_DEEP_CALLS)
  --edit-budget N       WT-relative edit cap for every stage (default: $EDIT_BUDGET)
  --wide-budget N       edit cap for the opt-in budget15 cell (default: $EXPLORE_WIDE_BUDGET)
  --irmsd-bar A         acceptance bar in angstroms (default: $IRMSD_BAR)
  --dockq-bin PATH      DockQ executable (default: PATH, else ~/.local/bin/DockQ)
  --skip-audit          do not run the iRMSD audit after a structure stage
  --output-dir PATH     default: results/p17_consolidated_<timestamp>
  --dry-run             print the plan; create nothing, load nothing
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --prior-run) PRIOR_RUN="$2"; shift 2 ;;
        --seed-dir) SEED_DIR="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --alpha-rung) ALPHA_RUNG="$2"; shift 2 ;;
        --alpha-seeds) ALPHA_SEEDS="$2"; shift 2 ;;
        --seeds-per-cell) SEEDS_PER_CELL="$2"; shift 2 ;;
        --cells) SEEDED_CELLS="$2"; shift 2 ;;
        --norms-repeats) NORMS_REPEATS="$2"; shift 2 ;;
        --explore-cells) EXPLORE_CELLS="$2"; shift 2 ;;
        --explore-seeds) EXPLORE_SEEDS="$2"; shift 2 ;;
        --deep-calls) EXPLORE_DEEP_CALLS="$2"; shift 2 ;;
        --edit-budget) EDIT_BUDGET="$2"; shift 2 ;;
        --wide-budget) EXPLORE_WIDE_BUDGET="$2"; shift 2 ;;
        --irmsd-bar) IRMSD_BAR="$2"; shift 2 ;;
        --dockq-bin) DOCKQ_BIN="$2"; shift 2 ;;
        --skip-audit) SKIP_AUDIT=true; shift ;;
        --output-dir) OUT_ROOT="$2"; OUT_ROOT_GIVEN=true; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_consolidated_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
EXPLORE_SEED_LIST=$(seq 0 $((EXPLORE_SEEDS - 1)) | tr '\n' ' ')
[[ -z "$SEED_DIR" && -n "$PRIOR_RUN" ]] && SEED_DIR="$PRIOR_RUN/seeds"

if [[ -z "$DOCKQ_BIN" ]]; then
    DOCKQ_BIN="$(command -v DockQ || true)"
    [[ -z "$DOCKQ_BIN" ]] && DOCKQ_BIN="$HOME/.local/bin/DockQ"
fi

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

for s in $STAGES; do
    [[ " $VALID_STAGES " == *" $s "* ]] || { echo "Unknown stage: $s" >&2; exit 2; }
done
if has_stage audit && [[ -z "${OUT_ROOT_GIVEN:-}" ]]; then
    echo "Stage 'audit' re-reads an existing run, so it needs --output-dir." >&2
    exit 2
fi
if (has_stage salvage || has_stage seeded) && [[ -z "$PRIOR_RUN" ]]; then
    echo "Stages 'salvage' and 'seeded' need --prior-run (the 2026-10-07 run)." >&2
    exit 2
fi

echo "P17 consolidated -- $STAGES"
echo "  devices:     $DEVICES (${#DEV[@]})"
echo "  output:      $OUT_ROOT"
[[ -n "$PRIOR_RUN" ]] && echo "  prior run:   $PRIOR_RUN"
echo
echo "  Cheapest first, on purpose. Each stage narrows what the next can mean,"
echo "  so stopping after any one leaves something interpretable behind."
echo "  The stage that decides whether the rest is interpretable at all is"
echo "  'alpha': it is the only task in the queue with a known answer the"
echo "  predictor demonstrably scores correctly."
echo

if "$DRY_RUN"; then
    echo "Dry run."
    has_stage free && cat <<EOF

  [free]  seconds, no GPU
    $PYTHON_BIN $SCRIPT_DIR/p17_saliency_rank.py <a damaged-start run with saliency.csv> --out $OUT_ROOT/free/testb.json
    reproducibility read over $PRIOR_RUN/{parent,colabdesign}
EOF
    has_stage norms && cat <<EOF

  [norms]  ~2 min, 1 GPU
    $PYTHON_BIN $SCRIPT_DIR/p17_term_gradients.py --out $OUT_ROOT/norms/at_reference.json --weight-registry $NORMS_REGISTRY_WEIGHT --repeats $NORMS_REPEATS ...
    and again at the best JN.1 winner, since a gradient is a local object
EOF
    has_stage salvage && cat <<EOF

  [salvage]  ~10 min
    bash $SCRIPT_DIR/run_p17_heldout_retry.sh --run $PRIOR_RUN/bindcraft/run --target jn1
EOF
    has_stage alpha && cat <<EOF

  [alpha]  ~2-4 h  <- THE CONTROL (then the $IRMSD_BAR A iRMSD audit)
    bash $SCRIPT_DIR/run_p17_alpha_recovery.sh --output-dir $OUT_ROOT/alpha \\
        --select-rung $ALPHA_RUNG --edit-budget $ALPHA_RUNG \\
        --search-seeds "$ALPHA_SEEDS" --save-saliency ...
    then Test B again, on this run's own 5-edit rung
EOF
    has_stage explore && cat <<EOF

  [explore]  ~6 h  <- is the ceiling search-limited, and do pair moves help
    base      arity 1, $SCORE_CALLS calls          the matched baseline
    calls600  arity 1, $EXPLORE_DEEP_CALLS calls          3x depth, same operator
    arity2    arity 2, $SCORE_CALLS calls          every feasible position-disjoint pair
    budget15  arity 1, budget $EXPLORE_WIDE_BUDGET          opt-in; changes the PROBLEM, not the method
    cells: $EXPLORE_CELLS   seeds/cell: $EXPLORE_SEEDS
    then: p17_capri_audit.py $OUT_ROOT/explore --irmsd-threshold $IRMSD_BAR
    read the BEST iRMSD, not the mean: the question is whether the ceiling moves
EOF
    has_stage seeded && cat <<EOF

  [seeded]  ~4-5 h
    per cell in $SEEDED_CELLS: p17_alpha_recovery.py jn1 --start-sequence <seed>
        --budget-anchor reference --search-seeds $SEEDS ...
    seeds read from $SEED_DIR
    then: p17_capri_audit.py $OUT_ROOT --cells $SEEDED_CELLS --irmsd-threshold $IRMSD_BAR
EOF
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

SUMMARY=()
note() { SUMMARY+=("$1"); }

# The 10 A bar, computed rather than left as an instruction in the closing
# text. All three structure stages produce held-out PDBs and iRMSD is this
# project's acceptance criterion -- but jn1_recovery.csv carries ipSAE and pose
# RMSD, not iRMSD, so without this step each stage hands back the inputs to its
# own headline question and stops one command short of answering it. `explore`
# is the clearest case: its whole purpose is to compare the BEST iRMSD across
# cells.
#
# A missing DockQ is a SKIP with its reason, never a stage failure. DockQ pins
# numpy<2 and cannot share this environment, so it lives in an isolated tool
# install that a fresh cluster checkout will not have; the searches and the
# structures are the expensive part and they are already on disk by the time
# this runs. The skip note carries the exact command to finish the job later.
#
# Usage: capri_audit <label> <root> <out.csv> [extra args for the audit]
capri_audit() {
    local label="$1" root="$2" out="$3"; shift 3
    local log="${out%.csv}.log"
    if "$SKIP_AUDIT"; then
        note "$label/capri: SKIPPED (--skip-audit)"
        return 0
    fi
    if [[ ! -d "$root" ]]; then
        note "$label/capri: SKIPPED (no $root)"
        return 0
    fi
    # The two layouts p17_capri_audit.py accepts. A root with neither holds no
    # held-out structures at all, which is a skip -- the stage may have been
    # excluded or may have failed before its held-out step.
    if ! compgen -G "$root/heldout/shard_*" >/dev/null \
       && ! compgen -G "$root/*/run/heldout/shard_*" >/dev/null; then
        note "$label/capri: SKIPPED (no held-out structures under $root)"
        return 0
    fi
    if [[ ! -x "$DOCKQ_BIN" ]]; then
        echo "  no DockQ executable at $DOCKQ_BIN -- skipping the iRMSD audit." >&2
        echo "  install it isolated:  uv tool install dockq" >&2
        echo "  then audit everything this run produced, without re-running it:" >&2
        echo "    bash $0 --stages audit --output-dir $OUT_ROOT" >&2
        note "$label/capri: SKIPPED (no DockQ; afterwards: --stages audit --output-dir $OUT_ROOT)"
        return 0
    fi
    echo
    echo "-- iRMSD audit of $label against this project's $IRMSD_BAR A bar"
    echo "   per candidate, not per structure: the held-out seeds of one"
    echo "   candidate are correlated and are not three measurements."
    if "$PYTHON_BIN" "$SCRIPT_DIR/p17_capri_audit.py" "$root" \
        --out "$out" --dockq-bin "$DOCKQ_BIN" \
        --irmsd-threshold "$IRMSD_BAR" "$@" \
        2>&1 | tee "$log"; then
        note "$label/capri: PASSED ($out)"
    else
        note "$label/capri: FAILED (see $log)"
    fi
}

############################ stage: free ############################
if has_stage free; then
    echo "################ stage free (no GPU) ################"
    mkdir -p "$OUT_ROOT/free"

    # Test B wants a run that started from a damaged sequence AND saved
    # saliency. Nothing in the archive is guaranteed to be both, so look rather
    # than assume, and say plainly when there is nothing to read.
    echo "-- Test B: rank of the correct residue at the gradient calls that happened"
    found=""
    while IFS= read -r cfg; do
        run="$(dirname "$cfg")"
        [[ -f "$run/tables/saliency.csv" ]] || continue
        if "$PYTHON_BIN" - "$cfg" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
ref = c.get("reference_binder_sequence") or c.get("binder_sequence")
sys.exit(0 if ref != c.get("binder_sequence") else 1)
PY
        then found="$run"; break; fi
    done < <(find "$REPO_DIR/results" -name config.json -path '*/search/*' 2>/dev/null | sort)

    if [[ -n "$found" ]]; then
        echo "   reading $found"
        "$PYTHON_BIN" "$SCRIPT_DIR/p17_saliency_rank.py" "$found" \
            --out "$OUT_ROOT/free/testb.json" \
            2>&1 | tee "$OUT_ROOT/free/testb.log" && note "free/testb: read $found" \
            || note "free/testb: FAILED"
    else
        echo "   no archived run is both damaged-start and saliency-saving."
        echo "   Test B therefore runs after the alpha stage, which passes"
        echo "   --save-saliency and starts from a damaged rung by construction."
        note "free/testb: deferred to the alpha stage (no eligible archive)"
    fi

    # The accidental reproducibility measurement. colabdesign_r0 asked for zero
    # edits, so that cell's start IS the parent, and search.py expands
    # initial_sequences=None to the same repeated wt -- the cell is the control
    # by construction. Two identical configurations, 20 seeds each. Nobody
    # would have funded this experiment.
    if [[ -n "$PRIOR_RUN" ]]; then
        echo
        echo "-- reproducibility: two cells that are identical by construction"
        "$PYTHON_BIN" - "$PRIOR_RUN" "$OUT_ROOT/free/reproducibility.json" <<'PY' \
            2>&1 | tee "$OUT_ROOT/free/reproducibility.log" || true
import csv, json, statistics as st, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
cells = {}
for name in ("parent", "colabdesign"):
    path = root / name / "run" / "tables" / "jn1_recovery.csv"
    if not path.is_file():
        print(f"  {name}: no table at {path}")
        continue
    rows = [r for r in csv.DictReader(path.open()) if r.get("is_wt") == "False"]
    if not rows:
        print(f"  {name}: no non-WT winners")
        continue
    ipsae = [float(r["mean_ipsae"]) for r in rows]
    pose = [float(r["mean_pose_rmsd_A"]) for r in rows]
    cells[name] = dict(
        n=len(rows), mean_ipsae=st.fmean(ipsae), best_ipsae=max(ipsae),
        mean_pose=st.fmean(pose), best_pose=min(pose),
        sem_ipsae=(st.stdev(ipsae) / len(ipsae) ** 0.5 if len(ipsae) > 1 else None),
        sem_pose=(st.stdev(pose) / len(pose) ** 0.5 if len(pose) > 1 else None),
    )
    c = cells[name]
    print(f"  {name:<12} n={c['n']:<3} ipSAE {c['mean_ipsae']:.4f} "
          f"(best {c['best_ipsae']:.4f})  pose {c['mean_pose']:.2f} A "
          f"(best {c['best_pose']:.2f})")
verdict = None
if len(cells) == 2:
    a, b = cells["parent"], cells["colabdesign"]
    d_ip = abs(a["mean_ipsae"] - b["mean_ipsae"])
    d_po = abs(a["mean_pose"] - b["mean_pose"])
    sems = [v for v in (a["sem_ipsae"], b["sem_ipsae"]) if v]
    pooled = (sum(v * v for v in sems)) ** 0.5 if sems else None
    print()
    print(f"  gap: {d_ip:.4f} ipSAE, {d_po:.2f} A pose")
    if pooled:
        print(f"  pooled SEM on ipSAE: {pooled:.4f}  ->  {d_ip / pooled:.2f} sigma")
    verdict = (
        "These two cells are the same configuration: colabdesign's seed asked "
        "for zero edits, so its start is the parent, and the harness expands a "
        "null initial_sequences to the same repeated wt. Any gap here is "
        "run-to-run variation of the whole pipeline at fixed settings, and it "
        "bounds how much of section 30's sigma values is trajectory noise "
        "rather than a treatment effect."
    )
    print()
    print("  " + verdict)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(cells=cells, interpretation=verdict), indent=2) + "\n")
PY
        note "free/reproducibility: read parent vs colabdesign"
    fi
fi

############################ stage: norms ############################
if has_stage norms; then
    echo
    echo "################ stage norms (1 GPU, ~2 min) ################"
    mkdir -p "$OUT_ROOT/norms"
    dev="${DEV[0]}"

    echo "-- at the reference sequence"
    if CUDA_VISIBLE_DEVICES="$dev" "$PYTHON_BIN" "$SCRIPT_DIR/p17_term_gradients.py" \
        --out "$OUT_ROOT/norms/at_reference.json" \
        --edit-budget "$EDIT_BUDGET" \
        --weight-registry "$NORMS_REGISTRY_WEIGHT" \
        --registry-contact-distance "$REGISTRY_CUTOFF" \
        --sampling-steps "$STEPS" --opendde-dtype "$DTYPE" \
        --repeats "$NORMS_REPEATS" \
        2>&1 | tee "$OUT_ROOT/norms/at_reference.log"; then
        note "norms/at_reference: PASSED"
    else
        note "norms/at_reference: FAILED"
    fi

    # A gradient is a local object (§28.3). A term that is quiet at the parent
    # may not be quiet five edits away, and five edits away is where the search
    # actually spends its time.
    if [[ -n "$PRIOR_RUN" ]]; then
        # The recovery CSV carries candidate_id and metrics but no sequence
        # column; the winners themselves are in each search run's summary.json
        # as `best_sequence`, ranked by `best_score` (the search maximises it).
        winner=$("$PYTHON_BIN" - "$PRIOR_RUN" <<'PY' || true
import json, sys
from pathlib import Path
best, seq = None, ""
for path in Path(sys.argv[1]).glob("*/run/search/*/summary.json"):
    try:
        d = json.loads(path.read_text())
    except Exception:
        continue
    s, v = d.get("best_sequence"), d.get("best_score")
    if not s or v is None:
        continue
    if best is None or float(v) > best:
        best, seq = float(v), s
print(seq)
PY
)
        if [[ -n "$winner" ]]; then
            echo
            echo "-- at the best JN.1 winner on disk"
            if CUDA_VISIBLE_DEVICES="$dev" "$PYTHON_BIN" "$SCRIPT_DIR/p17_term_gradients.py" \
                --out "$OUT_ROOT/norms/at_winner.json" --sequence "$winner" \
                --edit-budget "$EDIT_BUDGET" \
                --weight-registry "$NORMS_REGISTRY_WEIGHT" \
                --registry-contact-distance "$REGISTRY_CUTOFF" \
                --sampling-steps "$STEPS" --opendde-dtype "$DTYPE" \
                --repeats "$NORMS_REPEATS" \
                2>&1 | tee "$OUT_ROOT/norms/at_winner.log"; then
                note "norms/at_winner: PASSED"
            else
                note "norms/at_winner: FAILED"
            fi
        else
            echo "   no winner sequence column found; skipping the second point."
            note "norms/at_winner: skipped (no sequence column in the prior tables)"
        fi
    fi
fi

############################ stage: salvage ############################
if has_stage salvage; then
    echo
    echo "################ stage salvage (~10 min) ################"
    target="$PRIOR_RUN/bindcraft/run"
    if [[ ! -d "$target/search" ]]; then
        echo "  no completed search at $target/search; nothing to salvage." >&2
        note "salvage: SKIPPED (no $target/search)"
    elif [[ -f "$target/tables/jn1_recovery.csv" ]]; then
        echo "  $target already has a recovery table; nothing to salvage."
        note "salvage: SKIPPED (table already present)"
    else
        echo "  re-running only the held-out stage; the searches stay untouched."
        if bash "$SCRIPT_DIR/run_p17_heldout_retry.sh" \
            --run "$target" --target jn1 --devices "$DEVICES" \
            --opendde-dtype "$DTYPE" \
            2>&1 | tee "$OUT_ROOT/salvage.log"; then
            note "salvage: PASSED"
        else
            note "salvage: FAILED (see salvage.log)"
        fi
    fi
fi

############################ stage: alpha ############################
if has_stage alpha; then
    echo
    echo "################ stage alpha -- THE CONTROL (~2-4 h) ################"
    echo "  Damage the Alpha binder by $ALPHA_RUNG edits and ask for it back."
    echo "  A correct answer exists here and the predictor scores it 0.795 ipSAE"
    echo "  at 1.98-2.96 A. Recovery makes every JN.1 null informative; failure"
    echo "  makes none of them mean anything."
    echo
    if bash "$SCRIPT_DIR/run_p17_alpha_recovery.sh" \
        --output-dir "$OUT_ROOT/alpha" \
        --devices "$DEVICES" \
        --select-rung "$ALPHA_RUNG" --edit-budget "$ALPHA_RUNG" \
        --search-seeds "$ALPHA_SEEDS" \
        --steps "$STEPS" --opendde-dtype "$DTYPE" \
        --aggregation "$AGGREGATION" \
        --max-score-calls "$SCORE_CALLS" \
        --max-gradient-calls "$GRADIENT_CALLS" \
        --max-proposals "$PROPOSALS" \
        --save-saliency \
        2>&1 | tee "$OUT_ROOT/alpha.log"; then
        note "alpha: PASSED"
    else
        note "alpha: FAILED (see alpha.log)"
    fi

    # Test B belongs here: this stage starts from a damaged rung by
    # construction and now passes --save-saliency, which is the combination
    # nothing in the archive was guaranteed to have.
    echo
    echo "-- Test B on this run's own $ALPHA_RUNG-edit rung"
    for cfg in "$OUT_ROOT"/alpha/search/*/config.json; do
        [[ -f "$cfg" ]] || continue
        run="$(dirname "$cfg")"
        [[ -f "$run/tables/saliency.csv" ]] || continue
        "$PYTHON_BIN" "$SCRIPT_DIR/p17_saliency_rank.py" "$run" \
            --out "$OUT_ROOT/alpha/testb_$(basename "$run").json" \
            2>&1 | tee -a "$OUT_ROOT/alpha/testb.log" || true
        break
    done
    [[ -f "$OUT_ROOT/alpha/testb.log" ]] && note "alpha/testb: read" \
        || note "alpha/testb: no saliency.csv was written"

    # run_p17_alpha_recovery.sh puts heldout/ at its own root rather than under
    # a cell, which is the single-run layout p17_capri_audit.py accepts.
    capri_audit alpha "$OUT_ROOT/alpha" "$OUT_ROOT/alpha/capri.csv"
fi

############################ stage: explore ############################
if has_stage explore; then
    echo
    echo "################ stage explore (~6 h) ################"
    echo "  Does the ~6.6 A ceiling move? Three cells, one code version."
    echo "  base     arity 1, $SCORE_CALLS calls   the matched baseline"
    echo "  calls600 arity 1, $EXPLORE_DEEP_CALLS calls   3x depth, same operator"
    echo "  arity2   arity 2, $SCORE_CALLS calls   position-disjoint pair moves"
    echo
    echo "  Read the BEST iRMSD per cell, not the mean. Section 30 found the"
    echo "  means 5-6 A apart and the bests only 0.4 A apart, so the open"
    echo "  question is the ceiling and the mean cannot answer it."
    echo

    EXPLORE_COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
                    --max-proposals "$PROPOSALS" --target-entropy "$ENTROPY"
                    --acceptance-temperature "$ACCEPT_TEMP"
                    --registry-contact-distance "$REGISTRY_CUTOFF" --save-saliency)

    for c in $EXPLORE_CELLS; do
        echo
        echo "---------------- cell $c ----------------"
        budget="$EDIT_BUDGET"
        case "$c" in
            base)     arity=1; calls="$SCORE_CALLS" ;;
            calls600) arity=1; calls="$EXPLORE_DEEP_CALLS" ;;
            arity2)   arity=2; calls="$SCORE_CALLS" ;;
            budget15) arity=1; calls="$SCORE_CALLS"; budget="$EXPLORE_WIDE_BUDGET" ;;
            *) echo "  unknown explore cell: $c" >&2
               note "explore/$c: SKIPPED (unknown cell)"; continue ;;
        esac
        echo "  move arity $arity, $calls score calls, budget $budget, $EXPLORE_SEEDS seeds"
        if [[ "$budget" != "$EDIT_BUDGET" ]]; then
            echo "  NOTE: this cell changes the PROBLEM, not the method. Its result"
            echo "  is not comparable to the other cells on the 10 A bar -- a wider"
            echo "  budget searches a different space. Read it on its own."
        fi
        if [[ -f "$OUT_ROOT/explore/$c/run/tables/jn1_recovery.csv" ]]; then
            note "explore/$c: SKIPPED (already has a table)"
            continue
        fi
        mkdir -p "$OUT_ROOT/explore/$c"
        # The gradient ceiling tracks the score ceiling. One gradient serves
        # three to four proposals, so a depth cell that kept the old gradient
        # cap would stop on the gradient budget instead of on the one being
        # varied, and the comparison would measure the wrong thing.
        if "$PYTHON_BIN" "$DRIVER" jn1 \
            --output-dir "$OUT_ROOT/explore/$c/run" --policies "$POLICY" \
            --edit-budget "$budget" --weight-registry "$WEIGHT_REGISTRY" \
            --move-arity "$arity" \
            --max-score-calls "$calls" --max-gradient-calls "$calls" \
            --search-seeds $EXPLORE_SEED_LIST "${EXPLORE_COMMON[@]}" \
            2>&1 | tee "$OUT_ROOT/explore/$c/run.log"; then
            note "explore/$c: PASSED (arity $arity, $calls calls, budget $budget)"
        else
            note "explore/$c: FAILED (see explore/$c/run.log)"
        fi
    done

    # Was the pair operator used, or only offered? A cell that offered 147,117
    # pair moves and sampled none of them would otherwise look identical to the
    # baseline and be read as a null for the operator. Read this file by
    # CROSS-CELL CONTRAST, not by its own counts: an arity-1 exchange at the
    # edit-budget cap also changes two positions and logs the same
    # move_positions: 2, so the counts alone do not separate the two operators
    # within one cell. See docs/P17_JN1.md section 31.4.
    if [[ -d "$OUT_ROOT/explore" ]]; then
        echo
        echo "-- was the pair operator used?"
        "$PYTHON_BIN" - "$OUT_ROOT/explore" <<"PY" | tee "$OUT_ROOT/explore/move_usage.txt" || true
import json, sys
from collections import Counter
from pathlib import Path

for log in sorted(Path(sys.argv[1]).glob("*/run/search/*/logs/events.jsonl")):
    # explore/<cell>/run/search/<run>/logs/events.jsonl
    cell = log.parents[4].name
    run = log.parents[1].name
    sizes, kept = Counter(), Counter()
    with log.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") != "proposal":
                continue
            size = event.get("move_positions")
            if size is None:
                continue
            sizes[size] += 1
            if event.get("accepted"):
                kept[size] += 1
    if not sizes:
        continue
    total = sum(sizes.values())
    parts = ", ".join(
        f"{n}-position: {sizes[n]} proposed ({100 * sizes[n] / total:.0f}%), "
        f"{kept[n]} kept"
        for n in sorted(sizes)
    )
    print(f"  {cell:<10} {run:<20} {parts}")
PY
    fi

    # One call for every cell, so the per-cell BEST iRMSD values -- the number
    # the ceiling question turns on -- land in one printout next to each other
    # and in one CSV, with the pooled ALL row underneath.
    capri_audit explore "$OUT_ROOT/explore" "$OUT_ROOT/explore/capri.csv"
fi

############################ stage: seeded ############################
if has_stage seeded; then
    echo
    echo "################ stage seeded (~4-5 h) ################"
    COMMON=(--devices "$DEVICES" --steps "$STEPS" --opendde-dtype "$DTYPE"
            --max-score-calls "$SCORE_CALLS" --max-gradient-calls "$GRADIENT_CALLS"
            --max-proposals "$PROPOSALS" --target-entropy "$ENTROPY"
            --acceptance-temperature "$ACCEPT_TEMP"
            --registry-contact-distance "$REGISTRY_CUTOFF" --save-saliency)

    for c in $SEEDED_CELLS; do
        echo
        echo "---------------- cell $c ----------------"
        # Any replicate, not _r0 specifically: on 2026-10-07 nine seeding runs
        # over eight GPUs doubled up device 0, both of those died, and one was
        # apgm_r0 -- which skipped the whole apgm cell while r1 and r2 sat
        # finished and unused.
        seed_json=""
        for r in 0 1 2 3; do
            if [[ -f "$SEED_DIR/${c}_r${r}.json" ]]; then
                seed_json="$SEED_DIR/${c}_r${r}.json"; break
            fi
        done
        if [[ -z "$seed_json" ]]; then
            echo "  no seed for $c in $SEED_DIR; skipping." >&2
            note "seeded/$c: SKIPPED (no seed)"
            continue
        fi
        if [[ -f "$OUT_ROOT/$c/run/tables/jn1_recovery.csv" ]]; then
            note "seeded/$c: SKIPPED (already has a table)"
            continue
        fi
        seq=$("$PYTHON_BIN" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["seed_sequence"])' \
            "$seed_json")
        spent=$("$PYTHON_BIN" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["hamming_from_parent"])' \
            "$seed_json")
        echo "  seed: $seed_json ($spent of $EDIT_BUDGET edits already spent)"
        if [[ "$spent" == "0" ]]; then
            echo "  NOTE: a zero-edit seed IS the parent, so this cell is the"
            echo "  parent control by construction and carries no information"
            echo "  about continuous seeding. Running it measures reproducibility."
        fi
        mkdir -p "$OUT_ROOT/$c"
        if "$PYTHON_BIN" "$SCRIPT_DIR/p17_alpha_recovery.py" jn1 \
            --output-dir "$OUT_ROOT/$c/run" --policies population \
            --edit-budget "$EDIT_BUDGET" --weight-registry "$WEIGHT_REGISTRY" \
            --start-sequence "$seq" --budget-anchor reference \
            --search-seeds $SEEDS "${COMMON[@]}" \
            2>&1 | tee "$OUT_ROOT/$c/run.log"; then
            note "seeded/$c: PASSED"
        else
            note "seeded/$c: FAILED (see $c/run.log)"
        fi
    done

    # The seeded cells sit directly under OUT_ROOT beside free/ norms/ alpha/
    # explore/, none of which match <cell>/run/heldout, but name the cells
    # anyway rather than relying on that.
    capri_audit seeded "$OUT_ROOT" "$OUT_ROOT/seeded_capri.csv" --cells $SEEDED_CELLS
fi

############################ stage: audit ############################
if has_stage audit; then
    echo
    echo "################ stage audit (minutes, no GPU) ################"
    echo "  Re-reading $OUT_ROOT against the $IRMSD_BAR A bar. Reads structures,"
    echo "  runs no model, and overwrites only capri.csv and capri.log."
    capri_audit alpha "$OUT_ROOT/alpha" "$OUT_ROOT/alpha/capri.csv"
    capri_audit explore "$OUT_ROOT/explore" "$OUT_ROOT/explore/capri.csv"
    capri_audit seeded "$OUT_ROOT" "$OUT_ROOT/seeded_capri.csv" --cells $SEEDED_CELLS
fi

echo
echo "################ Done ################"
if ((${#SUMMARY[@]})); then printf '%s\n' "${SUMMARY[@]/#/  }"; fi
cat <<EOF

Read in this order, because each one changes how the next reads:

  $OUT_ROOT/norms/at_reference.json
      Per-term gradient norms and pairwise cosines. If registry's norm is
      orders below contact's and pose's, §27.2's null was a scale artefact and
      the term was never actually tried. The edit term reads 0.000 by
      construction at any one-hot inside the budget, so it says nothing about
      §30.4 either way -- do not read it as the hinge being slack.
      Read the COSINES as carefully as the norms: two terms at cosine ~1 are one
      direction carrying two weights, and a term's norm then overstates what it
      contributes that its neighbour did not. On 2026-10-12 contact|registry was
      +0.942, which is the mechanism behind §27.2's null.
      Compare against at_winner.json: a term quiet at the parent and loud five
      edits out is a different story from one that is quiet throughout. On
      2026-10-12 every structure term's norm fell 4-7x from reference to winner
      while ablang2 held flat, making the sequence prior ~37% of the gradient
      five edits out -- i.e. the structure signal is largely spent by the time
      the budget is.

  $OUT_ROOT/alpha/tables/recovery.csv  +  alpha/capri.log
      THE CONTROL. Did the machinery recover a damaged binder when the answer
      exists and the predictor can see it? Everything else in this project is
      conditional on this. recovery.csv gives ipSAE and pose RMSD; capri.log
      gives the iRMSD bar, and on the Alpha rung the two should agree -- a
      recovered binder that misses the bar would mean the confidence readout
      and the interface disagree on a task with a known answer.

  $OUT_ROOT/alpha/testb_*.json
      Whether the first-order ranking ever surfaced the correct residue. A high
      rank never sampled points at top-k enumeration; a low rank means a better
      sampler over the same ranking cannot help.

  $OUT_ROOT/explore/capri.log  +  explore/move_usage.txt
      THE CEILING QUESTION. capri.log prints mean/median/BEST iRMSD per cell
      and the count inside the $IRMSD_BAR A bar, per candidate. Read the BEST,
      not the mean: section 30 found the cell means 5-6 A apart and the bests
      only 0.4 A apart, so the mean cannot answer this.
        calls600 beats base on BEST  ->  the ceiling is search-limited, and more
                                         depth buys reach. Section 30.3's
                                         "discrete search saturates" was over-
                                         reached: it holds at budget 2 where
                                         147,117 sequences were enumerated, not
                                         at budget 5 where 200 calls sample
                                         7e-10 of a 2.9e11 space.
        calls600 ties base on BEST   ->  the ceiling is structural, and it is
                                         the budget, the objective or the
                                         predictor. Not the search.
        arity2 beats base            ->  combination effects are real and the
                                         accept test was blocking them. Report
                                         it as the ACCEPT TEST never seeing the
                                         intermediate state, NOT as the gradient
                                         having found an interaction -- a pair's
                                         predicted delta is the exact sum of its
                                         parts, so the ranking cannot see synergy.
      Check move_usage.txt first. A cell that offered 147,117 pair moves and
      sampled none of them looks identical to the baseline, and that is an
      unused operator rather than a null result.

  $OUT_ROOT/free/reproducibility.json
      Two cells that are the same configuration. Bounds how much of §30's
      sigma values is trajectory noise.

  $OUT_ROOT/seeded_capri.log  and  $PRIOR_RUN/bindcraft/
      The continuous hand-off, against the parent control. §30.3 predicts a
      null: discrete search alone already reached 99.9% of an exhaustive
      optimum at this budget, and every synthetic hand-off came in marginally
      worse. Aggregate per candidate, not per structure -- three held-out seeds
      of one candidate are correlated and are not three measurements.

The iRMSD audit runs automatically after alpha, explore and seeded, and
reports this project's bar FIRST: candidates with iRMSD <= $IRMSD_BAR A,
counted per candidate rather than per structure, because the held-out seeds of
one candidate are correlated and are not three measurements. fnat and DockQ are
printed but do not gate. On the 2026-10-07 screen that read gradient 16/40
against uniform 4/40, best 6.66 A. If the summary above says a capri step was
SKIPPED for want of DockQ, install it with \`uv tool install dockq\` and re-run
just that one command -- it reads structures and runs no model.

GPU-free post-hoc, on any of the above:

  python examples/p17_search_trajectory.py <run>/search/*
  python examples/p17_site_gate_audit.py <run>

Scale bar (§19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. §30's gradient cells reached mean
ipSAE 0.567 and put 16 of 40 candidates inside the 10 A bar, best 6.66 A,
against 4 of 40 for uniform proposals. Every structure is CAPRI-incorrect at
best DockQ 0.174, which is a stricter standard than this project's own.

Establishes nothing about binding.
EOF
