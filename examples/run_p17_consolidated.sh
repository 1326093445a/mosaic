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
#   seeded    ~4-5 h, N GPUs     the continuous cells that never ran
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
# WHY `free` AND `norms` COME FIRST. §26.5's per-term gradient norms now gate
# three separate questions: §27.2's registry null (a term two orders of
# magnitude below its neighbours was present, not tried), which term produced
# §30.2's 5.17 A iRMSD gain, and whether the EditBudget hinge can hold at the
# real objective's gradient scale, which §30.3 showed decides whether a relaxed
# optimum stays inside its budget. One forward-plus-backward pass per term.
# Running the expensive stages before this number is known is how §27 happened.
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
# NOT IN THIS SCRIPT, DELIBERATELY. Two changes that would plausibly matter more
# than anything above are CODE changes and have no business in a launcher:
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
#
# Establishes nothing about binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

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

ALL_STAGES="free norms salvage alpha seeded"
STAGES="$ALL_STAGES"
PRIOR_RUN=""
SEED_DIR=""
OUT_ROOT=""
DRY_RUN=false

usage() {
    sed -n '2,84p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --stages "a b"        subset of: $ALL_STAGES  (default: all, in that order)
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
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$REPO_DIR/results/p17_consolidated_$(date +%Y%m%d_%H%M%S)"
DEVICES="${DEVICES//[[:space:]]/}"
IFS=',' read -r -a DEV <<< "$DEVICES"
SEEDS=$(seq 0 $((SEEDS_PER_CELL - 1)) | tr '\n' ' ')
[[ -z "$SEED_DIR" && -n "$PRIOR_RUN" ]] && SEED_DIR="$PRIOR_RUN/seeds"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

for s in $STAGES; do
    [[ " $ALL_STAGES " == *" $s "* ]] || { echo "Unknown stage: $s" >&2; exit 2; }
done
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
    $PYTHON_BIN $SCRIPT_DIR/p17_term_gradients.py --out $OUT_ROOT/norms/at_parent.json --weight-registry $NORMS_REGISTRY_WEIGHT --repeats $NORMS_REPEATS ...
    and again at the best JN.1 winner, since a gradient is a local object
EOF
    has_stage salvage && cat <<EOF

  [salvage]  ~10 min
    bash $SCRIPT_DIR/run_p17_heldout_retry.sh --run $PRIOR_RUN/bindcraft/run --target jn1
EOF
    has_stage alpha && cat <<EOF

  [alpha]  ~2-4 h  <- THE CONTROL
    bash $SCRIPT_DIR/run_p17_alpha_recovery.sh --output-dir $OUT_ROOT/alpha \\
        --select-rung $ALPHA_RUNG --edit-budget $ALPHA_RUNG \\
        --search-seeds "$ALPHA_SEEDS" --save-saliency ...
    then Test B again, on this run's own 5-edit rung
EOF
    has_stage seeded && cat <<EOF

  [seeded]  ~4-5 h
    per cell in $SEEDED_CELLS: p17_alpha_recovery.py jn1 --start-sequence <seed>
        --budget-anchor reference --search-seeds $SEEDS ...
    seeds read from $SEED_DIR
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
fi

echo
echo "################ Done ################"
if ((${#SUMMARY[@]})); then printf '%s\n' "${SUMMARY[@]/#/  }"; fi
cat <<EOF

Read in this order, because each one changes how the next reads:

  $OUT_ROOT/norms/at_reference.json
      Per-term gradient norms and pairwise cosines. If registry's norm is
      orders below contact's and pose's, §27.2's null was a scale artefact and
      the term was never actually tried. If the edit term's norm is comparable
      to the others', §30.4's hinge bind is real at the objective's own scale.
      Compare against at_winner.json: a term quiet at the parent and loud five
      edits out is a different story from one that is quiet throughout.

  $OUT_ROOT/alpha/tables/recovery.csv
      THE CONTROL. Did the machinery recover a damaged binder when the answer
      exists and the predictor can see it? Everything else in this project is
      conditional on this.

  $OUT_ROOT/alpha/testb_*.json
      Whether the first-order ranking ever surfaced the correct residue. A high
      rank never sampled points at top-k enumeration; a low rank means a better
      sampler over the same ranking cannot help.

  $OUT_ROOT/free/reproducibility.json
      Two cells that are the same configuration. Bounds how much of §30's
      sigma values is trajectory noise.

  $OUT_ROOT/<cell>/run/tables/jn1_recovery.csv  and  $PRIOR_RUN/bindcraft/
      The continuous hand-off, against the parent control. §30.3 predicts a
      null: discrete search alone already reached 99.9% of an exhaustive
      optimum at this budget, and every synthetic hand-off came in marginally
      worse. Aggregate per candidate, not per structure -- three held-out seeds
      of one candidate are correlated and are not three measurements.

GPU-free post-hoc, on any of the above:

  python examples/p17_search_trajectory.py <run>/search/*
  python examples/p17_site_gate_audit.py <run>
  python examples/p17_capri_audit.py <run> --out capri.csv
      Reports this project's bar FIRST: candidates with iRMSD <= 10 A, counted
      per candidate rather than per structure, because the held-out seeds of
      one candidate are correlated and are not three measurements. fnat and
      DockQ are printed but do not gate. On the 2026-10-07 screen that read
      gradient 16/40 against uniform 4/40, best 6.66 A.

Scale bar (§19.1): P17+Alpha measured ipSAE 0.795 and pose 1.98-2.96 A;
P17+JN.1 ipSAE 0.000-0.163 and pose 22-58 A. §30's gradient cells reached mean
ipSAE 0.567 and put 16 of 40 candidates inside the 10 A bar, best 6.66 A,
against 4 of 40 for uniform proposals. Every structure is CAPRI-incorrect at
best DockQ 0.174, which is a stricter standard than this project's own.

Establishes nothing about binding.
EOF
