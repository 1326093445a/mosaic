#!/usr/bin/env bash
# Isolate the EditBudget hinge in the continuous seeding stage, on the real loss.
#
# WHY THIS EXISTS AS ITS OWN SCRIPT. Section 31.10 of docs/P17_JN1.md documented
# an eight-run contrast and no launcher implemented it:
# `run_p17_continuous_stage.sh` runs 3 methods x 3 replicates at ONE hinge
# setting, which is not a contrast at all. It also assigned 9 runs round-robin
# over 8 devices, putting run 9 back on device 0 -- the exact doubling-up its
# own header records as having killed `apgm_r0` on 2026-10-07. This script
# refuses to start if the run count exceeds the device count.
#
# WHAT IT ANSWERS, AND WHAT IT CANNOT. The CPU toy loss in
# tests/test_continuous_seed.py shows the hinge CAN hold a relaxation at the
# parent; only OpenDDE can say whether it did. Read
# `candidate_substitutions_considered`, NOT `hamming_from_parent`, which is
# capped at the edit budget and so reports "5" for anything from 5 upwards.
#
#   hinge-off arms want >> 5 and hinge-on arms want <= 5
#       -> the hinge suppresses the final argmax substitutions FOR THIS method,
#          init scale, seed and schedule, and section 30.3's truncation question
#          is live: more substitutions are wanted than the budget allows, so
#          `project_to_budget` decides WHICH of them survive -- on margins the
#          optimisation produced, so the two jointly determine the hand-off.
#   runs 7-8 move but 1-6 do not
#       -> a lower init scale permits movement WITH THE HINGE OFF. Not "the
#          scale, not the hinge": there is no hinge-on cell at scale 1.0, so
#          that 2x2 is incomplete and a hinge effect there stays untested.
#   both arms near 0
#       -> NOT evidence about OpenDDE's gradient. The same toy loss, with no
#          model in it, produces 0 in both arms at init scale 5.0 on a 24-step
#          budget, and simplex_APGM produces 0 from a scale-5.0 start at any
#          length. It would establish only that few argmax substitutions
#          happened inside this budget, which separates a bad gradient from an
#          optimiser that never moved not at all. Check the scale-1.0 arms
#          first; raising the step count is then the cheapest next candidate,
#          not a conclusion this result forces.
#
# ONE SEED PER CELL. This is a screen, not an estimate: there is no within-cell
# variance to compare a between-cell difference against.
#
# This runs the CURRENT code. It cannot attribute the historical 2026-10-12
# null, which was produced by the pre-2026-10-13 code paths (single-stage
# colabdesign with hard=False and temp annealed to 0.01; bindcraft with its
# four stage lengths hardcoded). Those are gone.
#
#   bash examples/run_p17_hinge_probe.sh --dry-run
#   bash examples/run_p17_hinge_probe.sh            # ~25 min on 8 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
SEEDER="$SCRIPT_DIR/p17_continuous_seed.py"

DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
STEPS=125
EDIT_BUDGET=5
SAMPLING_STEPS=64
DTYPE=bf16
LR=0.1
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/results/p17_hinge_probe_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=false
ALLOW_BUSY_GPUS=false

# method:hinge_weight:init_logit_scale
# Rows 1-6 vary ONLY the hinge, at the init scale the stage shipped before
# 2026-10-13, so the contrast is against the historical configuration.
# Rows 7-8 vary ONLY the scale with the hinge held off, because the toy says
# the hinge is necessary but not sufficient and the scale is what decides
# whether simplex_APGM moves at all.
RUNS=(
  apgm:5.0:5.0        apgm:0.0:5.0
  colabdesign:5.0:5.0 colabdesign:0.0:5.0
  bindcraft:5.0:5.0   bindcraft:0.0:5.0
  apgm:0.0:1.0        colabdesign:0.0:1.0
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-busy-gpus) ALLOW_BUSY_GPUS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --output-dir) OUT_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n "2,$(( $(grep -n '^set -euo' "${BASH_SOURCE[0]}" | head -1 | cut -d: -f1) - 1 ))p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

source "$SCRIPT_DIR/lib_gpu_capacity.sh"

IFS=',' read -r -a DEV <<< "$DEVICES"
# Distinctness, not just count. `--devices 0,0,0,0,0,0,0,0` has eight entries
# and passed the count check while putting all eight jobs on GPU 0 -- the exact
# doubling-up this script exists to refuse.
for d in "${DEV[@]}"; do
    if [[ -z "$d" || ! "$d" =~ ^[0-9]+$ ]]; then
        echo "Refusing to start: --devices has an empty or non-numeric entry" >&2
        echo "('$d' in '$DEVICES')." >&2
        exit 2
    fi
done
if (( $(printf '%s\n' "${DEV[@]}" | sort -u | wc -l) != ${#DEV[@]} )); then
    echo "Refusing to start: --devices repeats a device ('$DEVICES'), which would" >&2
    echo "put more than one process on the same GPU. That is what killed apgm_r0" >&2
    echo "on 2026-10-07, and the count check alone does not catch it." >&2
    exit 2
fi
if (( ${#RUNS[@]} > ${#DEV[@]} )); then
    echo "Refusing to start: ${#RUNS[@]} runs over ${#DEV[@]} devices would put two" >&2
    echo "processes on one GPU. That is what killed apgm_r0 on 2026-10-07. Pass" >&2
    echo "--devices with at least ${#RUNS[@]} entries, or trim RUNS." >&2
    exit 2
fi

echo "P17 hinge probe -- ${#RUNS[@]} runs, one per GPU, $STEPS steps each"
echo "  devices: $DEVICES"
echo "  output:  $OUT_ROOT"
# The seeder inherits JAX's defaults here rather than having a fraction set for
# it, so report against the same 0.90 the stage launcher uses: it is what the
# runtime preallocates unless the caller says otherwise.
if ! "$DRY_RUN"; then
    gpu_capacity_check "$DEVICES" \
        "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" "$ALLOW_BUSY_GPUS"
fi
echo

# The tags are needed before the loop so the reporter can check for the files
# it EXPECTS rather than for whatever JSON happens to be lying around.
EXPECTED=()
for spec in "${RUNS[@]}"; do
    IFS=: read -r m h s <<< "$spec"
    # tag:method:hinge:scale -- the reporter checks each result against the
    # settings its run was ASKED for. Eight mock results all reporting hinge 0,
    # including the weight-five arms, were previously accepted as a contrast.
    EXPECTED+=("${m}_h${h}_s${s}:${m}:${h}:${s}")
done

if "$DRY_RUN"; then
    echo "Dry run: not creating $OUT_ROOT and not writing provenance."
    echo
else

# A FRESH directory, not merely one without provenance. Result JSONs left by an
# earlier run with no provenance.json beside them were accepted, and this script
# then wrote its own provenance over them -- so another run's outputs acquired
# this run's commit, which is worse than having no provenance at all.
if [[ -e "$OUT_ROOT" ]] && [[ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]]; then
    echo "Refusing to start: $OUT_ROOT exists and is not empty, so this run's" >&2
    echo "provenance would be written over, or alongside, files it did not" >&2
    echo "produce -- leaving them attributed to this commit. Pass --output-dir" >&2
    echo "with a fresh path. Found:" >&2
    ls -A "$OUT_ROOT" | sed 's/^/        /' >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"

# `--ignore-submodules=untracked`, because plain `git status --porcelain`
# reports ` M OpenDDE` whenever the nested OpenDDE checkout has any untracked
# content -- which it permanently does, since OpenDDE must never be committed
# into this repo. Without any flag `git_dirty` is the constant `true` on a
# perfectly clean tree, which makes the field worth nothing: it cannot tell a
# citable run from an uncitable one, which is its entire job.
#
# `untracked`, NOT `all`. `all` would also hide a tracked edit inside OpenDDE
# and a changed OpenDDE revision -- the two dependency changes most likely to
# move a result. `untracked` drops only the untracked-file noise. And neither
# flag says anything about the INSTALLED dependency: the five patches under
# patches/ are applied to the site-packages copy at runtime, and nothing here
# records what that copy was.
DIRTY="$(git -C "$REPO_DIR" status --porcelain --ignore-submodules=untracked 2>/dev/null)"

# Provenance, written BEFORE anything runs. The base-library corrections of
# 2026-10-13 (bin centers, the interface-score broadcast, the MCMC reverse path
# and key handling, the APGM best_x contract) all change numbers, so a result
# is only comparable with another from the same revision. A dirty tree is
# recorded as such rather than silently ignored.
{
    echo "{"
    echo "  \"started\": \"$(date -Iseconds)\","
    echo "  \"git_commit\": \"$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"git_dirty\": $( [[ -n "$DIRTY" ]] && echo true || echo false ),"
    echo "  \"git_dirty_entries\": $( [[ -n "$DIRTY" ]] && echo "$DIRTY" | wc -l || echo 0 ),"
    echo "  \"host\": \"$(hostname)\","
    echo "  \"devices\": \"$DEVICES\","
    echo "  \"steps\": $STEPS,"
    echo "  \"edit_budget\": $EDIT_BUDGET,"
    echo "  \"runs\": \"${RUNS[*]}\""
    echo "}"
} > "$OUT_ROOT/provenance.json"
echo "  provenance: $OUT_ROOT/provenance.json"
if [[ -n "$DIRTY" ]]; then
    echo "  NOTE: the working tree is dirty, so this run is not reproducible from" >&2
    echo "        a commit alone. Commit first if the result is to be cited:" >&2
    echo "$DIRTY" | sed 's/^/          /' >&2
fi
echo

fi  # not a dry run

i=0
pids=()
for spec in "${RUNS[@]}"; do
    IFS=: read -r method hinge scale <<< "$spec"
    tag="${method}_h${hinge}_s${scale}"
    cmd=("$PYTHON_BIN" "$SEEDER" --method "$method" --seed 0 --steps "$STEPS"
         --init parent --init-logit-scale "$scale"
         --edit-budget "$EDIT_BUDGET" --weight-edit-budget "$hinge"
         --no-logit-norm --no-stochastic-hard --hard-temp 1.0
         --sampling-steps "$SAMPLING_STEPS" --opendde-dtype "$DTYPE"
         --lr "$LR" --out "$OUT_ROOT/$tag.json")
    if "$DRY_RUN"; then
        echo "  [GPU ${DEV[$i]}] ${cmd[*]}"
    else
        echo "  [GPU ${DEV[$i]}] $tag"
        CUDA_VISIBLE_DEVICES="${DEV[$i]}" "${cmd[@]}" \
            >"$OUT_ROOT/$tag.log" 2>&1 &
        pids+=($!)
    fi
    i=$((i + 1))
done

if "$DRY_RUN"; then
    echo
    echo "Dry run: nothing executed."
    exit 0
fi

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo
# Said BEFORE the table, because a reader who sees numbers first has already
# formed a conclusion by the time they reach the caveat.
if ((failed)); then
    echo "WARNING: $failed of ${#RUNS[@]} runs FAILED -- see $OUT_ROOT/*.log" >&2
    echo "Any table below covers only the runs that produced a JSON." >&2
    echo >&2
fi

echo "################ results ################"
echo "read 'wanted', not 'edits' -- edits is capped at the budget"
# Named result files, not a glob: a `*.json` glob also picked up
# provenance.json and then reported the script's OWN metadata as having come
# from obsolete code. The reporter's exit status is kept, because a reporter
# that fails silently turns a broken run into an apparently clean one.
report_failed=0
"$PYTHON_BIN" - "$OUT_ROOT" "${EXPECTED[@]}" <<'MARK' || report_failed=1
import json, sys
from pathlib import Path

root, specs = Path(sys.argv[1]), sys.argv[2:]
rows, absent, stale, mismatched = [], [], [], []
for spec in specs:
    tag, method, hinge, scale = spec.split(":")
    path = root / f"{tag}.json"
    if not path.exists():
        absent.append(tag)
        continue
    try:
        d = json.loads(path.read_text())
    except Exception as exc:
        stale.append(f"{path.name}: unreadable ({exc})")
        continue
    gaps = [k for k in ("candidate_substitutions_considered", "weight_edit_budget",
                        "init_logit_scale") if k not in d]
    if gaps:
        stale.append(
            f"{path.name}: MISSING {gaps} -- this JSON came from pre-2026-10-13 "
            "code, so the node is not running what this script expects"
        )
        continue
    # A result that does not carry the settings its run was asked for is not
    # part of this contrast, whatever its numbers say.
    wrong = []
    if str(d.get("method")) != method:
        wrong.append(f"method {d.get('method')!r} != {method!r}")
    for key, want in (("weight_edit_budget", hinge), ("init_logit_scale", scale)):
        try:
            if float(d.get(key)) != float(want):
                wrong.append(f"{key} {d.get(key)} != {want}")
        except (TypeError, ValueError):
            wrong.append(f"{key} {d.get(key)!r} is not a number")
    if wrong:
        mismatched.append(f"{path.name}: " + "; ".join(wrong))
        continue
    rows.append((d["method"], d["weight_edit_budget"], d["init_logit_scale"],
                 d["candidate_substitutions_considered"], d["hamming_from_parent"]))

if rows:
    print(f"  {'method':<12} {'hinge':>6} {'scale':>6} {'wanted':>7} {'edits':>6}")
    for m, h, s, w, e in sorted(rows):
        print(f"  {m:<12} {h:>6} {s:>6} {w:>7} {e:>6}")

for line in stale:
    print(f"  {line}", file=sys.stderr)
for line in mismatched:
    print(f"  SETTINGS MISMATCH {line} -- this run did not execute the "
          "configuration the matrix asked for", file=sys.stderr)
if absent:
    print(f"  NO RESULT FILE for {len(absent)} of {len(specs)} runs: "
          f"{', '.join(absent)}", file=sys.stderr)
if len(rows) != len(specs):
    print(f"  {len(rows)} of {len(specs)} runs usable, so the table above is "
          "NOT the eight-way contrast this script was asked for", file=sys.stderr)
sys.exit(1 if (absent or stale or mismatched or not rows) else 0)
MARK
if ((report_failed)); then
    echo >&2
    echo "The reporter failed or found no usable results, so the table above (if" >&2
    echo "any) is NOT the comparison this script was asked for." >&2
    failed=$((failed + 1))
fi
echo
if ((failed)); then
    echo "FAILED: $failed of ${#RUNS[@]} runs did not complete, so this is NOT the" >&2
    echo "comparison the script was asked for. Fix them and re-run before reading" >&2
    echo "anything into the table above. Logs: $OUT_ROOT/*.log" >&2
    exit 1
fi

echo "Nothing here establishes binding."
