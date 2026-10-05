#!/usr/bin/env bash
# Run the test suite the way it actually has to be run on this repo.
#
# Three things a bare `pytest` gets wrong here, each measured rather than
# assumed:
#
#   1. Optional dependencies are not installed here, and they fail in two
#      different ways. A test module that cannot even be IMPORTED aborts the
#      whole run rather than itself, so tests/test_esmfold2_multisample.py
#      (`esmjfold2`) makes a bare pytest collect nothing. Two further tests
#      import theirs inside the test body, so they merely fail. Both kinds are
#      skipped here, and only while the dependency is genuinely unusable -- so
#      installing one brings its tests back with no edit to this script.
#      "Unusable" is wider than "not installed", because a package can import
#      and still be unusable: on the cluster `jpromera` is installed, and
#      test_promera fails inside it at `tinyprot.msa`, which raises unless a
#      taxonomy LMDB has been downloaded. So each entry names the module whose
#      *import* is the real precondition, and any failure to import it -- a
#      missing package or a missing data file -- skips that test.
#
#   2. Nothing pins the backend, and JAX preallocates 75% of EVERY visible
#      device, so a bare pytest on an 8-GPU node claims all eight and kills
#      whatever is training there. Measured on 2026-10-04: running the suite
#      beside one scoring job made the suite report 8 failures instead of 2 and
#      OOM-killed the job. So this runs on the CPU by default, which costs one
#      skipped test (test_confidence_search's memory-stats case, which needs a
#      backend that reports them) and is in fact faster: 39s against 52s.
#      `--gpu` opts back in, on ONE device, without preallocating, and only
#      after checking that device is free.
#
#   3. pyproject's addopts deselects the `slow` markers, which is the right
#      default but hides that 238 tests did not run. The count is printed.
#
# With those three handled, a clean checkout passes with no failures, so any
# failure reported here is a real one. The two that a bare pytest reports on
# main -- test_cache's ESMFold2 MSA override and test_promera's structure
# writer -- are both missing optional dependencies, not defects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

# module -> how to skip what needs it. `ignore` takes a file whose import
# fails at collection; `deselect` takes a single test that imports its
# dependency in the body and so only fails. tests/test_cache.py gets a
# deselect rather than an ignore because its other three tests do pass.
OPTIONAL_DEPS=(
    "esmjfold2:ignore:tests/test_esmfold2_multisample.py"
    "esmjfold2:deselect:tests/test_cache.py::test_esmfold_msa_cache_follows_runtime_override"
    "jpromera:ignore:tests/test_promera.py"
    # Installed on the cluster but unusable without `python -m tinyprot.init`.
    "tinyprot.msa:ignore:tests/test_promera.py"
)

SLOW=false
ALLOW_BUSY=false
DRY_RUN=false
USE_GPU=false
DEVICE=0
PYTEST_ARGS=()

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Usage: bash run_tests.sh [options] [-- pytest args...]

Options:
  --slow             also run the tests marked slow (adds ~239 tests)
  --gpu [N]          run on GPU N (default 0) instead of the CPU. Needed only
                     for the one memory-stats test that the CPU backend skips.
                     Never takes more than the one device.
  --allow-busy-gpus  with --gpu, run even though that device holds a process
  --dry-run          print the pytest command and exit
  -h, --help         this text

The default is CPU-only, so this is safe to run on a shared node while
something else is using the GPUs.

Anything after -- goes to pytest, so a single file or -k filter works:
  bash run_tests.sh -- tests/test_followup_arms.py -v
  bash run_tests.sh -- -k decoy
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slow) SLOW=true; shift ;;
        --gpu)
            USE_GPU=true
            # Optional argument: a bare --gpu means device 0.
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then DEVICE="$2"; shift 2; else shift; fi
            ;;
        --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "No interpreter at $PYTHON_BIN." >&2
    echo "The suite needs the repo venv: gemmi and jax are not in the base env." >&2
    exit 2
fi
cd "$SCRIPT_DIR"

echo "Test suite"
echo "  python:   $PYTHON_BIN"

# 1. Skip what an absent optional dependency breaks -- and only while it is
# absent, so this needs no edit when one gets installed.
SKIPS=()
declare -A DEP_SEEN=()
for entry in "${OPTIONAL_DEPS[@]}"; do
    module="${entry%%:*}"
    rest="${entry#*:}"
    kind="${rest%%:*}"
    target="${rest#*:}"
    # Both kinds name a file, a deselect suffixing it with ::test_name.
    [[ -f "${target%%::*}" ]] || continue
    if [[ -z "${DEP_SEEN[$module]:-}" ]]; then
        if "$PYTHON_BIN" -c "import $module" >/dev/null 2>&1; then
            DEP_SEEN[$module]=present
            echo "  deps:     $module present, so its tests are included"
        else
            DEP_SEEN[$module]=missing
            echo "  deps:     $module unavailable, so its tests are skipped"
        fi
    fi
    [[ "${DEP_SEEN[$module]}" == missing ]] || continue
    # Two modules can guard the same file, so the same flag must not be added
    # twice.
    case "$kind" in
        ignore)
            [[ " ${SKIPS[*]-} " == *" --ignore=$target "* ]] && continue
            SKIPS+=("--ignore=$target") ;;
        deselect)
            [[ " ${SKIPS[*]-} " == *" $target "* ]] && continue
            SKIPS+=("--deselect" "$target") ;;
        *) echo "Bad OPTIONAL_DEPS entry: $entry" >&2; exit 2 ;;
    esac
done

# 2. Backend. CPU by default so this never competes for a device; with --gpu,
# exactly one device and no preallocation, after checking that device is free.
if ! "$USE_GPU"; then
    export JAX_PLATFORMS=cpu
    export CUDA_VISIBLE_DEVICES=""
    echo "  backend:  cpu, no GPU touched (pass --gpu for the one skipped test)"
else
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    # Without this JAX takes 75% of the device up front, which is what makes a
    # concurrent job die rather than simply slow down.
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    echo "  backend:  gpu $DEVICE only, preallocation off"
    if command -v nvidia-smi >/dev/null 2>&1; then
        # An absent device prints to stdout and exits nonzero, so the exit code
        # is what separates "not there" from "there and idle".
        if ! PIDS="$(nvidia-smi --id="$DEVICE" --query-compute-apps=pid \
                     --format=csv,noheader 2>&1)"; then
            echo "  ERROR: device $DEVICE does not exist on this host." >&2
            "$DRY_RUN" || exit 2
            PIDS=""
        fi
        COUNT="$(printf '%s' "$PIDS" | grep -c . || true)"
        if [[ "${COUNT:-0}" -gt 0 ]]; then
            echo "  WARNING: device $DEVICE already holds a process." >&2
            if ! "$ALLOW_BUSY" && ! "$DRY_RUN"; then
                echo "GPU tests fail under contention and can OOM-kill the other" >&2
                echo "job (measured: 8 failures busy vs 2 free). Use a different" >&2
                echo "--gpu N, drop --gpu to run on the CPU, or pass" >&2
                echo "--allow-busy-gpus and read the failures with that in mind." >&2
                exit 2
            fi
        else
            echo "  gpus:     device $DEVICE is free"
        fi
    else
        echo "  WARNING: nvidia-smi unavailable, so occupancy was NOT checked." >&2
    fi
fi

# 3. pyproject sets addopts = -m 'not slow'; an empty -m overrides it.
MARKERS=()
if "$SLOW"; then
    MARKERS=(-m "")
    echo "  markers:  all, including slow"
else
    echo "  markers:  not slow (pass --slow to include them)"
fi

COMMAND=("$PYTHON_BIN" -m pytest "${SKIPS[@]+"${SKIPS[@]}"}"
         "${MARKERS[@]+"${MARKERS[@]}"}" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}")

if "$DRY_RUN"; then
    echo
    echo "Dry run. Would run:"
    printf '  %q' "${COMMAND[@]}"; echo
    exit 0
fi

echo
set +e
"${COMMAND[@]}"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
    echo "All selected tests passed."
else
    echo "pytest exited $STATUS. Tests needing absent optional dependencies" >&2
    echo "were already skipped, so these failures are real." >&2
fi
exit $STATUS
