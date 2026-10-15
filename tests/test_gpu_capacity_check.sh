#!/usr/bin/env bash
# `gpu_capacity_check` decides whether a 2-hour run is allowed to start, so its
# arithmetic and its refusal are worth pinning. nvidia-smi is stubbed; the
# numbers in the first case are the real gpu21-sublab reading from 2026-10-14,
# where a 0.90 preallocation fitted with 1,583 MiB to spare and nothing said so.
set -uo pipefail
cd "$(dirname "$0")/.."
source examples/lib_gpu_capacity.sh

pass=0; fail=0
check() { if [[ "$2" == "$3" ]]; then echo "  PASS  $1"; pass=$((pass+1));
          else echo "  FAIL  $1 (expected '$2', got '$3')"; fail=$((fail+1)); fi; }

run_case() {  # rows, devices, fraction, allow_busy
    local rows="$1"
    nvidia-smi() {
        [[ "$*" == *"--query-gpu"* ]] || return 1
        printf '%s\n' "$rows"
    }
    ( gpu_capacity_check "$2" "$3" "${4:-false}" ) 2>&1
    echo "exit:$?"
}

BUSY_BUT_FITS=$'0, 143771, 8287\n1, 143771, 11127\n2, 143771, 12795'
out=$(run_case "$BUSY_BUT_FITS" "0,1,2" 0.90 false)
check "a busy node with room starts" "exit:0" "$(tail -1 <<< "$out")"
check "the tightest margin is reported" "1583" \
    "$(grep -oE 'margin +\+?[0-9]+' <<< "$out" | tail -1 | grep -oE '[0-9]+')"

# The same node at a fraction it cannot hold.
out=$(run_case "$BUSY_BUT_FITS" "0,1,2" 0.95 false)
check "a fraction that does not fit is refused" "exit:2" "$(tail -1 <<< "$out")"
case "$out" in *"DOES NOT FIT"*) echo "  PASS  names the device"; pass=$((pass+1)) ;;
    *) echo "  FAIL  does not name the device"; fail=$((fail+1)) ;; esac
case "$out" in *"XLA_PYTHON_CLIENT_MEM_FRACTION=0.60"*)
    echo "  PASS  suggests the fix"; pass=$((pass+1)) ;;
    *) echo "  FAIL  no suggestion"; fail=$((fail+1)) ;; esac

out=$(run_case "$BUSY_BUT_FITS" "0,1,2" 0.95 true)
check "--allow-busy-gpus overrides" "exit:0" "$(tail -1 <<< "$out")"
case "$out" in *"Expect an OOM"*) echo "  PASS  warns on override"; pass=$((pass+1)) ;;
    *) echo "  FAIL  silent override"; fail=$((fail+1)) ;; esac

# Only the assigned devices matter: GPU 3 is full but not in use here.
FULL_SPARE=$'0, 143771, 1000\n3, 143771, 143000'
out=$(run_case "$FULL_SPARE" "0" 0.90 false)
check "an unassigned full device is ignored" "exit:0" "$(tail -1 <<< "$out")"

out=$(run_case "$FULL_SPARE" "0,3" 0.90 false)
check "an assigned full device is refused" "exit:2" "$(tail -1 <<< "$out")"

# No nvidia-smi at all: a CPU box is not a reason to refuse to start.
out=$( nvidia-smi() { return 127; }
       ( gpu_capacity_check "0,1" 0.90 false ) 2>&1; echo "exit:$?" )
check "an unreadable nvidia-smi does not block" "exit:0" "$(tail -1 <<< "$out")"
check "and says nothing" "" "$(grep -c 'gpu capacity' <<< "$out" | grep -v '^0$')"

echo; echo "  $pass passed, $fail failed"
[[ "$fail" == 0 ]]
