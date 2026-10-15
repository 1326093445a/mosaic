#!/usr/bin/env bash
# Will what this run preallocates actually fit on the devices it was given?
#
# WHY THIS EXISTS. Both GPU launchers validated the device LIST -- distinct,
# numeric, enough entries -- and then assumed the devices were idle. On
# 2026-10-14 the JN.1 seed stage launched onto eight H200s that were already
# carrying a 23-hour job at 8-12 GB each. It fitted, with 1,582 MiB of margin
# against a 0.90 preallocation, but nothing said so: the margin had to be
# reconstructed afterwards from a `nvidia-smi` dump. The useful check is not
# "is anything running there" -- the answer is often yes and harmless -- but
# "does what I am about to claim fit in what is free".
#
# `run_tests.sh` refuses a busy device outright, which is right for a test
# suite that wants one device briefly and can wait. It is wrong for a shared
# cluster node where a 2-hour run has to coexist, so this reports the margin
# and refuses only when the arithmetic does not work.
#
# Usage:
#   source "$SCRIPT_DIR/lib_gpu_capacity.sh"
#   gpu_capacity_check "$DEVICES" "$MEM_FRACTION" "$ALLOW_BUSY"
#
# Exits 2 on a device that cannot hold the request, unless allow_busy is
# "true". Says nothing and returns 0 when nvidia-smi is unavailable: a CPU box
# or a container without the tool is not a reason to refuse to start.

gpu_capacity_check() {
    local devices="$1" fraction="$2" allow_busy="${3:-false}"
    command -v nvidia-smi >/dev/null 2>&1 || return 0

    local query
    query=$(nvidia-smi --query-gpu=index,memory.total,memory.used \
                       --format=csv,noheader,nounits 2>/dev/null) || return 0
    [[ -n "$query" ]] || return 0

    local -A total=() used=()
    local index mem_total mem_used
    while IFS=, read -r index mem_total mem_used; do
        index="${index// /}"; mem_total="${mem_total// /}"; mem_used="${mem_used// /}"
        [[ "$index" =~ ^[0-9]+$ ]] || continue
        total[$index]="$mem_total"
        used[$index]="$mem_used"
    done <<< "$query"

    local tight=0 reported=0 device
    IFS=',' read -r -a _caps_dev <<< "$devices"
    for device in "${_caps_dev[@]}"; do
        [[ -n "${total[$device]:-}" ]] || continue
        local want free margin
        want=$(awk -v t="${total[$device]}" -v f="$fraction" 'BEGIN{printf "%d", t*f}')
        free=$(( ${total[$device]} - ${used[$device]} ))
        margin=$(( free - want ))
        if (( reported == 0 )); then
            echo "  gpu capacity:   fraction $fraction of each device"
            reported=1
        fi
        printf '    GPU %-3s free %7s MiB, this run wants %7s MiB, margin %+8s MiB%s\n' \
            "$device" "$free" "$want" "$margin" \
            "$( (( margin < 0 )) && echo '  <-- DOES NOT FIT' )"
        (( margin < 0 )) && tight=1
    done

    if (( tight )) && [[ "$allow_busy" != "true" ]]; then
        echo >&2
        echo "Refusing to start: at least one device cannot hold a $fraction" >&2
        echo "preallocation alongside what is already resident. Lower it with" >&2
        echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 bash <this script> ..." >&2
        echo "pick other devices with --devices, or pass --allow-busy-gpus to" >&2
        echo "proceed anyway and have the OOM recorded against this decision." >&2
        exit 2
    fi
    if (( tight )); then
        echo "  WARNING: proceeding onto a device that cannot hold the request" >&2
        echo "           (--allow-busy-gpus). Expect an OOM at model load." >&2
    fi
    return 0
}
