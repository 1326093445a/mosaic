#!/usr/bin/env bash
# Paths follow this checkout (including /storage/frank/mosaic on the cluster).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# One worker per allocated H200: reserve a large pool up front to reduce
# fragmentation. Explicit caller settings take precedence. Do not set both
# memory-fraction aliases: newer JAX rejects that combination.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
fi
exec .venv/bin/python examples/p17_pose_experiment.py "$@"
