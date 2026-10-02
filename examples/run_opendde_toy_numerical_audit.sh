#!/usr/bin/env bash
# Fixed toy-input audit; eight GPUs run two probes x four seeds by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
cd "$REPO_DIR"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/opendde_toy_numerical_audit.py" "$@"
