#!/usr/bin/env bash
# Real model forward/backward checks on a fixed, five-residue software fixture.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
cd "$REPO_DIR"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/opendde_toy_backward.py" "$@"
