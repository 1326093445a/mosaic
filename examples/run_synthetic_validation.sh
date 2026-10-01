#!/usr/bin/env bash
# Weight-free component checks. Does not launch structure prediction or search.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python unavailable: $PYTHON_BIN; set PYTHON_BIN to an installed environment." >&2
    exit 2
fi
cd "$REPO_DIR"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/synthetic_validation.py" "$@"
