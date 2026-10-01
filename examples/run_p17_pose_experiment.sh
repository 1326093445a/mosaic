#!/usr/bin/env bash
# Paths follow this checkout (including /storage/frank/mosaic on the cluster).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
exec .venv/bin/python examples/p17_pose_experiment.py "$@"
