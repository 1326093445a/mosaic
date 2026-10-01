#!/usr/bin/env bash
# Forward-only numerical controls; no optimization stage follows.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_p17_wt_validation.sh" \
    --steps 64 --aggregation-modes original stable "$@"
