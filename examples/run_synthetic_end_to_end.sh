#!/usr/bin/env bash
# End-to-end synthetic validation only; no biological model or checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_synthetic_attention_cluster.sh" \
    --fixed-devices --validation-suite --torch-backend cpu \
    --seeds 0,1,2 --sizes small,medium,large --precision-modes strict \
    --rounds 5 --repeats 5 \
    --variants fp32,original_mixed_bf16,qkv_rounding_only "$@"
