#!/usr/bin/env bash
# Use CUDA_VISIBLE_DEVICES in the calling environment to select the GPU.
# All arguments pass through to p17_confidence_search.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Help must not mutate installed dependencies.
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
    exec .venv/bin/python examples/p17_confidence_search.py "$@"
fi
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
.venv/bin/python patches/patch_jopendde_bf16_dtype.py
PYTHONUNBUFFERED=1 exec .venv/bin/python examples/p17_confidence_search.py "$@"
