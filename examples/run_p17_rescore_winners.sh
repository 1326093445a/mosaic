#!/usr/bin/env bash
# Use CUDA_VISIBLE_DEVICES to select ONE allocated GPU. All args go to Python.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h || "$arg" == --dry-run ]]; then
        exec .venv/bin/python examples/p17_rescore_winners.py "$@"
    fi
done
.venv/bin/python patches/patch_jopendde_outer_product_mean.py
.venv/bin/python patches/patch_jopendde_structural_token_expander.py
.venv/bin/python patches/patch_jopendde_bf16_dtype.py
.venv/bin/python patches/patch_jopendde_aggregation.py
.venv/bin/python patches/patch_jopendde_padding.py
PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda exec .venv/bin/python examples/p17_rescore_winners.py "$@"
