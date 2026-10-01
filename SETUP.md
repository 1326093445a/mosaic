# OpenDDE setup and validation

This guide covers the OpenDDE work in this checkout. Start with
[the current status and next steps](docs/p17_status_and_next_steps.md) for the
latest findings; [the detailed record](docs/p17_jn1_redesign.md) contains history
and references.

## Environment

The project requires Python 3.12. Dependencies and source overrides are recorded
in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). The existing installation
command from the repository root is:

```bash
uv sync --group jax-cuda
```

For CPU-only local checks, the alternative dependency group is `jax-cpu`.
Full-model GPU memory requirements depend on the inputs and compiled path;
small-kernel tests do not establish that a full prediction will fit.

The repository currently selects CPU Torch wheels. A JAX CUDA installation
therefore does not establish native Torch CUDA availability. Preserve the working
cluster environment and check both frameworks before assuming that a fresh
installation matches it:

```bash
.venv/bin/python - <<'PY'
import jax
import torch
import jopendde
import mosaic
print('JAX:', jax.__version__, jax.devices())
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('JOpenDDE:', jopendde.__file__)
print('Mosaic:', mosaic.__file__)
PY
```

These are environment checks, not model validation. The full controls also need
the OpenDDE checkpoint, its inference assets and the reference input used by the
existing validator. The nested `OpenDDE/` checkout has its own repository history.

## Dependency patches and local checks

The WT validation runner applies five source-checked patches before worker imports:
outer-product mean, structural-token expansion, BF16 dtype handling, aggregation
and padding. Their implementations are in [patches/](patches/). Reinstalling a
dependency can replace patched files; use the launcher to reapply its patch set.

The runner also prepares the schema-2 atom-template cache once on CPU before
spawning GPU workers. Watch `logs/template_cache.log` during that preparation.

[Test notes](tests/README.md) describe the small numerical regression checks.
The recorded CPU, GPU-kernel and featurizer results are summarized in the
[current handoff](docs/p17_status_and_next_steps.md#implemented-fixes-and-verification).
They do not verify full-model geometry or gradients.

## Cluster handoff

Use the [current validation instructions](docs/p17_status_and_next_steps.md#next-cluster-validation)
for the next run and its outputs. The scripts resolve paths from the checkout,
so the cluster checkout at `/storage/frank/mosaic` needs no `/home/yfeng17`
path substitution. The numerical preset performs forward controls only.

The WT shell wrapper defaults to GPU preallocation and a 0.90 memory fraction,
while respecting explicit caller settings. Avoid setting both memory-fraction
aliases simultaneously. Saved worker memory reports describe JAX allocator usage;
they are not a complete accounting of every process on a device.
