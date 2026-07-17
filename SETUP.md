# Setup guide for the guided-partial-diffusion fork

This guide is for the modifications added in commit `5fad377`: guided partial
diffusion of BoltzGen, edit-budgeted greedy polish, and the VHH design workflow
in `src/mosaic/workflows/boltzgen_vhh_guided.py`. The matching file under
`examples/` is a compatibility launcher. This supplements the upstream `README.md`.

If you only want to run the existing mosaic examples (no partial diffusion),
the upstream README is sufficient — this fork is a strict superset.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python | **3.12.x** (pinned in `pyproject.toml`) |
| GPU | CUDA-capable; ≥24 GB VRAM recommended for full pipeline. 8 GB works for v0/v1 smoke only. |
| CUDA drivers | 12.x. WSL2 works (we tested on Driver 576.02 / CUDA 12.9). |
| Disk | ~10 GB for the virtualenv + ~6 GB for BoltzGen weights downloaded on first run |
| Git | with SSH key registered on GitHub if cloning your fork via SSH |

**VRAM budget per pipeline mode:**
- v0: BoltzGen diffusion + JAX BoltzGen-IF decode
- v1: above + differentiable JAX BoltzGen-IF edit-budget guidance
- v2: above + ESM2, AbLang2, and optional Boltz2 guidance
- v3 / v4: above + sequence search and Boltz2 refolding

The guided modes are intended for large-memory accelerator nodes. Measure the
actual compiled peak on the target complex before launching large seed sweeps.

---

## 2. Install `uv` (the package manager mosaic uses)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc to persist
uv --version    # should print uv 0.11.x or newer
```

`uv` is what the upstream README recommends. It handles the JAX-CUDA wheels
and dependency overrides automatically.

---

## 3. Clone the fork

```bash
git clone git@github.com:1326093445a/mosaic.git
cd mosaic
```

(Or `https://github.com/1326093445a/mosaic.git` if you don't have SSH set up.)

Single branch (`main`), our work sits as commit `5fad377` on top of upstream.

---

## 4. Create the virtualenv and install dependencies

The exact dependency versions are pinned in `uv.lock` (committed). Reproducible
across machines.

```bash
uv sync --group jax-cuda
```

This will:

1. Create a `.venv/` in the repo (Python 3.12)
2. Install all dependencies (mosaic itself in editable mode, boltzgen, joltzgen,
   joltz, esm2quinox, esmj, ablang, jablang, protenix, jax with CUDA12 plugin,
   torch CPU-only, etc.)
3. Take 2–5 minutes depending on network

Variants:
- `uv sync --group jax-cpu` if you want CPU-only (slow, useful for syntax checks)
- `uv sync --group jax-tpu` for TPU machines

If your machine has CUDA driver issues, `uv sync` will still finish — failures
only surface when JAX actually tries to allocate GPU memory.

---

## 5. Verify the install

```bash
uv run python -c "
import jax
print('JAX:', jax.__version__, jax.default_backend(), jax.devices())
import joltzgen
from joltzgen import AtomDiffusion
assert hasattr(AtomDiffusion, 'preconditioned_network_forward')
print('joltzgen OK')
import mosaic
print('mosaic OK')
"
```

Expected output:
```
JAX: 0.10.0 gpu [CudaDevice(id=0)]
joltzgen OK
mosaic OK
```

If `jax.devices()` shows `[CpuDevice(id=0)]`, the CUDA plugin isn't being picked
up — see Troubleshooting below.

---

## 6. v0 smoke test (vanilla partial diffusion)

Verifies the full BoltzGen + partial-diffusion pipeline runs end-to-end with no
classifier guidance. The first run will download BoltzGen weights (~770 MB) +
`mols.zip` (~390 MB) into `~/.boltz/`.

Set `MOSAIC_SMOKE_STRUCTURE` to a local protein PDB/mmCIF containing chain A;
residues 50-54 must exist because this smoke test marks them designable.

```bash
uv run python << 'PY'
import os
from pathlib import Path
import time, jax, jax.numpy as jnp
from mosaic.models.boltzgen import (
    load_boltzgen, load_features_and_structure_writer, Sampler,
    guided_partial_diffusion, build_atom_partial_mask,
)
print('loading BoltzGen...'); t0=time.time()
boltzgen = load_boltzgen()
print(f'  {time.time()-t0:.1f}s')

structure_path = Path(os.environ['MOSAIC_SMOKE_STRUCTURE']).expanduser().resolve()
yaml_str = """
entities:
  - file:
      path: INPUT_STRUCTURE
      include: [{chain: {id: A}}]
      design: [{chain: {id: A, res_index: '50,51,52,53,54'}}]
""".replace("INPUT_STRUCTURE", structure_path.name)
features, _ = load_features_and_structure_writer(
    yaml_string=yaml_str, files={structure_path.name: structure_path},
)
sampler = Sampler.from_features(
    model=boltzgen, features=features, key=jax.random.key(0),
    deterministic=True, recycling_steps=1,
)
designable_token_mask = jnp.array(features['design_mask'][0], dtype=bool)
atom_partial_mask = build_atom_partial_mask(features, designable_token_mask)
initial_coords = jnp.array(features['coords'])[0, 0]
atom_pad_mask = jnp.array(features['atom_pad_mask'][0], dtype=jnp.float32)

print('vanilla partial diffusion (5 steps)...'); t0=time.time()
x_final = guided_partial_diffusion(
    sampler=sampler, structure_module=boltzgen.structure_module,
    initial_coords=initial_coords, atom_partial_mask=atom_partial_mask,
    atom_mask=atom_pad_mask, num_sampling_steps=5, start_sigma_frac=0.3,
    step_scale=2.0, noise_scale=0.88, guidance_fn=None, key=jax.random.key(0),
)
print(f'  {time.time()-t0:.1f}s')
delta = jnp.abs(x_final - initial_coords)
df = float((delta * (atom_partial_mask < 0.5)[:, None].astype(jnp.float32)).max())
dd = float((delta * (atom_partial_mask > 0.5)[:, None].astype(jnp.float32)).max())
print(f'  frozen Δ={df:.4f} | designable Δ={dd:.4f}')
assert df < 1e-3, 'frozen atoms moved'
print('PASS')
PY
```

Expected: `frozen Δ=0.0000 | designable Δ≈8` and `PASS`. End-to-end ~90 s on a
4060 (45 s model load + 35 s trunk JIT + 12 s sampling).

---

## 7. v1 smoke test (JAX BoltzGen-IF guidance)

The focused slow test loads the official checkpoint and verifies that a loss on
the soft JAX BoltzGen-IF sequence produces a finite, nonzero atom-coordinate
gradient:

```bash
uv run pytest -q -m slow tests/test_boltzgen_if_jax.py
```

The ordinary test suite also checks Torch/JAX encoder and decoder parity:

```bash
uv run pytest -q
```

---

## 8. Run the full driver

```bash
uv run python examples/boltzgen_vhh_guided.py \
    --mode v3 \
    --complex-cif <path-to-Ab-Ag-complex.cif> \
    --binder-chain <heavy-chain-id> \
    --target-chains <target-chain-id-1> [<target-chain-id-2> ...] \
    --cdr-indices <space-separated-1-indexed-positions> \
    --budget 7 \
    --output-dir ./vhh_designs
```

Modes:
- `--mode v0` — vanilla partial diffusion only (no guidance, no polish)
- `--mode v1` — EditBudget-only guidance, no polish
- `--mode v2` — full multi-model guidance, no polish
- `--mode v3` — adds edit-budgeted greedy polish (full pipeline minus refold)
- `--mode v4` — adds Boltz2 refolding + iPTM/ipSAE ranking

Outputs land in `--output-dir`:
- `config.json` — config snapshot
- `iterations.csv` — per-outer-loop metrics
- `pareto_front.csv` — best sequence at each edit count 0..budget
- `refold_ranked.csv` (v4 only) — Boltz2-ranked candidates

---

## 9. Troubleshooting

### `jax.devices()` shows `CpuDevice` instead of `CudaDevice`

```bash
nvidia-smi   # confirm GPU is visible
uv run python -c "import jax_cuda12_pjrt; print(jax_cuda12_pjrt.__file__)"
```

If `nvidia-smi` works but JAX can't find the GPU, your CUDA driver may be too
old or too new for the wheels in the lockfile. Try:
```bash
uv pip install -U "jax[cuda12]"
```
which installs the current wheels and may resolve compatibility.

### "Could not get kernel mode driver version" warning on WSL2

Harmless. The kernel driver version string format is non-standard on WSL but
JAX still finds the GPU. Ignore.

### Triton kernel error `tt.dot op expected ...`

We saw this on the 4060 — it's a non-fatal kernel selection issue in JAX-Triton.
Sampling still completes. Ignore unless an OperationFailed exception is raised.

### OOM during sampling on small GPU

The full `--mode v3`/`v4` pipeline wants ≥16 GB VRAM. On smaller GPUs:
- Use `--mode v0` or `v1` for development (fits in 8 GB)
- Reduce `recycling_steps` from 3 to 1
- Reduce `num_sampling_steps` (default 200 → 50)
- Reduce `polish_batch_size` from 16 to 4 in the config

### `mols.zip` download fails

The download URL is hardcoded in `mosaic.models.boltzgen.load_boltzgen`. If
HuggingFace is unreachable, manually download
`https://huggingface.co/datasets/boltzgen/inference-data/resolve/main/mols.zip`
and place it at `~/.boltz/mols.zip`.

### "boltzgen 0.3.1" installed, no partial-diffusion support

The PyPI release of `boltzgen` doesn't have the partial-diffusion changes from
the original author's local fork. **This doesn't matter for our pipeline** —
`mosaic.models.boltzgen.guided_partial_diffusion` re-implements partial diffusion
in JAX on top of joltzgen's existing `preconditioned_network_forward`. We do
not depend on `boltzgen.sample()` having `partial_diffusion_mask`.

---

## 10. What our changes added (high-level reference)

| File | Public API |
|---|---|
| `src/mosaic/losses/transformations.py` | `EditBudget`, `FixedLogitsSequenceLoss` |
| `src/mosaic/optimizers.py` | `edit_budgeted_greedy_descent` (returns `(best_seq, best_val, pareto_front)`) |
| `src/mosaic/models/boltzgen.py` | `guided_partial_diffusion`, `build_atom_partial_mask`, `_center` |
| `src/mosaic/models/boltzgen_if.py` | Native BoltzGen-IF checkpoint loading, feature preparation, and discrete decoding |
| `src/mosaic/models/boltzgen_if_jax.py` | JAX BoltzGen-IF checkpoint conversion, geometric GNN, autoregressive decode, and coordinate gradients |
| `src/mosaic/workflows/boltzgen_vhh_guided.py` | CLI workflow with `--mode v0..v4`, `VHHDesignConfig`, `run`, `refold_pareto_with_boltz2` |
| `examples/boltzgen_vhh_guided.py` | Backward-compatible launcher for the packaged workflow |

The mathematical foundation: partial diffusion is sampling-time orchestration on
top of an unmodified denoiser, so the entire change set lives in mosaic; no
joltzgen fork needed. The only joltzgen API surface used is
`AtomDiffusion.preconditioned_network_forward`.

The workflow uses the JAX port of the official BoltzGen-IF checkpoint for
per-step soft-sequence guidance, final-backbone decoding, and the fixed sequence
prior during greedy/MCMC search. ABMPNN is not used by this workflow.

---

## 11. Pulling future upstream changes

```bash
git remote add upstream https://github.com/escalante-bio/mosaic.git
git fetch upstream
git merge upstream/main      # resolve any conflicts
git push origin main
```

Our changes are isolated to 4 files plus the new driver, so conflicts should be
minimal unless upstream rewrites `boltzgen.py`, `optimizers.py`, or
`transformations.py`.
