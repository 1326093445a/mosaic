"""Standalone smoke test for the mosaic OpenDDE integration
(src/mosaic/models/opendde.py, src/mosaic/losses/opendde.py).

PR #77 ("add opendde") added the wrapper without wiring it into a workflow,
and prior OpenDDE usage in this project went through a separate external
CLI + conda environment rather than this JAX port. Run this before trusting
the guidance/refold wiring in
src/mosaic/workflows/boltzgen_vhh_guided.py (build_opendde_guidance_loss,
refold_pareto_with_opendde) -- it isolates the "does the environment /
checkpoint / JAX conversion actually work" question from the much larger
compile the full driver pays.

What this DOES verify: that OpenDDEModelAbag loads, featurizes, forwards,
that the real guidance objective has a finite/nonzero gradient, and that an
actual step along the negative gradient of a smooth distogram probe lowers
its re-evaluated loss without changing the discrete residue identities.

What this does NOT verify: whether the resulting gradient is "good" for
guidance in the deeper sense discussed in
docs/guidance_design_notes.md/guidance_search_summary.md -- staying close to
BoltzGen's learned manifold, being well-scaled relative to the other
objectives, actually correlating with better binders. Those need real
trajectories and the Phase 2 diagnostics machinery
(src/mosaic/diagnostics.py), not a synthetic single-step check like this one.

Requires: `uv sync --group jax-cuda` (jopendde/opendde are regular
dependencies in pyproject.toml, but were not installed in the .venv this
feature was developed against -- confirm the sync picks them up), a GPU, and
the opendde_abag.pt checkpoint (auto-downloaded on first use, or reuse an
existing ~/.cache/opendde/checkpoint/opendde_abag.pt if present).

Usage:
    uv run python examples/opendde_smoke_test.py
    # or, if `uv run` hangs on this machine (see SETUP.md's known floating
    # git-dependency issue): .venv/bin/python examples/opendde_smoke_test.py
"""
import os

# XLA's autotuner benchmarks candidate kernels by transiently grabbing large
# scratch buffers during compilation -- on this size of GPU that alone can
# trigger a spurious CUDA OOM before the real forward pass ever runs. Set
# before jax's backend initializes (must happen before the `import jax`
# below); mirrors the same self-set boltzgen_vhh_guided.py's run() now does.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import time

import jax
import jax.numpy as jnp

print("=== OpenDDE smoke test ===", flush=True)

t0 = time.time()
print("[1/5] importing mosaic.models.opendde...", flush=True)
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.structure_prediction import TargetChain
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print(
    "[2/5] constructing OpenDDEModelAbag() -- loads opendde_abag.pt and "
    "builds/loads the cached per-residue templates (slow on first run, "
    "fast after; see build_opendde_templates/_get_templates)...",
    flush=True,
)
model = OpenDDEModelAbag()
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

# A short synthetic binder + target -- enough to exercise cross-chain features
# without making this wiring smoke compete with production-sized predictions.
BINDER_LEN = 15
TARGET_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"

t0 = time.time()
print("[3/5] featurizing a synthetic binder+target complex...", flush=True)
features, _ = model.binder_features(BINDER_LEN, [TargetChain(TARGET_SEQ, use_msa=False)])
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print(
    "[4/5] building the distogram-only guidance loss (DistogramIPTMProxy, "
    "1 recycle) -- this is the exact same build_distogram_only_loss path "
    "build_opendde_guidance_loss uses in the real driver, not a stand-in...",
    flush=True,
)
from mosaic.losses.structure_prediction import DistogramIPTMProxy

loss_term = model.build_distogram_only_loss(
    loss=DistogramIPTMProxy(),
    features=features,
    recycling_steps=1,
)

key = jax.random.key(0)
soft_seq = jax.nn.softmax(jax.random.normal(key, (BINDER_LEN, 20)))
value, aux = loss_term(soft_seq, key)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

print(f"\nloss value: {float(value):.4f}", flush=True)
print(f"aux: {jax.tree.map(lambda x: float(x) if x.ndim == 0 else x.shape, aux)}", flush=True)

assert jnp.isfinite(value), f"loss value is not finite: {value}"
loss_fn = lambda s: loss_term(s, key)[0]
grad = jax.grad(loss_fn)(soft_seq)
assert jnp.all(jnp.isfinite(grad)), "gradient contains non-finite values"
assert jnp.any(grad != 0), "gradient is all-zero -- guidance would be a no-op"
print("finite/nonzero check: PASS (necessary, not sufficient -- see below)", flush=True)

t0 = time.time()
print(
    "\n[5/5] descent-direction check through a smooth distogram contact "
    "probe. DistogramIPTMProxy itself uses hard top-k pair selection, so a "
    "finite coordinate perturbation can cross a selection boundary. Use a "
    "smooth probe and verify that an actual step along -gradient lowers its "
    "re-evaluated loss without changing any residue argmax.",
    flush=True,
)
from mosaic.common import LossTerm


class SmoothContactProbe(LossTerm):
    def __call__(self, sequence, output, key):
        binder_len = sequence.shape[0]
        logits = output.distogram_logits[:binder_len, binder_len:]
        contact_mask = output.distogram_bins < 8.0
        contact_probability = jax.nn.softmax(logits, axis=-1)[..., contact_mask].sum(-1)
        value = -jnp.log(contact_probability + 1e-6).mean()
        return value, {"smooth_contact_probe": value}


probe_term = model.build_distogram_only_loss(
    loss=SmoothContactProbe(),
    features=features,
    recycling_steps=1,
)
loss_fn = lambda s: probe_term(s, key)[0]
grad = jax.grad(loss_fn)(soft_seq)
assert jnp.all(jnp.isfinite(grad))
assert jnp.any(grad != 0)

direction = grad / (jnp.max(jnp.abs(grad)) + 1e-12)
step_size = 1e-3
minus = soft_seq - step_size * direction
plus = soft_seq + step_size * direction
original_ids = jnp.argmax(soft_seq, axis=-1)
assert jnp.array_equal(jnp.argmax(minus, axis=-1), original_ids)
assert jnp.array_equal(jnp.argmax(plus, axis=-1), original_ids)

loss_minus = float(loss_fn(minus))
loss_plus = float(loss_fn(plus))
print(
    f"  loss(-grad)={loss_minus:.6f}  loss(+grad)={loss_plus:.6f}",
    flush=True,
)
assert loss_minus < loss_plus, (
    "stepping along -gradient did not lower the smooth probe relative to "
    "+gradient; do not trust this gradient for guidance"
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

print(
    f"\nPASS: OpenDDEModelAbag loads, featurizes, forwards, its real "
    f"DistogramIPTMProxy guidance is finite/nonzero, and a smooth distogram "
    f"probe decreases under an actual -gradient step with residue identities "
    f"held fixed.",
    flush=True,
)
