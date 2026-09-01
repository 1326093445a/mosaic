"""Standalone smoke test for the OpenDDE POST-REFOLD path
(refold_pareto_with_opendde in
src/mosaic/workflows/boltzgen_vhh_guided.py) -- the coverage gap flagged in
review: examples/opendde_smoke_test.py only exercises the distogram-only
in-loop guidance path (build_distogram_only_loss), which is architecturally
a *different* OpenDDE code path from refolding (full diffusion coordinate
sampling + confidence head via opendde_forward_from_trunk, not the
trunk-only distogram head).

This script exercises the same pieces refold_pareto_with_opendde uses on
each Pareto candidate: target_only_features -> trunk -> full forward
(coordinates + PAE) -> ipTM/ipSAE scoring -> to_structure() -> CIF write,
plus the interface/RMSD metric functions -- and specifically checks the one
assumption refold_pareto_with_opendde's hardcoded refolded_binder_chain_id
= "A" / refolded_target_chain_ids = ["B", ...] convention depends on: that
StructureModelOutput.to_structure() assigns chain letters in asym_id order,
and OpenDDE's featurizer assigns asym_id in input-chain order (binder
first). That assumption was never verified against a real OpenDDE output
before this script existed.

No real complex CIF is available for this check, so this uses two
independently-sampled forward passes of the SAME synthetic sequences as a
stand-in "original" and "refolded" structure pair -- enough to exercise
every function in the pipeline with real (not synthetic/hand-built) gemmi
Structures, even though the RMSD numbers themselves aren't scientifically
meaningful (two independent structure predictions of the same sequence, not
a real refold-vs-parent comparison).

Requires: same as examples/opendde_smoke_test.py (uv sync --group jax-cuda,
GPU, opendde_abag.pt checkpoint).

Usage:
    .venv/bin/python examples/opendde_refold_smoke_test.py

    # Optional: raise sampling depth on larger GPUs.
    MOSAIC_OPENDDE_SMOKE_STEPS=8 .venv/bin/python examples/opendde_refold_smoke_test.py
"""
import os

# See examples/opendde_smoke_test.py for why -- this test additionally hits a
# second, distinct autotune-related failure: refold_pareto_with_opendde's
# eqx.filter_jit(jax.vmap(...)) scoring path compiles a >1GB kernel binary
# whose CUDA module load competes with JAX's preallocated memory arena.
# Disabling autotuning avoids both. Must be set before `import jax` below.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

print("=== OpenDDE refold-path smoke test ===", flush=True)

t0 = time.time()
print("[1/7] importing...", flush=True)
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.losses.opendde import opendde_forward_from_trunk
from mosaic.losses.structure_prediction import (
    IPTMLoss, BinderTargetIPSAE, TargetBinderIPSAE, IPSAE_min,
)
from mosaic.structure_prediction import TargetChain
from mosaic.legacy.boltzgen_vhh_guided import (
    interface_geometry_metrics,
    target_aligned_rmsd_metrics,
    write_structure_cif,
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[2/7] constructing OpenDDEModelAbag()...", flush=True)
model = OpenDDEModelAbag()
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

BINDER_LEN = 15
BINDER_SEQ = "QVQLQESGGGLVQAG"[:BINDER_LEN]
TARGET_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV"
FULL_FORWARD_STEPS = int(os.environ.get("MOSAIC_OPENDDE_SMOKE_STEPS", "1"))

t0 = time.time()
print(
    "[3/7] featurizing with target_only_features (real, discrete binder "
    "sequence -- this is the refold path, not the soft-PSSM guidance path)...",
    flush=True,
)
feat, _ = model.target_only_features(
    [TargetChain(BINDER_SEQ, use_msa=False), TargetChain(TARGET_SEQ, use_msa=False)]
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[4/7] running trunk + one full forward (coords+PAE)...", flush=True)
s_inputs, s, z = model.model.get_pairformer_output(feat, 1)

def _forward(key):
    return opendde_forward_from_trunk(
        model.model, feat, s_inputs, s, z, key,
        n_step=FULL_FORWARD_STEPS,
        dense_atom_to_atom37=model.dense_atom_to_atom37,
        pae_bin_params=model.pae_bin_params,
        plddt_bin_params=model.plddt_bin_params,
    )

out_pred = _forward(jax.random.key(1))
jax.block_until_ready(out_pred.pae)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[5/7] scoring with the same ipTM/ipSAE losses refold_pareto_with_opendde uses...", flush=True)
binder_placeholder = jnp.zeros((BINDER_LEN, 20))
iptm_loss, bt_ipsae_loss, tb_ipsae_loss, ipsae_min_loss = (
    IPTMLoss(), BinderTargetIPSAE(pae_cutoff=12.0), TargetBinderIPSAE(pae_cutoff=12.0),
    IPSAE_min(pae_cutoff=12.0),
)
_, iptm_aux = iptm_loss(sequence=binder_placeholder, output=out_pred, key=jax.random.key(2))
_, bt_aux = bt_ipsae_loss(sequence=binder_placeholder, output=out_pred, key=jax.random.key(3))
_, tb_aux = tb_ipsae_loss(sequence=binder_placeholder, output=out_pred, key=jax.random.key(4))
_, ipsae_min_aux = ipsae_min_loss(sequence=binder_placeholder, output=out_pred, key=jax.random.key(5))
for name, aux in [("iptm", iptm_aux), ("bt_ipsae", bt_aux), ("tb_ipsae", tb_aux), ("ipsae_min", ipsae_min_aux)]:
    v = float(list(aux.values())[0])
    assert jnp.isfinite(v), f"{name} is not finite: {v}"
    print(f"  {name}: {v:.4f}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print(
    "[6/7] to_structure() + chain-mapping check + interface/RMSD metrics + "
    "CIF write...",
    flush=True,
)
structure_pred = out_pred.to_structure()
structure_ref = structure_pred.clone()

chain_names = [c.name for c in structure_pred[0]]
print(f"  chain names: {chain_names}", flush=True)
assert chain_names == ["A", "B"], (
    f"expected asym_id-ordered chains ['A', 'B'] (binder first, per "
    f"OpenDDEFeatures/binder_features placing the binder at asym_id 0), "
    f"got {chain_names} -- refold_pareto_with_opendde's hardcoded "
    f"refolded_binder_chain_id='A'/refolded_target_chain_ids=['B',...] "
    f"convention assumes exactly this ordering. If this assertion fails, "
    f"that convention is wrong and every refolded ranking metric is being "
    f"computed on the wrong chains."
)

binder_chain = structure_pred[0]["A"]
target_chain = structure_pred[0]["B"]
assert len(binder_chain) == BINDER_LEN, (
    f"binder chain has {len(binder_chain)} residues, expected {BINDER_LEN}"
)
assert len(target_chain) == len(TARGET_SEQ), (
    f"target chain has {len(target_chain)} residues, expected {len(TARGET_SEQ)}"
)
print(
    f"  chain lengths OK: binder(A)={len(binder_chain)}, target(B)={len(target_chain)}",
    flush=True,
)

interface_metrics = interface_geometry_metrics(
    structure_pred, binder_chain_id="A", target_chain_ids=["B"],
)
print(f"  interface_geometry_metrics: {interface_metrics}", flush=True)
for k, v in interface_metrics.items():
    if isinstance(v, float):
        assert v == v, f"interface metric {k} is NaN"  # NaN != NaN

rmsd_metrics = target_aligned_rmsd_metrics(
    structure_ref, structure_pred,
    original_binder_chain_id="A", original_target_chain_ids=["B"],
    refolded_binder_chain_id="A", refolded_target_chain_ids=["B"],
    cdr_residue_indices=None,
)
print(f"  target_aligned_rmsd_metrics: {rmsd_metrics}", flush=True)
for k, v in rmsd_metrics.items():
    if isinstance(v, float):
        assert jnp.isfinite(v), f"rmsd metric {k} is not finite: {v}"

import gemmi

work_dir = Path(tempfile.mkdtemp(prefix="opendde_refold_smoke_"))
cif_path = work_dir / "smoke_test.cif"
write_structure_cif(structure_pred, cif_path)
assert cif_path.exists() and cif_path.stat().st_size > 0, "CIF was not written"
reparsed = gemmi.read_structure(str(cif_path))
reparsed.setup_entities()
reparsed_chains = [c.name for c in reparsed[0]]
assert reparsed_chains == ["A", "B"], (
    f"CIF round-trip changed chain names: {reparsed_chains}"
)
print(f"  CIF written and re-parsed OK ({cif_path.stat().st_size} bytes)", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

print(
    "\nPASS (component level): target_only_features, trunk+full forward, "
    "ipTM/ipSAE scoring, to_structure()'s chain-ordering convention, "
    "interface_geometry_metrics, target_aligned_rmsd_metrics, and CIF "
    "write+reparse all work as refold_pareto_with_opendde assumes.",
    flush=True,
)

# ---------------------------------------------------------------------------
# Whole-function check: call refold_pareto_with_opendde itself, not just its
# pieces -- covers batched vmap sampling, best-sample selection,
# rmsd-pass-based ranking, and the three-CSV output, none of which the
# component-level checks above exercise.
# ---------------------------------------------------------------------------
# Free the component-level test's forward-pass buffers before the
# whole-function check below -- they're not needed anymore (structure_ref is
# still needed just below, to seed the synthetic "original complex" CIF), and
# this whole run is tight on a single 24GB GPU.
import gc

del out_pred, structure_pred
gc.collect()
jax.clear_caches()

t0 = time.time()
print(
    "\n[7/7] calling refold_pareto_with_opendde() itself (whole-function "
    "check)...",
    flush=True,
)
from mosaic.common import TOKENS
from mosaic.legacy.boltzgen_vhh_guided import (
    VHHDesignConfig,
    refold_pareto_with_opendde,
)

# structure_ref (from forward pass #1) stands in for "the original complex";
# refold_pareto_with_opendde will refold BINDER_SEQ against it and compare.
complex_cif_path = work_dir / "complex.cif"
write_structure_cif(structure_ref, complex_cif_path)

output_dir = work_dir / "refold_output"
cfg = VHHDesignConfig(
    complex_cif_path=complex_cif_path,
    binder_chain_id="A",
    target_chain_ids=["B"],
    cdr_residue_indices=list(range(1, BINDER_LEN + 1)),
    output_dir=output_dir,
    seed=0,
    recycling_steps=1,
    refold_sampling_steps=FULL_FORWARD_STEPS,
    refold_num_samples=1,
    refold_batch_size=1,
    ipsae_pae_cutoff=12.0,
    refold_rmsd_threshold=0.0,  # <=0 disables the pass/fail filter (see its docstring)
)
seq_ids = np.array([TOKENS.index(c) for c in BINDER_SEQ], dtype=np.int32)
pareto = {0: (0.42, seq_ids)}  # one Pareto candidate at edit_count=0
binder_token_indices = jnp.arange(BINDER_LEN)

refold_pareto_with_opendde(pareto, cfg, binder_token_indices)

ranked_csv = output_dir / "refold_ranked.csv"
best_csv = output_dir / "refold_best_by_edit_count.csv"
all_csv = output_dir / "refold_all_samples.csv"
for path in (ranked_csv, best_csv, all_csv):
    assert path.exists() and path.stat().st_size > 0, f"{path.name} was not written"

import polars as pl

ranked = pl.read_csv(ranked_csv)
all_samples = pl.read_csv(all_csv)
print(f"  refold_ranked.csv: {len(ranked)} row(s)", flush=True)
print(f"  refold_all_samples.csv: {len(all_samples)} row(s)", flush=True)
assert len(all_samples) == cfg.refold_num_samples, (
    f"expected {cfg.refold_num_samples} sampled rows, got {len(all_samples)}"
)
assert len(ranked) == 1, f"expected exactly 1 Pareto candidate ranked, got {len(ranked)}"
top = ranked.row(0, named=True)
for col in ("ipsae_min", "iptm", "bt_ipsae", "tb_ipsae"):
    assert top[col] == top[col], f"{col} is NaN in refold_ranked.csv"
print(f"  top row: ipsae_min={top['ipsae_min']:.4f}, iptm={top['iptm']:.4f}", flush=True)

refolded_cifs = list((output_dir / "refolded_cifs").glob("*.cif"))
assert len(refolded_cifs) == 1, f"expected 1 refolded CIF, found {len(refolded_cifs)}"
print(f"  refolded CIF: {refolded_cifs[0].name}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

print(
    "\nPASS (whole-function): refold_pareto_with_opendde() itself runs "
    "end to end -- batched vmap sampling, best-sample selection, "
    "rmsd-pass-based ranking, and refold_ranked.csv/"
    "refold_best_by_edit_count.csv/refold_all_samples.csv/refolded_cifs/ "
    "output all work.",
    flush=True,
)
