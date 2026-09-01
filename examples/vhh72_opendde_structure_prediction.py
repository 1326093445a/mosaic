"""Sanity check: can OpenDDE predict the real VHH72 -> WT RBD complex with
any confidence at all?

Everything else in this project's OpenDDE testing (build_distogram_only_loss
for in-loop guidance, the refold path) assumes OpenDDE's signal about this
complex is meaningful. If OpenDDE can't predict this real, published
complex (VHH72 x SARS-CoV-2 WT RBD -- a real, known binder, not a random
pair) with reasonable confidence, that's a prior question that undercuts
everything built on top of it -- no point trusting a gradient derived from a
model that doesn't think this pair binds.

Runs OpenDDE's REAL structure-prediction path (target_only_features -> full
diffusion sampling -> confidence heads, via opendde_forward_from_trunk) on
the discrete, real sequences -- not the distogram-only guidance path used
elsewhere, and not the poly-Trp designable placeholder.

IMPORTANT: opendde_forward_from_trunk MUST be called under eqx.filter_jit at
real complex sizes, not eagerly. An earlier version of this script called it
eagerly and OOM'd catastrophically (223GB requested for the full 334-residue
complex) -- isolated (see docs/guidance_alphaseq_testing_notes.md) to
expand_to_structural_tokens specifically: without JIT, XLA can't fuse
operations across the structural-token refiner's transformer-style layers,
so every intermediate over the (N_struct, N_struct, C) pair representation
gets fully materialized with no fusion. Under eqx.filter_jit the identical
computation runs in ~7s with a normal, small memory footprint. This is NOT a
correctness bug in jopendde or a real memory ceiling -- production code
(refold_pareto_with_opendde) was never affected because it always wraps this
call in eqx.filter_jit already; the standalone smoke tests
(opendde_smoke_test.py, opendde_refold_smoke_test.py) call it eagerly too,
but only ever "worked" because they use tiny synthetic sequences (~56
residues) where unfused eager execution happens to still fit. Do not trust
an eager call as representative of whether a complex is tractable -- always
JIT before drawing conclusions about memory.

Multiple independent samples are drawn (stochastic diffusion sampling) to
check consistency, not just a single point estimate -- wide variance across
samples would itself be a sign OpenDDE is uncertain about this complex.

Usage:
    .venv/bin/python examples/vhh72_opendde_structure_prediction.py
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import functools
import tempfile
import time
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

print("=== Can OpenDDE predict VHH72 -> WT RBD with confidence? ===", flush=True)

t0 = time.time()
print("[1/6] importing...", flush=True)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
# JIT-compiling opendde_forward_from_trunk (see module docstring) fixed the
# catastrophic 223GB blowup, but the full 334-residue complex still
# genuinely needs ~45-50GB for full diffusion sampling (n_step=8) + full
# confidence heads -- more than allocator/mem-fraction tuning can close on
# one 24GB GPU. That earlier single-sample raw-torch CLI run (35s, no issue)
# used real kernel fusion (cuequivariance triangle kernels) this JAX port
# doesn't have equivalents for, so it isn't directly comparable on memory.
# Falling back to the CDR-interface crop, same as the gradient test.
COMPLEX_CIF = REPO_ROOT / "vhh72_wt_wt_rbd_cropped.cif"
IPSAE_PAE_CUTOFF = 12.0
N_SAMPLES = 3

t0 = time.time()
print("[2/6] extracting real VHH72 + WT RBD sequences from the CIF...", flush=True)
st = gemmi.read_structure(str(COMPLEX_CIF))
st.setup_entities()
binder_seq = gemmi.one_letter_code([r.name for r in st[0]["A"]]).upper()
target_seq = gemmi.one_letter_code([r.name for r in st[0]["A2"]]).upper()
print(f"  binder (VHH72, {len(binder_seq)} aa): {binder_seq}", flush=True)
print(f"  target (WT RBD, {len(target_seq)} aa): {target_seq}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[3/6] loading OpenDDEModelAbag()...", flush=True)
model = OpenDDEModelAbag()
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[4/6] featurizing with target_only_features (real, discrete sequences)...", flush=True)
feat, _ = model.target_only_features(
    [TargetChain(binder_seq, use_msa=False), TargetChain(target_seq, use_msa=False)]
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[5/6] running trunk + full forward (coords + confidence heads)...", flush=True)
s_inputs, s, z = model.model.get_pairformer_output(feat, 1)

iptm_loss, bt_ipsae_loss, tb_ipsae_loss, ipsae_min_loss = (
    IPTMLoss(), BinderTargetIPSAE(pae_cutoff=IPSAE_PAE_CUTOFF),
    TargetBinderIPSAE(pae_cutoff=IPSAE_PAE_CUTOFF), IPSAE_min(pae_cutoff=IPSAE_PAE_CUTOFF),
)
binder_placeholder = jnp.zeros((len(binder_seq), 20))

# MUST be jitted at this size -- see module docstring. eqx.filter_jit
# because opendde_forward_from_trunk closes over the eqx.Module `model.model`
# and Features pytree, not just plain arrays.
_forward_jit = eqx.filter_jit(
    functools.partial(
        opendde_forward_from_trunk,
        n_step=8,
        dense_atom_to_atom37=model.dense_atom_to_atom37,
        pae_bin_params=model.pae_bin_params,
        plddt_bin_params=model.plddt_bin_params,
    )
)

samples = []
for i in range(N_SAMPLES):
    out = _forward_jit(model.model, feat, s_inputs, s, z, jax.random.key(1000 + i))
    _, iptm_aux = iptm_loss(sequence=binder_placeholder, output=out, key=jax.random.key(2000 + i))
    _, bt_aux = bt_ipsae_loss(sequence=binder_placeholder, output=out, key=jax.random.key(3000 + i))
    _, tb_aux = tb_ipsae_loss(sequence=binder_placeholder, output=out, key=jax.random.key(4000 + i))
    _, ipsae_min_aux = ipsae_min_loss(sequence=binder_placeholder, output=out, key=jax.random.key(5000 + i))
    plddt_mean = float(jnp.mean(out.plddt)) if hasattr(out, "plddt") and out.plddt is not None else None
    row = {
        "sample": i,
        "iptm": float(iptm_aux["iptm"]),
        "bt_ipsae": float(bt_aux["bt_ipsae"]),
        "tb_ipsae": float(tb_aux["tb_ipsae"]),
        "ipsae_min": float(ipsae_min_aux["ipsae_min"]),
        "plddt_mean": plddt_mean,
    }
    samples.append((row, out))
    print(f"  sample {i}: iptm={row['iptm']:.4f} bt_ipsae={row['bt_ipsae']:.4f} "
          f"tb_ipsae={row['tb_ipsae']:.4f} ipsae_min={row['ipsae_min']:.4f} "
          f"plddt_mean={plddt_mean}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[6/6] interface geometry + writing predicted structures for visual inspection...", flush=True)
work_dir = Path(tempfile.mkdtemp(prefix="vhh72_opendde_pred_"))
out_dir = REPO_ROOT / "vhh72_opendde_predictions"
out_dir.mkdir(exist_ok=True)

for row, out in samples:
    structure = out.to_structure()
    chain_names = [c.name for c in structure[0]]
    assert chain_names == ["A", "B"], f"unexpected chain order: {chain_names}"
    metrics = interface_geometry_metrics(structure, binder_chain_id="A", target_chain_ids=["B"])
    row.update(metrics)
    cif_path = out_dir / f"vhh72_opendde_predicted_sample{row['sample']}.cif"
    write_structure_cif(structure, cif_path)
    print(f"  sample {row['sample']}: {metrics}", flush=True)
    print(f"    wrote {cif_path}", flush=True)

iptm_vals = [r["iptm"] for r, _ in samples]
ipsae_vals = [r["ipsae_min"] for r, _ in samples]
print(f"\nacross {N_SAMPLES} samples: iptm mean={np.mean(iptm_vals):.4f} std={np.std(iptm_vals):.4f}, "
      f"ipsae_min mean={np.mean(ipsae_vals):.4f} std={np.std(ipsae_vals):.4f}", flush=True)
print(f"done ({time.time() - t0:.1f}s)", flush=True)

print(
    "\nDone. This does NOT tell us whether the gradient is well-behaved "
    "(see vhh72_gradient_path_comparison.py for that) -- it tells us "
    "whether OpenDDE thinks this real, known-binding complex is even a "
    "plausible, confident prediction in the first place. Compare the ipTM/"
    "ipSAE numbers above against typical thresholds (commonly ~0.8+ ipTM, "
    "~0.5+ ipSAE for confident real interfaces in the literature -- not "
    "validated by this project specifically) and eyeball the written CIFs "
    "against the real vhh72_wt_wt_rbd.cif structure.",
    flush=True,
)
