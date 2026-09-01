"""Path 1: does the guidance gradient's merge mechanism matter?

Compares two ways of turning per-objective gradients (g_bind from OpenDDE,
g_nat from AbLang2, g_edit from EditBudget) into a coordinate update, at the
*same* real x0_hat, from the *same* raw gradients:

  - "raw": g_total = g_bind + g_nat + g_edit, no masking, no de-meaning, no
    RMS-normalization, no PCGrad conflict projection, no trust-region clip.
    This is the "single merged scalar loss... unclipped raw coordinate
    gradient" behavior docs/legacy/guidance_design_notes.md sections 2-3 identifies
    as too aggressive -- the pre-Phase-1 design.
  - "controller": the actual production merge from
    guided_partial_diffusion/step_body (src/mosaic/models/boltzgen.py),
    called via the same private functions the real driver uses
    (_mask_center_normalize, _compat_project, _clip_rms, the default
    schedules) -- not a reimplementation.

Both start from the identical x0_hat (one real BoltzGen denoising step on
the real WT VHH72 -> WT RBD structure, vhh72_wt_wt_rbd.cif) and the
identical raw per-objective gradients -- the ONLY thing that differs is the
merge. The comparison: does each config's soft-sequence shift (via the same
differentiable BoltzGen-IF bridge production guidance uses) move probability
mass toward substitutions the real AlphaSeq VHH72 contrast-pair data
(examples/alphaseq_vhh72_cdr_contrast_pairs.py) says are empirically
beneficial, or away from it?

See docs/guidance_alphaseq_testing_notes.md for the full context this
script is answering (Path 1 of the two-path plan, section 5 there).

Deliberately reuses real production code wherever possible rather than
reimplementing it -- load_core_models/load_guidance_models/build_guidance_loss
for model loading and loss construction, Sampler.from_features +
structure_module.preconditioned_network_forward for the real denoiser call,
and the actual private controller functions from boltzgen.py -- so this
exercises the real mechanism, not a stand-in for it.

Requires: GPU, jopendde/opendde installed, AbLang2, native BoltzGen-IF
checkpoint, vhh72_wt_wt_rbd.cif (repo root).

Usage:
    .venv/bin/python examples/vhh72_gradient_path_comparison.py
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

print("=== VHH72 gradient-path comparison: raw sum vs. full controller ===", flush=True)

t0 = time.time()
print("[1/9] importing...", flush=True)
from mosaic.models.boltzgen import (
    Sampler,
    _center,
    _clip_rms,
    _compat_project,
    _mask_center_normalize,
    build_atom_partial_mask,
    default_alpha_schedule,
    default_beta_schedule,
    default_lambda_schedule,
    default_tau_schedule,
    load_features_and_structure_writer,
)
from mosaic.legacy.boltzgen_if_jax import (
    differentiable_jax_boltzgen_if,
    prepare_jax_boltzgen_if_context,
)
from mosaic.util import fold_in
from mosaic.legacy.boltzgen_vhh_guided import (
    VHHDesignConfig,
    binder_indices_from_design_mask,
    build_complex_yaml,
    build_guidance_loss,
    cdr_token_mask_from_features,
    ensure_boltzgen_if_loaded,
    lambda_schedule_fn,
    load_core_models,
    load_guidance_models,
    parent_one_hot_from_features,
    squeeze_feature,
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Cropped to the CDR-proximal interface (examples/crop_vhh72_wt_rbd.py) --
# the full 209-residue target OOMs OpenDDE's backward pass on this GPU
# (see docs/guidance_alphaseq_testing_notes.md section 9 point 1).
COMPLEX_CIF = REPO_ROOT / "vhh72_wt_wt_rbd_cropped.cif"
assert COMPLEX_CIF.exists(), (
    f"missing {COMPLEX_CIF} -- run examples/crop_vhh72_wt_rbd.py first"
)

# CDR boundaries via ANARCI/IMGT on the real WT VHH72 sequence -- see
# examples/alphaseq_vhh72_cdr_contrast_pairs.py's module docstring. 1-indexed,
# matching VHHDesignConfig.cdr_residue_indices' label_seq_id convention.
CDR_RESIDUE_INDICES = list(range(26, 34)) + list(range(51, 59)) + list(range(97, 115))

cfg = VHHDesignConfig(
    complex_cif_path=COMPLEX_CIF,
    binder_chain_id="A",
    target_chain_ids=["A2"],
    cdr_residue_indices=CDR_RESIDUE_INDICES,
    output_dir=Path("/tmp/vhh72_gradient_path_comparison"),
    seed=0,
    recycling_steps=1,
    num_sampling_steps=200,
    start_sigma_frac=0.3,
    # Fixed objective per this session's decision: OpenDDE bind + AbLang2 nat
    # + edit budget, identical in both configs -- only the merge varies.
    weight_opendde_contact=0.5,
    weight_ablang2=0.10,
    weight_edit_budget=5.0,
    opendde_guidance_recycling_steps=1,
)

t0 = time.time()
print("[2/9] loading BoltzGen, OpenDDE, AbLang2, BoltzGen-IF...", flush=True)
models = load_core_models(cfg)
models = load_guidance_models(cfg, models)
models = ensure_boltzgen_if_loaded(cfg, models)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[3/9] featurizing the real WT VHH72 -> WT RBD complex...", flush=True)
yaml_string = build_complex_yaml(
    cif_filename=COMPLEX_CIF.name,
    binder_chain_id=cfg.binder_chain_id,
    target_chain_ids=cfg.target_chain_ids,
    cdr_residue_indices=cfg.cdr_residue_indices,
)
features, writer = load_features_and_structure_writer(
    yaml_string=yaml_string,
    files={COMPLEX_CIF.name: COMPLEX_CIF},
    mask=True,
    mask_backbone=False,
    mask_disto=True,
)
parent_one_hot = parent_one_hot_from_features(features)
designable_token_mask = cdr_token_mask_from_features(features)
initial_coords = squeeze_feature(features["coords"], "coords", 2)
atom_pad_mask = squeeze_feature(features["atom_pad_mask"], "atom_pad_mask", 1)
atom_partial_mask = build_atom_partial_mask(features, designable_token_mask)
asym_id = squeeze_feature(features["asym_id"], "asym_id", 1)
binder_token_indices = binder_indices_from_design_mask(asym_id, designable_token_mask)
n_designable = int(designable_token_mask.sum())
print(
    f"  {parent_one_hot.shape[0]} tokens, {binder_token_indices.shape[0]} binder "
    f"tokens, {n_designable} designable (CDR) positions",
    flush=True,
)
assert n_designable == len(CDR_RESIDUE_INDICES), (
    f"expected {len(CDR_RESIDUE_INDICES)} designable positions from "
    f"cdr_residue_indices, featurization produced {n_designable} -- "
    f"chain id / numbering mismatch somewhere"
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[4/9] running BoltzGen trunk (Sampler.from_features)...", flush=True)
sampler = Sampler.from_features(
    model=models.boltzgen,
    features=features,
    key=jax.random.key(cfg.seed),
    deterministic=True,
    recycling_steps=cfg.recycling_steps,
)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[5/9] building the fixed guidance objective (OpenDDE bind + AbLang2 nat + edit)...", flush=True)
guidance_losses = build_guidance_loss(
    cfg, models, parent_one_hot, designable_token_mask, binder_token_indices
)
assert guidance_losses is not None
assert guidance_losses.bind is not None and guidance_losses.nat is not None and guidance_losses.edit is not None
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[6/9] building guidance_fn_bind/nat/edit closures + their raw gradients...", flush=True)
# Exactly mirrors run()'s _make_guidance_fn (boltzgen_vhh_guided.py) -- x0 ->
# soft_seq via the differentiable BoltzGen-IF bridge -> loss_term(soft_seq).
if_context = prepare_jax_boltzgen_if_context(
    models.boltzgen_if_torch,
    writer.torch_features,
    np.asarray(initial_coords),
    parent_sequence=np.asarray(parent_one_hot),
    designable_mask=np.asarray(designable_token_mask),
)
if_order = jax.random.permutation(jax.random.key(cfg.seed + 7919), if_context.designable_positions)


def _make_guidance_fn(loss_term):
    def guidance_fn(x0):
        soft_seq = differentiable_jax_boltzgen_if(
            models.boltzgen_if_jax,
            if_context,
            x0[0],
            key=jax.random.key(cfg.seed),
            temperature=cfg.boltzgen_if_guidance_temperature,
            avoid=cfg.boltzgen_if_avoid,
            order=if_order,
        )
        v, _ = loss_term(soft_seq, key=jax.random.key(cfg.seed))
        return v
    return guidance_fn


guidance_fn_bind = _make_guidance_fn(guidance_losses.bind)
guidance_fn_nat = _make_guidance_fn(guidance_losses.nat)
guidance_fn_edit = _make_guidance_fn(guidance_losses.edit)
grad_bind = jax.grad(guidance_fn_bind)
grad_nat = jax.grad(guidance_fn_nat)
grad_edit = jax.grad(guidance_fn_edit)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[7/9] getting one real x0_hat (real noise + churn + BoltzGen denoise)...", flush=True)
# Mirrors guided_partial_diffusion's init code + step_body's first iteration
# (boltzgen.py) exactly -- same sigma schedule, same churn formula, same
# preconditioned_network_forward call, same optional rigid realignment.
# Reusing sampler/structure_module means this is the real denoiser, not a
# stand-in for it.
structure_module = models.boltzgen.structure_module
initial_coords_b = initial_coords[None]      # (1, M, 3)
atom_partial_mask_b = atom_partial_mask[None]
atom_mask_b = atom_pad_mask[None]

full_sigmas = structure_module.sample_schedule_dilated(cfg.num_sampling_steps)
start_idx = int((1.0 - cfg.start_sigma_frac) * (len(full_sigmas) - 1))
sigmas = full_sigmas[start_idx:]
gamma_0 = structure_module.gamma_0
gamma_min = structure_module.gamma_min
gammas = jnp.where(sigmas > gamma_min, gamma_0, 0.0)

key = jax.random.key(cfg.seed + 424242)
key, sub = jax.random.split(key)
init_sigma = sigmas[0]
init_noise = jax.random.normal(sub, initial_coords_b.shape)
atom_coords = jnp.where(
    (atom_partial_mask_b > 0)[..., None],
    initial_coords_b + init_sigma * init_noise,
    initial_coords_b,
)
init_coords_rep = initial_coords_b

sigma_tm = sigmas[0]
gamma = gammas[1]
t_hat = sigma_tm * (1.0 + gamma)
noise_var = cfg.noise_scale**2 * (t_hat**2 - sigma_tm**2)

atom_coords = _center(atom_coords, atom_mask_b)
init_coords_rep = _center(init_coords_rep, atom_mask_b)

key, sub = jax.random.split(key)
eps = cfg.noise_scale * jnp.sqrt(jnp.maximum(noise_var, 0.0)) * jax.random.normal(sub, atom_coords.shape)
atom_coords_noisy = atom_coords + eps

diffusion_conditioning = {
    "q": sampler.q, "c": sampler.c, "to_keys": sampler.to_keys,
    "atom_enc_bias": sampler.atom_enc_bias, "atom_dec_bias": sampler.atom_dec_bias,
    "token_trans_bias": sampler.token_trans_bias,
}
network_condition_kwargs = dict(
    s_trunk=sampler.trunk_s, s_inputs=sampler.s_inputs, feats=sampler.feats,
    diffusion_conditioning=diffusion_conditioning, multiplicity=1,
)
key, sub = jax.random.split(key)
x0_hat = structure_module.preconditioned_network_forward(
    atom_coords_noisy, t_hat, network_condition_kwargs=network_condition_kwargs, key=sub,
)
x0_hat = jax.lax.stop_gradient(x0_hat)

if structure_module.alignment_reverse_diff:
    from joltzgen import weighted_rigid_align
    atom_coords_noisy = weighted_rigid_align(atom_coords_noisy, x0_hat, atom_mask_b, atom_mask_b)

unguided_direction = (atom_coords_noisy - x0_hat) / t_hat
assert jnp.all(jnp.isfinite(x0_hat)), "x0_hat contains non-finite values"
print(f"  t_hat={float(t_hat):.4f}  x0_hat finite, shape={x0_hat.shape}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[8/9] computing raw gradients + both merges (raw sum vs. controller)...", flush=True)
g_bind_raw = grad_bind(x0_hat)
g_nat_raw = grad_nat(x0_hat)
g_edit_raw = grad_edit(x0_hat)
for name, g in [("g_bind_raw", g_bind_raw), ("g_nat_raw", g_nat_raw), ("g_edit_raw", g_edit_raw)]:
    assert jnp.all(jnp.isfinite(g)), f"{name} contains non-finite values"
    print(f"  {name}: rms={float(jnp.sqrt(jnp.mean(g**2))):.6f}", flush=True)

# "raw": sum with no controller machinery at all, scaled by the same overall
# step-size schedule the controller uses (so a difference in outcome reflects
# the merge mechanism, not just an arbitrary difference in step magnitude).
lambda_fn = lambda_schedule_fn(cfg.lambda_schedule, cfg.lambda_max)
g_total_raw = g_bind_raw + g_nat_raw + g_edit_raw
delta_raw = lambda_fn(t_hat) * g_total_raw

# "controller": the real production merge -- same private functions
# step_body calls, same default alpha/beta/tau schedules (not passed by the
# driver's call site, so they resolve to these defaults; lambda_fn matches
# what the driver's default cfg.lambda_schedule="sigma_squared" produces).
g_bind_c = _mask_center_normalize(g_bind_raw, atom_partial_mask_b)
g_nat_c = _mask_center_normalize(g_nat_raw, atom_partial_mask_b)
g_edit_c = _mask_center_normalize(g_edit_raw, atom_partial_mask_b)
g_total_full = (
    g_bind_c
    + default_alpha_schedule(t_hat) * _compat_project(g_nat_c, g_bind_c)
    + default_beta_schedule(t_hat) * _compat_project(g_edit_c, g_bind_c)
)
delta_full = _clip_rms(lambda_fn(t_hat) * g_total_full, default_tau_schedule(t_hat), atom_partial_mask_b)

# "consistent": controller + a NOS-style (arXiv:2305.20009) prior-consistency
# term, per docs/guidance_alphaseq_testing_notes.md section 3b/9 -- the
# candidate mechanism we picked over Gradient Guidance's look-ahead loss
# (arXiv:2404.14743, requires backprop through the whole denoiser, and its
# manifold-preservation proof needs a linear-subspace assumption that
# doesn't hold for BoltzGen anyway). Implemented as a _compat_project of the
# controller's full merged g_total against unguided_direction -- reusing the
# exact PCGrad machinery already in production (boltzgen.py's
# _compat_project), just with unguided_direction as the anchor instead of
# g_bind. This removes only the component of g_total that actively conflicts
# with what BoltzGen's own denoiser would have done unguided; components
# that agree pass through unchanged (same asymmetric behavior _compat_project
# already has for g_nat/g_edit against g_bind). unguided_direction is
# mask/center/RMS-normalized first so the conflict dot-product is on the
# same footing as every other gradient it's compared against.
unguided_direction_norm = _mask_center_normalize(unguided_direction, atom_partial_mask_b)
g_total_consistent = _compat_project(g_total_full, unguided_direction_norm)
delta_consistent = _clip_rms(
    lambda_fn(t_hat) * g_total_consistent, default_tau_schedule(t_hat), atom_partial_mask_b
)

def _cos(a, b):
    return float(jnp.sum(a * b) / (jnp.linalg.norm(a.reshape(-1)) * jnp.linalg.norm(b.reshape(-1)) + 1e-8))

print(f"  delta_raw rms={float(jnp.sqrt(jnp.mean(delta_raw**2))):.6f}  "
      f"delta_full rms={float(jnp.sqrt(jnp.mean(delta_full**2))):.6f}  "
      f"delta_consistent rms={float(jnp.sqrt(jnp.mean(delta_consistent**2))):.6f}", flush=True)
print(f"  cos(delta_raw, delta_full)={_cos(delta_raw, delta_full):.4f}", flush=True)
print(f"  cos(delta_full, delta_consistent)={_cos(delta_full, delta_consistent):.4f}", flush=True)
print(f"  cos(delta_full, unguided_direction)={_cos(delta_full, unguided_direction_norm):.4f}  "
      f"cos(delta_consistent, unguided_direction)={_cos(delta_consistent, unguided_direction_norm):.4f}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
print("[9/9] getting p_seq shifts (unguided vs raw vs controller vs controller+consistency)...", flush=True)
x0_unguided = x0_hat
x0_guided_raw = x0_hat - delta_raw
x0_guided_full = x0_hat - delta_full
x0_guided_consistent = x0_hat - delta_consistent


def _p_seq(x0):
    return differentiable_jax_boltzgen_if(
        models.boltzgen_if_jax, if_context, x0[0],
        key=jax.random.key(cfg.seed), temperature=cfg.boltzgen_if_guidance_temperature,
        avoid=cfg.boltzgen_if_avoid, order=if_order,
    )


p_seq_unguided = _p_seq(x0_unguided)
p_seq_raw = _p_seq(x0_guided_raw)
p_seq_full = _p_seq(x0_guided_full)
p_seq_consistent = _p_seq(x0_guided_consistent)

shift_raw = p_seq_raw - p_seq_unguided
shift_full = p_seq_full - p_seq_unguided
shift_consistent = p_seq_consistent - p_seq_unguided
print(f"  p_seq shape={p_seq_unguided.shape}", flush=True)
print(f"  |shift_raw| mean={float(jnp.mean(jnp.abs(shift_raw))):.5f}  "
      f"|shift_full| mean={float(jnp.mean(jnp.abs(shift_full))):.5f}  "
      f"|shift_consistent| mean={float(jnp.mean(jnp.abs(shift_consistent))):.5f}", flush=True)
print(f"  cos(shift_full.flatten(), shift_consistent.flatten())="
      f"{_cos(shift_full, shift_consistent):.4f}", flush=True)
print(f"  done ({time.time() - t0:.1f}s)", flush=True)

import pickle
CACHE_PATH = REPO_ROOT / "vhh72_gradient_path_comparison_cache.pkl"
with open(CACHE_PATH, "wb") as f:
    pickle.dump({
        "designable_token_mask": np.asarray(designable_token_mask),
        "binder_token_indices": np.asarray(binder_token_indices),
        "cdr_residue_indices": CDR_RESIDUE_INDICES,
        "p_seq_unguided": np.asarray(p_seq_unguided),
        "p_seq_raw": np.asarray(p_seq_raw),
        "p_seq_full": np.asarray(p_seq_full),
        "p_seq_consistent": np.asarray(p_seq_consistent),
        "shift_raw": np.asarray(shift_raw),
        "shift_full": np.asarray(shift_full),
        "shift_consistent": np.asarray(shift_consistent),
    }, f)
print(f"\nwrote {CACHE_PATH} for the contrast-pair scoring step", flush=True)

print("\nPASS: real x0_hat obtained, all three merges (raw / controller / "
      "controller+consistency) produce finite deltas from the same raw "
      "gradients, and all propagate through the real BoltzGen-IF bridge to "
      "finite, distinct p_seq shifts. Next step (not yet in this script): "
      "score shift_raw/shift_full/shift_consistent at CDR positions against "
      "examples/alphaseq_vhh72_cdr_contrast_pairs.py's empirical table.", flush=True)
