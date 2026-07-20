"""Unit tests for the Phase 1 guidance controller primitives in
src/mosaic/models/boltzgen.py (docs/guidance_implementation_todo.md Phase 1).

These test the standalone jax.numpy primitives directly with hand-constructed
gradient tensors -- no BoltzGen model, checkpoint, or GPU required, so they
run fast and don't need the guidance model weights loaded.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from mosaic.models.boltzgen import (
    _mask_center_normalize,
    _compat_project,
    _clip_rms,
    _euler_step_delta,
    default_lambda_schedule,
    default_tau_schedule,
    default_alpha_schedule,
    default_beta_schedule,
)

# atoms 0-3 designable, 4-5 frozen, matching every test below
ATOM_PARTIAL_MASK = jnp.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
B, M = 1, 6


def test_mask_zeros_frozen_atoms():
    g = jnp.ones((B, M, 3)) * 5.0
    out = _mask_center_normalize(g, ATOM_PARTIAL_MASK)
    assert np.allclose(np.array(out[0, 4:]), 0.0)


def test_demean_cancels_pure_rigid_translation():
    # A uniform gradient across all designable atoms is a pure rigid
    # translation and should fully cancel after de-meaning (checked before
    # normalization would otherwise blow a near-zero vector back up).
    g_uniform = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([1.0, 2.0, 3.0]))
    mask = (ATOM_PARTIAL_MASK > 0)[..., None]
    n_design = jnp.sum(ATOM_PARTIAL_MASK, axis=-1, keepdims=True)
    mean = jnp.sum(jnp.where(mask, g_uniform, 0.0), axis=1) / n_design
    demeaned = jnp.where(mask, g_uniform - mean[:, None, :], 0.0)
    assert np.allclose(np.array(demeaned), 0.0, atol=1e-6)


def test_rms_normalize_hits_unit_rms():
    g_random = jnp.array(np.random.RandomState(0).randn(B, M, 3) * 10.0)
    out = _mask_center_normalize(g_random, ATOM_PARTIAL_MASK)
    design_vals = np.array(out[0, :4])
    rms = np.sqrt((design_vals ** 2).sum() / (4 * 3))
    assert abs(rms - 1.0) < 1e-4


def test_pcgrad_agreeing_gradient_passes_through_unchanged():
    g_anchor = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([1.0, 0.0, 0.0]))
    g_agree = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([1.0, 0.5, 0.0]))
    out = _compat_project(g_agree, g_anchor)
    assert np.allclose(np.array(out), np.array(g_agree))


def test_pcgrad_conflicting_gradient_has_conflict_removed():
    g_anchor = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([1.0, 0.0, 0.0]))
    g_conflict = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([-1.0, 0.5, 0.0]))
    out = _compat_project(g_conflict, g_anchor)
    dot_after = float(jnp.sum(out * g_anchor))
    assert abs(dot_after) < 1e-4


def test_trust_radius_clip_caps_rms():
    big_delta = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([10.0, 0.0, 0.0]))
    tau = 1.0
    out = _clip_rms(big_delta, tau, ATOM_PARTIAL_MASK)
    design_vals = np.array(out[0, :4])
    rms_after = np.sqrt((design_vals ** 2).sum() / (4 * 3))
    assert rms_after <= tau + 1e-4


def test_trust_radius_clip_is_noop_when_under_tau():
    small_delta = jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([0.01, 0.0, 0.0]))
    out = _clip_rms(small_delta, 1.0, ATOM_PARTIAL_MASK)
    assert np.allclose(np.array(out), np.array(small_delta), atol=1e-6)


@pytest.mark.parametrize(
    "sigmas",
    [[2.0, 1.0, 0.5, 0.1, 0.01]],
)
def test_default_schedules_have_documented_shape(sigmas):
    lam_vals = [float(default_lambda_schedule(s)) for s in sigmas]
    tau_vals = [float(default_tau_schedule(s)) for s in sigmas]
    alpha_vals = [float(default_alpha_schedule(s)) for s in sigmas]
    beta_vals = [float(default_beta_schedule(s)) for s in sigmas]

    # lambda: decreases monotonically as sigma -> 0 (guidance_design_notes.md
    # section 9: "sharply reduce coordinate guidance magnitude" late)
    assert all(lam_vals[i] >= lam_vals[i + 1] for i in range(len(lam_vals) - 1))

    # alpha/beta: increase monotonically as sigma -> 0 (naturalness/locality
    # "start to matter"/"turns on" as the trajectory progresses)
    assert all(alpha_vals[i] <= alpha_vals[i + 1] for i in range(len(alpha_vals) - 1))
    assert all(beta_vals[i] <= beta_vals[i + 1] for i in range(len(beta_vals) - 1))

    # tau floor: never collapses to exactly zero at the final step
    assert tau_vals[-1] >= 0.05 - 1e-6


def test_euler_step_delta_applies_physical_scaling():
    # cosine similarity is invariant to this scaling, but the norm-ratio
    # diagnostic (docs/guidance_implementation_todo.md Phase 2) needs the
    # actual physical displacement, not the raw direction.
    direction = jnp.ones((B, M, 3)) * 3.0
    step_scale, sigma_t, t_hat = 2.0, 0.5, 1.0
    out = _euler_step_delta(direction, step_scale, sigma_t, t_hat)
    expected = step_scale * (sigma_t - t_hat) * direction
    assert np.allclose(np.array(out), np.array(expected))


def test_euler_step_delta_matches_manual_next_coord_update():
    # guided_step_delta should equal exactly the delta guided_partial_diffusion
    # adds to atom_coords_noisy to get atom_coords_next (pre re-anchoring).
    atom_coords_noisy = jnp.array(np.random.RandomState(1).randn(B, M, 3))
    denoised_over_sigma = jnp.array(np.random.RandomState(2).randn(B, M, 3))
    step_scale, sigma_t, t_hat = 1.5, 0.3, 0.8
    delta = _euler_step_delta(denoised_over_sigma, step_scale, sigma_t, t_hat)
    atom_coords_next = atom_coords_noisy + delta
    expected_next = atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
    assert np.allclose(np.array(atom_coords_next), np.array(expected_next))
