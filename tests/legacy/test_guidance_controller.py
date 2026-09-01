"""Unit tests for the Phase 1 guidance controller primitives in
src/mosaic/models/boltzgen.py (docs/guidance_implementation_todo.md Phase 1).

These test the standalone jax.numpy primitives directly with hand-constructed
gradient tensors -- no BoltzGen model, checkpoint, or GPU required, so they
run fast and don't need the guidance model weights loaded.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.models.boltzgen import (
    _mask_center_normalize,
    _compat_project,
    _clip_rms,
    _euler_step_delta,
    _merge_aux_gradients,
    _nos_iterative_merge,
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


# --- section 12a: real iterative NOS-style consistency mechanism -----------
# See docs/guidance_alphaseq_testing_notes.md section 12a for why a one-shot
# squared-distance penalty is mathematically a no-op (or, in a two-stage
# variant, a provable rescale of the auxiliary gradient) and why genuine
# iteration is required instead. These tests use a simple quadratic
# single-objective loss (gradient = 2*(x - target), which genuinely changes
# magnitude/direction as x moves) so re-evaluation across inner steps is
# meaningfully different from repeating one fixed gradient.

def _quadratic_bind_loss(target):
    def loss_bind(x):
        return jnp.sum(((x - target) ** 2) * (ATOM_PARTIAL_MASK > 0)[..., None])
    return loss_bind


# Per-atom-varying target (NOT a uniform rigid translation) -- a uniform
# target would make grad_bind's raw gradient identical across all 4
# designable atoms, which _mask_center_normalize's de-mean step correctly
# zeroes out entirely (see test_demean_cancels_pure_rigid_translation
# above), leaving nothing for these tests to actually exercise.
_ZERO_TARGET = jnp.zeros((B, M, 3)).at[:, :4, :].set(
    jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
)
_ALPHA0 = lambda t: 0.0
_BETA0 = lambda t: 0.0


def test_merge_aux_gradients_matches_manual_pcgrad_merge():
    grad_bind = jax.grad(_quadratic_bind_loss(_ZERO_TARGET))
    grad_nat = jax.grad(_quadratic_bind_loss(jnp.zeros((B, M, 3)).at[:, :4, :].set(jnp.array([0.0, 2.0, 0.0]))))
    x0 = jnp.zeros((B, M, 3))
    alpha_fn, beta_fn = (lambda t: 0.5), (lambda t: 0.0)

    g_total, g_bind, g_nat, g_edit = _merge_aux_gradients(
        x0, grad_bind, grad_nat, None, ATOM_PARTIAL_MASK, alpha_fn, beta_fn, 1.0
    )
    g_bind_manual = _mask_center_normalize(grad_bind(x0), ATOM_PARTIAL_MASK)
    g_nat_manual = _mask_center_normalize(grad_nat(x0), ATOM_PARTIAL_MASK)
    g_total_manual = g_bind_manual + 0.5 * _compat_project(g_nat_manual, g_bind_manual)

    assert np.allclose(np.array(g_total), np.array(g_total_manual))
    assert np.allclose(np.array(g_bind), np.array(g_bind_manual))
    assert np.allclose(np.array(g_edit), 0.0)


_TAU_GENEROUS = lambda t: 10.0  # loose enough not to bind in tests that aren't testing the clip itself


def test_nos_iterative_merge_first_step_matches_one_shot():
    # At inner step 0, x0_i == anchor exactly, so the consistency gradient
    # (2*(x0_i - anchor)) is exactly zero regardless of lambda_kl -- one
    # inner step must reduce exactly to one one-shot merge step (as long as
    # the per-step clip doesn't bind, hence the generous tau here).
    grad_bind = jax.grad(_quadratic_bind_loss(_ZERO_TARGET))
    x0_hat = jnp.zeros((B, M, 3))
    step_size = 0.1

    x0_final, g_bind_last, _, _ = _nos_iterative_merge(
        x0_hat, grad_bind, None, None, ATOM_PARTIAL_MASK, _ALPHA0, _BETA0, _TAU_GENEROUS, 1.0,
        n_inner_steps=1,
        inner_step_fn=lambda t: step_size,
        lambda_kl_fn=lambda t: 5.0,  # irrelevant at step 0
        noise_fn=lambda t: 0.0,
        key=jax.random.key(0),
    )
    g_total_one_shot, g_bind_one_shot, _, _ = _merge_aux_gradients(
        x0_hat, grad_bind, None, None, ATOM_PARTIAL_MASK, _ALPHA0, _BETA0, 1.0
    )
    expected = x0_hat - step_size * g_total_one_shot
    assert np.allclose(np.array(x0_final), np.array(expected), atol=1e-5)
    assert np.allclose(np.array(g_bind_last), np.array(g_bind_one_shot), atol=1e-5)


def test_nos_iterative_merge_pure_reevaluation_matches_naive_repeat_for_one_objective():
    # Documented, verified nuance (section 12a): for a SINGLE objective, once
    # _mask_center_normalize rescales every re-evaluated gradient to unit
    # RMS, the direction along a straight-line path to the loss's minimum is
    # invariant -- so K re-evaluated steps with lambda_kl=0 coincide with K
    # copies of the one-shot step. This is NOT a bug; it's why the
    # consistency term (not bare re-evaluation) is the real source of
    # iteration-dependent behavior in this single-objective case -- PCGrad's
    # asymmetric conflict projection is the other source, when >1 objective
    # is active (production always has at least bind, usually more).
    grad_bind = jax.grad(_quadratic_bind_loss(_ZERO_TARGET))
    x0_hat = jnp.zeros((B, M, 3))
    step_size, n_steps = 0.1, 5

    x0_final, *_ = _nos_iterative_merge(
        x0_hat, grad_bind, None, None, ATOM_PARTIAL_MASK, _ALPHA0, _BETA0, _TAU_GENEROUS, 1.0,
        n_inner_steps=n_steps,
        inner_step_fn=lambda t: step_size,
        lambda_kl_fn=lambda t: 0.0,
        noise_fn=lambda t: 0.0,
        key=jax.random.key(0),
    )
    g_total_one_shot, *_ = _merge_aux_gradients(
        x0_hat, grad_bind, None, None, ATOM_PARTIAL_MASK, _ALPHA0, _BETA0, 1.0
    )
    naive_repeated = x0_hat - (n_steps * step_size) * g_total_one_shot
    assert np.allclose(np.array(x0_final), np.array(naive_repeated), atol=1e-3)


def test_nos_consistency_term_pulls_back_toward_anchor():
    # Stable regime: the consistency-only recursion is
    # (x_i+1 - anchor) = (1 - 2*step*lambda_kl) * (x_i - anchor), which
    # diverges when 2*step*lambda_kl > 1 (found directly while testing this
    # mechanism -- step=0.1, lambda_kl=50 blew a small perturbation up to a
    # distance of ~2000 over 5 steps instead of shrinking it). step=0.1,
    # lambda_kl=2.0 keeps 2*step*lambda_kl=0.4, safely stable.
    grad_bind = jax.grad(_quadratic_bind_loss(_ZERO_TARGET))
    x0_hat = jnp.zeros((B, M, 3))
    common = dict(
        grad_bind=grad_bind, grad_nat=None, grad_edit=None,
        atom_partial_mask=ATOM_PARTIAL_MASK, alpha_fn=_ALPHA0, beta_fn=_BETA0, tau_fn=_TAU_GENEROUS, t_hat=1.0,
        n_inner_steps=5, inner_step_fn=lambda t: 0.1, noise_fn=lambda t: 0.0,
    )
    x0_far, *_ = _nos_iterative_merge(x0_hat, lambda_kl_fn=lambda t: 0.0, key=jax.random.key(0), **common)
    x0_pulled, *_ = _nos_iterative_merge(x0_hat, lambda_kl_fn=lambda t: 2.0, key=jax.random.key(0), **common)

    dist_far = float(jnp.sqrt(jnp.sum((x0_far - x0_hat) ** 2)))
    dist_pulled = float(jnp.sqrt(jnp.sum((x0_pulled - x0_hat) ** 2)))
    assert dist_pulled < dist_far


def test_nos_inner_step_clip_bounds_unstable_hyperparameters():
    # The same unstable (step=0.1, lambda_kl=50) combination that exploded
    # to a distance of ~2000 over 5 steps before the per-inner-step clip was
    # added must now stay bounded, thanks to _clip_rms capping each raw
    # inner step at tau_fn(t_hat)/n_inner_steps.
    grad_bind = jax.grad(_quadratic_bind_loss(_ZERO_TARGET))
    x0_hat = jnp.zeros((B, M, 3))
    tau_fn = lambda t: 1.0
    n_steps = 5

    x0_final, *_ = _nos_iterative_merge(
        x0_hat, grad_bind, None, None, ATOM_PARTIAL_MASK, _ALPHA0, _BETA0, tau_fn, 1.0,
        n_inner_steps=n_steps,
        inner_step_fn=lambda t: 0.1,
        lambda_kl_fn=lambda t: 50.0,
        noise_fn=lambda t: 0.0,
        key=jax.random.key(0),
    )
    dist = float(jnp.sqrt(jnp.sum((x0_final - x0_hat) ** 2)))
    # worst case: n_steps capped displacements of RMS tau_fn(t_hat)/n_steps each,
    # summed with maximally-unfavorable directions -- generous bound, just
    # ruling out the ~2000-scale blowup seen without the clip.
    assert dist <= tau_fn(1.0) * 2
