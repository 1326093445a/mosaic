"""Tests for the real look-ahead guidance mechanism (docs/
guidance_alphaseq_testing_notes.md section 12b) -- both the standalone
closure-builder in src/mosaic/models/guidance_lookahead.py, and its wiring
into step_body/guided_partial_diffusion in src/mosaic/models/boltzgen.py.

The wiring tests use a small, fully synthetic fake `structure_module` /
`Sampler` (a deterministic nonlinear function standing in for the real
BoltzGen denoiser) rather than the real model -- no GPU/checkpoint needed,
matching the style of tests/test_guidance_controller.py, but exercising the
real guided_partial_diffusion/step_body code path end to end (not just the
isolated primitive), since the sign/scale derivation for this mechanism is
the most subtle part of section 12 and a wiring mistake would not be caught
by testing build_lookahead_grad_fn in isolation.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.models.boltzgen import Sampler, guided_partial_diffusion
from mosaic.models.guidance_lookahead import build_lookahead_grad_fn

B, M = 1, 6
ATOM_PARTIAL_MASK = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # unbatched, matches guided_partial_diffusion's own doc
ATOM_MASK = jnp.ones((M,))


def _fake_denoiser(coords):
    # Deterministic, genuinely nonlinear (tanh) -- so its Jacobian is
    # nontrivial and look-ahead's through-denoiser gradient can differ from
    # a stop-gradient gradient evaluated at the same output.
    return jnp.tanh(coords * 0.3) * 1.5


def _quadratic_loss(target):
    def loss(x0):
        return jnp.sum(((x0 - target) ** 2) * (ATOM_PARTIAL_MASK > 0)[..., None])
    return loss


_TARGET = jnp.zeros((M, 3)).at[:4, :].set(
    jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
)


def test_build_lookahead_grad_fn_matches_direct_composition():
    loss = _quadratic_loss(_TARGET)
    grad_la = build_lookahead_grad_fn(_fake_denoiser, loss)
    x = jnp.array(np.random.RandomState(0).randn(M, 3))

    expected = jax.grad(lambda coords: loss(_fake_denoiser(coords)))(x)
    assert np.allclose(np.array(grad_la(x)), np.array(expected))


def test_build_lookahead_grad_fn_differs_from_stopgrad_gradient():
    # The whole point of look-ahead: differentiating through the denoiser's
    # Jacobian must give a genuinely different gradient than the existing
    # one-shot/NOS paths' d(loss)/d(x0_hat) computed with x0_hat frozen.
    loss = _quadratic_loss(_TARGET)
    x = jnp.array(np.random.RandomState(1).randn(M, 3))
    x0_hat = _fake_denoiser(x)

    grad_lookahead = build_lookahead_grad_fn(_fake_denoiser, loss)(x)
    grad_stopgrad = jax.grad(loss)(x0_hat)  # what the one-shot/NOS paths compute
    assert not np.allclose(np.array(grad_lookahead), np.array(grad_stopgrad), atol=1e-4)


def _fake_sampler_and_structure_module():
    dummy = jnp.zeros((1,))

    class _FakeStructureModule:
        gamma_0 = 0.0
        gamma_min = 0.0
        alignment_reverse_diff = False

        def sample_schedule_dilated(self, num_sampling_steps):
            return jnp.linspace(1.0, 0.05, num_sampling_steps + 1)

        def preconditioned_network_forward(self, atom_coords_noisy, t_hat, *, network_condition_kwargs, key):
            return _fake_denoiser(atom_coords_noisy)

    sampler = Sampler(
        trunk_s=dummy, s_inputs=dummy, feats={}, q=dummy, c=dummy,
        to_keys=dummy, atom_enc_bias=dummy, atom_dec_bias=dummy, token_trans_bias=dummy,
    )
    return sampler, _FakeStructureModule()


def test_lookahead_disabled_matches_prior_one_shot_behavior():
    # guidance_lookahead=False (default) must be byte-identical to the
    # pre-existing one-shot path -- regression safety for the refactor.
    sampler, structure_module = _fake_sampler_and_structure_module()
    initial_coords = jnp.array(np.random.RandomState(2).randn(M, 3))
    common = dict(
        sampler=sampler, structure_module=structure_module, initial_coords=initial_coords,
        atom_partial_mask=ATOM_PARTIAL_MASK, atom_mask=ATOM_MASK,
        num_sampling_steps=4, start_sigma_frac=1.0, step_scale=1.0, noise_scale=0.0,
        guidance_fn_bind=_quadratic_loss(_TARGET),
    )
    out_default = guided_partial_diffusion(key=jax.random.key(0), **common)
    out_explicit_false = guided_partial_diffusion(key=jax.random.key(0), guidance_lookahead=False, **common)
    assert np.allclose(np.array(out_default), np.array(out_explicit_false))


def test_lookahead_enabled_runs_and_changes_result():
    sampler, structure_module = _fake_sampler_and_structure_module()
    initial_coords = jnp.array(np.random.RandomState(3).randn(M, 3))
    common = dict(
        sampler=sampler, structure_module=structure_module, initial_coords=initial_coords,
        atom_partial_mask=ATOM_PARTIAL_MASK, atom_mask=ATOM_MASK,
        num_sampling_steps=4, start_sigma_frac=1.0, step_scale=1.0, noise_scale=0.0,
        guidance_fn_bind=_quadratic_loss(_TARGET),
        guidance_lambda_fn=lambda t: 0.5,
    )
    out_unguided = guided_partial_diffusion(key=jax.random.key(0), guidance_fn_bind=None, **{
        k: v for k, v in common.items() if k != "guidance_fn_bind"
    })
    out_lookahead = guided_partial_diffusion(key=jax.random.key(0), guidance_lookahead=True, **common)
    assert not np.allclose(np.array(out_unguided), np.array(out_lookahead), atol=1e-4)


def test_lookahead_actually_reduces_the_guidance_loss():
    # The property "runs and changes the result" (previous test) does not
    # catch a wrong-sign implementation -- an earlier version of this
    # mechanism changed the result, just in the direction that made the toy
    # loss WORSE (caught by review: guidance_loss(x0_hat)=9.12 vs 12.67 after
    # one guided step, the sign-flipped bug; fixed to 5.17, a real decrease).
    # This test directly checks the one property that actually matters: does
    # enabling look-ahead guidance make the final structure score BETTER on
    # the same loss it was guided by, relative to no guidance at all.
    sampler, structure_module = _fake_sampler_and_structure_module()
    initial_coords = jnp.array(np.random.RandomState(5).randn(M, 3))
    loss = _quadratic_loss(_TARGET)
    common = dict(
        sampler=sampler, structure_module=structure_module, initial_coords=initial_coords,
        atom_partial_mask=ATOM_PARTIAL_MASK, atom_mask=ATOM_MASK,
        num_sampling_steps=4, start_sigma_frac=1.0, step_scale=1.0, noise_scale=0.0,
        guidance_fn_bind=loss,
        guidance_lambda_fn=lambda t: 0.5,
    )
    out_unguided = guided_partial_diffusion(key=jax.random.key(1), guidance_fn_bind=None, **{
        k: v for k, v in common.items() if k != "guidance_fn_bind"
    })
    out_lookahead = guided_partial_diffusion(key=jax.random.key(1), guidance_lookahead=True, **common)

    loss_unguided = float(loss(out_unguided))
    loss_lookahead = float(loss(out_lookahead))
    assert loss_lookahead < loss_unguided, (
        f"look-ahead should reduce the guidance loss it's guiding by "
        f"(unguided={loss_unguided:.4f}, lookahead={loss_lookahead:.4f})"
    )


def test_lookahead_and_nos_are_mutually_exclusive():
    sampler, structure_module = _fake_sampler_and_structure_module()
    initial_coords = jnp.array(np.random.RandomState(4).randn(M, 3))
    with pytest.raises(ValueError, match="mutually exclusive"):
        guided_partial_diffusion(
            sampler=sampler, structure_module=structure_module, initial_coords=initial_coords,
            atom_partial_mask=ATOM_PARTIAL_MASK, atom_mask=ATOM_MASK,
            num_sampling_steps=4, start_sigma_frac=1.0, step_scale=1.0, noise_scale=0.0,
            guidance_fn_bind=_quadratic_loss(_TARGET),
            guidance_nos_inner_steps=2,
            guidance_lookahead=True,
            key=jax.random.key(0),
        )
