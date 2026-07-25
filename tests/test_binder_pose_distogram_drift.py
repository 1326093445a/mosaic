"""Tests for BinderPoseDistogramDrift (src/mosaic/losses/structure_prediction.py)
-- the cheap, coordinate-free pose-anchor loss for use inside a hallucination
search loop. Pure JAX/numpy, no model/GPU needed.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.losses.structure_prediction import BinderPoseDistogramDrift, StructureModelOutput

N_BINDER, N_TARGET = 3, 2
N = N_BINDER + N_TARGET
N_BINS = 128
BINS = jnp.linspace(2.0, 30.0, N_BINS)  # fine bins so nearest-bin quantization error is negligible


def _make_output_with_expected_distances(target_binder_target_dist):
    """Construct distogram_logits so that softmax(logits) puts ~all mass on
    the single bin closest to each entry of target_binder_target_dist
    (shape [N_BINDER, N_TARGET]) -- makes the expected distance exactly
    controllable in a test, not just approximately."""
    logits = jnp.full((N, N, N_BINS), -1e4)
    for i in range(N_BINDER):
        for j in range(N_TARGET):
            bin_idx = int(jnp.argmin(jnp.abs(BINS - target_binder_target_dist[i, j])))
            logits = logits.at[i, N_BINDER + j, bin_idx].set(1e4)
    return StructureModelOutput(
        distogram_logits=logits,
        distogram_bins=BINS,
        plddt=jnp.zeros((N,)),
        pae=jnp.zeros((N, N)),
        pae_logits=jnp.zeros((N, N, 4)),
        pae_bins=jnp.zeros((4,)),
        structure_coordinates=jnp.zeros((N, 3)),
        backbone_coordinates=jnp.zeros((N, 4, 3)),
        full_sequence=jnp.zeros((N, 20)),
        asym_id=jnp.concatenate([jnp.zeros(N_BINDER), jnp.ones(N_TARGET)]),
        residue_idx=jnp.arange(N),
        atom37_coords=jnp.zeros((N, 37, 3)),
        atom37_mask=jnp.zeros((N, 37)),
    )


REFERENCE = jnp.array(np.random.RandomState(0).uniform(5.0, 20.0, (N_BINDER, N_TARGET)))
SEQUENCE = jnp.zeros((N_BINDER, 20))


def test_zero_when_predicted_matches_reference_exactly():
    loss_fn = BinderPoseDistogramDrift(REFERENCE, tolerance=2.0)
    output = _make_output_with_expected_distances(REFERENCE)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    assert float(aux["binder_target_distogram_drift"]) == pytest.approx(0.0, abs=1e-1)  # bin quantization
    assert float(violation) == pytest.approx(0.0, abs=1e-1)


def test_within_tolerance_is_exactly_zero():
    perturbed = REFERENCE + 0.5  # small, uniform shift, within tolerance
    loss_fn = BinderPoseDistogramDrift(REFERENCE, tolerance=2.0)
    output = _make_output_with_expected_distances(perturbed)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    assert float(aux["binder_target_distogram_drift"]) > 0.0
    assert float(violation) == pytest.approx(0.0, abs=1e-1)


def test_beyond_tolerance_produces_positive_violation():
    perturbed = REFERENCE + 8.0  # large, uniform shift, well beyond tolerance
    tol = 2.0
    loss_fn = BinderPoseDistogramDrift(REFERENCE, tolerance=tol)
    output = _make_output_with_expected_distances(perturbed)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    drift = float(aux["binder_target_distogram_drift"])
    assert drift > tol
    assert float(violation) == pytest.approx(drift - tol, abs=1e-1)


def test_gradient_is_zero_within_tolerance_and_nonzero_beyond_it():
    loss_fn = BinderPoseDistogramDrift(REFERENCE, tolerance=2.0)

    def loss_of_shift(shift):
        output = _make_output_with_expected_distances(REFERENCE + shift)
        v, _ = loss_fn(SEQUENCE, output, key=None)
        return v

    # Use real (non-saturated) softmax logits for the gradient check --
    # the extreme-logit trick above is only for exact-value tests.
    def loss_of_shift_smooth(shift, scale=8.0):
        target = REFERENCE + shift
        idx = jnp.arange(N_BINS)
        logits = jnp.full((N, N, N_BINS), 0.0)
        for i in range(N_BINDER):
            for j in range(N_TARGET):
                dist_to_target = jnp.abs(BINS - target[i, j])
                logits = logits.at[i, N_BINDER + j, :].set(-scale * dist_to_target)
        struct_output = StructureModelOutput(
            distogram_logits=logits,
            distogram_bins=BINS,
            plddt=jnp.zeros((N,)), pae=jnp.zeros((N, N)),
            pae_logits=jnp.zeros((N, N, 4)), pae_bins=jnp.zeros((4,)),
            structure_coordinates=jnp.zeros((N, 3)), backbone_coordinates=jnp.zeros((N, 4, 3)),
            full_sequence=jnp.zeros((N, 20)),
            asym_id=jnp.concatenate([jnp.zeros(N_BINDER), jnp.ones(N_TARGET)]),
            residue_idx=jnp.arange(N), atom37_coords=jnp.zeros((N, 37, 3)), atom37_mask=jnp.zeros((N, 37)),
        )
        v, _ = loss_fn(SEQUENCE, struct_output, key=None)
        return v

    grad_small = jax.grad(loss_of_shift_smooth)(0.1)
    grad_large = jax.grad(loss_of_shift_smooth)(8.0)
    assert float(grad_small) == pytest.approx(0.0, abs=1e-3)
    assert float(grad_large) > 0.0
