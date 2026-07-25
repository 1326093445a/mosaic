"""Tests for BinderPoseRMSD (src/mosaic/losses/structure_prediction.py),
the target-aligned pose-anchoring hinge loss for hallucination-style
design. Pure JAX/numpy, no model/GPU needed -- synthetic
StructureModelOutput instances with only backbone_coordinates populated
meaningfully (everything else is an unused placeholder for this loss).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.losses.structure_prediction import BinderPoseRMSD, StructureModelOutput

N_BINDER, N_TARGET = 4, 3
N = N_BINDER + N_TARGET


def _make_output(backbone_ca_and_rest):
    """backbone_ca_and_rest: (N, 3) Calpha coords -- everything else in
    StructureModelOutput is an unused placeholder for this loss."""
    backbone_coordinates = jnp.zeros((N, 4, 3)).at[:, 1, :].set(backbone_ca_and_rest)
    return StructureModelOutput(
        distogram_logits=jnp.zeros((N, N, 4)),
        distogram_bins=jnp.zeros((4,)),
        plddt=jnp.zeros((N,)),
        pae=jnp.zeros((N, N)),
        pae_logits=jnp.zeros((N, N, 4)),
        pae_bins=jnp.zeros((4,)),
        structure_coordinates=jnp.zeros((N, 3)),
        backbone_coordinates=backbone_coordinates,
        full_sequence=jnp.zeros((N, 20)),
        asym_id=jnp.concatenate([jnp.zeros(N_BINDER), jnp.ones(N_TARGET)]),
        residue_idx=jnp.arange(N),
        atom37_coords=jnp.zeros((N, 37, 3)),
        atom37_mask=jnp.zeros((N, 37)),
    )


def _random_rigid(key, coords):
    """Apply a random SO(3) rotation + translation to a (M,3) point set."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (3, 3))
    Q, _ = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.linalg.det(Q))  # ensure proper rotation, det=+1
    t = jax.random.normal(k2, (3,)) * 5.0
    return coords @ Q + t


REFERENCE = jnp.array(np.random.RandomState(0).randn(N, 3) * 3.0)
REFERENCE_BINDER, REFERENCE_TARGET = REFERENCE[:N_BINDER], REFERENCE[N_BINDER:]
SEQUENCE = jnp.zeros((N_BINDER, 20))  # only sequence.shape[0] (binder_len) is used


def test_zero_when_pose_exactly_matches_reference():
    loss_fn = BinderPoseRMSD(REFERENCE_BINDER, REFERENCE_TARGET, rmsd_tolerance=2.0)
    output = _make_output(REFERENCE)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    assert float(violation) == pytest.approx(0.0, abs=1e-4)
    assert float(aux["binder_pose_rmsd"]) == pytest.approx(0.0, abs=1e-4)


def test_invariant_to_rigid_transform_of_whole_complex():
    # The key correctness property: if the model just outputs the entire
    # predicted complex in some arbitrary frame (a real, expected behavior,
    # not a bug), target-alignment must cancel it out completely -- this is
    # exactly why alignment happens on the target, not skipped, and not
    # done over the whole complex.
    transformed = _random_rigid(jax.random.key(0), REFERENCE)
    loss_fn = BinderPoseRMSD(REFERENCE_BINDER, REFERENCE_TARGET, rmsd_tolerance=2.0)
    output = _make_output(transformed)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    assert float(violation) == pytest.approx(0.0, abs=1e-3)
    assert float(aux["binder_pose_rmsd"]) == pytest.approx(0.0, abs=1e-3)


def test_within_tolerance_is_exactly_zero():
    # Perturb only the binder (post target-alignment) by less than tolerance.
    small_shift = REFERENCE_BINDER + jnp.ones((N_BINDER, 3)) * 0.5  # small, uniform nudge
    coords = jnp.concatenate([small_shift, REFERENCE_TARGET], axis=0)
    loss_fn = BinderPoseRMSD(REFERENCE_BINDER, REFERENCE_TARGET, rmsd_tolerance=2.0)
    output = _make_output(coords)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    assert float(aux["binder_pose_rmsd"]) > 0.0  # real drift happened
    assert float(violation) == pytest.approx(0.0, abs=1e-4)  # but hinge stays at 0


def test_beyond_tolerance_grows_linearly_and_matches_hinge_exactly():
    big_shift = REFERENCE_BINDER + jnp.ones((N_BINDER, 3)) * 5.0
    coords = jnp.concatenate([big_shift, REFERENCE_TARGET], axis=0)
    tol = 2.0
    loss_fn = BinderPoseRMSD(REFERENCE_BINDER, REFERENCE_TARGET, rmsd_tolerance=tol)
    output = _make_output(coords)
    violation, aux = loss_fn(SEQUENCE, output, key=None)
    rmsd = float(aux["binder_pose_rmsd"])
    assert rmsd > tol
    assert float(violation) == pytest.approx(rmsd - tol, abs=1e-4)


def test_gradient_is_zero_within_tolerance_and_nonzero_beyond_it():
    loss_fn = BinderPoseRMSD(REFERENCE_BINDER, REFERENCE_TARGET, rmsd_tolerance=2.0)

    def loss_of_binder_shift(shift_scale):
        shifted = REFERENCE_BINDER + jnp.ones((N_BINDER, 3)) * shift_scale
        coords = jnp.concatenate([shifted, REFERENCE_TARGET], axis=0)
        output = _make_output(coords)
        v, _ = loss_fn(SEQUENCE, output, key=None)
        return v

    grad_small = jax.grad(loss_of_binder_shift)(0.1)  # well within 2.0 tolerance
    grad_large = jax.grad(loss_of_binder_shift)(5.0)  # well beyond tolerance
    assert float(grad_small) == pytest.approx(0.0, abs=1e-5)
    assert float(grad_large) > 0.0
