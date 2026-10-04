"""Geometry and alignment-derivative checks for the pose RMSD objective.

These close the gaps left by `test_binder_pose_rmsd.py`, which already covers
whole-scene rigid invariance, relative translation, and the tolerance hinge.
What was untested: whether orientation and internal deformation are separated,
and whether a derivative failure in the Kabsch alignment is distinguishable
from a valid gradient.

Toy arrays only: no checkpoint, no model, no biological input. CPU is enough.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.losses.structure_prediction import BinderPoseRMSD, StructureModelOutput
from mosaic.util import calculate_rmsd, kabsch_conditioning

N_BINDER, N_TARGET = 4, 4
N = N_BINDER + N_TARGET

# A regular tetrahedron has isotropic centered covariance, so the Kabsch
# covariance has three equal singular values. The composite rotation map stays
# differentiable there -- the finite differences below return a nonzero
# derivative -- but the individual U and Vt factors it is built from do not
# have well-defined derivatives when singular values coincide, so an
# implementation that differentiates through them separately cannot compute it.
ISOTROPIC_TARGET = (
    jnp.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    * 3.0
)
GENERIC_TARGET = jnp.array(
    [[0.0, 0.0, 0.0], [4.0, 0.5, 0.0], [0.3, 5.0, 1.0], [1.0, 2.0, 6.0]]
)
BINDER = jnp.array(
    [[0.0, 0.0, 8.0], [1.5, 0.0, 8.0], [0.0, 1.5, 8.0], [1.0, 1.0, 9.0]]
)
SEQUENCE = jnp.zeros((N_BINDER, 20))  # only binder_len is read


def _make_output(ca):
    backbone_coordinates = jnp.zeros((N, 4, 3)).at[:, 1, :].set(ca)
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


def _rotation_z(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(predicted_binder, predicted_target, reference_target, tolerance=0.0):
    loss_fn = BinderPoseRMSD(BINDER, reference_target, rmsd_tolerance=tolerance)
    coords = jnp.concatenate([predicted_binder, predicted_target], axis=0)
    _, aux = loss_fn(SEQUENCE, _make_output(coords), key=None)
    return aux


def test_binder_rotation_changes_pose_without_changing_internal_shape():
    """Orientation must register as pose error while shape error stays zero."""
    centroid = BINDER.mean(axis=0)
    rotated = (BINDER - centroid) @ _rotation_z(0.6) + centroid

    aux = _pose(rotated, GENERIC_TARGET, GENERIC_TARGET)
    pose_rmsd = float(aux["binder_pose_rmsd"])
    internal_rmsd = float(calculate_rmsd(rotated, BINDER))

    # Compare against the displacement the rotation actually produces rather
    # than an arbitrary threshold: the target is unmoved, so target alignment
    # is the identity and the pose number must equal the raw displacement.
    expected = float(jnp.sqrt(jnp.mean(jnp.sum((rotated - BINDER) ** 2, axis=-1))))
    assert expected > 0.1, "the probe rotation must actually move the binder"
    assert pose_rmsd == pytest.approx(expected, rel=1e-4), (
        "a rotated binder must not look correctly placed"
    )
    assert internal_rmsd == pytest.approx(0.0, abs=1e-4), (
        "rotation preserves internal shape, so the independently aligned "
        "RMSD must stay zero -- this is the quantity that would hide the error"
    )
    assert float(aux["pose_target_fit_rmsd"]) == pytest.approx(0.0, abs=1e-4)


def test_deformation_at_fixed_placement_is_reported_as_shape_error():
    """Deforming the binder about its own centroid leaves placement intact."""
    centroid = BINDER.mean(axis=0)
    deformed = centroid + (BINDER - centroid) * jnp.array([1.8, 1.8, 1.8])
    assert deformed.mean(axis=0) == pytest.approx(centroid, abs=1e-5)

    aux = _pose(deformed, GENERIC_TARGET, GENERIC_TARGET)

    assert float(calculate_rmsd(deformed, BINDER)) > 0.3, "shape changed"
    assert float(aux["binder_pose_rmsd"]) > 0.3, (
        "coordinate RMSD mixes shape into the pose number; this records that "
        "deliberately, so a pose change is never read as placement alone"
    )


def _target_rotation_derivative(reference_target, angle):
    """d(pose RMSD)/d(angle) for a rotation of the *predicted target*.

    The rotation enters only through Kabsch's R and t, so this exercises the
    SVD pullback rather than the direct binder path.
    """

    def f(a):
        predicted_target = reference_target @ _rotation_z(a)
        shifted_binder = BINDER + jnp.array([3.0, 0.0, 0.0])
        coords = jnp.concatenate([shifted_binder, predicted_target], axis=0)
        loss_fn = BinderPoseRMSD(BINDER, reference_target, rmsd_tolerance=0.0)
        value, _ = loss_fn(SEQUENCE, _make_output(coords), key=None)
        return value

    # JAX defaults to float32 here, where a 1e-6 step is below the resolution
    # of f: the difference quantizes to a few ULPs and the quotient becomes a
    # power of two times 1e6 rather than a derivative. 1e-3 keeps truncation
    # error near 1e-6 while staying well above float32 round-off.
    h = 1e-3
    autodiff = float(jax.grad(f)(angle))
    central = float((f(angle + h) - f(angle - h)) / (2 * h))
    return autodiff, central


@pytest.mark.parametrize("angle", [0.0, 0.3])
def test_well_conditioned_alignment_gradient_matches_finite_differences(angle):
    autodiff, central = _target_rotation_derivative(GENERIC_TARGET, angle)
    assert abs(central) > 1e-3, "the probe must have a nonzero true derivative"
    assert autodiff == pytest.approx(central, rel=2e-2)
    assert bool(kabsch_conditioning(GENERIC_TARGET, GENERIC_TARGET)["well_conditioned"])


DEGENERATE_ANGLES = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
BROKEN_KINDS = ("zeroed", "nonfinite", "wrong")


def _classify_derivative(autodiff, central):
    """How the autodiff value relates to the true derivative at this point."""
    if not np.isfinite(autodiff):
        return "nonfinite"
    if abs(autodiff) < abs(central) / 1e4:
        return "zeroed"
    if abs(autodiff - central) <= 0.15 * abs(central):
        return "approximate"
    return "wrong"


def _degenerate_classifications():
    observed = {}
    for angle in DEGENERATE_ANGLES:
        autodiff, central = _target_rotation_derivative(ISOTROPIC_TARGET, angle)
        assert abs(central) > 1e-3, (
            f"the probe at {angle} rad must have a nonzero true derivative; "
            f"got {central}"
        )
        observed[angle] = _classify_derivative(autodiff, central)
    return observed


def test_degenerate_alignment_target_gradient_is_unreliable():
    """Records a known limitation. This does NOT validate gradient correctness.

    At a degenerate (isotropic) alignment the target-side pose derivative is
    *unreliable*, and an earlier revision of this test understated it as
    "silently zeroed" while asserting a near-zero value at one angle. Measured
    here, the same code path produces four different behaviors, and which one
    appears depends on the **backend**:

    | angle (rad) | CPU | GPU (CudaDevice, this host) |
    |---|---|---|
    | 0.0 | zeroed | zeroed |
    | 0.1 | zeroed | approximate (within ~9%) |
    | 0.2 | zeroed | **NaN** |
    | 0.3 | zeroed | approximate (within ~7%) |
    | 0.5 | **wrong** (+0.529 vs +0.011) | zeroed |
    | 1.0 | zeroed | zeroed |

    Consequences, which are the point of this test:

    - The defect is **not** detectable by checking for a zero: on GPU it
      returns a NaN at 0.2 rad and a plausible value at 0.1/0.3 rad, and on
      CPU it returns a value ~46x the truth at 0.5 rad.
    - It is **not** detectable by checking for a NaN either: `zero_nan_pullback`
      suppresses most nonfinite pullbacks, so the usual result is finite and
      wrong, but it does not always, and a NaN does reach the caller.
    - Agreement at one probe angle establishes nothing, and agreement on one
      backend establishes nothing about the other.

    The true derivative exists at every angle here — the finite differences
    compute it. So this is an autodiff failure, not a nonsmooth point.
    `kabsch_conditioning` (tested below) is the only reliable detector.

    Exact per-angle classifications are platform- and precision-dependent, so
    this asserts the robust claim — degenerate alignments are not trustworthy
    — and reports the observed map when the behavior changes.
    """
    observed = _degenerate_classifications()
    broken = {a: k for a, k in observed.items() if k in BROKEN_KINDS}
    assert broken, (
        "known limitation no longer reproduces: every degenerate angle now "
        f"agrees with finite differences ({observed}). If the alignment "
        "gradient was fixed, replace this test rather than deleting it."
    )


def test_conditioning_flags_the_degenerate_case_the_gradient_hides():
    degenerate = kabsch_conditioning(ISOTROPIC_TARGET, ISOTROPIC_TARGET)
    healthy = kabsch_conditioning(GENERIC_TARGET, GENERIC_TARGET)

    assert not bool(degenerate["well_conditioned"])
    assert float(degenerate["min_relative_gap"]) == pytest.approx(0.0, abs=1e-6)
    assert bool(healthy["well_conditioned"])
    assert float(healthy["min_relative_gap"]) > 1e-2
    # The forward value is unremarkable in both cases: only the derivative is.
    assert np.all(np.isfinite(np.asarray(degenerate["singular_values"])))


def test_conditioning_also_flags_a_rank_deficient_target():
    """Collinear CA atoms leave the rotation about that axis unconstrained."""
    collinear = jnp.stack([jnp.arange(4.0), jnp.zeros(4), jnp.zeros(4)], axis=1)
    report = kabsch_conditioning(collinear, collinear)
    assert not bool(report["well_conditioned"])
    assert float(report["smallest_relative_value"]) == pytest.approx(0.0, abs=1e-6)
