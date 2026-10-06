"""`BinderTargetRegistry`: does a named-pair restraint detect a flipped pose?

The measured failure (docs/P17_JN1.md section 25) is a binder at the correct
epitope, presenting its CDRs, rotated 156-178 degrees. Every aggregate site
criterion passes on those predictions. These tests check that the pairwise
restraint separates them and that the existing aggregate term does not --
which is the whole premise for adding it.

Synthetic distograms built from explicit coordinates. No model, no checkpoint.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.losses.structure_prediction import (  # noqa: E402
    BinderTargetContact,
    BinderTargetRegistry,
    StructureModelOutput,
    reference_contact_pairs,
)

N_BINDER, N_TARGET = 8, 10
N = N_BINDER + N_TARGET
BINS = jnp.linspace(2.0, 22.0, 64)

# A binder laid out along x, facing a target row 6 A away. Reference pairing is
# therefore i <-> i, which a 180-degree flip of the binder exchanges for
# i <-> (N_BINDER-1-i): the same residues, the same distances, new partners.
BINDER = np.stack([np.arange(N_BINDER) * 4.0, np.zeros(N_BINDER), np.zeros(N_BINDER)], -1)
TARGET = np.stack(
    [np.arange(N_TARGET) * 4.0, np.full(N_TARGET, 6.0), np.zeros(N_TARGET)], -1
)


def flipped(binder):
    """Reverse the binder along x about its own centre: same shape, new registry."""
    c = binder.mean(0)
    out = binder.copy()
    out[:, 0] = 2 * c[0] - binder[:, 0]
    return out


def distogram_from(binder, target, sharpness=40.0):
    """One-hot-ish distogram peaked at each pair's true distance."""
    coords = np.concatenate([binder, target], 0)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    logits = -sharpness * (np.asarray(BINS)[None, None, :] - d[:, :, None]) ** 2
    return jnp.asarray(logits)


def make_output(binder, target):
    return StructureModelOutput(
        distogram_logits=distogram_from(binder, target),
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


SEQ = jnp.zeros((N_BINDER, 20))
PAIRS = reference_contact_pairs(BINDER, TARGET, contact_distance=8.0)


def test_reference_pairs_are_the_diagonal_partners():
    """Sanity: the reference pairing is i <-> i, plus near neighbours."""
    assert PAIRS.size > 0
    assert PAIRS.shape[1] == 2
    for i, j in PAIRS:
        assert abs(int(i) - int(j)) <= 1, "geometry should pair residue i with target i"


def test_registry_penalises_a_flip_that_preserves_every_distance():
    """The decisive test: same residues, same distances, exchanged partners."""
    loss = BinderTargetRegistry(pairs=jnp.asarray(PAIRS), contact_distance=8.0)
    ref, _ = loss(SEQ, make_output(BINDER, TARGET), key=None)
    flip, _ = loss(SEQ, make_output(flipped(BINDER), TARGET), key=None)
    assert float(flip) > float(ref) + 1.0, (
        f"registry must separate a flip: reference {float(ref):.3f} vs "
        f"flipped {float(flip):.3f}"
    )


def test_the_aggregate_contact_term_does_not_separate_the_same_flip():
    """Why the new term is needed rather than a reweighting of the old one.

    `BinderTargetContact` asks whether each binder residue is near *any*
    epitope column, which a flip leaves satisfied. This pins that blindness.
    """
    agg = BinderTargetContact(
        epitope_idx=list(range(N_TARGET)), contact_distance=8.0
    )
    ref, _ = agg(SEQ, make_output(BINDER, TARGET), key=None)
    flip, _ = agg(SEQ, make_output(flipped(BINDER), TARGET), key=None)
    assert abs(float(flip) - float(ref)) < 0.1, (
        "the aggregate term is expected to be nearly flip-invariant here; if "
        f"this fails the premise changed (ref {float(ref):.3f}, flip {float(flip):.3f})"
    )


def test_registry_rewards_the_reference_pose_most():
    """Translating the binder away must be worse than leaving it in place.

    The shift is negative in y: the target sits at y = +6, so a positive
    shift moves the binder *through* it and back into contact. An earlier
    revision of this test shifted the wrong way and recorded distances of 2 A
    at "shift 4" -- the loss was right and the fixture was wrong.
    """
    loss = BinderTargetRegistry(pairs=jnp.asarray(PAIRS), contact_distance=8.0)
    ref, _ = loss(SEQ, make_output(BINDER, TARGET), key=None)
    previous = float(ref)
    for shift in (4.0, 10.0, 20.0):
        moved = BINDER - np.array([0.0, shift, 0.0])
        away, _ = loss(SEQ, make_output(moved, TARGET), key=None)
        assert float(away) >= previous, (
            f"moving {shift} A away should not score better (got "
            f"{float(away):.3f} against {previous:.3f})"
        )
        previous = float(away)
    assert previous > float(ref) + 1.0, "the farthest pose must be clearly worse"


def test_repel_pairs_bite_on_the_flipped_pose_and_not_on_the_reference():
    """The off-registry half must cost something only when those pairs form.

    The forbidden pairs here are the partners a flip creates: binder 0 with
    the far end of the target, and vice versa. In the reference pose they are
    ~36 A apart, so the penalty is correctly ~0 -- an earlier revision of this
    test asserted a penalty there and was wrong about the geometry, not about
    the loss. On the flipped pose those pairs are in contact and it bites.
    """
    repel = jnp.asarray([[0, N_BINDER - 1], [N_BINDER - 1, 0]])
    cold = BinderTargetRegistry(pairs=jnp.asarray(PAIRS), repel_pairs=repel,
                                repel_weight=0.0)
    hot = BinderTargetRegistry(pairs=jnp.asarray(PAIRS), repel_pairs=repel,
                               repel_weight=5.0)

    ref_out = make_output(BINDER, TARGET)
    c_ref, aux_cold = cold(SEQ, ref_out, key=None)
    h_ref, aux_hot = hot(SEQ, ref_out, key=None)
    assert "registry_repel_p" not in aux_cold, "inert at weight zero"
    assert "registry_repel_p" in aux_hot
    assert float(h_ref) == pytest.approx(float(c_ref), abs=1e-3), (
        "pairs 36 A apart must cost nothing in the reference pose"
    )

    flip_out = make_output(flipped(BINDER), TARGET)
    c_flip, _ = cold(SEQ, flip_out, key=None)
    h_flip, aux = hot(SEQ, flip_out, key=None)
    assert float(aux["registry_repel_p"]) > 0.5, (
        "the flip should put real probability on the forbidden pairs"
    )
    assert float(h_flip) > float(c_flip) + 1.0, (
        "and that must raise the loss relative to the unweighted term"
    )


def test_gradient_flows_to_the_distogram():
    """It must be differentiable where the search needs it."""
    import jax

    loss = BinderTargetRegistry(pairs=jnp.asarray(PAIRS))
    logits = distogram_from(BINDER, TARGET)

    def f(lg):
        out = make_output(BINDER, TARGET)
        out = type(out)(**{**{k: getattr(out, k) for k in out.__dataclass_fields__},
                           "distogram_logits": lg})
        return loss(SEQ, out, key=None)[0]

    g = jax.grad(f)(logits)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.abs(np.asarray(g)).sum() > 0, "gradient must be nonzero"


def test_subsets_restrict_the_pair_set():
    full = reference_contact_pairs(BINDER, TARGET, 8.0)
    sub = reference_contact_pairs(BINDER, TARGET, 8.0, binder_subset=[0, 1],
                                  target_subset=[0, 1])
    assert sub.shape[0] < full.shape[0]
    assert set(map(tuple, sub.tolist())) <= set(map(tuple, full.tolist()))
    assert all(int(i) in (0, 1) and int(j) in (0, 1) for i, j in sub)


def test_empty_pair_set_is_visible_to_the_caller():
    """An epitope with no reference contacts must not silently become a no-op."""
    far = reference_contact_pairs(BINDER, TARGET + np.array([0.0, 500.0, 0.0]), 8.0)
    assert far.shape[0] == 0


# --- Real reference geometry -------------------------------------------------
#
# The toy tests above establish the mechanism. These use the actual P17/JN.1
# reference and the flip geometries measured in docs/P17_JN1.md section 25:
# ~160-178 degrees of rotation with a 13-15 A centroid shift, and the CDRs
# still engaging the epitope at 56-76% of the interface. A flip that turns the
# paratope away is not the observed failure and both terms catch it; the
# flips that matter are the ones that keep the CDRs in contact.

REFERENCE = Path(__file__).resolve().parent.parent / "P17_JN1.pdb"
real = pytest.mark.skipif(not REFERENCE.is_file(), reason="P17_JN1.pdb absent")


def _real_setup():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from p17_alpha_reference import load_complex
    from p17_hallucination_search import CDR_RESIDUE_INDICES_1IDX

    j = load_complex(REFERENCE, "B", "T")
    cdr = np.array(sorted(i - 1 for i in CDR_RESIDUE_INDICES_1IDX), dtype=np.int32)
    return j["binder_ca"], j["target_ca"], cdr


def _rot_about(points, axis, pivot, theta=np.pi):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return (points - pivot) @ R.T + pivot


def _real_output(binder, target):
    nb, nt = len(binder), len(target)
    n = nb + nt
    bins = jnp.linspace(2.0, 22.0, 64)
    coords = np.concatenate([binder, target], 0)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    logits = jnp.asarray(-20.0 * (np.asarray(bins)[None, None, :] - d[:, :, None]) ** 2)
    z = jnp.zeros
    return StructureModelOutput(
        distogram_logits=logits, distogram_bins=bins, plddt=z((n,)),
        pae=z((n, n)), pae_logits=z((n, n, 4)), pae_bins=z((4,)),
        structure_coordinates=z((n, 3)), backbone_coordinates=z((n, 4, 3)),
        full_sequence=z((n, 20)),
        asym_id=jnp.concatenate([jnp.zeros(nb), jnp.ones(nt)]),
        residue_idx=jnp.arange(n), atom37_coords=z((n, 37, 3)),
        atom37_mask=z((n, 37)),
    )


@real
def test_registry_is_near_zero_at_the_real_reference():
    """Its baseline is interpretable, unlike the aggregate term's.

    The pair set is defined *from* the reference, so the reference scores ~0
    and any value is readable as distance from reference registry. The
    aggregate term scores in the hundreds at the reference because it asks
    every CDR residue to sit near one of five epitope residues, which the real
    structure does not satisfy -- so most of its magnitude is irreducible.
    """
    rb, rt, cdr = _real_setup()
    pairs = reference_contact_pairs(rb, rt, 8.0, binder_subset=cdr)
    assert len(pairs) > 10, f"reference should have a real interface, got {len(pairs)}"
    reg = BinderTargetRegistry(pairs=jnp.asarray(pairs), contact_distance=8.0)
    value, aux = reg(jnp.zeros((len(rb), 20)), _real_output(rb, rt), key=None)
    assert float(value) < 1.0, f"reference registry should be ~0, got {float(value)}"
    assert "registry_log_p" in aux


@real
def test_registry_beats_the_aggregate_term_on_paratope_preserving_flips():
    """The measured failure mode, and the reason this term was added.

    Both terms catch a flip that turns the paratope away from the target.
    Only the registry term strongly penalises a flip that keeps the CDRs on
    the epitope and merely exchanges which CDR residue meets which epitope
    residue -- which is what section 25 measured in the saved predictions.
    """
    rb, rt, cdr = _real_setup()
    epi = np.array([114, 116, 145, 147, 149], dtype=np.int32)
    pairs = reference_contact_pairs(rb, rt, 8.0, binder_subset=cdr)
    seq = jnp.zeros((len(rb), 20))
    reg = BinderTargetRegistry(pairs=jnp.asarray(pairs), contact_distance=8.0)
    agg = BinderTargetContact(paratope_idx=jnp.asarray(cdr),
                              epitope_idx=jnp.asarray(epi), contact_distance=8.0)

    base_r = float(reg(seq, _real_output(rb, rt), key=None)[0])
    base_a = float(agg(seq, _real_output(rb, rt), key=None)[0])

    paratope = rb[cdr].mean(0)
    approach = rt.mean(0) - rb.mean(0)
    approach = approach / np.linalg.norm(approach)
    perp = np.cross(approach, [0.0, 0.0, 1.0])
    perp = perp / np.linalg.norm(perp)

    for axis, label in ((approach, "spin about approach axis"),
                        (perp, "tumble through paratope")):
        flip = _rot_about(rb, axis, paratope)
        # confirm the flip is the observed kind: CDRs still at the epitope
        cdr_epi = np.linalg.norm(flip[cdr][:, None] - rt[epi][None], axis=-1).min()
        assert cdr_epi < 6.0, f"{label}: CDRs must stay engaged, got {cdr_epi:.1f} A"

        d_reg = float(reg(seq, _real_output(flip, rt), key=None)[0]) - base_r
        d_agg = float(agg(seq, _real_output(flip, rt), key=None)[0]) - base_a
        assert d_reg > 0, f"{label}: registry must penalise the flip"
        assert d_reg > 2.0 * d_agg, (
            f"{label}: registry should be the more sensitive term "
            f"(registry +{d_reg:.0f} against aggregate +{d_agg:.0f})"
        )
