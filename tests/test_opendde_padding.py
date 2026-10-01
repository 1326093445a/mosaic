"""Synthetic array-accounting checks; no checkpoint or protein input needed."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jopendde.features import Features
from jopendde.transformer import aggregate_atom_to_token

from mosaic.losses.opendde import (
    OpenDDEAtomTemplates,
    OpenDDEDesignFeatures,
    _rebuild_dense_trunk,
    set_binder_sequence,
)
from mosaic.models.opendde import _atom_template_shapes


def _synthetic_design():
    # Two residue tokens, four structural tokens. The first residue's six-slot
    # allocation shrinks to four atoms; the target's four atoms follow it.
    a2t = jnp.array([0] * 6 + [1] * 4)
    pos = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    d_lm, v_lm, pad = _rebuild_dense_trunk(pos, a2t)
    feat = Features(
        token_index=jnp.arange(2),
        residue_index=jnp.zeros(2, dtype=jnp.int32),
        asym_id=jnp.arange(2),
        entity_id=jnp.arange(2),
        sym_id=jnp.zeros(2, dtype=jnp.int32),
        restype=jax.nn.one_hot(jnp.zeros(2, dtype=jnp.int32), 32),
        token_bonds=jnp.zeros((2, 2)),
        has_frame=jnp.ones(2),
        frame_atom_index=jnp.array([[0, 1, 2], [6, 7, 8]]),
        profile=jax.nn.one_hot(jnp.zeros(2, dtype=jnp.int32), 32),
        deletion_mean=jnp.zeros(2),
        relp=jnp.zeros((2, 2, 1)),
        ref_pos=pos,
        ref_charge=jnp.zeros(10, dtype=jnp.int32),
        ref_mask=jnp.ones(10),
        ref_element=jnp.zeros((10, 128)),
        ref_atom_name_chars=jnp.zeros((10, 4, 64)),
        atom_to_token_idx=a2t,
        atom_to_tokatom_idx=jnp.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]),
        is_ligand=jnp.zeros(10),
        distogram_rep_atom_mask=jnp.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
        pae_rep_atom_mask=jnp.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
        d_lm=d_lm,
        v_lm=v_lm,
        pad_info=pad,
        structural_token_index=jnp.arange(4),
        subtoken_role_id=jnp.array([1, 2, 1, 2]),
        parent_residue_idx=jnp.array([0, 0, 1, 1]),
        prev_parent_residue_idx=jnp.full(4, -1),
        next_parent_residue_idx=jnp.full(4, -1),
        atom_to_structural_token_idx=jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]),
        atom_to_structural_tokatom_idx=jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
        structural_has_frame=jnp.ones(4),
        structural_frame_atom_index=jnp.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [6, 7, 9]]
        ),
        structural_distogram_rep_atom_mask=jnp.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1]),
        structural_pae_rep_atom_mask=jnp.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1]),
    )
    float_fields = {"ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars"}
    arrays = {
        name: jnp.zeros(shape, dtype=jnp.float32 if name in float_fields else jnp.int32)
        for name, shape in _atom_template_shapes().items()
    }
    arrays["n_atoms"] = jnp.full((2, 20), 4, dtype=jnp.int32)
    arrays["a_struct_tok"] = arrays["a_struct_tok"].at[:, :, 3].set(1)
    arrays["s_valid"] = jnp.ones((2, 20, 2), dtype=jnp.int32)
    arrays["s_has_frame"] = jnp.ones((2, 20, 2), dtype=jnp.int32)
    arrays["s_disto_off"] = arrays["s_disto_off"].at[:, :, 1].set(3)
    arrays["s_pae_off"] = arrays["s_pae_off"].at[:, :, 1].set(3)
    return OpenDDEDesignFeatures(
        features=feat,
        atom_templates=OpenDDEAtomTemplates(**arrays),
        binder_length=1,
        binder_atom_alloc=6,
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_padding_cannot_contribute_to_either_token_aggregation(compiled):
    design = _synthetic_design()
    refresh = eqx.filter_jit(set_binder_sequence) if compiled else set_binder_sequence
    feat = refresh(jax.nn.one_hot(jnp.array([0]), 20), design, jax.random.key(0))
    active = np.asarray(feat.ref_mask) > 0.5
    assert active.sum() == 8
    np.testing.assert_array_equal(feat.ref_pos[4:8], design.features.ref_pos[6:10])

    for indices, n_token in (
        (feat.atom_to_token_idx, 2),
        (feat.atom_to_structural_token_idx, 4),
    ):
        # Large padding values make both numerator leakage and incorrect
        # contribution counts visible in the real downstream aggregation.
        values = jnp.where(feat.ref_mask[:, None] > 0.5, 1.0, 999.0)
        observed = aggregate_atom_to_token(values, indices, n_token)
        np.testing.assert_array_equal(observed, np.ones((n_token, 1)))
        assert np.all(np.asarray(indices)[~active] >= n_token)


@pytest.mark.parametrize("compiled", [False, True])
def test_absent_subtoken_does_not_claim_a_representative_atom(compiled):
    design = _synthetic_design()
    tmpl = design.atom_templates
    tmpl = eqx.tree_at(
        lambda t: (t.s_valid, t.a_struct_tok, t.s_disto_off, t.s_pae_off),
        tmpl,
        (
            tmpl.s_valid.at[:, :, 1].set(0),
            jnp.zeros_like(tmpl.a_struct_tok),
            tmpl.s_disto_off.at[:, :, 0].set(1).at[:, :, 1].set(0),
            tmpl.s_pae_off.at[:, :, 0].set(1).at[:, :, 1].set(0),
        ),
    )
    design = eqx.tree_at(lambda d: d.atom_templates, design, tmpl)
    refresh = eqx.filter_jit(set_binder_sequence) if compiled else set_binder_sequence
    feat = refresh(jax.nn.one_hot(jnp.array([0]), 20), design, jax.random.key(0))
    for mask in (
        feat.structural_distogram_rep_atom_mask,
        feat.structural_pae_rep_atom_mask,
    ):
        np.testing.assert_array_equal(mask[:4], [0, 1, 0, 0])
    np.testing.assert_array_equal(feat.structural_has_frame, [1, 0, 1, 1])


@pytest.mark.parametrize("compiled", [False, True])
def test_frame_validity_comes_from_the_current_template(compiled):
    design = _synthetic_design()
    tmpl = eqx.tree_at(
        lambda t: t.s_has_frame,
        design.atom_templates,
        jnp.zeros_like(design.atom_templates.s_has_frame),
    )
    design = eqx.tree_at(lambda d: d.atom_templates, design, tmpl)
    refresh = eqx.filter_jit(set_binder_sequence) if compiled else set_binder_sequence
    feat = refresh(jax.nn.one_hot(jnp.array([0]), 20), design, jax.random.key(0))
    np.testing.assert_array_equal(feat.structural_has_frame, [0, 0, 1, 1])


@pytest.mark.parametrize("compiled", [False, True])
def test_structural_expansion_threads_masks_through_the_refiner(compiled):
    from types import SimpleNamespace
    from jopendde.model import OpenDDE

    feat = _synthetic_design().features
    feat = eqx.tree_at(
        lambda f: f.atom_to_structural_token_idx,
        feat,
        jnp.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 3]),
    )
    expanded = jnp.array([[1.0], [999.0], [3.0], [5.0]])

    def refiner(s, z, pair_mask, extra_attn_bias):
        assert pair_mask is not None
        # Deliberately create nonzero output on the empty query too. The real
        # expansion boundary must mask both its inputs and returned state.
        return jax.nn.softmax(extra_attn_bias, axis=-1) @ s, z + 17

    model = SimpleNamespace(
        structural_token_expander=lambda *a: (
            expanded,
            expanded,
            jnp.ones((4, 4, 1)),
            jnp.zeros((4, 4)),
        ),
        generate_relp=lambda f: f.relp,
        structural_token_refiner=refiner,
    )

    def expand(features):
        return OpenDDE.expand_to_structural_tokens(model, features, None, None, None)

    run = eqx.filter_jit(expand) if compiled else expand
    _, si, s, z, bias = run(feat)
    np.testing.assert_array_equal(si[1], [0])
    np.testing.assert_allclose(s, [[3], [0], [3], [3]], atol=1e-6)
    assert not np.asarray(z)[1].any()
    assert not np.asarray(z)[:, 1].any()
    np.testing.assert_array_equal(jax.nn.softmax(bias, axis=-1)[:, 1], np.zeros(4))
