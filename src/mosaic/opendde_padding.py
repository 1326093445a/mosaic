"""Padding masks for numerical inference; reference-conformer masks are separate."""

import jax.numpy as jnp

from jopendde.transformer import rearrange_qk_to_dense_trunk


def active_atoms(indices, n_token):
    return (indices >= 0) & (indices < n_token)


def occupied_tokens(indices, n_token):
    destination = jnp.where(active_atoms(indices, n_token), indices, n_token)
    # Integer counts have no floating-point reduction-order ambiguity.
    return jnp.zeros(n_token, dtype=jnp.int32).at[destination].add(1, mode="drop") > 0


def atom_key_bias(mask, n_queries, n_keys, dtype):
    _, (keys,), _ = rearrange_qk_to_dense_trunk(
        [mask],
        [mask],
        [-1],
        [-1],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )
    return jnp.where(keys[:, None, None, :], 0.0, -1e9).astype(dtype)


def mask_structural_inputs(indices, s_inputs, s, z, bias):
    valid = occupied_tokens(indices, s.shape[-2])
    pair_mask = valid[:, None] & valid[None, :]
    return (
        jnp.where(valid[:, None], s_inputs, 0),
        jnp.where(valid[:, None], s, 0),
        jnp.where(pair_mask[..., None], z, 0),
        jnp.where(valid[None, :], bias, jnp.asarray(-1e9, bias.dtype)),
        pair_mask,
    )
