from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from boltzgen.model.modules.inverse_fold import MLPAttnGNN, MLPAttnGNNDecoder

from mosaic.models.boltzgen_if_jax import (
    BoltzGenIFContext,
    DecoderLayer,
    EncoderLayer,
    differentiable_jax_boltzgen_if,
    load_jax_boltzgen_if,
)


def test_encoder_layer_matches_torch_eval():
    torch.manual_seed(4)
    layer = MLPAttnGNN(
        node_dim=8,
        pair_dim=6,
        hidden_dim=8,
        dropout=0.0,
        softmax_dropout=0.0,
        transformation_scale_factor=1.0,
        num_heads=2,
    ).eval()
    jax_layer = EncoderLayer.from_torch(layer)
    source = torch.tensor([0, 1, 2, 3, 0, 2])
    destination = torch.tensor([0, 0, 1, 1, 2, 3])
    edge_index = torch.stack((source, destination))
    nodes = torch.randn(4, 8)
    pairs = torch.randn(6, 6)

    with torch.inference_mode():
        torch_nodes, torch_pairs = layer(nodes, pairs, edge_index)
    jax_nodes, jax_pairs = jax_layer(
        jnp.asarray(nodes.numpy()),
        jnp.asarray(pairs.numpy()),
        jnp.asarray(source.numpy()),
        jnp.asarray(destination.numpy()),
    )

    np.testing.assert_allclose(jax_nodes, torch_nodes.numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(jax_pairs, torch_pairs.numpy(), rtol=2e-4, atol=2e-4)


def test_decoder_position_matches_torch_eval():
    torch.manual_seed(5)
    layer = MLPAttnGNNDecoder(
        node_dim=8,
        pair_dim=6,
        hidden_dim=8,
        dropout=0.0,
        softmax_dropout=0.0,
        transformation_scale_factor=1.0,
        num_heads=2,
    ).eval()
    jax_layer = DecoderLayer.from_torch(layer)
    node = torch.randn(1, 8)
    neighbors = torch.randn(5, 14)

    with torch.inference_mode():
        torch_result = layer.sample(node, neighbors)
    jax_result = jax_layer.sample_position(
        jnp.asarray(node.numpy()[0]), jnp.asarray(neighbors.numpy())
    )
    np.testing.assert_allclose(jax_result, torch_result.numpy()[0], rtol=2e-5, atol=2e-5)


@pytest.mark.slow
def test_complete_checkpoint_has_coordinate_gradient():
    checkpoint = Path("~/.boltz/boltzgen1_ifold.ckpt").expanduser()
    if not checkpoint.exists():
        pytest.skip("official BoltzGen-IF checkpoint is not cached")
    model, _ = load_jax_boltzgen_if(checkpoint, torch_device="cpu")

    num_nodes = 5
    atom_count = num_nodes * 4
    token_to_bb4 = np.zeros((num_nodes, 4, atom_count), dtype=np.float32)
    for residue in range(num_nodes):
        for atom in range(4):
            token_to_bb4[residue, atom, residue * 4 + atom] = 1.0
    parent = np.eye(20, dtype=np.float32)[[0, 1, 2, 3, 4]]
    context = BoltzGenIFContext(
        s_inputs=jnp.zeros((num_nodes, 384), dtype=jnp.float32),
        token_to_bb4_atoms=jnp.asarray(token_to_bb4),
        feature_asym_id=jnp.zeros(num_nodes, dtype=jnp.int32),
        feature_residue_index=jnp.arange(num_nodes, dtype=jnp.int32),
        token_bonds=jnp.zeros((num_nodes, num_nodes), dtype=jnp.float32),
        type_bonds=jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32),
        parent_sequence=jnp.asarray(parent),
        designable_mask=jnp.asarray([False, True, True, False, False]),
        designable_positions=jnp.asarray([1, 2], dtype=jnp.int32),
        valid_indices=jnp.arange(num_nodes, dtype=jnp.int32),
        full_parent_sequence=jnp.asarray(parent),
    )
    coords = jnp.arange(atom_count * 3, dtype=jnp.float32).reshape(atom_count, 3) / 10

    def objective(atom_coords):
        soft_sequence = differentiable_jax_boltzgen_if(
            model,
            context,
            atom_coords,
            key=jax.random.key(0),
            temperature=0.3,
            avoid="C",
            order=jnp.asarray([1, 2]),
        )
        return -jnp.log(soft_sequence[1, 17]) - jnp.log(soft_sequence[2, 18])

    gradient = jax.grad(objective)(coords)
    assert gradient.shape == coords.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0
