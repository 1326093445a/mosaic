"""JAX forward implementation of the official BoltzGen inverse-folding model.

The upstream model is PyTorch and places its inference path under
``torch.no_grad``. This module converts the trained linear/normalization weights
to JAX and reproduces the geometric GNN plus autoregressive decoder. KNN graph
selection and sampled residue identities are discrete; logits remain
differentiable with respect to input atom coordinates within the selected graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from boltzgen.data import const
from jaxtyping import Array, Float, Int

from mosaic.common import TOKENS
from mosaic.legacy.boltzgen_if import BoltzGenIFResult, load_boltzgen_if, prepare_boltzgen_if_features


def _array(value: torch.Tensor) -> jax.Array:
    return jnp.asarray(value.detach().float().cpu().numpy())


class Linear(eqx.Module):
    weight: Float[Array, "out in"]
    bias: Float[Array, "out"] | None

    @classmethod
    def from_torch(cls, layer: torch.nn.Linear) -> "Linear":
        bias = None if layer.bias is None else _array(layer.bias)
        return cls(_array(layer.weight), bias)

    def __call__(self, x):
        result = x @ self.weight.T
        return result if self.bias is None else result + self.bias


class BatchNorm(eqx.Module):
    weight: Float[Array, "D"]
    bias: Float[Array, "D"]
    mean: Float[Array, "D"]
    variance: Float[Array, "D"]
    eps: float = eqx.field(static=True)

    @classmethod
    def from_torch(cls, layer: torch.nn.modules.batchnorm._BatchNorm) -> "BatchNorm":
        return cls(
            _array(layer.weight),
            _array(layer.bias),
            _array(layer.running_mean),
            _array(layer.running_var),
            float(layer.eps),
        )

    def __call__(self, x):
        normalized = (x - self.mean) * jax.lax.rsqrt(self.variance + self.eps)
        return normalized * self.weight + self.bias


class ThreeLayerMLP(eqx.Module):
    layers: tuple[Linear, Linear, Linear]

    @classmethod
    def from_torch(cls, module: torch.nn.Sequential) -> "ThreeLayerMLP":
        return cls(tuple(Linear.from_torch(module[index]) for index in (0, 2, 4)))

    def __call__(self, x):
        x = jax.nn.gelu(self.layers[0](x), approximate=False)
        x = jax.nn.gelu(self.layers[1](x), approximate=False)
        return self.layers[2](x)


class ResidualBlock(eqx.Module):
    first: Linear
    second: Linear
    norm: BatchNorm

    @classmethod
    def from_torch(cls, module: torch.nn.Sequential) -> "ResidualBlock":
        return cls(
            Linear.from_torch(module[0]),
            Linear.from_torch(module[2]),
            BatchNorm.from_torch(module[4]),
        )

    def __call__(self, x):
        return self.norm(self.second(jax.nn.gelu(self.first(x), approximate=False)))


def _segment_sum(values, indices, num_segments):
    return jax.ops.segment_sum(values, indices, num_segments=num_segments)


def _segment_softmax(values, indices, num_segments):
    maxima = jax.ops.segment_max(values, indices, num_segments=num_segments)
    centered = values - maxima[indices]
    exponentials = jnp.exp(centered)
    totals = _segment_sum(exponentials, indices, num_segments)
    return exponentials / (totals[indices] + 1e-10)


class EncoderLayer(eqx.Module):
    attn_weight: ThreeLayerMLP
    attn_value: ThreeLayerMLP
    attn_output: Linear
    attn_output_norm: BatchNorm
    node_ffn: ResidualBlock
    edge_ffn: ResidualBlock
    num_heads: int = eqx.field(static=True)

    @classmethod
    def from_torch(cls, layer) -> "EncoderLayer":
        return cls(
            ThreeLayerMLP.from_torch(layer.attn_weight_mlp),
            ThreeLayerMLP.from_torch(layer.attn_value_mlp),
            Linear.from_torch(layer.attn_output_linear[0]),
            BatchNorm.from_torch(layer.attn_output_linear[2]),
            ResidualBlock.from_torch(layer.attn_FFN),
            ResidualBlock.from_torch(layer.edge_FFN),
            int(layer.num_heads),
        )

    def __call__(self, s, z, src, dst):
        num_nodes = s.shape[0]
        z = z + self.edge_ffn(jnp.concatenate((s[src], s[dst], z), axis=-1))
        weights = self.attn_weight(jnp.concatenate((s[dst], s[src], z), axis=-1))
        values = self.attn_value(jnp.concatenate((s[src], z), axis=-1))
        weights = _segment_softmax(weights, dst, num_nodes)
        messages = weights[..., None] * values[:, None, :]
        messages = _segment_sum(messages, dst, num_nodes).reshape(num_nodes, -1)
        s = s + self.attn_output_norm(self.attn_output(messages))
        s = s + self.node_ffn(s)
        return s, z


class DecoderLayer(eqx.Module):
    attn_weight: ThreeLayerMLP
    attn_value: ThreeLayerMLP
    attn_output: Linear
    attn_output_norm: BatchNorm
    node_ffn: ResidualBlock

    @classmethod
    def from_torch(cls, layer) -> "DecoderLayer":
        return cls(
            ThreeLayerMLP.from_torch(layer.attn_weight_mlp),
            ThreeLayerMLP.from_torch(layer.attn_value_mlp),
            Linear.from_torch(layer.attn_output_linear[0]),
            BatchNorm.from_torch(layer.attn_output_linear[2]),
            ResidualBlock.from_torch(layer.attn_FFN),
        )

    def sample_position(self, s_position, neighbor_representation):
        count = neighbor_representation.shape[0]
        query = jnp.broadcast_to(s_position, (count, s_position.shape[-1]))
        weights = jax.nn.softmax(
            self.attn_weight(jnp.concatenate((query, neighbor_representation), axis=-1)),
            axis=0,
        )
        values = self.attn_value(neighbor_representation)
        message = (weights[..., None] * values[:, None, :]).sum(axis=0).reshape(-1)
        s_position = s_position + self.attn_output_norm(self.attn_output(message))
        return s_position + self.node_ffn(s_position)


@dataclass(frozen=True)
class BoltzGenIFContext:
    """Static per-complex features consumed by :class:`JaxBoltzGenIF`."""

    s_inputs: jax.Array
    token_to_bb4_atoms: jax.Array
    feature_asym_id: jax.Array
    feature_residue_index: jax.Array
    token_bonds: jax.Array
    type_bonds: jax.Array
    parent_sequence: jax.Array
    designable_mask: jax.Array
    designable_positions: jax.Array
    valid_indices: jax.Array
    full_parent_sequence: jax.Array


@dataclass(frozen=True)
class JaxBoltzGenIFResult:
    logits: jax.Array
    sequence_ids: jax.Array
    order: jax.Array

    @property
    def sequence(self) -> str:
        return "".join(TOKENS[int(index)] for index in np.asarray(self.sequence_ids))


class JaxBoltzGenIF(eqx.Module):
    """Official BoltzGen-IF weights evaluated with JAX operations."""

    node_projection: Linear
    pair_projection: Linear
    encoder_layers: tuple[EncoderLayer, ...]
    decoder_layers: tuple[DecoderLayer, ...]
    sequence_projection: Linear
    predictor: Linear
    gaussian_offsets: Float[Array, "G"]
    gaussian_coefficient: float = eqx.field(static=True)
    topk: int = eqx.field(static=True)
    residue_radius: int = eqx.field(static=True)
    num_bond_types: int = eqx.field(static=True)

    @classmethod
    def from_torch(cls, model) -> "JaxBoltzGenIF":
        encoder = model.inverse_folding_encoder
        decoder = model.structure_module
        offsets = _array(encoder.distance_gaussian_smearing.offset)
        return cls(
            Linear.from_torch(encoder.linear_token_to_node),
            Linear.from_torch(encoder.linear_token_to_pair),
            tuple(EncoderLayer.from_torch(layer) for layer in encoder.encoder_layers),
            tuple(DecoderLayer.from_torch(layer) for layer in decoder.decoder_layers),
            Linear.from_torch(decoder.seq_to_s),
            Linear.from_torch(decoder.predictor),
            offsets,
            float(encoder.distance_gaussian_smearing.coeff),
            int(encoder.topk),
            int(encoder.r_max),
            len(const.bond_types) + 1,
        )

    def _graph_and_pair_features(self, context: BoltzGenIFContext, coords):
        token_to_bb4 = context.token_to_bb4_atoms
        backbone = jnp.einsum("nkm,mc->nkc", token_to_bb4, coords)
        centers = backbone[:, 1]
        num_nodes = centers.shape[0]
        neighbor_count = min(self.topk, num_nodes)
        # Graph topology is intentionally discrete. Stopping its coordinate
        # derivative also avoids undefined gradients at self-distance zero.
        distances = jax.lax.stop_gradient(
            jnp.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        )
        # Rows are destination nodes and columns are candidate source nodes.
        src_by_dst = jax.lax.top_k(-distances, neighbor_count)[1]
        dst_by_dst = jnp.broadcast_to(
            jnp.arange(num_nodes)[:, None], src_by_dst.shape
        )
        src = src_by_dst.reshape(-1)
        dst = dst_by_dst.reshape(-1)

        pair_deltas = backbone[src, None, :, :] - backbone[dst, :, None, :]
        # sqrt(max(...)) has a zero derivative for coincident atom pairs rather
        # than the NaN derivative of a raw Euclidean norm at exactly zero.
        pair_distances = jnp.sqrt(
            jnp.maximum(jnp.sum(pair_deltas**2, axis=-1), 1e-12)
        ).reshape(src.shape[0], -1)
        smeared = jnp.exp(
            self.gaussian_coefficient
            * (pair_distances[..., None] - self.gaussian_offsets) ** 2
        ).reshape(src.shape[0], -1)

        same_chain = context.feature_asym_id[src] == context.feature_asym_id[dst]
        same_residue = (
            context.feature_residue_index[src]
            == context.feature_residue_index[dst]
        )
        residue_delta = context.feature_residue_index[dst] - context.feature_residue_index[src]
        residue_delta = jnp.clip(
            residue_delta + self.residue_radius,
            0,
            2 * self.residue_radius,
        )
        residue_delta = jnp.where(
            same_chain,
            residue_delta,
            2 * self.residue_radius + 1,
        )
        relative = jax.nn.one_hot(residue_delta, 2 * self.residue_radius + 2)
        edge_attributes = jnp.concatenate(
            (relative, same_chain[:, None], same_residue[:, None]), axis=-1
        )
        bond = jnp.concatenate(
            (
                context.token_bonds[dst, src, None],
                jax.nn.one_hot(context.type_bonds[dst, src], self.num_bond_types),
            ),
            axis=-1,
        )
        pair_features = jnp.concatenate((smeared, edge_attributes, bond), axis=-1)
        return backbone, src_by_dst, src, dst, pair_features

    def encode(self, context: BoltzGenIFContext, coords):
        _, src_by_dst, src, dst, pair_features = self._graph_and_pair_features(
            context, coords
        )
        s = self.node_projection(context.s_inputs)
        z = self.pair_projection(pair_features)
        for layer in self.encoder_layers:
            s, z = layer(s, z, src, dst)
        z_by_dst = z.reshape(s.shape[0], src_by_dst.shape[1], z.shape[-1])
        return s, z_by_dst, src_by_dst

    def __call__(
        self,
        context: BoltzGenIFContext,
        coords: Float[Array, "M 3"],
        *,
        key,
        temperature: float | None = None,
        avoid: str = "C",
        order: Int[Array, "D"] | None = None,
    ) -> JaxBoltzGenIFResult:
        s, z_by_dst, src_by_dst = self.encode(context, coords)
        parent = context.parent_sequence
        designable = context.designable_mask
        decoded = jnp.where(designable[:, None], 0.0, parent)
        logits = jnp.where(designable[:, None], 0.0, parent)

        if order is None:
            order = jax.random.permutation(key, context.designable_positions)
        order = jnp.asarray(order, dtype=jnp.int32)

        blocked = jnp.zeros((len(TOKENS),), dtype=bool)
        for residue in avoid:
            if residue not in TOKENS:
                raise ValueError(f"Unknown residue in BoltzGen-IF avoid set: {residue}")
            blocked = blocked.at[TOKENS.index(residue)].set(True)

        def decode_position(carry, inputs):
            decoded_sequence, all_logits, random_key = carry
            position, step_index = inputs
            neighbors = src_by_dst[position]
            neighbor_sequence = jnp.zeros(
                (neighbors.shape[0], const.num_tokens), dtype=decoded_sequence.dtype
            ).at[
                :, const.canonicals_offset : const.canonicals_offset + len(TOKENS)
            ].set(decoded_sequence[neighbors])
            residue_representation = self.sequence_projection(neighbor_sequence)
            neighbor_representation = jnp.concatenate(
                (z_by_dst[position], s[neighbors] + residue_representation), axis=-1
            )
            s_position = s[position]
            for layer in self.decoder_layers:
                s_position = layer.sample_position(s_position, neighbor_representation)
            position_logits = self.predictor(s_position)
            canonical_logits = position_logits[
                const.canonicals_offset : const.canonicals_offset + len(TOKENS)
            ]
            canonical_logits = jnp.where(blocked, -1e6, canonical_logits)
            random_key, sample_key = jax.random.split(random_key)
            if temperature is None or temperature <= 0:
                residue_id = jnp.argmax(canonical_logits)
            else:
                residue_id = jax.random.categorical(
                    jax.random.fold_in(sample_key, step_index),
                    canonical_logits / temperature,
                )
            residue = jax.nn.one_hot(residue_id, len(TOKENS))
            decoded_sequence = decoded_sequence.at[position].set(residue)
            all_logits = all_logits.at[position].set(canonical_logits)
            return (decoded_sequence, all_logits, random_key), None

        (decoded, logits, _), _ = jax.lax.scan(
            decode_position,
            (decoded, logits, key),
            (order, jnp.arange(order.shape[0], dtype=jnp.int32)),
        )
        sequence_ids = jnp.argmax(decoded, axis=-1)
        return JaxBoltzGenIFResult(logits, sequence_ids, order)


def prepare_jax_boltzgen_if_context(
    torch_model,
    torch_features: dict[str, Any],
    coords: np.ndarray,
    *,
    parent_sequence: np.ndarray,
    designable_mask: np.ndarray,
) -> BoltzGenIFContext:
    """Extract static, valid-token features and precompute IF input embeddings."""
    device = next(torch_model.parameters()).device
    features = prepare_boltzgen_if_features(
        torch_features,
        coords,
        parent_sequence=parent_sequence,
        designable_mask=designable_mask,
        device=device,
    )
    with torch.inference_mode():
        s_inputs = torch_model.input_embedder(features)
    valid = (features["token_resolved_mask"].bool() & features["token_pad_mask"].bool())[0]
    valid_indices = torch.where(valid)[0]
    token_to_bb4 = features["token_to_bb4_atoms"][0, valid]
    token_bonds = features["token_bonds"][0][valid_indices][:, valid_indices, 0]
    type_bonds = features["type_bonds"][0][valid_indices][:, valid_indices]

    parent = np.asarray(parent_sequence, dtype=np.float32)[np.asarray(valid.cpu())]
    design = np.asarray(designable_mask, dtype=bool)[np.asarray(valid.cpu())]
    return BoltzGenIFContext(
        _array(s_inputs[0, valid]),
        _array(token_to_bb4),
        jnp.asarray(features["feature_asym_id"][0, valid].cpu().numpy()),
        jnp.asarray(features["feature_residue_index"][0, valid].cpu().numpy()),
        _array(token_bonds),
        jnp.asarray(type_bonds.cpu().numpy()),
        jnp.asarray(parent),
        jnp.asarray(design),
        jnp.asarray(np.flatnonzero(design), dtype=jnp.int32),
        jnp.asarray(valid_indices.cpu().numpy()),
        jnp.asarray(parent_sequence, dtype=jnp.float32),
    )


def decode_with_jax_boltzgen_if(
    model: JaxBoltzGenIF,
    context: BoltzGenIFContext,
    coords: np.ndarray | jax.Array,
    *,
    seed: int,
    temperature: float | None = 0.3,
    avoid: str = "C",
    order: Int[Array, "D"] | None = None,
) -> BoltzGenIFResult:
    """Decode with JAX and expand valid-token results to the full feature shape."""
    result = model(
        context,
        jnp.asarray(coords),
        key=jax.random.key(seed),
        temperature=temperature,
        avoid=avoid,
        order=order,
    )
    full_logits = jnp.zeros_like(context.full_parent_sequence)
    full_logits = full_logits.at[context.valid_indices].set(result.logits)
    full_ids = jnp.argmax(context.full_parent_sequence, axis=-1)
    full_ids = full_ids.at[context.valid_indices].set(result.sequence_ids)
    return BoltzGenIFResult(
        sequence_ids=np.asarray(full_ids, dtype=np.int32),
        logits=np.asarray(full_logits, dtype=np.float32),
    )


def differentiable_jax_boltzgen_if(
    model: JaxBoltzGenIF,
    context: BoltzGenIFContext,
    coords: Float[Array, "M 3"],
    *,
    key,
    temperature: float = 0.3,
    avoid: str = "C",
    order: Int[Array, "D"] | None = None,
) -> Float[Array, "N 20"]:
    """Return a full soft sequence differentiable with respect to coordinates.

    Autoregressive residue choices are argmax-discrete context. The conditional
    logits at every designable position remain differentiable, and their softmax
    probabilities are inserted into the fixed full parent sequence.
    """
    if temperature <= 0:
        raise ValueError("BoltzGen-IF guidance temperature must be > 0")
    result = model(
        context,
        coords,
        key=key,
        temperature=None,
        avoid=avoid,
        order=order,
    )
    probabilities = jax.nn.softmax(result.logits / temperature, axis=-1)
    valid_sequence = jnp.where(
        context.designable_mask[:, None],
        probabilities,
        context.parent_sequence,
    )
    return context.full_parent_sequence.at[context.valid_indices].set(valid_sequence)


def load_jax_boltzgen_if(
    checkpoint=None,
    *,
    torch_device: str = "cpu",
) -> tuple[JaxBoltzGenIF, Any]:
    """Load official weights and return both JAX and Torch IF models.

    The Torch object is retained only to prepare static input-embedder features
    and to support numerical parity checks. Coordinate-dependent IF evaluation
    is performed by :class:`JaxBoltzGenIF`.
    """
    torch_model = load_boltzgen_if(checkpoint, device=torch_device)
    return JaxBoltzGenIF.from_torch(torch_model), torch_model
