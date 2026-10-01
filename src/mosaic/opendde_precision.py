"""Explicit OpenDDE compute precision; does not alter global JAX settings."""

import jax
import jax.numpy as jnp
from jopendde.backend import LayerNorm, Linear


def compute_dtype(name):
    if name not in ("fp32", "bf16"):
        raise ValueError(f"unsupported OpenDDE compute dtype: {name!r}")
    return jnp.float32 if name == "fp32" else jnp.bfloat16


def cast_float_arrays(tree, dtype):
    """Preserve integer indices, masks and static metadata."""
    return jax.tree.map(
        lambda x: (
            x.astype(dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        tree,
    )


class BF16Linear(Linear):
    """Cast float32 geometry-derived inputs at the neural projection boundary.

    Without this, noisy coordinates and confidence distances promote BF16
    activations back to float32 and can cause scan-carry dtype mismatches.
    """

    def __call__(self, x):
        return super().__call__(x.astype(jnp.bfloat16))


class BF16LayerNorm(LayerNorm):
    """Compute normalization statistics in float32, then restore activation dtype."""

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight.astype(jnp.float32)
        if self.bias is not None:
            x = x + self.bias.astype(jnp.float32)
        return x.astype(dtype)


def prepare_opendde_model(model, name):
    """Opt-in BF16 weights/activations with float32 normalization reductions.

    FP32 mode preserves the existing model, including its separately configured
    attention-core precision. Apply once to an original checkpoint model.
    """
    dtype = compute_dtype(name)
    if name == "fp32":
        return model

    def convert(x):
        if isinstance(x, Linear):
            return BF16Linear(
                weight=cast_float_arrays(x.weight, dtype),
                bias=cast_float_arrays(x.bias, dtype),
            )
        if isinstance(x, LayerNorm):
            return BF16LayerNorm(
                weight=cast_float_arrays(x.weight, dtype),
                bias=cast_float_arrays(x.bias, dtype),
                eps=x.eps,
            )
        return cast_float_arrays(x, dtype)

    return jax.tree.map(
        convert, model, is_leaf=lambda x: isinstance(x, (Linear, LayerNorm))
    )
