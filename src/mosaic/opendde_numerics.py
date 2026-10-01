"""Small, checkpoint-free numerical kernels used by the OpenDDE adapter."""

import os

import jax
import jax.numpy as jnp


def aggregation_mode():
    """Read at JAX trace time; use a fresh process for each control arm."""
    mode = os.environ.get("MOSAIC_OPENDDE_AGGREGATION", "stable")
    if mode not in ("original", "stable"):
        raise ValueError("MOSAIC_OPENDDE_AGGREGATION must be original or stable")
    return mode


def stable_segment_mean(values, indices, num_segments):
    """Mean of [..., atoms, channels] by token, with fixed reduction order.

    Sort integer assignments stably, perform a segmented tree scan, then write
    each segment once. This avoids conflicting floating-point scatter-adds.
    Half-precision inputs accumulate in float32; output retains input dtype.
    Negative/out-of-range indices are padding, and empty segments return zero.
    ``num_segments`` must be static when compiled.
    """
    if values.ndim < 2 or indices.ndim != 1 or values.shape[-2] != indices.shape[0]:
        raise ValueError("expected [..., atoms, channels] and [atoms] indices")
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("segment indices must be integers")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("segment values must be floating point")
    if num_segments < 0:
        raise ValueError("num_segments must be nonnegative")
    shape = values.shape[:-2] + (num_segments, values.shape[-1])
    if values.shape[-2] == 0 or num_segments == 0:
        return jnp.zeros(shape, dtype=values.dtype)

    valid = (indices >= 0) & (indices < num_segments)
    ids = jnp.where(valid, indices, num_segments)
    order = jnp.argsort(ids, stable=True)
    ids = ids[order]
    dtype = jnp.promote_types(values.dtype, jnp.float32)
    x = jnp.take(values, order, axis=-2).astype(dtype)
    x = jnp.where((ids < num_segments)[:, None], x, 0)
    start = jnp.concatenate([jnp.array([True]), ids[1:] != ids[:-1]])
    flags = jnp.broadcast_to(start[:, None], x.shape[:-1] + (1,))

    def combine(left, right):
        left_sum, left_start = left
        right_sum, right_start = right
        return (
            jnp.where(right_start, right_sum, left_sum + right_sum),
            left_start | right_start,
        )

    sums, _ = jax.lax.associative_scan(combine, (x, flags), axis=-2)
    positions = jnp.arange(ids.shape[0])
    starts = jax.lax.associative_scan(jnp.maximum, jnp.where(start, positions, 0))
    counts = positions - starts + 1
    end = jnp.concatenate([ids[:-1] != ids[1:], jnp.array([True])])
    destination = jnp.where(end & (ids < num_segments), ids, num_segments)
    means = sums / counts[:, None].astype(dtype)
    # Only the final element of each valid segment writes an in-range index.
    return (
        jnp.zeros(shape, dtype=dtype)
        .at[..., destination, :]
        .set(means, mode="drop")
        .astype(values.dtype)
    )
