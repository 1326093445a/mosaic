"""Numerical and derivative checks on synthetic data, without model weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.opendde_numerics import stable_segment_mean


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_segment_mean_matches_independent_reference(dtype, compiled):
    indices = np.array([2, 0, 2, 4, -1, 1, 2, 9, 0], dtype=np.int32)
    x = jnp.asarray(np.random.default_rng(4).normal(size=(2, 9, 3)), dtype=dtype)
    host = np.asarray(x, dtype=np.float32)
    expected = np.zeros((2, 5, 3), dtype=np.float32)
    for token in (0, 1, 2, 4):
        expected[:, token] = host[:, indices == token].mean(axis=1)
    run = (
        jax.jit(stable_segment_mean, static_argnums=2)
        if compiled
        else stable_segment_mean
    )
    actual = run(x, jnp.asarray(indices), 5)
    assert actual.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(jnp.asarray(expected, dtype=dtype), dtype=np.float32),
        atol=2e-7,
        rtol=2e-6,
    )


def test_padding_is_ignored_even_when_nonfinite_and_empty_segments_are_zero():
    x = jnp.array([[2.0], [jnp.nan], [6.0], [jnp.inf]])
    ids = jnp.array([1, -1, 1, 3])
    result = jax.jit(stable_segment_mean, static_argnums=2)(x, ids, 3)
    np.testing.assert_array_equal(result, [[0], [4], [0]])
    assert stable_segment_mean(jnp.empty((0, 2)), jnp.empty(0, dtype=int), 3).shape == (
        3,
        2,
    )


def test_segment_mean_derivative_matches_finite_differences():
    ids = jnp.array([1, 0, 1, -1, 5, 0])
    x = jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 7
    direction = jnp.asarray(
        np.random.default_rng(9).normal(size=x.shape), dtype=x.dtype
    )

    def objective(v):
        return jnp.square(stable_segment_mean(v, ids, 3)).sum()

    gradient = jax.jit(jax.grad(objective))(x)
    eps = 1e-3
    finite_difference = (
        objective(x + eps * direction) - objective(x - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), finite_difference, rtol=2e-3, atol=2e-4
    )
    np.testing.assert_array_equal(gradient[jnp.array([3, 4])], np.zeros((2, 2)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_same_input_repeats_are_identical(dtype):
    rng = np.random.default_rng(99)
    ids = jnp.asarray(np.repeat(np.arange(128, dtype=np.int32), 14))
    x = jnp.asarray(rng.normal(size=(len(ids), 128)), dtype=dtype)
    run = jax.jit(lambda v: stable_segment_mean(v, ids, 128))
    first = np.asarray(run(x))
    for _ in range(10):
        np.testing.assert_array_equal(np.asarray(run(x)), first)
