"""The shape contract between `p17_term_gradients.py` and `SetPositions`.

The §26.5 instrument wraps each term in `SetPositions` to differentiate it the
way the real objective is differentiated. `SetPositions.__call__` asserts its
input is `(len(variable_positions), 20)` -- ONLY the variable rows -- and the
instrument was handing it the full-length one-hot, so every invocation died on
that assertion after loading a 656M-parameter checkpoint. Nothing caught it,
because the failure needs a GPU the assertion never reaches.

These tests reach it without one: a trivial loss term, a tiny sequence, and
the same wrapping the instrument does.
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from mosaic.common import TOKENS, LossTerm  # noqa: E402
from mosaic.losses.transformations import SetPositions  # noqa: E402

import p17_term_gradients as tg  # noqa: E402

LENGTH = 12
DESIGNABLE = [3, 5, 7, 9]


class WeightedSum(LossTerm):
    """A stand-in with a known, position-dependent gradient."""

    weights: jnp.ndarray

    def __call__(self, seq, *, key):
        return (seq * self.weights).sum(), {}


def problem():
    rng = np.random.default_rng(0)
    wildtype = jnp.asarray(rng.integers(0, len(TOKENS), LENGTH), dtype=jnp.int32)
    designable = jnp.asarray(DESIGNABLE, dtype=jnp.int32)
    weights = jnp.asarray(rng.normal(size=(LENGTH, len(TOKENS))))
    return wildtype, designable, weights


def wrapped():
    wildtype, designable, weights = problem()
    return SetPositions(wildtype, designable, WeightedSum(weights)), wildtype, designable


def test_setpositions_rejects_the_full_length_one_hot(tmp_path):
    """The exact failure: this is what the instrument used to pass."""
    restricted, wildtype, _ = wrapped()
    full = jax.nn.one_hot(wildtype, len(TOKENS))
    assert full.shape == (LENGTH, len(TOKENS))
    with pytest.raises(AssertionError):
        restricted(full, key=jax.random.key(0))


def test_setpositions_takes_the_designable_rows_and_returns_their_gradient():
    restricted, wildtype, designable = wrapped()
    x = jax.nn.one_hot(wildtype, len(TOKENS))[designable]
    assert x.shape == (len(DESIGNABLE), len(TOKENS))
    fn = eqx.filter_jit(eqx.filter_value_and_grad(restricted, has_aux=True))
    (value, _), grad = fn(x, key=jax.random.key(0))
    assert np.isfinite(float(value))
    # one row per designable position, and no row for a fixed one
    assert np.asarray(grad).shape == (len(DESIGNABLE), len(TOKENS))


def test_the_restricted_gradient_matches_the_designable_rows_of_the_full_one():
    """Restricting the INPUT must give the same numbers as slicing the output.

    This is what makes the reported norms comparable to the search's: the
    search differentiates through SetPositions too.
    """
    wildtype, designable, weights = problem()
    restricted = SetPositions(wildtype, designable, WeightedSum(weights))
    x = jax.nn.one_hot(wildtype, len(TOKENS))[designable]
    _, grad = eqx.filter_value_and_grad(restricted, has_aux=True)(
        x, key=jax.random.key(0)
    )
    # the same term differentiated at full length, then sliced
    full_term = WeightedSum(weights)
    _, full_grad = eqx.filter_value_and_grad(full_term, has_aux=True)(
        jax.nn.one_hot(wildtype, len(TOKENS)), key=jax.random.key(0)
    )
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(full_grad)[np.asarray(DESIGNABLE)], rtol=1e-6
    )


# --------------------------------------------------------------------------
# the reporting helpers, which now receive an already-restricted gradient
# --------------------------------------------------------------------------

def test_designable_norm_is_the_norm_of_what_it_is_given():
    grad = np.arange(12, dtype=float).reshape(4, 3)
    assert tg.designable_norm(grad) == pytest.approx(np.linalg.norm(grad))


def test_cosine_is_computed_over_the_whole_restricted_gradient():
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert tg.cosine(a, a) == pytest.approx(1.0)
    assert tg.cosine(a, b) == pytest.approx(0.0)
    assert tg.cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_refuses_a_zero_gradient_rather_than_dividing_by_zero():
    a = np.array([[1.0, 0.0]])
    assert tg.cosine(a, np.zeros_like(a)) is None
    assert tg.cosine(np.zeros_like(a), a) is None
