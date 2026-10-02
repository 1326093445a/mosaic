"""Check the independent analytic reference without importing either backend."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_analytic_input_gradient_against_componentwise_differences(
    monkeypatch, scale, seed
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    numerical = importlib.import_module("synthetic_attention_numerics")
    x, weights, _ = numerical.fixture(seed)
    x = x * scale
    _, _, analytic = numerical.numpy_reference(x, weights)
    estimated = np.empty_like(x)
    step = 1e-5
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        estimated[index] = (
            numerical.numpy_reference(plus, weights)[0]
            - numerical.numpy_reference(minus, weights)[0]
        ) / (2 * step)
    np.testing.assert_allclose(analytic, estimated, atol=2e-8, rtol=2e-6)
