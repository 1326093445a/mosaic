"""Independent numerical checks for the toy audit's interpretation metrics."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("opendde_toy_numerical_audit").compare_derivatives


def test_exact_quadratic_directional_derivative(compare):
    # f(x)=x*x, x=2; symmetric differences give f'(2)=4 exactly.
    eps = 0.125
    result = compare([4, 4], [(2 + eps) ** 2] * 3, [(2 - eps) ** 2] * 3, [4, 4, 4], eps)
    assert result["finite_difference_mean"] == pytest.approx(4)
    assert result["absolute_derivative_gap"] == pytest.approx(0)
    assert result["signal_exceeds_3x_observed_resolution"]


def test_noisy_overlap_does_not_imply_resolved_signal(compare):
    result = compare(
        [1.0, 1.2, 0.8], [1.01, 1.21, 0.81], [0.99, 1.19, 0.79], [1, 1, 1], 0.01
    )
    assert result["sampled_derivative_ranges_overlap"]
    assert not result["signal_exceeds_3x_observed_resolution"]
    assert result["signal_to_observed_resolution"] < 1


def test_float32_rounding_is_not_zero_noise_floor(compare):
    value = np.float32(8192)
    plus = np.float32(value + 0.0001)
    minus = np.float32(value - 0.0001)
    result = compare([value] * 3, [plus] * 3, [minus] * 3, [0.1] * 3, 0.001)
    assert result["largest_same_input_loss_range"] == 0
    assert result["float32_ulp"] > 0
    assert not result["signal_exceeds_3x_observed_resolution"]
    assert result["absolute_derivative_gap"] > 0


@pytest.mark.parametrize("bad", [[1], [1, np.nan], [1, np.inf]])
def test_invalid_observations_are_rejected(compare, bad):
    with pytest.raises(ValueError):
        compare(bad, [1, 1], [1, 1], [1, 1], 0.01)
