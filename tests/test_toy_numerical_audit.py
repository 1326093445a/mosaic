"""Independent numerical checks for the toy audit's interpretation metrics."""

import importlib
import json
import sys
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


def test_cpu_plan_defaults_to_one_seed_and_two_serial_probes(
    compare, monkeypatch, capsys
):
    audit = importlib.import_module(compare.__module__)
    monkeypatch.setattr(sys, "argv", ["audit", "--cpu", "--dry-run"])
    assert audit.main() == 0
    output = capsys.readouterr().out
    assert "CPU (serial)" in output
    jobs = json.loads(output[output.index("[\n") :])
    assert [(j["probe"], j["seed"]) for j in jobs] == [
        ("coordinate", 0),
        ("confidence", 0),
    ]


def test_cpu_cannot_silently_accept_gpu_selection(compare, monkeypatch):
    audit = importlib.import_module(compare.__module__)
    monkeypatch.setattr(sys, "argv", ["audit", "--cpu", "--devices", "0", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2


def test_fingerprints_include_values_shapes_and_types(compare):
    audit = importlib.import_module(compare.__module__)
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    digest = audit.fingerprint_arrays({"x": original})["sha256"]
    assert digest == audit.fingerprint_arrays({"x": original.copy()})["sha256"]
    for changed in (original + 1, original.reshape(3, 2), original.astype(np.float64)):
        assert digest != audit.fingerprint_arrays({"x": changed})["sha256"]


def test_host_seed_resets_all_three_generators(compare):
    import random
    import torch

    audit = importlib.import_module(compare.__module__)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:

        def draw(seed):
            audit.seed_host_generators(seed)
            return random.random(), np.random.random(), float(torch.rand(()))

        assert draw(23) == draw(23)
        assert draw(24) != draw(23)
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def test_float32_identity_nonoverlap_is_not_a_correctness_verdict(compare):
    x, eps = np.float32(1), np.float32(0.001)
    result = compare([x] * 3, [x + eps] * 3, [x - eps] * 3, [1] * 3, float(eps))
    assert result["finite_difference_mean"] == pytest.approx(1, abs=2e-5)
    assert not result["sampled_derivative_ranges_overlap"]


def test_final_scalar_resolution_does_not_exclude_internal_rounding(compare):
    # Exactly representable BF16 values after rounding 1 +/- 0.01, then
    # converting back to FP32. The final-scalar ULP misses this quantization.
    result = compare([1] * 3, [1.0078125] * 3, [0.98828125] * 3, [1] * 3, 0.01)
    assert result["signal_exceeds_3x_observed_resolution"]
    assert result["finite_difference_mean"] == 0.9765625
    assert result["absolute_derivative_gap"] > 0.02


@pytest.mark.parametrize("changed", [False, True])
def test_observed_repeatability_does_not_certify_derivatives(compare, changed):
    audit = importlib.import_module(compare.__module__)
    eps = 0.125
    comparison = compare(
        [4] * 3, [(2 + eps) ** 2] * 3, [(2 - eps) ** 2] * 3, [4] * 3, eps
    )
    gradients = np.ones((3, 2, 3))
    if changed:
        gradients[1, 0, 0] += 0.01
    evidence = audit.summarize_numerical_evidence(
        [4] * 3, [4] * 3, gradients, [comparison]
    )
    repeats = evidence["repeatability"]
    assert repeats["forward_losses_identical"]
    assert repeats["backward_losses_identical"]
    assert repeats["gradients_identical"] is (not changed)
    agreement = evidence["derivative_agreement"]
    assert agreement["min_absolute_derivative_gap"] == 0
    assert agreement["max_absolute_derivative_gap"] == 0
    assert agreement["status"] == "not_assessed"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_batch_completion_is_separate_from_validation(
    compare, monkeypatch, tmp_path, exit_code
):
    audit = importlib.import_module(compare.__module__)

    class FinishedWorker:
        pid = 123

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return exit_code

    # Exercise parent reporting only; no model or checkpoint is loaded.
    monkeypatch.setattr(audit.subprocess, "Popen", FinishedWorker)
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--cpu", "--output", str(output)])
    assert audit.main() == exit_code
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["audit_completed"] is (exit_code == 0)
    assert summary["gradient_validation_status"] == "not_assessed"
    assert "gradient_validated" not in summary
    assert summary["derivative_agreement"]["status"] == "not_assessed"
