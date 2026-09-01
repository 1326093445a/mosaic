"""Unit tests for the Phase 2 guidance-diagnostics aggregation in
src/mosaic/diagnostics.py (docs/legacy/guidance_implementation_todo.md Phase 2).

All synthetic: hand-constructed diagnostics dicts shaped like what
guided_partial_diffusion(..., return_diagnostics=True) actually returns
(leading num_steps axis, batch axis of size 1, (M, 3) atom/coordinate
axes), so these run with no BoltzGen model, checkpoint, or GPU.
"""
import json

import numpy as np
import pytest

from mosaic.legacy.diagnostics import (
    per_step_metrics,
    summarize,
    correlate_with_outcomes,
    write_report,
    format_summary_text,
)

STEPS, M = 5, 4


def _diag(**overrides):
    zeros = np.zeros((STEPS, 1, M, 3))
    d = {
        "unguided_direction": zeros.copy(),
        "guided_direction": zeros.copy(),
        "unguided_step_delta": zeros.copy(),
        "guided_step_delta": zeros.copy(),
        "g_bind": zeros.copy(),
        "g_nat": zeros.copy(),
        "g_edit": zeros.copy(),
        "t_hat": np.linspace(2.0, 0.1, STEPS),
        "sigma_t": np.linspace(1.5, 0.05, STEPS),
    }
    d.update(overrides)
    return d


def test_per_step_metrics_identical_directions_give_cosine_one():
    v = np.ones((STEPS, 1, M, 3))
    diag = _diag(unguided_direction=v, guided_direction=v.copy())
    out = per_step_metrics(diag)
    assert np.allclose(out["cos_guided_unguided"], 1.0, atol=1e-5)


def test_per_step_metrics_orthogonal_directions_give_cosine_zero():
    a = np.zeros((STEPS, 1, M, 3))
    a[:, 0, 0, 0] = 1.0
    b = np.zeros((STEPS, 1, M, 3))
    b[:, 0, 0, 1] = 1.0
    diag = _diag(unguided_direction=a, guided_direction=b)
    out = per_step_metrics(diag)
    assert np.allclose(out["cos_guided_unguided"], 0.0, atol=1e-5)


def test_per_step_metrics_norm_ratio_matches_known_scale():
    unguided = np.ones((STEPS, 1, M, 3))
    guided = np.ones((STEPS, 1, M, 3)) * 3.0  # 3x larger displacement
    diag = _diag(unguided_step_delta=unguided, guided_step_delta=guided)
    out = per_step_metrics(diag)
    assert np.allclose(out["norm_ratio"], 3.0, atol=1e-4)


def test_per_step_metrics_cos_bind_nat_is_nan_when_nat_unused():
    diag = _diag()  # g_nat all zero (guidance_fn_nat was None)
    out = per_step_metrics(diag)
    assert np.all(np.isnan(out["cos_bind_nat"]))


def test_per_step_metrics_cos_bind_nat_is_real_when_nat_active():
    g_bind = np.zeros((STEPS, 1, M, 3))
    g_bind[:, 0, 0, 0] = 1.0
    g_nat = np.zeros((STEPS, 1, M, 3))
    g_nat[:, 0, 0, 0] = 1.0  # identical to g_bind -> cos = 1
    diag = _diag(g_bind=g_bind, g_nat=g_nat)
    out = per_step_metrics(diag)
    assert np.allclose(out["cos_bind_nat"], 1.0, atol=1e-5)


def test_per_step_metrics_rejects_batch_size_other_than_one():
    diag = _diag()
    diag["unguided_direction"] = np.zeros((STEPS, 2, M, 3))
    with pytest.raises(ValueError, match="batch size"):
        per_step_metrics(diag)


def test_per_step_metrics_no_designable_keys_without_mask():
    # Backward compat: no atom_partial_mask -> no *_designable keys at all.
    diag = _diag()
    out = per_step_metrics(diag)
    assert "cos_guided_unguided_designable" not in out
    assert "norm_ratio_designable" not in out


def test_per_step_metrics_whole_complex_dilutes_designable_disagreement():
    # Regression guard for the diagnostics-masking bug: 3 of 4 atoms are
    # "frozen" (identical direction -> perfect agreement), 1 atom is
    # "designable" and fully disagrees (opposite direction). The
    # whole-complex cosine should be diluted by the frozen majority; the
    # designable-only cosine should show the real, full disagreement.
    unguided = np.zeros((STEPS, 1, M, 3))
    guided = np.zeros((STEPS, 1, M, 3))
    unguided[:, 0, :3, 0] = 1.0  # atoms 0,1,2: unguided points +x
    guided[:, 0, :3, 0] = 1.0    # atoms 0,1,2: guided agrees exactly
    unguided[:, 0, 3, 0] = 1.0   # atom 3 (designable): unguided points +x
    guided[:, 0, 3, 0] = -1.0    # atom 3 (designable): guided points -x

    diag = _diag(unguided_direction=unguided, guided_direction=guided)
    atom_partial_mask = np.array([0.0, 0.0, 0.0, 1.0])  # only atom 3 designable

    out = per_step_metrics(diag, atom_partial_mask=atom_partial_mask)

    assert np.allclose(out["cos_guided_unguided"], 0.5, atol=1e-5)
    assert np.allclose(out["cos_guided_unguided_designable"], -1.0, atol=1e-5)


def test_per_step_metrics_rejects_mask_shape_mismatch():
    diag = _diag()
    with pytest.raises(ValueError, match="atom_partial_mask"):
        per_step_metrics(diag, atom_partial_mask=np.array([1.0, 1.0]))


def test_per_step_metrics_rejects_all_zero_mask():
    diag = _diag()
    with pytest.raises(ValueError, match="no designable atoms"):
        per_step_metrics(diag, atom_partial_mask=np.zeros(M))


def test_summarize_prefers_designable_metrics_when_present():
    unguided = np.zeros((STEPS, 1, M, 3))
    guided = np.zeros((STEPS, 1, M, 3))
    unguided[:, 0, :3, 0] = 1.0
    guided[:, 0, :3, 0] = 1.0
    unguided[:, 0, 3, 0] = 1.0
    guided[:, 0, 3, 0] = -1.0
    diag = _diag(unguided_direction=unguided, guided_direction=guided)
    atom_partial_mask = np.array([0.0, 0.0, 0.0, 1.0])

    per_step = per_step_metrics(diag, atom_partial_mask=atom_partial_mask)
    summary = summarize(per_step)

    # Primary stat reflects the designable-region (real) disagreement...
    assert np.isclose(summary["mean_cos_guided_unguided"], -1.0, atol=1e-5)
    # ...while the whole-complex (diluted) value is still reported, labeled.
    assert np.isclose(
        summary["mean_cos_guided_unguided_whole_complex"], 0.5, atol=1e-5
    )


def test_summarize_whole_complex_key_present_without_mask_too():
    # Even with no mask, mean_cos_guided_unguided_whole_complex is always
    # present (redundant with mean_cos_guided_unguided in that case, but a
    # stable key across both call styles).
    diag = _diag()
    summary = summarize(per_step_metrics(diag))
    assert summary["mean_cos_guided_unguided_whole_complex"] == pytest.approx(
        summary["mean_cos_guided_unguided"]
    )


def test_summarize_disagreement_fraction():
    # 2 of 5 steps orthogonal (cos=0, below threshold), 3 identical (cos=1)
    unguided = np.ones((STEPS, 1, M, 3))
    guided = np.ones((STEPS, 1, M, 3))
    guided[0:2] = 0.0
    guided[0:2, 0, 0, 1] = 1.0  # orthogonal to unguided's all-ones for those steps
    diag = _diag(unguided_direction=unguided, guided_direction=guided)
    per_step = per_step_metrics(diag)
    summary = summarize(per_step, disagreement_cos_threshold=0.5)
    assert summary["frac_strong_disagreement"] == pytest.approx(2 / 5)


def test_summarize_sigma_bins_partition_all_steps():
    diag = _diag()
    per_step = per_step_metrics(diag)
    summary = summarize(per_step, n_sigma_bins=3)
    total = sum(b["n_steps"] for b in summary["by_sigma_bin"])
    assert total == STEPS


def test_correlate_with_outcomes_detects_perfect_linear_relationship():
    records = [
        {"summary": {"mean_cos_guided_unguided": c, "mean_norm_ratio": 1.0,
                      "frac_strong_disagreement": 0.0},
         "outcome": {"ipsae": c, "rmsd": 1.0}}  # unused, zero-variance
        for c in [0.1, 0.5, 0.9, 1.0]
    ]
    out = correlate_with_outcomes(records)
    assert out["mean_cos_guided_unguided_vs_ipsae"] == pytest.approx(1.0, abs=1e-6)
    assert out["mean_cos_guided_unguided_vs_rmsd"] is None  # zero-variance outcome


def test_correlate_with_outcomes_returns_none_below_three_records():
    records = [
        {"summary": {"mean_cos_guided_unguided": 0.5, "mean_norm_ratio": 1.0,
                      "frac_strong_disagreement": 0.0},
         "outcome": {"ipsae": 0.8, "rmsd": 2.0}},
        {"summary": {"mean_cos_guided_unguided": 0.9, "mean_norm_ratio": 1.2,
                      "frac_strong_disagreement": 0.1},
         "outcome": {"ipsae": 0.6, "rmsd": 3.0}},
    ]
    out = correlate_with_outcomes(records)
    assert out["mean_cos_guided_unguided_vs_ipsae"] is None
    assert out["n_records"] == 2


def test_write_report_round_trips_through_json(tmp_path):
    diag = _diag()
    per_step = per_step_metrics(diag)
    summary = summarize(per_step)
    records = [{"label": "outer_0", "summary": summary, "outcome": None}]
    out_path = tmp_path / "report.json"
    report = write_report(records, out_path)
    reloaded = json.loads(out_path.read_text())
    assert reloaded == report
    assert reloaded["n_trajectories"] == 1
    assert reloaded["n_trajectories_with_outcome"] == 0
    assert reloaded["outcome_correlations"] is None


def test_format_summary_text_contains_key_stats():
    diag = _diag()
    summary = summarize(per_step_metrics(diag))
    text = format_summary_text("outer_0", summary)
    assert "outer_0" in text
    assert "cos(guided,unguided)" in text
    assert "norm_ratio" in text
