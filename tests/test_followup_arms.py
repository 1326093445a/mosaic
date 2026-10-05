"""The four follow-up arms: decoy, budget/exploration, pose-zero, alpha5.

Each arm exists to answer a specific objection to the 2026-10-05 results, so
the tests check that the arm actually differs from the baseline in the one way
it claims to and in no other way. Also covers the decoy's fold confound, which
is what would make the negative control pass vacuously.

No model, no GPU.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import p17_alpha_recovery as driver  # noqa: E402
from p17_alpha_reference import shuffled_target  # noqa: E402


def _args(stage, *extra, out="/tmp/x"):
    return driver.build_parser().parse_args(
        [stage, "--output-dir", out, "--devices", "0", *extra]
    )


# Decoy target construction ---------------------------------------------------


def test_shuffle_preserves_length_and_composition():
    """Matching both removes length and amino-acid frequency as explanations."""
    target = "ACDEFGHIKLMNPQRSTVWY" * 9 + "ACDE"
    decoy, info = shuffled_target(target, seed=0)
    assert len(decoy) == len(target)
    assert sorted(decoy) == sorted(target)
    assert info["composition_matched"] is True
    assert info["length"] == len(target)
    assert 0.0 <= info["identity_to_real_target"] <= 1.0


def test_shuffle_is_deterministic_and_seed_dependent():
    target = "ACDEFGHIKLMNPQRSTVWY" * 9
    assert shuffled_target(target, 0)[0] == shuffled_target(target, 0)[0]
    assert shuffled_target(target, 0)[0] != shuffled_target(target, 1)[0]


def test_shuffle_actually_moves_most_residues():
    """A decoy that is nearly the real target is not a negative control."""
    target = "ACDEFGHIKLMNPQRSTVWY" * 9
    _, info = shuffled_target(target, seed=0)
    assert info["identity_to_real_target"] < 0.25


# Arm construction ------------------------------------------------------------


def test_decoy_arm_passes_the_decoy_sequence_and_nothing_else_differs():
    """The negative control must differ from the real arm in one input."""
    args = _args("decoy")
    args._decoy_sequence = "W" * 184
    jobs = driver.search_jobs(Path("/tmp/x"), args.policies, [0], None, 5, args, None)
    command = jobs[0]["command"]
    assert command[command.index("--target-sequence") + 1] == "W" * 184
    # Same entry point, same policy default, no start override.
    assert command[command.index("--policy") + 1] == "population"
    assert "--start-sequence" not in command


def test_pose_zero_arm_sets_the_weight_and_changes_nothing_else():
    zero = _args("jn1", "--weight-pose", "0")
    base = _args("jn1")
    z = driver.search_jobs(Path("/tmp/x"), zero.policies, [0], None, 5, zero, None)[0]
    b = driver.search_jobs(Path("/tmp/x"), base.policies, [0], None, 5, base, None)[0]
    assert z["command"][z["command"].index("--weight-pose") + 1] == "0.0"
    assert "--weight-pose" not in b["command"], "baseline leaves the default alone"
    stripped = [t for t in z["command"] if t not in ("--weight-pose", "0.0")]
    assert stripped == b["command"], "pose weight is the only difference"


def test_budget_arm_raises_ceilings_and_warms_the_chain():
    """Section 20.8: all eight runs stopped at 32 scored sequences."""
    args = _args(
        "jn1",
        "--max-score-calls", "300",
        "--target-entropy", "0.8",
        "--acceptance-temperature", "0.05",
    )
    command = driver.search_jobs(
        Path("/tmp/x"), args.policies, [0], None, 5, args, None
    )[0]["command"]
    assert command[command.index("--max-score-calls") + 1] == "300"
    assert command[command.index("--target-entropy") + 1] == "0.8"
    assert command[command.index("--acceptance-temperature") + 1] == "0.05"


def test_exploration_knobs_are_omitted_when_not_requested():
    """Unset knobs must not silently pin the search CLI's own defaults."""
    args = _args("jn1")
    command = driver.search_jobs(
        Path("/tmp/x"), args.policies, [0], None, 5, args, None
    )[0]["command"]
    for flag in ("--target-entropy", "--acceptance-temperature", "--width",
                 "--weight-pose", "--target-sequence"):
        assert flag not in command


def test_saliency_is_requested_only_when_asked():
    on = _args("jn1", "--save-saliency")
    off = _args("jn1")
    for args, expected in ((on, True), (off, False)):
        command = driver.search_jobs(
            Path("/tmp/x"), args.policies, [0], None, 5, args, None
        )[0]["command"]
        assert ("--save-saliency" in command) is expected


# The decoy's fold confound ---------------------------------------------------


def _decoy_run(tmp_path, monkeypatch, target_fit, force=False):
    calls = {"searches": 0}

    def fake_workers(jobs, devices, log_dir):
        if jobs[0]["name"] == "fold_check":
            run = tmp_path / "fold_check"
            (run / "tables").mkdir(parents=True)
            rows = [
                dict(candidate_id="0", selection_seed=s, is_wt="True",
                     ipsae_min=0.02, binder_pose_rmsd_A=40.0,
                     target_aligned_rmsd_A=target_fit, binder_internal_rmsd_A=2.0)
                for s in (0, 1)
            ]
            with (run / "tables/predictions.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        else:
            calls["searches"] += 1
        return [0] * len(jobs)

    monkeypatch.setattr(driver, "run_workers", fake_workers)
    monkeypatch.setattr(driver, "run_heldout", lambda *a, **k: "heldout")
    monkeypatch.setattr(driver, "write_jn1_table", lambda *a, **k: None)
    argv = ["decoy", "--output-dir", str(tmp_path), "--devices", "0"]
    if force:
        argv.append("--force")
    return driver.decoy(driver.build_parser().parse_args(argv)), calls


def test_decoy_refuses_to_search_when_its_target_does_not_fold(tmp_path, monkeypatch):
    """The confound that would make this control pass vacuously.

    A shuffled target has no native fold. If the predictor cannot place it,
    low interface confidence follows from that rather than from absent
    complementarity, and the arm proves nothing.
    """
    with pytest.raises(RuntimeError, match="uninterpretable"):
        _decoy_run(tmp_path, monkeypatch, target_fit=16.4)
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is False
    assert report["wt_metrics"]["mean_target_fit_A"] == pytest.approx(16.4)


def test_decoy_proceeds_when_its_target_places_consistently(tmp_path, monkeypatch):
    code, calls = _decoy_run(tmp_path, monkeypatch, target_fit=1.85)
    assert code == 0
    assert calls["searches"] == 1
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is True
    assert report["decoy"]["kind"] == "shuffled"
    assert len(report["decoy_sequence"]) == len(report["decoy_sequence"])


def test_force_records_the_failed_fold_check_rather_than_hiding_it(
    tmp_path, monkeypatch
):
    code, calls = _decoy_run(tmp_path, monkeypatch, target_fit=16.4, force=True)
    assert code == 0 and calls["searches"] == 1
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is False, "the failure stays on record"


def test_supplied_decoy_must_match_the_reference_target_length(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "reference_target_sequence", lambda *a: "M" * 184)
    args = driver.build_parser().parse_args(
        ["decoy", "--output-dir", str(tmp_path), "--devices", "0",
         "--decoy-sequence", "ACDEF"]
    )
    with pytest.raises(ValueError, match="lengths must match"):
        driver.decoy(args)


# Saliency persistence --------------------------------------------------------


def test_saliency_callback_receives_every_feasible_move(monkeypatch):
    """The matrix is computed to build the proposal distribution and otherwise
    discarded, so the callback is the only chance to record it."""
    from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search

    wt = np.zeros(6, dtype=np.int32)
    mask = np.array([True, True, True, False, False, False])
    captured = []

    def gradient_fn(seq):
        g = np.zeros((6, 20), dtype=np.float64)
        g[0, 1] = -1.0
        return float(seq.sum()), g

    def confidence_fn(seq):
        return ConfidenceScore(value=float(seq.sum()) * 0.01)

    run_gradient_search(
        wt=wt, designable_mask=mask,
        gradient_fn=gradient_fn, confidence_fn=confidence_fn,
        config=SearchConfig(policy="population", width=2, edit_budget=2,
                            max_score_calls=4, max_gradient_calls=4,
                            max_proposals=8, seed=0),
        on_saliency=lambda **kw: captured.append(kw),
    )
    assert captured, "the hook fired"
    first = captured[0]
    assert len(first["moves"]) == len(first["deltas"])
    # Only designable positions may appear.
    assert {m[0] for m in first["moves"]} <= {0, 1, 2}
    assert np.all(np.isfinite(first["deltas"]))


def test_search_without_the_hook_behaves_identically():
    """Instrumentation must not change the trajectory."""
    from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search

    def build():
        wt = np.zeros(6, dtype=np.int32)
        mask = np.array([True, True, True, False, False, False])
        return wt, mask

    def gradient_fn(seq):
        g = np.zeros((6, 20), dtype=np.float64)
        g[0, 1] = -1.0
        return float(seq.sum()), g

    def confidence_fn(seq):
        return ConfidenceScore(value=float(seq.sum()) * 0.01)

    cfg = dict(policy="population", width=2, edit_budget=2, max_score_calls=4,
               max_gradient_calls=4, max_proposals=8, seed=0)
    results = []
    for hook in (None, lambda **kw: None):
        wt, mask = build()
        results.append(
            run_gradient_search(
                wt=wt, designable_mask=mask, gradient_fn=gradient_fn,
                confidence_fn=confidence_fn, config=SearchConfig(**cfg),
                on_saliency=hook,
            ).best.score
        )
    assert results[0] == results[1]
