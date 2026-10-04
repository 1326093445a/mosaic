"""Decision logic for the P17+Alpha recovery control driver.

The expensive stages are stubbed. What matters here is what the driver
*decides*: which damaged rung becomes the starting point, when it refuses to
proceed, and how recovery is reported against the run's own reference -- all
of which gate a multi-day GPU run.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import p17_alpha_recovery as driver  # noqa: E402


def write_predictions(run_dir, per_seed):
    (run_dir / "tables").mkdir(parents=True)
    rows = [
        dict(
            candidate_id="0",
            selection_seed=seed,
            ipsae_min=ipsae,
            binder_pose_rmsd_A=pose,
            target_aligned_rmsd_A=1.8,
            binder_internal_rmsd_A=2.0,
        )
        for seed, ipsae, pose in per_seed
    ]
    with (run_dir / "tables/predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_start_metrics_aggregate_over_seeds(tmp_path):
    write_predictions(tmp_path, [(0, 0.80, 2.0), (1, 0.70, 3.0)])
    metrics = driver.read_start_metrics(tmp_path)
    assert metrics["seeds"] == 2
    assert metrics["mean_ipsae"] == pytest.approx(0.75)
    assert metrics["worst_ipsae"] == pytest.approx(0.70), "worst confidence is the min"
    assert metrics["worst_pose_rmsd_A"] == pytest.approx(3.0), "worst pose is the max"


def test_start_metrics_ignore_non_start_candidates(tmp_path):
    """Only candidate 0 is the run's own start point."""
    write_predictions(tmp_path, [(0, 0.80, 2.0)])
    path = tmp_path / "tables/predictions.csv"
    rows = list(csv.DictReader(path.open()))
    rows.append(dict(rows[0], candidate_id="7", ipsae_min="0.01"))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert driver.read_start_metrics(tmp_path)["mean_ipsae"] == pytest.approx(0.80)


def test_start_metrics_absent_table_is_none_not_a_crash(tmp_path):
    assert driver.read_start_metrics(tmp_path / "missing") is None


def test_score_command_pins_the_alpha_reference_and_contact_epitope(tmp_path):
    command = driver.score_command(tmp_path, None, 64, "bf16")
    assert "--complex" in command and "P17_Alpha.pdb" in " ".join(command)
    assert command[command.index("--epitope-mode") + 1] == "contact"
    assert command[command.index("--sampling-steps") + 1] == "64"
    assert "--start-sequence" not in command, "the reference scores its own sequence"

    damaged = driver.score_command(tmp_path, "ACDE", 64, "bf16")
    assert damaged[damaged.index("--start-sequence") + 1] == "ACDE"


def _ladder(tmp_path, counts=(2, 5)):
    payload = dict(
        rungs=[
            dict(n_edits=c, sequence="A" * 123, substitutions=[], hamming_from_wt=c)
            for c in counts
        ]
    )
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(payload))
    return path


def _calibrate(tmp_path, monkeypatch, scores):
    """Run `calibrate` with the GPU stage replaced by fixed scores."""
    root = tmp_path / "calibrate"

    def fake_workers(jobs, devices, log_dir):
        for job in jobs:
            run_dir = root / job["name"]
            write_predictions(run_dir, scores[job["name"]])
        return [0] * len(jobs)

    monkeypatch.setattr(driver, "run_workers", fake_workers)
    args = driver.build_parser().parse_args(
        [
            "calibrate",
            "--ladder", str(_ladder(tmp_path)),
            "--output-dir", str(root),
            "--devices", "0",
        ]
    )
    code = driver.calibrate(args)
    return code, json.loads((root / "calibration.json").read_text())


def test_calibration_recommends_the_mildest_damaged_rung(tmp_path, monkeypatch):
    """Recovery should start from the gentlest damage that actually worked."""
    code, payload = _calibrate(
        tmp_path,
        monkeypatch,
        {
            "reference": [(0, 0.80, 2.0), (1, 0.79, 2.5)],
            "damaged_02": [(0, 0.05, 30.0), (1, 0.03, 35.0)],
            "damaged_05": [(0, 0.01, 40.0), (1, 0.00, 45.0)],
        },
    )
    assert code == 0
    assert payload["recommended_rung"] == 2
    assert all(r["reached_non_binding"] for r in payload["rungs"])
    assert payload["rungs"][0]["ipsae_drop_from_reference"] == pytest.approx(0.755)


def test_a_rung_that_still_scores_like_a_binder_is_not_recommended(
    tmp_path, monkeypatch
):
    code, payload = _calibrate(
        tmp_path,
        monkeypatch,
        {
            "reference": [(0, 0.80, 2.0), (1, 0.79, 2.5)],
            "damaged_02": [(0, 0.70, 3.0), (1, 0.68, 3.2)],
            "damaged_05": [(0, 0.02, 40.0), (1, 0.01, 45.0)],
        },
    )
    assert code == 0
    assert payload["rungs"][0]["reached_non_binding"] is False
    assert payload["recommended_rung"] == 5, "skips the rung that is still binder-like"


def test_calibration_refuses_when_no_rung_reached_the_non_binding_regime(
    tmp_path, monkeypatch
):
    """Recovery from a starting point that still scores like a binder is not
    a recovery measurement, so this must stop rather than pick the best of a
    bad set."""
    code, payload = _calibrate(
        tmp_path,
        monkeypatch,
        {
            "reference": [(0, 0.80, 2.0)],
            "damaged_02": [(0, 0.70, 3.0)],
            "damaged_05": [(0, 0.60, 4.0)],
        },
    )
    assert code == 3
    assert payload["recommended_rung"] is None


def test_calibration_fails_loudly_if_the_reference_itself_did_not_score(
    tmp_path, monkeypatch
):
    """Without the reference there is no recovery target to measure against."""
    root = tmp_path / "calibrate"

    def fake_workers(jobs, devices, log_dir):
        codes = []
        for job in jobs:
            if job["name"] == "reference":
                codes.append(1)
                continue
            write_predictions(root / job["name"], [(0, 0.02, 40.0)])
            codes.append(0)
        return codes

    monkeypatch.setattr(driver, "run_workers", fake_workers)
    args = driver.build_parser().parse_args(
        [
            "calibrate",
            "--ladder", str(_ladder(tmp_path)),
            "--output-dir", str(root),
            "--devices", "0",
        ]
    )
    with pytest.raises(RuntimeError, match="undamaged reference failed"):
        driver.calibrate(args)


def test_search_refuses_without_a_usable_rung(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                reference=dict(mean_ipsae=0.8),
                rungs=[dict(n_edits=2, scored=True, mean_ipsae=0.7)],
                recommended_rung=None,
            )
        )
    )
    args = driver.build_parser().parse_args(
        [
            "search",
            "--ladder", str(_ladder(tmp_path)),
            "--calibration", str(calibration),
            "--output-dir", str(tmp_path / "out"),
            "--devices", "0",
        ]
    )
    with pytest.raises(RuntimeError, match="non-binding regime"):
        driver.search(args)


def test_search_refuses_a_rung_absent_from_the_ladder(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                reference=dict(mean_ipsae=0.8),
                rungs=[dict(n_edits=99, scored=True, mean_ipsae=0.01)],
                recommended_rung=99,
            )
        )
    )
    args = driver.build_parser().parse_args(
        [
            "search",
            "--ladder", str(_ladder(tmp_path)),
            "--calibration", str(calibration),
            "--output-dir", str(tmp_path / "out"),
            "--devices", "0",
        ]
    )
    with pytest.raises(RuntimeError, match="not in the ladder"):
        driver.search(args)
