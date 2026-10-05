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


def test_score_command_takes_the_reference_it_is_given(tmp_path):
    command = driver.score_command(
        tmp_path, None, 64, "bf16", driver.ALPHA_COMPLEX_PDB
    )
    assert "--complex" in command and "P17_Alpha.pdb" in " ".join(command)
    assert command[command.index("--epitope-mode") + 1] == "contact"
    assert command[command.index("--sampling-steps") + 1] == "64"
    assert "--start-sequence" not in command, "the reference scores its own sequence"

    damaged = driver.score_command(
        tmp_path, "ACDE", 64, "bf16", driver.ALPHA_COMPLEX_PDB
    )
    assert damaged[damaged.index("--start-sequence") + 1] == "ACDE"


def test_score_command_without_a_reference_leaves_the_script_default(tmp_path):
    """`None` means JN.1, the search script's own reference.

    This used to be impossible to express: the Alpha reference was hardcoded
    here, so the decoy arm's fold check validated a JN.1-length decoy against
    Alpha's longer target and exited 2.
    """
    command = driver.score_command(tmp_path, None, 64, "bf16", None)
    assert "--complex" not in command
    assert "--epitope-mode" not in command


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


# JN.1 stage -------------------------------------------------------------------


def _stub_merge(monkeypatch):
    """Neutralize the shard merge, which needs real shard directories.

    Patches the real module's attribute rather than installing a fake module
    in `sys.modules`: a fake there is not restored between test files and
    broke the rescore tests in `test_confidence_search.py`.
    """
    import p17_rescore_winners

    monkeypatch.setattr(
        p17_rescore_winners, "merge_shard_tables", lambda *a, **k: None
    )


def test_jn1_jobs_omit_the_complex_flag_and_the_start_sequence(tmp_path):
    """Omitting --complex *is* the JN.1 path, and WT is already non-binding."""
    args = driver.build_parser().parse_args(
        ["jn1", "--output-dir", str(tmp_path), "--devices", "0"]
    )
    jobs = driver.search_jobs(
        tmp_path, args.policies, args.search_seeds, None, args.edit_budget, args, None
    )
    for job in jobs:
        command = job["command"]
        assert "--complex" not in command, "JN.1 is the default reference"
        assert "--epitope-mode" not in command, "hotspot epitope applies here"
        assert "--start-sequence" not in command, "WT is the starting point"
        assert command[command.index("--edit-budget") + 1] == "5"


def test_alpha_jobs_pin_the_alpha_reference_and_the_damaged_start(tmp_path):
    args = driver.build_parser().parse_args(
        [
            "search",
            "--ladder", str(tmp_path / "l.json"),
            "--calibration", str(tmp_path / "c.json"),
            "--output-dir", str(tmp_path),
            "--devices", "0",
        ]
    )
    jobs = driver.search_jobs(
        tmp_path, args.policies, [0], "DAMAGED", 5, args, driver.REPO / "P17_Alpha.pdb"
    )
    command = jobs[0]["command"]
    assert "P17_Alpha.pdb" in " ".join(command)
    assert command[command.index("--epitope-mode") + 1] == "contact"
    assert command[command.index("--start-sequence") + 1] == "DAMAGED"


def test_population_is_the_default_policy_for_both_stages(tmp_path):
    """Section 17's standing decision; section 19.4 deprioritizes the contrast."""
    for argv in (
        ["jn1", "--output-dir", str(tmp_path)],
        [
            "search",
            "--ladder", "l.json",
            "--calibration", "c.json",
            "--output-dir", str(tmp_path),
        ],
    ):
        assert driver.build_parser().parse_args(argv).policies == ["population"]


def test_independent_remains_available_when_asked_for(tmp_path):
    args = driver.build_parser().parse_args(
        [
            "jn1",
            "--output-dir", str(tmp_path),
            "--policies", "population", "independent",
        ]
    )
    jobs = driver.search_jobs(
        tmp_path, args.policies, [0], None, 5, args, None
    )
    assert {j["name"] for j in jobs} == {"population_seed0", "independent_seed0"}


def test_default_seeds_spend_the_freed_workers_on_replication(tmp_path):
    """One policy x four seeds, not two policies x two seeds."""
    args = driver.build_parser().parse_args(
        ["jn1", "--output-dir", str(tmp_path)]
    )
    assert args.search_seeds == [0, 1, 2, 3]
    jobs = driver.search_jobs(
        tmp_path, args.policies, args.search_seeds, None, 5, args, None
    )
    assert len(jobs) == 4
    assert all(j["name"].startswith("population_") for j in jobs)


def test_jn1_table_is_relative_to_this_runs_own_wt(tmp_path, monkeypatch):
    """No known ceiling here, so the table reports gains, not fractions."""
    heldout = tmp_path / "heldout"
    (heldout / "tables").mkdir(parents=True)
    rows = []
    for candidate, ipsae, pose in (("0", 0.01, 40.0), ("1", 0.25, 30.0)):
        for seed in driver.HELDOUT_SEEDS:
            rows.append(
                dict(
                    candidate_id=candidate,
                    selection_seed=seed,
                    ipsae_min=ipsae,
                    binder_pose_rmsd_A=pose,
                    target_aligned_rmsd_A=1.85,
                    binder_internal_rmsd_A=2.0,
                )
            )
    with (heldout / "tables/predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _stub_merge(monkeypatch)

    driver.write_jn1_table(tmp_path)
    summary = list(csv.DictReader((tmp_path / "tables/jn1_recovery.csv").open()))
    wt = next(r for r in summary if r["is_wt"] == "True")
    winner = next(r for r in summary if r["is_wt"] == "False")
    assert float(wt["ipsae_gain_over_wt"]) == pytest.approx(0.0)
    assert float(winner["ipsae_gain_over_wt"]) == pytest.approx(0.24)
    assert float(winner["pose_change_vs_wt_A"]) == pytest.approx(-10.0)
    assert "ipsae_recovery_fraction" not in winner, "no known ceiling for JN.1"
    context = json.loads((tmp_path / "tables/jn1_context.json").read_text())
    assert context["wt_baseline"]["mean_ipsae"] == pytest.approx(0.01)
    assert "no solution is known to exist" in context["interpretation"].lower()


def test_jn1_table_requires_the_wt_candidate(tmp_path, monkeypatch):
    heldout = tmp_path / "heldout"
    (heldout / "tables").mkdir(parents=True)
    rows = [
        dict(
            candidate_id="1",
            selection_seed=seed,
            ipsae_min=0.2,
            binder_pose_rmsd_A=30.0,
            target_aligned_rmsd_A=1.85,
            binder_internal_rmsd_A=2.0,
        )
        for seed in driver.HELDOUT_SEEDS
    ]
    with (heldout / "tables/predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _stub_merge(monkeypatch)
    with pytest.raises(ValueError, match="no WT candidate"):
        driver.write_jn1_table(tmp_path)


# Held-out stage, and the reference that broke it on 2026-10-04 ---------------


def test_alpha_heldout_passes_the_alpha_reference_to_rescore(tmp_path):
    """The 2026-10-04 failure: rescore defaults to JN.1 and refused the run.

    It verifies archived binder/target sequences against whatever reference it
    loads. Alpha's target is 195 residues on chain A against JN.1's 184 on
    chain T, so without --complex it correctly refuses rather than scoring the
    wrong complex.
    """
    captured = {}

    def fake_workers(jobs, devices, log_dir):
        captured["jobs"] = jobs
        return [0] * len(jobs)

    import p17_alpha_recovery as d

    original = d.run_workers
    d.run_workers = fake_workers
    try:
        d.run_heldout(tmp_path, ["0"], "bf16", d.REPO / "P17_Alpha.pdb")
    finally:
        d.run_workers = original

    command = captured["jobs"][0]["command"]
    assert "P17_Alpha.pdb" in " ".join(command)
    assert command[command.index("--binder-chain") + 1] == "B"
    assert command[command.index("--target-chain") + 1] == "A"


def test_jn1_heldout_omits_the_complex_flag(tmp_path):
    captured = {}

    def fake_workers(jobs, devices, log_dir):
        captured["jobs"] = jobs
        return [0] * len(jobs)

    import p17_alpha_recovery as d

    original = d.run_workers
    d.run_workers = fake_workers
    try:
        d.run_heldout(tmp_path, ["0"], "bf16", None)
    finally:
        d.run_workers = original

    assert "--complex" not in captured["jobs"][0]["command"]


def test_heldout_failure_says_not_to_repeat_the_search(tmp_path, monkeypatch):
    """The searches are the expensive part and must not be discarded."""
    monkeypatch.setattr(driver, "run_workers", lambda jobs, d, log: [1] * len(jobs))
    with pytest.raises(RuntimeError) as excinfo:
        driver.run_heldout(tmp_path, ["0"], "bf16", None)
    message = str(excinfo.value)
    assert "re-run the `heldout` stage rather than the search" in message
    assert "shard_0" in message


def test_heldout_stage_writes_to_a_retry_directory_by_default(tmp_path):
    """A re-run must not overwrite the failed attempt's shards."""
    args = driver.build_parser().parse_args(
        ["heldout", "--output-dir", str(tmp_path), "--target", "jn1"]
    )
    assert args.suffix == "_retry"
    captured = {}

    def fake_workers(jobs, devices, log_dir):
        captured["jobs"] = jobs
        return [0] * len(jobs)

    import p17_alpha_recovery as d

    original = d.run_workers
    d.run_workers = fake_workers
    try:
        stage = d.run_heldout(tmp_path, ["0"], "bf16", None, "_retry")
    finally:
        d.run_workers = original
    assert stage == "heldout_retry"
    assert "heldout_retry" in captured["jobs"][0]["command"][
        captured["jobs"][0]["command"].index("--output-dir") + 1
    ]


def test_heldout_refuses_when_no_search_runs_exist(tmp_path):
    args = driver.build_parser().parse_args(
        ["heldout", "--output-dir", str(tmp_path), "--target", "jn1"]
    )
    with pytest.raises(RuntimeError, match="no completed search runs"):
        driver.heldout(args)


def test_alpha_heldout_requires_a_calibration_file(tmp_path):
    """Alpha's table is relative to the reference and damaged metrics."""
    with pytest.raises(SystemExit):
        driver.main(
            ["heldout", "--output-dir", str(tmp_path), "--target", "alpha"]
        )
