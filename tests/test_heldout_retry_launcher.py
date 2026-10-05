"""The held-out retry launcher: preflight only, no model and no GPU.

Every case stops at --dry-run or at a refusal, so none of them loads a
checkpoint. `nvidia-smi` is stubbed on PATH, because what is under test is how
its output is read rather than what this host happens to have free.
"""

import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "examples/run_p17_heldout_retry.sh"


def stub_smi(tmp_path, *, busy_pids="", absent=False):
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    script = directory / "nvidia-smi"
    body = (
        '      echo "No devices were found"; exit 6 ;;\n'
        if absent
        else f'      printf "%s" {busy_pids!r}; exit 0 ;;\n'
    )
    script.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n  case "$arg" in\n'
        "    --query-compute-apps=pid)\n" + body + "  esac\ndone\nexit 0\n"
    )
    script.chmod(0o755)
    return directory


def invoke(tmp_path, *args, busy_pids="", absent=False):
    env = dict(os.environ)
    env["PATH"] = (
        f"{stub_smi(tmp_path, busy_pids=busy_pids, absent=absent)}"
        f"{os.pathsep}{env['PATH']}"
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), *map(str, args)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )


def make_run(tmp_path, name, *, searches=2, calibration=False):
    run = tmp_path / name
    for index in range(searches):
        directory = run / "search" / f"population_seed{index}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}")
    if searches == 0:
        (run / "search").mkdir(parents=True)
    if calibration:
        (run / "calibrate").mkdir(parents=True)
        (run / "calibrate/calibration.json").write_text("{}")
    return run


# Preconditions --------------------------------------------------------------


def test_run_is_required(tmp_path):
    result = invoke(tmp_path, "--dry-run")
    assert result.returncode == 2
    assert "--run is required" in result.stderr


def test_a_missing_run_directory_is_refused(tmp_path):
    result = invoke(tmp_path, "--run", tmp_path / "nope", "--dry-run")
    assert result.returncode == 2
    assert "No such run directory" in result.stderr


def test_a_run_without_finished_searches_cannot_be_retried(tmp_path):
    """The searches are the precondition, and the driver would only say so
    after loading a model."""
    run = make_run(tmp_path, "decoy", searches=0)
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 2
    assert "needs finished searches" in result.stderr


def test_completed_searches_are_counted(tmp_path):
    run = make_run(tmp_path, "decoy", searches=4)
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "searches: 4 completed" in result.stdout


# Target inference -----------------------------------------------------------


@pytest.mark.parametrize("name", ["decoy", "budget", "posezero"])
def test_non_alpha_arms_are_scored_against_jn1(tmp_path, name):
    run = make_run(tmp_path, name)
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "target:   jn1 (inferred" in result.stdout
    assert "--target  jn1" in result.stdout


def test_an_alpha_arm_infers_alpha_and_finds_its_calibration(tmp_path):
    run = make_run(tmp_path, "alpha5", calibration=True)
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "target:   alpha (inferred" in result.stdout
    assert "--calibration" in result.stdout
    assert "calibration.json" in result.stdout


def test_an_alpha_arm_without_a_calibration_is_refused(tmp_path):
    """The Alpha table needs that run's measured reference and damaged start,
    which cannot be reconstructed here."""
    run = make_run(tmp_path, "alpha5", calibration=False)
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 2
    assert "needs --calibration" in result.stderr


def test_an_explicit_target_overrides_the_inference(tmp_path):
    run = make_run(tmp_path, "decoy", calibration=True)
    result = invoke(
        tmp_path, "--run", run, "--target", "alpha", "--devices", "0", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    assert "target:   alpha" in result.stdout
    assert "inferred" not in result.stdout


def test_an_unknown_target_is_refused(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(
        tmp_path, "--run", run, "--target", "omicron", "--devices", "0", "--dry-run"
    )
    assert result.returncode == 2
    assert "must be jn1 or alpha" in result.stderr


# The stage directory --------------------------------------------------------


def test_the_failed_attempt_is_preserved_by_default(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert "--suffix  _retry" in result.stdout
    assert "the failed attempt is preserved" in result.stdout


def test_the_suffix_can_be_chosen(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(
        tmp_path, "--run", run, "--suffix", "_third", "--devices", "0", "--dry-run"
    )
    assert "--suffix  _third" in result.stdout


def test_a_dry_run_creates_nothing(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(tmp_path, "--run", run, "--devices", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not (run / "logs").exists()
    assert not list(run.glob("heldout*"))


# Occupancy ------------------------------------------------------------------


def test_a_busy_device_stops_the_retry(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(tmp_path, "--run", run, "--devices", "0", busy_pids="99\n")
    assert result.returncode == 2
    assert "already hold a process" in result.stderr
    assert "Applied five OpenDDE patches" not in result.stdout


def test_a_busy_device_can_be_overridden(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(
        tmp_path, "--run", run, "--devices", "0", "--allow-busy-gpus", "--dry-run",
        busy_pids="99\n",
    )
    assert result.returncode == 0, result.stderr
    assert "already hold a process" in result.stderr


def test_an_absent_device_is_fatal_and_not_an_occupancy_problem(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(
        tmp_path, "--run", run, "--devices", "0,5", "--allow-busy-gpus",
        absent=True,
    )
    assert result.returncode == 2
    assert "do not exist on this host" in result.stderr
    assert "already hold a process" not in result.stderr


def test_free_devices_are_reported(tmp_path):
    run = make_run(tmp_path, "decoy")
    result = invoke(tmp_path, "--run", run, "--devices", "0,1,2,3", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "all 4 requested devices exist and are free" in result.stdout


def test_help_names_the_three_contract_failures_it_exists_for():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.count("2026-10-0") >= 3
    assert "--run PATH" in result.stdout
