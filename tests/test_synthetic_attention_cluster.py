"""Scheduler planning and child-process isolation; no GPU or model required."""

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("synthetic_attention_cluster")


def arguments(**overrides):
    values = dict(
        cpu=False,
        devices="0,1,2,3,4,5,6,7",
        seeds=None,
        sizes=None,
        precision_modes=None,
        rounds=None,
        repeats=3,
        torch_backend="match",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_plan_is_bounded_and_has_paired_fresh_processes(runner):
    devices, jobs = runner.make_plan(arguments())
    assert len(devices) == 8
    assert len(jobs) == 48
    groups = {}
    for job in jobs:
        groups.setdefault((job["seed"], job["size"], job["precision_mode"]), []).append(
            job["round"]
        )
    assert len(groups) == 24
    assert all(sorted(rounds) == [0, 1] for rounds in groups.values())
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_cpu_default_is_one_worker(runner):
    devices, jobs = runner.make_plan(arguments(cpu=True))
    assert devices == ["cpu"]
    assert [(j["seed"], j["size"], j["precision_mode"]) for j in jobs] == [
        (0, "small", "strict")
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"devices": "0,0"},
        {"sizes": "unknown"},
        {"seeds": "-1"},
        {"precision_modes": "fp16"},
    ],
)
def test_invalid_plans_rejected(runner, overrides):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        runner.make_plan(arguments(**overrides))


def test_gpu_job_selects_explicit_backend_device_and_unique_output(runner, tmp_path):
    args = arguments()
    _, jobs = runner.make_plan(args)
    cmd = runner.command(jobs[0], tmp_path, "7", args)
    assert cmd[cmd.index("--backend") + 1] == "cuda"
    assert cmd[cmd.index("--device") + 1] == "7"
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "workers" / jobs[0]["name"])
    env = runner.worker_env("7", False)
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_cleanup_stops_only_the_spawned_process_group(runner):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        runner.stop_process(proc)
        assert proc.poll() is not None
        runner.stop_process(proc)  # Idempotent cleanup after completion.
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_preflight_failure_retains_summary_and_archive(runner, monkeypatch, tmp_path):
    import json
    import tarfile

    class FailedPreflight:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(runner.subprocess, "Popen", FailedPreflight)
    args = arguments(cpu=True)
    args.output = tmp_path / "failed_preflight"
    args.dry_run = False
    args.no_archive = False
    args.worker_timeout_minutes = 1
    args.telemetry_seconds = 15
    devices, jobs = runner.make_plan(args)
    assert runner.run(args, devices, jobs) == 1
    summary = json.loads((args.output / "summary.json").read_text())
    assert not summary["batch_completed"]
    assert not summary["all_synthetic_controls_passed"]
    assert "Preflight failed" in summary["failure"]
    archive = args.output.with_name(args.output.name + ".tar.gz")
    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert "failed_preflight/metadata/runner_error.txt" in names
    assert "failed_preflight/READ_ME.txt" in names
    assert "failed_preflight/checksums.sha256" in names


def test_cpu_reference_keeps_jax_on_requested_gpu(runner, tmp_path):
    args = arguments(torch_backend="cpu")
    _, jobs = runner.make_plan(args)
    cmd = runner.command(jobs[0], tmp_path, "7", args)
    assert cmd[cmd.index("--backend") + 1] == "cuda"
    assert cmd[cmd.index("--torch-backend") + 1] == "cpu"
    assert cmd[cmd.index("--device") + 1] == "7"
    assert runner.effective_torch_backend(False, "cpu") == "cpu"
    assert runner.effective_torch_backend(False, "match") == "cuda"


@pytest.mark.parametrize(
    "platform,selection,error",
    [
        ("gpu", "cpu", None),
        ("gpu", "match", "CUDA-enabled PyTorch required"),
        ("cpu", "cpu", "Expected one gpu JAX device"),
    ],
)
def test_preflight_policy_never_silently_falls_back(
    runner, monkeypatch, capsys, platform, selection, error
):
    # Test capability-policy branching, not actual GPU execution.
    fake_jax = SimpleNamespace(
        __version__="test", devices=lambda: [SimpleNamespace(platform=platform)]
    )
    fake_torch = SimpleNamespace(
        __version__="test+cpu",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    code = runner.preflight_source(False, selection)
    if error:
        with pytest.raises(RuntimeError, match=error):
            exec(code, {})
    else:
        exec(code, {})
    assert '"torch_cuda_available": false' in capsys.readouterr().out


def test_fixed_device_plan_covers_every_gpu_in_every_round(runner):
    devices, jobs = runner.make_plan(
        arguments(
            fixed_devices=True,
            seeds="0",
            sizes="small",
            precision_modes="strict",
            rounds=5,
        )
    )
    assert len(jobs) == 40
    for device in devices:
        assert sorted(
            j["round"] for j in jobs if j["assigned_device"] == device
        ) == list(range(5))
    assert len({j["name"] for j in jobs}) == 40


def test_busy_device_never_moves_its_repeat_to_another_gpu(runner):
    pending = [
        {"assigned_device": "0", "name": "pinned0"},
        {"assigned_device": "1", "name": "pinned1"},
    ]
    free = ["1", "2"]
    job, device = runner.take_ready_job(pending, free)
    assert (job["name"], device) == ("pinned1", "1")
    assert runner.take_ready_job(pending, free) is None
    assert pending[0]["name"] == "pinned0"
    free.append("0")
    assert runner.take_ready_job(pending, free)[1] == "0"


def test_shell_launcher_preserves_child_failure(tmp_path):
    import os

    interpreter = tmp_path / "failing_interpreter"
    interpreter.write_text("#!/usr/bin/env bash\nexit 17\n")
    interpreter.chmod(0o755)
    launcher = (
        Path(__file__).resolve().parents[1] / "examples/run_synthetic_end_to_end.sh"
    )
    result = subprocess.run(
        ["bash", str(launcher), "--cpu"],
        env={**os.environ, "PYTHON_BIN": str(interpreter)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17
