"""The test runner's own preflight, without ever invoking pytest recursively.

Every case here either stops at --dry-run or stops at a refusal, so none of
them starts a nested suite. The GPU cases stub `nvidia-smi` on PATH, because
the behaviour under test is how its output is read, not whether this host
happens to have a free device.
"""

import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "run_tests.sh"


def invoke(*args, path_prefix=None):
    env = dict(os.environ)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(RUNNER), *map(str, args)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )


def fake_nvidia_smi(tmp_path, *, busy_pids="", devices="0", absent=False):
    """A stub covering the three queries the runner makes."""
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    script = directory / "nvidia-smi"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    --query-gpu=index)\n"
        f'      printf "%s\\n" {devices!r} | tr "," "\\n"; exit 0 ;;\n'
        "    --query-compute-apps=pid)\n"
        + (
            '      echo "No devices were found"; exit 6 ;;\n'
            if absent
            else f'      printf "%s" {busy_pids!r}; exit 0 ;;\n'
        )
        + "  esac\n"
        "done\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    return directory


def test_the_default_touches_no_gpu_at_all(tmp_path):
    """The point of the CPU default: safe beside a job on a shared node.

    JAX preallocates 75% of every visible device, so a suite that let itself
    see all eight would kill whatever is training on them.
    """
    result = invoke("--dry-run", path_prefix=fake_nvidia_smi(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "cpu, no GPU touched" in result.stdout
    assert "device" not in result.stdout.split("Would run:")[0].lower() or (
        "does not exist" not in result.stdout
    )


def test_gpu_mode_takes_one_named_device_and_does_not_preallocate(tmp_path):
    bare = invoke("--dry-run", "--gpu", path_prefix=fake_nvidia_smi(tmp_path))
    assert bare.returncode == 0, bare.stderr
    assert "gpu 0 only, preallocation off" in bare.stdout

    chosen = invoke(
        "--dry-run", "--gpu", "5",
        path_prefix=fake_nvidia_smi(tmp_path, devices="5"),
    )
    assert chosen.returncode == 0, chosen.stderr
    assert "gpu 5 only, preallocation off" in chosen.stdout


def test_gpu_mode_refuses_a_device_that_is_not_on_the_host(tmp_path):
    result = invoke("--gpu", "3", path_prefix=fake_nvidia_smi(tmp_path, absent=True))
    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert "test session starts" not in result.stdout


def test_dry_run_skips_only_what_absent_dependencies_break(tmp_path):
    """A bare pytest collects nothing here; the skips are what make it run."""
    result = invoke("--dry-run", path_prefix=fake_nvidia_smi(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "--ignore=tests/test_esmfold2_multisample.py" in result.stdout
    assert "--deselect" in result.stdout
    assert "test_esmfold_msa_cache_follows_runtime_override" in result.stdout
    assert "--ignore=tests/test_promera.py" in result.stdout
    # test_cache's other three tests pass, so the file itself stays in.
    assert "--ignore=tests/test_cache.py" not in result.stdout


def test_a_present_dependency_brings_its_tests_back_with_no_edit(tmp_path):
    """The skips are conditional, not a hardcoded exclusion list."""
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "jpromera.py").write_text("")
    env_result = subprocess.run(
        ["bash", str(RUNNER), "--dry-run"],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(stub),
            "PATH": f"{fake_nvidia_smi(tmp_path)}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True, text=True, timeout=120,
    )
    assert env_result.returncode == 0, env_result.stderr
    assert "jpromera present" in env_result.stdout
    assert "--ignore=tests/test_promera.py" not in env_result.stdout


def test_slow_marker_override_is_an_empty_m_not_a_removed_addopts(tmp_path):
    plain = invoke("--dry-run", path_prefix=fake_nvidia_smi(tmp_path))
    slow = invoke("--dry-run", "--slow", path_prefix=fake_nvidia_smi(tmp_path))
    # `python -m pytest` always contains a -m, so the marker flag is the
    # empty-argument form specifically.
    assert "-m  ''" not in plain.stdout
    assert "not slow (pass --slow" in plain.stdout
    assert "-m  ''" in slow.stdout
    assert "including slow" in slow.stdout


def test_arguments_after_the_separator_reach_pytest(tmp_path):
    result = invoke(
        "--dry-run", "--", "tests/test_followup_arms.py", "-k", "decoy",
        path_prefix=fake_nvidia_smi(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    tail = result.stdout.split("Would run:")[1]
    assert "tests/test_followup_arms.py" in tail
    assert "-k  decoy" in tail


def test_a_busy_device_stops_the_run_before_pytest_starts(tmp_path):
    """Measured: 8 failures with the device busy against 2 with it free.

    The refusal has to come before pytest, both so the failures are not
    misread and so the suite does not OOM-kill whatever holds the device.
    Only reachable under --gpu, since the default never asks for a device.
    """
    result = invoke("--gpu", path_prefix=fake_nvidia_smi(tmp_path, busy_pids="4242\n"))
    assert result.returncode == 2
    assert "already holds a process" in result.stderr
    assert "test session starts" not in result.stdout, "pytest must not have run"


def test_a_busy_device_cannot_block_the_cpu_default(tmp_path):
    """A node with every GPU in use must still be able to run the suite."""
    result = invoke(
        "--dry-run", path_prefix=fake_nvidia_smi(tmp_path, busy_pids="4242\n")
    )
    assert result.returncode == 0, result.stderr
    assert "already hold" not in result.stderr
    assert "Would run:" in result.stdout


def test_a_busy_device_is_a_warning_under_the_override(tmp_path):
    result = invoke(
        "--gpu", "--allow-busy-gpus", "--dry-run",
        path_prefix=fake_nvidia_smi(tmp_path, busy_pids="4242\n"),
    )
    assert result.returncode == 0, result.stderr
    assert "already holds a process" in result.stderr
    assert "Would run:" in result.stdout


def test_an_absent_device_is_not_counted_as_busy(tmp_path):
    """nvidia-smi prints to stdout and exits 6 for a device that is not there.

    Counting its output as a process reported every absent device as busy in
    an earlier launcher, so the exit code is what has to be read.
    """
    result = invoke(
        "--dry-run", "--gpu", path_prefix=fake_nvidia_smi(tmp_path, absent=True)
    )
    assert "already holds a process" not in result.stderr


def test_unknown_option_is_refused_rather_than_passed_to_pytest(tmp_path):
    result = invoke("--not-an-option", path_prefix=fake_nvidia_smi(tmp_path))
    assert result.returncode == 2
    assert "Unknown option" in result.stderr


def test_help_explains_the_three_things_it_handles():
    result = invoke("--help")
    assert result.returncode == 0
    for expected in ("esmjfold2", "GPU", "slow", "CPU-only"):
        assert expected in result.stdout
