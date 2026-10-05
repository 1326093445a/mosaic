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


def optional_deps():
    """The script's own OPTIONAL_DEPS table, as (module, kind, target)."""
    lines = RUNNER.read_text().split("OPTIONAL_DEPS=(", 1)[1].split(")", 1)[0]
    entries = []
    for line in lines.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        module, kind, target = line.strip('"').split(":", 2)
        entries.append((module, kind, target))
    return entries


def importable(module):
    import subprocess

    return subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", f"import {module}"],
        capture_output=True, timeout=120,
    ).returncode == 0


def test_a_dependency_is_skipped_exactly_when_it_is_unusable(tmp_path):
    """Asserted against this host, not against a hardcoded environment.

    The first version of this test assumed the local machine's missing
    packages. On the cluster `esmjfold2` and `jpromera` are installed, so it
    failed there while the script was behaving correctly. What has to hold is
    the conditional: a skip appears iff importing the guarding module fails.
    """
    result = invoke("--dry-run", path_prefix=fake_nvidia_smi(tmp_path))
    assert result.returncode == 0, result.stderr
    plan = result.stdout.split("Would run:")[1]

    entries = optional_deps()
    assert entries, "the table should not be empty"
    # A target can be guarded by more than one module; it is skipped if any of
    # them is unusable.
    unusable = {}
    for module, kind, target in entries:
        unusable[(kind, target)] = unusable.get((kind, target), False) or (
            not importable(module)
        )
    for (kind, target), expected in unusable.items():
        flag = f"--ignore={target}" if kind == "ignore" else target
        assert (flag in plan) is expected, (
            f"{target} should {'be' if expected else 'not be'} skipped here"
        )
        if kind == "ignore":
            assert plan.count(flag) <= 1, "a target must not be skipped twice"

    # test_cache's other three tests pass either way, so the file itself is
    # never ignored wholesale.
    assert "--ignore=tests/test_cache.py " not in plan


def test_a_present_dependency_brings_its_tests_back_with_no_edit(tmp_path):
    """The skips are conditional, not a hardcoded exclusion list.

    test_promera is guarded by two modules, so satisfying only one must not
    bring it back -- that is the cluster's exact situation, where `jpromera`
    imports and `tinyprot.msa` raises for a missing taxonomy database.
    """
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "jpromera.py").write_text("")

    def run_with(stub_path):
        return subprocess.run(
            ["bash", str(RUNNER), "--dry-run"],
            cwd=REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(stub_path),
                "PATH": (
                    f"{fake_nvidia_smi(tmp_path)}{os.pathsep}{os.environ['PATH']}"
                ),
            },
            capture_output=True, text=True, timeout=180,
        )

    half = run_with(stub)
    assert half.returncode == 0, half.stderr
    assert "jpromera present" in half.stdout
    assert "--ignore=tests/test_promera.py" in half.stdout, (
        "tinyprot.msa still guards it, which is the cluster's situation"
    )

    package = stub / "tinyprot"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "msa.py").write_text("")
    both = run_with(stub)
    assert both.returncode == 0, both.stderr
    assert "tinyprot.msa present" in both.stdout
    assert "--ignore=tests/test_promera.py" not in both.stdout


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
