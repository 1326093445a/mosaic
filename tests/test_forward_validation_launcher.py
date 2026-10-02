"""Launcher preflight and exit handling, without model or GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

import gemmi
import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "examples/run_p17_forward_validation.sh"


def invoke(*args, cwd=REPO, env=None):
    return subprocess.run(
        ["bash", str(LAUNCHER), *map(str, args)],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_real_cif_dry_run_preserves_caller_paths_and_creates_no_output(tmp_path):
    reference = tmp_path / "input complex.cif"
    gemmi.read_structure(str(REPO / "P17_JN1.pdb")).make_mmcif_document().write_file(
        str(reference)
    )
    result = invoke(
        "--reference", reference.name,
        "--output-dir", "new output",
        "--devices", "2,5",
        "--dry-run",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "10 workers, 20 predictions; GPUs 2,5" in result.stdout
    assert str(reference) in result.stdout
    assert str(tmp_path / "new output") in result.stdout
    assert not (tmp_path / "new output").exists()


@pytest.mark.parametrize("problem", ["missing_reference", "existing_output"])
def test_preflight_rejects_invalid_paths_before_starting_workers(tmp_path, problem):
    output = tmp_path / "output"
    reference = REPO / "P17_JN1.pdb"
    if problem == "missing_reference":
        reference = tmp_path / "absent.cif"
    else:
        output.mkdir()
    result = invoke("--reference", reference, "--output-dir", output, "--dry-run")
    assert result.returncode != 0
    assert "WT-only:" not in result.stdout
    assert not (output / "plan.json").exists()


@pytest.fixture
def interpreter(tmp_path):
    """Replace only the model runner; execute real preflight/postflight Python."""
    path = tmp_path / "fixture-python"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('/p17_wt_validation.py'):\n"
        "    if os.environ.get('RUNNER_FAILURE'):\n"
        "        raise SystemExit(17)\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "    root.mkdir()\n"
        "    good = os.environ.get('BAD_GEOMETRY') != '1'\n"
        "    mapping = os.environ.get('BAD_MAPPING') != '1'\n"
        "    report = dict(completed=True, geometry_passed=good,\n"
        "        rows=[dict(completed=True, geometry_passed=good, mapping_agrees=mapping)])\n"
        "    (root / 'summary.json').write_text(json.dumps(report))\n"
        "    raise SystemExit(0)\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "fault,expected", [(None, 0), ("BAD_GEOMETRY", 1), ("BAD_MAPPING", 1), ("RUNNER_FAILURE", 17)]
)
def test_runner_and_geometry_failures_propagate(tmp_path, interpreter, fault, expected):
    output = tmp_path / "result"
    env = {"PYTHON_BIN": str(interpreter)}
    if fault:
        env[fault] = "1"
    result = invoke("--output-dir", output, env=env)
    assert result.returncode == expected, result.stderr
    if fault != "RUNNER_FAILURE":
        assert json.loads((output / "summary.json").read_text())["completed"]
    if fault in ("BAD_GEOMETRY", "BAD_MAPPING"):
        assert "Forward validation failed" in result.stderr
