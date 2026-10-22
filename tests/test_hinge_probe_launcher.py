"""What `examples/run_p17_hinge_probe.sh` must do before it is trusted.

An earlier version of this launcher reported success after all eight jobs
failed. The fixes below each come from a reproduced failure, so each one is
pinned here rather than left to a dry run that only checks the happy path:

- a dry run wrote `provenance.json` before testing `--dry-run`, so inspecting a
  planned run overwrote the metadata of a finished one;
- the reporter's exit status was discarded, so a reporter that crashed left the
  script exiting 0;
- the reporter globbed `*.json`, picked up `provenance.json`, and announced
  that the script's own metadata had come from obsolete code;
- nothing checked that the eight expected result files existed, so jobs that
  exited clean without writing anything also looked like a clean run.

The seeder is mocked: this is about the launcher's control flow, and the real
seeder needs a GPU and OpenDDE weights.
"""

import os
import subprocess
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "examples" / "run_p17_hinge_probe.sh"
DEVICES = "0,1,2,3,4,5,6,7"
N_RUNS = 8

MOCK = """#!/usr/bin/env bash
if [[ "$1" == "-" ]]; then exec %s "$@"; fi
out=""; method=""; hinge=""; scale=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --weight-edit-budget) hinge="$2"; shift 2 ;;
    --init-logit-scale) scale="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "$MOCK_MODE" in
  ok) printf '{"method":"%%s","weight_edit_budget":%%s,"init_logit_scale":%%s,\
"candidate_substitutions_considered":17,"hamming_from_parent":5}\\n' \
    "$method" "$hinge" "$scale" > "$out" ;;
  silent) exit 0 ;;
  zeros) printf '{"method":"%%s","weight_edit_budget":0.0,"init_logit_scale":0.0,\
"candidate_substitutions_considered":17,"hamming_from_parent":5}\\n' \
    "$method" > "$out" ;;
  corrupt) echo 'not json' > "$out" ;;
  stale) printf '{"method":"%%s"}\\n' "$method" > "$out" ;;
esac
"""


IDLE_EIGHT = """#!/usr/bin/env bash
case "$*" in
  *--query-gpu*) for i in 0 1 2 3 4 5 6 7; do echo "$i, 143771, 1024"; done ;;
  *) exit 1 ;;
esac
"""


def _stub_nvidia_smi(directory, body=IDLE_EIGHT):
    """Return a PATH prefix whose `nvidia-smi` is `body`."""
    binary = directory / "stubbin"
    binary.mkdir(exist_ok=True)
    stub = binary / "nvidia-smi"
    stub.write_text(body)
    stub.chmod(0o755)
    return str(binary)


@pytest.fixture
def probe(tmp_path):
    import sys

    mock = tmp_path / "mockpy"
    mock.write_text(MOCK % sys.executable)
    mock.chmod(0o755)

    def run(*args, mode="ok"):
        env = dict(os.environ, PYTHON_BIN=str(mock), MOCK_MODE=mode)
        env["PATH"] = _stub_nvidia_smi(tmp_path) + os.pathsep + env["PATH"]
        return subprocess.run(
            ["bash", str(PROBE), "--devices", DEVICES, *args],
            capture_output=True, text=True, env=env, timeout=300,
        )

    return run


def test_a_dry_run_creates_nothing(probe, tmp_path):
    out = tmp_path / "fresh"
    done = probe("--dry-run", "--output-dir", str(out))
    assert done.returncode == 0, done.stderr
    assert not out.exists(), "a dry run must not create its output directory"


def test_a_dry_run_does_not_overwrite_an_existing_provenance(probe, tmp_path):
    out = tmp_path / "finished"
    out.mkdir()
    (out / "provenance.json").write_text('{"git_commit": "ORIGINAL"}')
    probe("--dry-run", "--output-dir", str(out))
    assert "ORIGINAL" in (out / "provenance.json").read_text(), (
        "inspecting a planned run destroyed the record of a finished one"
    )


def test_a_real_run_refuses_an_output_directory_that_already_has_provenance(
    probe, tmp_path
):
    out = tmp_path / "finished"
    out.mkdir()
    (out / "provenance.json").write_text('{"git_commit": "ORIGINAL"}')
    done = probe("--output-dir", str(out))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "ORIGINAL" in (out / "provenance.json").read_text()


def test_eight_good_runs_report_eight_rows_and_exit_clean(probe, tmp_path):
    done = probe("--output-dir", str(tmp_path / "good"))
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [
        line for line in done.stdout.splitlines()
        if line.startswith(("  apgm", "  bindcraft", "  colabdesign"))
    ]
    assert len(rows) == N_RUNS, done.stdout
    assert "pre-2026-10-13" not in done.stdout + done.stderr, (
        "the reporter read its own provenance.json as a stale result"
    )


def test_jobs_that_exit_clean_without_writing_anything_still_fail(probe, tmp_path):
    done = probe("--output-dir", str(tmp_path / "silent"), mode="silent")
    assert done.returncode != 0, "a run that produced no results reported success"
    assert f"NO RESULT FILE for {N_RUNS} of {N_RUNS}" in done.stderr


def test_unreadable_results_fail_rather_than_being_skipped(probe, tmp_path):
    done = probe("--output-dir", str(tmp_path / "corrupt"), mode="corrupt")
    assert done.returncode != 0
    assert "unreadable" in done.stderr


def test_results_missing_the_new_keys_are_flagged_and_fail(probe, tmp_path):
    done = probe("--output-dir", str(tmp_path / "stale"), mode="stale")
    assert done.returncode != 0
    assert "pre-2026-10-13" in done.stderr


def test_more_runs_than_devices_is_refused(probe, tmp_path):
    env = dict(os.environ, PYTHON_BIN="/bin/false", MOCK_MODE="ok")
    done = subprocess.run(
        ["bash", str(PROBE), "--devices", "0,1", "--output-dir", str(tmp_path / "few")],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert done.returncode == 2
    assert "Refusing to start" in done.stderr




def test_an_output_directory_holding_other_results_is_refused(probe, tmp_path):
    """Worse than no provenance: another run's outputs taking this commit.

    Result JSONs with no `provenance.json` beside them used to be accepted, and
    the launcher then wrote its own provenance over them.
    """
    out = tmp_path / "someone_elses_results"
    out.mkdir()
    (out / "apgm_h5.0_s5.0.json").write_text('{"method": "apgm"}')
    done = probe("--output-dir", str(out), mode="silent")
    assert done.returncode == 2, done.stdout + done.stderr
    assert not (out / "provenance.json").exists(), (
        "an unrelated run's results acquired this run's provenance"
    )


def test_an_empty_output_directory_is_accepted(probe, tmp_path):
    """The refusal is about contents, not existence -- mkdir -p stays usable."""
    out = tmp_path / "empty"
    out.mkdir()
    done = probe("--output-dir", str(out))
    assert done.returncode == 0, done.stdout + done.stderr


def test_results_that_do_not_match_the_requested_settings_fail(probe, tmp_path):
    """Eight results all reporting hinge 0 are not a hinge contrast.

    The mock writes the settings it was given, so to reproduce this the mock is
    told to ignore them and report zeros instead.
    """
    done = probe("--output-dir", str(tmp_path / "wrong"), mode="zeros")
    assert done.returncode != 0, done.stdout
    assert "SETTINGS MISMATCH" in done.stderr
    assert "weight_edit_budget" in done.stderr


def test_repeated_devices_are_refused(probe, tmp_path):
    import os
    import subprocess

    env = dict(os.environ, PYTHON_BIN="/bin/false", MOCK_MODE="ok")
    done = subprocess.run(
        ["bash", str(PROBE), "--devices", "0,0,0,0,0,0,0,0",
         "--output-dir", str(tmp_path / "same_gpu")],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert "repeats a device" in done.stderr


def test_an_empty_device_entry_is_refused(probe, tmp_path):
    import os
    import subprocess

    env = dict(os.environ, PYTHON_BIN="/bin/false", MOCK_MODE="ok")
    done = subprocess.run(
        ["bash", str(PROBE), "--devices", "0,1,,3,4,5,6,7",
         "--output-dir", str(tmp_path / "blank_dev")],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert done.returncode == 2
    assert "empty or non-numeric" in done.stderr
