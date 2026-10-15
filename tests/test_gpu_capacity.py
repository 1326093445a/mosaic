"""Runs `tests/test_gpu_capacity_check.sh` so it cannot be forgotten.

The checks themselves are shell, because the thing under test is a sourced
bash function and a stubbed `nvidia-smi`. Wrapping it in pytest is what puts it
in the suite's count -- a standalone script nothing invokes is the same mistake
as the mmCIF fixture that sat unparsed for months (section 31.13).
"""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "test_gpu_capacity_check.sh"


def test_the_gpu_capacity_checks_pass():
    done = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # Not vacuous: the script must actually have run its cases.
    assert "passed, 0 failed" in done.stdout, done.stdout
    assert done.stdout.count("  PASS") >= 10, done.stdout
