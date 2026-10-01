"""Run existing synthetic component tests with device and memory reporting."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/test_opendde_numerics.py",
    "tests/test_binder_pose_rmsd.py",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def memory_snapshot(devices):
    result = {}
    for device in devices:
        try:
            stats = device.memory_stats()
            result[str(device)] = {"supported": stats is not None, "stats": stats}
        except Exception as exc:
            result[str(device)] = {"supported": False, "error": str(exc)}
    return result


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic component validation; no weights, predictions, or search. "
        "Memory measurements do not represent full-model backward memory."
    )
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--device", default="0", help="One CUDA device ID (default: 0)")
    backend.add_argument("--cpu", action="store_true", help="Run CPU checks instead")
    parser.add_argument("--output", type=Path, help="New output directory")
    args = parser.parse_args()
    if not args.cpu and not args.device.isdecimal():
        parser.error("--device must be one nonnegative CUDA device index")

    os.chdir(REPO)
    os.environ["JAX_PLATFORMS"] = "cpu" if args.cpu else "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Keep this run limited to the explicitly listed tests and installed pytest.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    sys.path.insert(0, str(REPO / "src"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (args.output or REPO / "results" / f"synthetic_validation_{stamp}_{os.getpid()}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "reports").mkdir()
    config = {
        "scope": "synthetic component tests only",
        "full_model_backward_validated": False,
        "memory_note": "JAX process counters only; peaks are cumulative and exclude "
        "failed allocations and allocations by other libraries.",
        "python": sys.version,
        "executable": sys.executable,
        "tests": TESTS,
        "requested_backend": os.environ["JAX_PLATFORMS"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True
        ).stdout,
    }
    write_json(root / "config.json", config)
    summary = {"scope": config["scope"], "passed": False, "exit_code": 1}
    devices = []
    start = time.monotonic()
    print(f"Repo: {REPO}\nOutput: {root}\nScope: {config['scope']}", flush=True)
    with (root / "logs" / "validation.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                import jax
                import pytest

                devices = jax.devices()
                expected = "cpu" if args.cpu else "gpu"
                if not devices or any(d.platform != expected for d in devices):
                    raise RuntimeError(f"Expected {expected}, received {devices}")
                config.update(jax_version=jax.__version__, devices=[str(d) for d in devices])
                write_json(root / "config.json", config)
                summary["memory_before"] = memory_snapshot(devices)
                print(f"Devices: {devices}")
                code = int(pytest.main([
                    "-q", "-o", "addopts=", "--confcutdir", str(REPO / "tests"),
                    f"--junitxml={root / 'reports' / 'tests.xml'}", *TESTS,
                ]))
                jax.effects_barrier()
                summary.update(exit_code=code, passed=code == 0)
            except Exception as exc:
                summary.update(exit_code=1, passed=False, error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                summary["memory_after"] = memory_snapshot(devices)
                summary["elapsed_seconds"] = time.monotonic() - start
                write_json(root / "summary.json", summary)
    print(f"Component checks {'passed' if summary['passed'] else 'failed'}: {root}")
    print("Full-model gradients, backward memory, and structure quality remain unvalidated.")
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
