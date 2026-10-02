"""Bounded multi-GPU runner for synthetic attention numerical diagnostics only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "examples" / "synthetic_attention_stages.py"
SOURCES = [
    WORKER,
    REPO / "examples" / "synthetic_attention_numerics.py",
    Path(__file__).resolve(),
    REPO / "examples" / "run_synthetic_attention_cluster.sh",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def csv_file(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choices(value, allowed=None, numeric=False):
    values = value.split(",")
    if not values or any(not x for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Use distinct comma-separated values")
    if numeric and any(not x.isdecimal() for x in values):
        raise argparse.ArgumentTypeError("Use nonnegative integer indices")
    if allowed and any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError(f"Allowed values: {','.join(allowed)}")
    return values


def make_plan(args):
    devices = ["cpu"] if args.cpu else choices(args.devices, numeric=True)
    seeds = choices(args.seeds or ("0" if args.cpu else "0,1,2,3"), numeric=True)
    sizes = choices(
        args.sizes or ("small" if args.cpu else "small,medium,large"),
        ("small", "medium", "large"),
    )
    modes = choices(
        args.precision_modes or ("strict" if args.cpu else "strict,native"),
        ("strict", "native"),
    )
    rounds = args.rounds if args.rounds is not None else (1 if args.cpu else 2)
    jobs = [
        {
            "name": f"r{round_index:02d}_seed{seed}_{size}_{mode}",
            "round": round_index,
            "seed": int(seed),
            "size": size,
            "precision_mode": mode,
        }
        for round_index in range(rounds)
        for size in sizes
        for seed in seeds
        for mode in modes
    ]
    return devices, jobs


def command(job, root, device, args):
    return [
        sys.executable,
        "-u",
        str(WORKER),
        "--output",
        str(root / "workers" / job["name"]),
        "--backend",
        "cpu" if args.cpu else "cuda",
        "--device",
        str(device),
        "--seed",
        str(job["seed"]),
        "--size",
        job["size"],
        "--precision-mode",
        job["precision_mode"],
        "--repeats",
        str(args.repeats),
    ]


def worker_env(device, cpu):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="" if cpu else str(device),
        JAX_PLATFORMS="cpu" if cpu else "cuda",
        PYTHONUNBUFFERED="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    return env


def stop_process(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def collect(root, statuses):
    comparisons, stage_rows, repeats, control_rows = [], [], [], []
    for job in statuses:
        folder = root / "workers" / job["name"]
        path = folder / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        identity = {
            k: job[k]
            for k in ("name", "round", "seed", "size", "precision_mode", "device")
        }
        for check in summary.get("checks", []):
            control_rows.append(
                {**identity, "check": check["name"], "passed": check["passed"]}
            )
        for row in summary.get("observations", []):
            base = {**identity, "variant": row["variant"], "mode": row["mode"]}
            compilation = row.get("original_jit_vs_eager", {}).get("input_gradient", {})
            comparisons.append(
                {
                    **base,
                    "loss": row["loss"],
                    "gradient_vs_torch_relative_l2": row["input_gradient_vs_torch"][
                        "relative_l2_error"
                    ],
                    "gradient_vs_torch_max_absolute": row["input_gradient_vs_torch"][
                        "max_absolute_error"
                    ],
                    "jit_vs_eager_gradient_relative_l2": compilation.get(
                        "relative_l2_error"
                    ),
                    "instrumentation_gradient_relative_l2": row[
                        "instrumentation_effect"
                    ]["input_gradient"]["relative_l2_error"],
                    "repeat_gradient_identical": row["repeat_gradient_identical"],
                    "repeat_output_identical": row["repeat_output_identical"],
                    "first_observed_value_stage": row.get(
                        "first_observed_value_stage_outside_fp32_tolerance"
                    ),
                }
            )
            for stage in row["stages"]:
                stage_rows.append(
                    {
                        **base,
                        "stage": stage["stage"],
                        "jax_dtype": stage["jax_dtype"],
                        "torch_dtype": stage["torch_dtype"],
                        "value_vs_torch_relative_l2": stage["values_vs_torch"][
                            "relative_l2_error"
                        ],
                        "gradient_vs_torch_relative_l2": stage[
                            "stage_gradient_vs_torch"
                        ]["relative_l2_error"],
                        "value_vs_eager_relative_l2": stage.get(
                            "values_vs_eager", {}
                        ).get("relative_l2_error"),
                        "gradient_vs_eager_relative_l2": stage.get(
                            "stage_gradient_vs_eager", {}
                        ).get("relative_l2_error"),
                    }
                )
            for repeat in row.get("repeats", []):
                repeats.append(
                    {
                        **base,
                        "repeat_index": repeat["repeat_index"],
                        "loss_absolute_gap": repeat["loss_absolute_gap"],
                        "gradient_relative_l2": repeat["gradient"]["relative_l2_error"],
                        "gradient_identical": repeat["gradient"]["identical"],
                    }
                )
    csv_file(root / "tables" / "workers.csv", statuses)
    csv_file(root / "tables" / "comparisons.csv", comparisons)
    csv_file(root / "tables" / "stages.csv", stage_rows)
    csv_file(root / "tables" / "repeatability.csv", repeats)
    csv_file(root / "tables" / "controls.csv", control_rows)
    # Compare identical numeric fixtures evaluated in separate worker processes.
    import numpy as np

    groups, fresh = {}, []
    for job in statuses:
        path = root / "workers" / job["name"] / "arrays.npz"
        if path.exists():
            groups.setdefault(
                (job["seed"], job["size"], job["precision_mode"]), []
            ).append(job)
    for group in groups.values():
        group.sort(key=lambda j: j["round"])
        first = group[0]
        with np.load(
            root / "workers" / first["name"] / "arrays.npz", allow_pickle=False
        ) as reference:
            keys = [k for k in reference.files if k.endswith("/input_gradient")]
            for job in group[1:]:
                with np.load(
                    root / "workers" / job["name"] / "arrays.npz", allow_pickle=False
                ) as current:
                    for key in keys:
                        a, b = (
                            current[key].astype(np.float64),
                            reference[key].astype(np.float64),
                        )
                        fresh.append(
                            {
                                "reference_worker": first["name"],
                                "worker": job["name"],
                                "array": key,
                                "reference_device": first["device"],
                                "device": job["device"],
                                "identical": bool(np.array_equal(a, b)),
                                "relative_l2_error": float(
                                    np.linalg.norm(a - b)
                                    / max(np.linalg.norm(b), np.finfo(np.float64).tiny)
                                ),
                            }
                        )
    csv_file(root / "tables" / "fresh_process_comparisons.csv", fresh)
    return {
        "comparison_rows": len(comparisons),
        "stage_rows": len(stage_rows),
        "fresh_process_comparison_rows": len(fresh),
    }


def run(args, devices, jobs):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (
        args.output
        or REPO / "results" / f"synthetic_attention_cluster_{stamp}_{os.getpid()}"
    ).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "scope": "synthetic numerical tests only",
                    "output": str(root),
                    "devices": devices,
                    "workers": len(jobs),
                    "variant_mode_cases": len(jobs) * 12,
                    "jobs": jobs,
                },
                indent=2,
            )
        )
        return 0
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("workers", "logs", "tables", "metadata", "sources"):
        (root / folder).mkdir()
    (root / "READ_ME.txt").write_text(
        "SYNTHETIC ATTENTION NUMERICAL REVIEW\n"
        "No protein model, checkpoint, or biological input was used.\n\n"
        "Start with summary.json and tables/workers.csv. A completed batch can\n"
        "contain failed synthetic controls; neither status validates model gradients.\n"
        "tables/comparisons.csv: backend, compilation, and instrumentation differences.\n"
        "tables/stages.csv: intermediate values, dtypes, and stage gradients.\n"
        "tables/repeatability.csv: repeated evaluations in each worker process.\n"
        "tables/fresh_process_comparisons.csv: matched fixtures across rounds.\n"
        "tables/controls.csv: individual synthetic control outcomes.\n"
        "workers/<name>/: config, summary, raw arrays, compiled IR, and event logs.\n"
        "logs/: worker stdout/stderr and preflight errors.\n"
        "metadata/: environment, GPU telemetry, and runner errors when applicable.\n"
        "sources/: exact copies of the scripts used for this run.\n"
        "status.json and runner.log: live progress. commands.sh: worker commands.\n\n"
        "Strict precision requests highest FP32 matmul precision and disables Torch\n"
        "TF32; native leaves framework defaults/settings intact and records them.\n"
        "Neither setting guarantees that different frameworks execute identical kernels.\n"
        "BF16 differences are descriptive. Exposing intermediates may alter compilation.\n"
        "Synthetic sizes are small=(4,5,3,2), medium=(64,32,16,16), and\n"
        "large=(256,64,32,32): rows, input width, query/key width, value width.\n\n"
        "The batch stops after its finite sweep; it does not wait until morning.\n"
        "On completion a sibling .tar.gz contains this entire directory, unless\n"
        "--no-archive was used. Upload that archive for review.\n"
        "Verify extracted files with: sha256sum -c checksums.sha256\n"
    )

    def log(message):
        line = f"{utc()} {message}"
        print(line, flush=True)
        with (root / "runner.log").open("a") as handle:
            handle.write(line + "\n")

    statuses = [
        {
            **j,
            "device": None,
            "state": "pending",
            "pid": None,
            "exit_code": None,
            "started_utc": None,
            "finished_utc": None,
            "seconds": None,
        }
        for j in jobs
    ]
    save(
        root / "plan.json",
        {
            "scope": "synthetic only",
            "devices": devices,
            "jobs": jobs,
            "repeats": args.repeats,
            "worker_timeout_minutes": args.worker_timeout_minutes,
        },
    )
    for source in SOURCES:
        shutil.copy2(source, root / "sources" / source.name)
    metadata = {
        "python": sys.version,
        "executable": sys.executable,
        "host": os.uname().nodename,
        "started_utc": utc(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }
    for key, cmd in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short"]),
    ):
        try:
            p = subprocess.run(
                cmd, cwd=REPO, text=True, capture_output=True, timeout=10
            )
            metadata[key] = {
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except Exception as exc:
            metadata[key] = {"error": str(exc)}
    save(root / "metadata" / "environment.json", metadata)
    log(f"Repo: {REPO}; output: {root}")
    log(
        f"Synthetic only: {len(jobs)} workers, {len(jobs) * 12} variant/mode cases; devices: {devices}"
    )
    log(
        "A finite sweep; no model/checkpoint loading and no deliberate overnight waiting."
    )
    active, free = [], list(devices)
    failure = None
    interrupted = False
    started = time.monotonic()
    next_telemetry = 0.0
    try:
        preflight_code = (
            "import json,jax,torch; d=jax.devices(); "
            f"assert len(d)==1 and d[0].platform=={'cpu' if args.cpu else 'gpu'!r},d; "
            + (
                ""
                if args.cpu
                else "assert torch.cuda.is_available(), 'CUDA-enabled PyTorch required'; "
            )
            + "print(json.dumps({'jax':jax.__version__,'torch':torch.__version__,'jax_devices':[str(x) for x in d],"
            "'torch_cuda':torch.version.cuda,'torch_cuda_available':torch.cuda.is_available()}))"
        )
        for device in devices:
            log(f"Preflight device {device}")
            with (root / "logs" / f"preflight_{device}.log").open("w") as handle:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-c", preflight_code],
                    cwd=REPO,
                    env=worker_env(device, args.cpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = proc.wait(timeout=120)
                finally:
                    stop_process(proc)
            if code:
                raise RuntimeError(
                    f"Preflight failed on device {device}; see logs/preflight_{device}.log"
                )
        pending = list(statuses)
        while pending or active:
            while pending and free:
                job, device = pending.pop(0), free.pop(0)
                cmd = command(job, root, device, args)
                with (root / "commands.sh").open("a") as handle:
                    handle.write(shlex.join(cmd) + "\n")
                with (root / "logs" / f"{job['name']}.log").open("w") as handle:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=REPO,
                        env=worker_env(device, args.cpu),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                job.update(
                    device=device, state="running", pid=proc.pid, started_utc=utc()
                )
                active.append((job, device, proc, time.monotonic()))
                log(f"Started {job['name']} on {device}, PID {proc.pid}")
                save(root / "status.json", statuses)
            now = time.monotonic()
            if not args.cpu and now >= next_telemetry:
                telemetry = {"utc": utc()}
                try:
                    p = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={','.join(devices)}",
                            "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    telemetry.update(
                        exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr
                    )
                except Exception as exc:
                    telemetry["error"] = str(exc)
                with (root / "metadata" / "gpu_telemetry.jsonl").open("a") as handle:
                    handle.write(json.dumps(telemetry) + "\n")
                next_telemetry = now + args.telemetry_seconds
            for job, device, proc, began in list(active):
                timed_out = (
                    proc.poll() is None
                    and time.monotonic() - began > args.worker_timeout_minutes * 60
                )
                if timed_out:
                    stop_process(proc)
                code = proc.poll()
                if code is None:
                    continue
                summary_path = root / "workers" / job["name"] / "summary.json"
                completed = summary_path.exists() and json.loads(
                    summary_path.read_text()
                ).get("experiment_completed", False)
                state = (
                    "timeout"
                    if timed_out
                    else (
                        "complete"
                        if code == 0 and completed
                        else "checks_failed"
                        if completed
                        else "error"
                    )
                )
                job.update(
                    state=state,
                    exit_code=code,
                    seconds=time.monotonic() - began,
                    finished_utc=utc(),
                )
                active.remove((job, device, proc, began))
                free.append(device)
                log(
                    f"Finished {job['name']}: {state}, exit {code}, {job['seconds']:.1f}s"
                )
                save(root / "status.json", statuses)
            if active:
                time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        failure = "Interrupted; active workers stopped, collected files retained"
        log(failure)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        (root / "metadata" / "runner_error.txt").write_text(traceback.format_exc())
        log(failure)
    finally:
        for job, _, proc, began in active:
            stop_process(proc)
            job.update(
                state="interrupted",
                exit_code=proc.returncode,
                seconds=time.monotonic() - began,
                finished_utc=utc(),
            )
        save(root / "status.json", statuses)
    try:
        counts = collect(root, statuses)
    except Exception as exc:
        counts = {"aggregation_error": str(exc)}
        failure = failure or f"Aggregation failed: {exc}"
        (root / "metadata" / "aggregation_error.txt").write_text(traceback.format_exc())
    summary = {
        "batch_completed": all(
            j["state"] in ("complete", "checks_failed") for j in statuses
        ),
        "all_synthetic_controls_passed": failure is None
        and all(j["state"] == "complete" for j in statuses),
        "elapsed_seconds": time.monotonic() - started,
        "failure": failure,
        "worker_state_counts": {
            state: sum(j["state"] == state for j in statuses)
            for state in sorted({j["state"] for j in statuses})
        },
        "full_model_gradient_validation_status": "not_assessed",
        "interpretation": "Synthetic controls only. Numerical differences are retained even when a control fails. "
        "A strict/native comparison changes framework FP32 precision policies; it does not guarantee identical kernels.",
        **counts,
    }
    save(root / "summary.json", summary)
    archive = root.with_name(root.name + ".tar.gz")
    log(f"Summary: {root / 'summary.json'}")
    if not args.no_archive:
        log(f"Creating archive: {archive}")
    # Nothing in root is mutated after creating the checksum manifest.
    with (root / "checksums.sha256").open("w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                with path.open("rb") as src:
                    digest = hashlib.file_digest(src, "sha256").hexdigest()
                handle.write(f"{digest}  {path.relative_to(root).as_posix()}\n")
    if not args.no_archive:
        partial = archive.with_suffix(archive.suffix + ".partial")
        with tarfile.open(partial, "w:gz") as handle:
            handle.add(root, arcname=root.name)
        partial.replace(archive)
        print(f"Download: {archive}", flush=True)
    return 130 if interrupted else 0 if summary["all_synthetic_controls_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Serial local test; defaults to one small worker",
    )
    parser.add_argument("--seeds")
    parser.add_argument("--sizes")
    parser.add_argument("--precision-modes")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-timeout-minutes", type=float, default=30)
    parser.add_argument("--telemetry-seconds", type=float, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        (args.rounds is not None and args.rounds < 1)
        or args.repeats < 2
        or args.worker_timeout_minutes <= 0
        or args.telemetry_seconds <= 0
    ):
        parser.error(
            "Positive rounds/time limits and at least two repeats are required"
        )
    try:
        devices, jobs = make_plan(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Gracefully preserve evidence on job-scheduler termination, too.
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    return run(args, devices, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
