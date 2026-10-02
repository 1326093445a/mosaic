"""Numerical conditioning audit of a fixed five-residue OpenDDE software fixture.

Separates the original coordinate and confidence probes. Repeats identical
forward/backward calls and both sides of finite differences using one fixed
model key per worker. No sequence optimization or arbitrary biological input.
Exit zero means the audit was collected, NOT that gradients were validated.
CPU mode runs serially and defaults to seed 0 for both probes.
Schema version 2 reports gradient correctness as "not_assessed", separately
from completion, observed repeatability, and derivative comparisons.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import tomllib
import traceback

REPO = Path(__file__).resolve().parents[1]
EPSILONS = (0.1, 0.03, 0.01, 0.003, 0.001)
FORWARD_REPEATS = 5
BACKWARD_REPEATS = 3
PAIR_REPEATS = 3


def seed_host_generators(seed):
    """Control setup randomness separately from the JAX evaluation key."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fingerprint_arrays(tree, *, include_arrays=False):
    """Hash numerical leaves and their paths/shapes/dtypes, not static code."""
    import jax
    import numpy as np

    aggregate = hashlib.sha256()
    arrays = []
    total_bytes = 0
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(value, "dtype") or not hasattr(value, "shape"):
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot be reproducibly fingerprinted")
        record = {
            "path": jax.tree_util.keystr(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        aggregate.update(json.dumps(record, sort_keys=True).encode())
        arrays.append(record)
        total_bytes += array.nbytes
    result = {
        "sha256": aggregate.hexdigest(),
        "array_count": len(arrays),
        "array_bytes": total_bytes,
    }
    if include_arrays:
        result["arrays"] = arrays
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def memory(devices):
    result = {}
    for device in devices:
        try:
            stats = device.memory_stats()
            result[str(device)] = {"supported": stats is not None, "stats": stats}
        except Exception as exc:
            result[str(device)] = {"supported": False, "error": str(exc)}
    return result


def compare_derivatives(baseline, plus, minus, autodiff, epsilon):
    """Descriptive sampled ranges, not confidence intervals or a correctness test."""
    import numpy as np

    arrays = [
        np.asarray(x, dtype=np.float64) for x in (baseline, plus, minus, autodiff)
    ]
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if any(x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError(
            "each comparison requires at least two finite scalar observations"
        )
    b, p, m, a = arrays
    mean_difference = float(p.mean() - m.mean())
    repeat_range = float(max(np.ptp(b), np.ptp(p), np.ptp(m)))
    # A heuristic resolution floor: host float64 subtraction cannot recover
    # information already lost when the model produced float32 scalars.
    ulp = float(
        max(
            np.spacing(np.abs(np.asarray(x, dtype=np.float32))).max() for x in (b, p, m)
        )
    )
    resolution = max(repeat_range, 2 * ulp)
    fd_low = float((p.min() - m.max()) / (2 * epsilon))
    fd_high = float((p.max() - m.min()) / (2 * epsilon))
    return {
        "epsilon": epsilon,
        "autodiff_mean": float(a.mean()),
        "autodiff_min": float(a.min()),
        "autodiff_max": float(a.max()),
        "finite_difference_mean": mean_difference / (2 * epsilon),
        "finite_difference_min": fd_low,
        "finite_difference_max": fd_high,
        "absolute_derivative_gap": abs(
            float(a.mean()) - mean_difference / (2 * epsilon)
        ),
        "loss_difference": mean_difference,
        "largest_same_input_loss_range": repeat_range,
        "float32_ulp": ulp,
        "signal_to_observed_resolution": abs(mean_difference) / resolution,
        "signal_exceeds_3x_observed_resolution": abs(mean_difference) > 3 * resolution,
        "sampled_derivative_ranges_overlap": fd_low <= float(a.max())
        and float(a.min()) <= fd_high,
    }


def summarize_numerical_evidence(baseline, backward_losses, gradients, comparisons):
    """Describe collected evidence without inventing a correctness criterion."""
    import numpy as np

    arrays = [np.asarray(x) for x in (baseline, backward_losses, gradients)]
    if any(x.ndim == 0 or len(x) < 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("repeatability requires at least two finite observations")
    gaps = [row["absolute_derivative_gap"] for row in comparisons]
    return {
        "repeatability": {
            "status": "observed",
            "forward_losses_identical": bool(np.all(arrays[0] == arrays[0][0])),
            "backward_losses_identical": bool(np.all(arrays[1] == arrays[1][0])),
            "gradients_identical": bool(np.all(arrays[2] == arrays[2][0])),
            "interpretation": "Exact equality over collected repeats only; "
            "does not establish gradient correctness.",
        },
        "derivative_agreement": {
            "status": "not_assessed",
            "comparisons_collected": len(comparisons),
            "min_absolute_derivative_gap": min(gaps) if gaps else None,
            "max_absolute_derivative_gap": max(gaps) if gaps else None,
            "interpretation": "Differences and sampled ranges are descriptive. "
            "No precision-aware acceptance criterion has been established.",
        },
    }


def environment():
    from packaging.requirements import Requirement

    versions = {}
    for package in ("jax", "jaxlib", "equinox", "numpy", "jopendde", "opendde"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    dependencies = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    requirement = next(
        Requirement(x) for x in dependencies if Requirement(x).name == "jax"
    )
    compatible = versions["jax"] is not None and requirement.specifier.contains(
        versions["jax"]
    )
    return {
        "python": sys.version,
        "packages": versions,
        "declared_jax_requirement": str(requirement),
        "jax_matches_declared_requirement": compatible,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "MOSAIC_OPENDDE_AGGREGATION",
                "PYTHONHASHSEED",
            )
        },
    }


def run_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else args.devices
    os.environ["JAX_PLATFORMS"] = "cpu" if args.cpu else "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, str(REPO / "src"))
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    devices = []
    report = {
        "audit_completed": False,
        "schema_version": 2,
        "gradient_validation_status": "not_assessed",
        "probe": args.probe,
        "seed": args.worker_seed,
        "repeatability": {"status": "not_measured"},
        "derivative_agreement": {"status": "not_assessed"},
    }
    config = {
        "input": {"first_chain_token_ids": [0, 7], "second_chain": "ACD"},
        "model": "OpenDDEModelAbag",
        "backend": "cpu" if args.cpu else "cuda",
        "compute_precision": "fp32",
        "logits": "selected token 4, all others 0; softmax before model",
        "probe": args.probe,
        "seed": args.worker_seed,
        "sampling_steps": 2,
        "recycling_steps": 1,
        "stop_grad_conf_coords": False,
        "epsilons": EPSILONS,
        "forward_repeats": FORWARD_REPEATS,
        "backward_repeats": BACKWARD_REPEATS,
        "repeats_per_perturbation": PAIR_REPEATS,
        "model_key": "same key for every evaluation within this worker",
        "host_random_seed": args.worker_seed,
        "host_seed_policy": "Python, NumPy, and PyTorch reset before model loading "
        "and again before toy feature generation; Python hash seed set at worker launch.",
        "scope": "fixed toy numerical audit, not structure quality or production memory",
        "interpretation": "Ranges describe a few observations, not statistical confidence "
        "intervals. Signal/resolution is heuristic and is not a gradient acceptance test. "
        "Probe scales are unchanged; no threshold tuning or optimization.",
        "memory_note": "Cumulative JAX process peaks, including setup/compilation; "
        "excludes failed and non-JAX allocations. Not incremental backward memory.",
    }
    save(root / "config.json", config)
    try:
        config["runtime"] = environment()
        save(root / "config.json", config)
        if not config["runtime"]["jax_matches_declared_requirement"]:
            print(
                "JAX version is outside the repository requirement; recording this environment as-is.",
                flush=True,
            )
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        from mosaic.common import LossTerm
        from mosaic.models.opendde import OpenDDEModelAbag
        from mosaic.structure_prediction import TargetChain

        devices = jax.devices()
        expected = "cpu" if args.cpu else "gpu"
        if len(devices) != 1 or devices[0].platform != expected:
            raise RuntimeError(f"Expected one {expected} device, got {devices}")
        report["device_kind"] = devices[0].device_kind

        def measured(label, fn):
            event = {"label": label, "before": memory(devices)}
            t0 = time.monotonic()
            try:
                value = fn()
                jax.block_until_ready(value)
                if not all(
                    np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value)
                ):
                    raise ValueError("Nonfinite model output")
                event["status"] = "ok"
                return value
            except Exception as exc:
                event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                event.update(seconds=time.monotonic() - t0, after=memory(devices))
                with (root / "memory.jsonl").open("a") as handle:
                    handle.write(json.dumps(event, default=str) + "\n")
                print(
                    f"{label}: {event['status']} ({event['seconds']:.2f}s)", flush=True
                )

        def record(label, value, **extra):
            with (root / "observations.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {"label": label, "value": float(value), **extra},
                        allow_nan=False,
                    )
                    + "\n"
                )

        print(
            f"Loading real model; probe={args.probe}, seed={args.worker_seed}",
            flush=True,
        )
        seed_host_generators(args.worker_seed)
        model = OpenDDEModelAbag(compute_precision="fp32")
        # Isolate feature randomness from loading/cache RNG consumption.
        # This controls the toy harness only.
        seed_host_generators(args.worker_seed)
        features, _ = model.binder_features(2, [TargetChain("ACD", use_msa=False)])
        fingerprints = {
            "model_arrays": fingerprint_arrays(model),
            "feature_arrays": fingerprint_arrays(features, include_arrays=True),
        }
        save(root / "setup_fingerprints.json", fingerprints)
        report["setup_array_hashes"] = {k: v["sha256"] for k, v in fingerprints.items()}

        class Probe(LossTerm):
            mode: str = eqx.field(static=True)

            def __call__(self, sequence, output, key):
                ca = output.backbone_coordinates[:, 1, :].astype(jnp.float32)
                centered = ca - ca.mean(axis=0)
                coordinate = jnp.square(centered).mean() / 100
                confidence = output.plddt.astype(jnp.float32).mean()
                value = coordinate if self.mode == "coordinate" else confidence
                return value, {"coordinate": coordinate, "confidence": confidence}

        loss = model.build_loss(
            loss=Probe(args.probe),
            features=features,
            recycling_steps=1,
            sampling_steps=2,
            stop_grad_conf_coords=False,
        )
        key = jax.random.key(args.worker_seed)
        logits = jnp.zeros((2, 20)).at[jnp.arange(2), jnp.array([0, 7])].set(4.0)
        report["evaluation_input_hash"] = fingerprint_arrays(
            (logits, jax.random.key_data(key))
        )["sha256"]

        def objective(x, model_loss, model_key):
            return model_loss(jax.nn.softmax(x, axis=-1), key=model_key)

        forward = eqx.filter_jit(objective)
        backward = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
        rng = np.random.default_rng(args.worker_seed)
        direction = rng.normal(size=(2, 20)).astype(np.float32)
        direction -= direction.mean(axis=-1, keepdims=True)
        direction /= np.linalg.norm(direction)
        delta = jnp.asarray(direction)
        baseline, gradient_values, gradients = [], [], []
        for repeat in range(FORWARD_REPEATS):
            value, aux = measured(
                f"baseline_{repeat}", lambda: forward(logits, loss, key)
            )
            baseline.append(float(value))
            record(
                f"baseline_{repeat}",
                value,
                components={k: float(np.asarray(v).mean()) for k, v in aux.items()},
            )
        for repeat in range(BACKWARD_REPEATS):
            (value, _), grad = measured(
                f"backward_{repeat}", lambda: backward(logits, loss, key)
            )
            gradients.append(np.asarray(grad))
            gradient_values.append(float(value))
            record(f"backward_{repeat}", value)
        gradients = np.asarray(gradients)
        ad = (gradients.astype(np.float64) * direction.astype(np.float64)).sum(
            axis=(1, 2)
        )
        np.savez_compressed(
            root / "gradients.npz", gradients=gradients, direction=direction
        )
        rows = []
        for eps in EPSILONS:
            xs = {"plus": logits + eps * delta, "minus": logits - eps * delta}
            if any(
                not np.array_equal(np.asarray(jnp.argmax(x, axis=-1)), [0, 7])
                for x in xs.values()
            ):
                raise ValueError("Perturbation changed discrete residue identities")
            values = {"plus": [], "minus": []}
            for repeat in range(PAIR_REPEATS):
                # Alternate order to make simple order effects visible.
                for side in ("plus", "minus") if repeat % 2 == 0 else ("minus", "plus"):
                    label = f"eps{eps}_{side}_{repeat}"
                    value, _ = measured(label, lambda: forward(xs[side], loss, key))
                    values[side].append(float(value))
                    record(label, value, epsilon=eps, side=side, repeat=repeat)
            row = {
                "probe": args.probe,
                "seed": args.worker_seed,
                **compare_derivatives(
                    baseline, values["plus"], values["minus"], ad, eps
                ),
            }
            rows.append(row)
            write_csv(root / "finite_differences.csv", rows)
        g64 = gradients.astype(np.float64)
        norms = np.linalg.norm(g64.reshape(BACKWARD_REPEATS, -1), axis=1)
        repeat_abs = np.linalg.norm(
            (g64 - g64[0]).reshape(BACKWARD_REPEATS, -1), axis=1
        )
        report.update(
            audit_completed=True,
            baseline_losses=baseline,
            backward_losses=gradient_values,
            baseline_loss_range=float(np.ptp(baseline)),
            gradient_norms=norms.tolist(),
            gradient_repeat_l2_differences=repeat_abs.tolist(),
            gradient_repeat_relative_l2=(repeat_abs / norms[0]).tolist()
            if norms[0] > 0
            else None,
            autodiff_directional_derivatives=ad.tolist(),
            finite_differences=rows,
            **summarize_numerical_evidence(baseline, gradient_values, gradients, rows),
            interpretation="Audit collected. No automatic gradient-correctness verdict; "
            "examine repeat ranges, float32 resolution, derivative gaps, and step-size dependence.",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        report.update(
            seconds=time.monotonic() - start, memory_at_completion=memory(devices)
        )
        save(root / "summary.json", report)
    return 0 if report["audit_completed"] else 1


def integers(value):
    parts = value.split(",")
    if (
        not parts
        or any(not p.isdecimal() for p in parts)
        or len(set(map(int, parts))) != len(parts)
    ):
        raise argparse.ArgumentTypeError(
            "use distinct nonnegative integers separated by commas"
        )
    return list(map(int, parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--devices", help="CUDA indices; defaults to 0,1,2,3,4,5,6,7")
    backend.add_argument(
        "--cpu", action="store_true", help="Use CPU only, one worker at a time"
    )
    parser.add_argument(
        "--seeds", type=integers, help="Defaults to 0 on CPU; 0,1,2,3 on GPU"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe", choices=("coordinate", "confidence"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    try:
        devices = ["cpu"] if args.cpu else integers(args.devices or "0,1,2,3,4,5,6,7")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.worker_seed is not None:
        if (
            len(devices) != 1
            or args.output is None
            or args.probe is None
            or args.worker_seed < 0
        ):
            parser.error(
                "worker requires one device, output, probe, and nonnegative seed"
            )
        return run_worker(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (
        args.output or REPO / "results" / f"opendde_toy_audit_{stamp}_{os.getpid()}"
    ).resolve()
    jobs = [
        {"probe": probe, "seed": seed, "name": f"{probe}_seed{seed}"}
        for seed in (
            args.seeds
            if args.seeds is not None
            else ([0] if args.cpu else [0, 1, 2, 3])
        )
        for probe in ("coordinate", "confidence")
    ]
    print(
        f"Output: {root}\nPlan: {len(jobs)} workers on {'CPU (serial)' if args.cpu else f'GPUs {devices}'}; fixed five-residue fixture.",
        flush=True,
    )
    print(
        "Each worker: 35 forward-only + 3 backward calls, including first-call compilation.\n"
        "Exit 0 means audit collected, not gradients validated.",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", {"jobs": jobs, "devices": devices})
    pending = list(jobs)
    running = []
    statuses = []
    free = list(devices)
    try:
        while pending or running:
            while pending and free:
                job, device = pending.pop(0), free.pop(0)
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    *(["--cpu"] if args.cpu else ["--devices", str(device)]),
                    "--worker-seed",
                    str(job["seed"]),
                    "--probe",
                    job["probe"],
                    "--output",
                    str(root / job["name"]),
                ]
                with (root / "logs" / f"{job['name']}.log").open("w") as log:
                    proc = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=REPO,
                        env={**os.environ, "PYTHONHASHSEED": str(job["seed"])},
                    )
                running.append((job, device, proc))
                print(
                    f"Started {job['name']} on {'CPU' if args.cpu else f'GPU {device}'} (PID {proc.pid})",
                    flush=True,
                )
            for job, device, proc in list(running):
                code = proc.poll()
                if code is not None:
                    statuses.append({**job, "device": device, "exit_code": code})
                    running.remove((job, device, proc))
                    free.append(device)
                    save(root / "status.json", statuses)
                    print(f"Finished {job['name']}: exit {code}", flush=True)
            if running:
                time.sleep(0.5)
    finally:
        for _, _, proc in running:
            proc.terminate()
        for job, device, proc in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            statuses.append({**job, "device": device, "exit_code": proc.returncode})
        save(root / "status.json", statuses)
    rows = []
    for job in jobs:
        path = root / job["name"] / "finite_differences.csv"
        if path.exists():
            with path.open() as handle:
                rows.extend(csv.DictReader(handle))
    write_csv(root / "finite_differences.csv", rows)
    completed = len(statuses) == len(jobs) and all(
        s["exit_code"] == 0 for s in statuses
    )
    save(
        root / "summary.json",
        {
            "audit_completed": completed,
            "schema_version": 2,
            "gradient_validation_status": "not_assessed",
            "repeatability": {"status": "see_worker_summaries"},
            "derivative_agreement": {"status": "not_assessed"},
            "workers": statuses,
            "scope": "fixed toy numerical conditioning audit",
        },
    )
    print(
        f"Audit {'collected' if completed else 'incomplete'}: {root / 'summary.json'}",
        flush=True,
    )
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
