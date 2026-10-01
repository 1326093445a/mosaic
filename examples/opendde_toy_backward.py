"""Fixed toy-input full-model derivative checks, without sequence optimization.

Uses the five-residue fixture from tests/test_model_smoke.py. This exercises
real model weights and the diffusion/confidence backward path, not realistic
folding, binding, or production-sized memory. No arbitrary protein input is
accepted. Eight workers are independent checks, not model parallelism.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
EPSILONS = (0.01, 0.003, 0.001)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def memory(devices):
    result = {}
    for device in devices:
        try:
            stats = device.memory_stats()
            result[str(device)] = {"supported": stats is not None, "stats": stats}
        except Exception as exc:
            result[str(device)] = {"supported": False, "error": str(exc)}
    return result


def run_worker(args):
    # Set before any JAX/model imports. A missing CUDA backend must fail loudly.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, str(REPO / "src"))
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "seed": args.worker_seed,
        "scope": "fixed toy full-model derivative",
    }
    devices = []
    start = time.monotonic()
    save(
        root / "config.json",
        {
            "input": {"first_chain_argmax_token_ids": [0, 7], "second_chain": "ACD"},
            "input_parameterization": "softmax logits: selected token 4, others 0",
            "model": "OpenDDEModelAbag",
            "physical_device_index": args.devices,
            "precision": "fp32",
            "recycling_steps": 1,
            "sampling_steps": 2,
            "stop_grad_conf_coords": False,
            "seed": args.worker_seed,
            "epsilons": EPSILONS,
            "finite_difference_rtol": 0.05,
            "finite_difference_atol": 0.001,
            "required_matching_epsilons": 2,
            "repeat_rtol": 1e-5,
            "repeat_atol": 1e-6,
            "threshold_note": "Fixed software smoke thresholds, not a biological acceptance criterion.",
            "scope_note": "Full architecture on five residues; not realistic structure quality "
            "or production memory. Derivatives are of the wrapper's relaxed input with "
            "discrete residue identities held fixed, not of mutations or model weights.",
            "memory_note": "JAX process counters only. Peaks accumulate across calls; "
            "failed allocations and non-JAX allocations are excluded.",
        },
    )
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        from mosaic.common import LossTerm
        from mosaic.models.opendde import OpenDDEModelAbag
        from mosaic.structure_prediction import TargetChain

        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError(f"Expected exactly one CUDA device; received {devices}")
        report.update(
            jax_version=jax.__version__,
            device=str(devices[0]),
            device_kind=devices[0].device_kind,
        )

        def measured(label, fn):
            event = {"label": label, "before": memory(devices)}
            t0 = time.monotonic()
            try:
                value = fn()
                jax.block_until_ready(value)
                event["status"] = "ok"
                return value
            except Exception as exc:
                event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                event.update(after=memory(devices), seconds=time.monotonic() - t0)
                with (root / "memory.jsonl").open("a") as handle:
                    handle.write(json.dumps(event, default=str) + "\n")
                print(
                    f"{label}: {event['status']} ({event['seconds']:.1f}s)", flush=True
                )

        print("Loading real OpenDDE weights for fixed toy input...", flush=True)
        model = OpenDDEModelAbag(compute_precision="fp32")
        features, _ = model.binder_features(2, [TargetChain("ACD", use_msa=False)])

        class Probe(LossTerm):
            def __call__(self, sequence, output, key):
                # Smooth, arbitrary software probe; not a binding/design objective.
                ca = output.backbone_coordinates[:, 1, :].astype(jnp.float32)
                centered = ca - ca.mean(axis=0)
                coordinate_probe = jnp.square(centered).mean() / 100
                confidence_probe = output.plddt.astype(jnp.float32).mean()
                value = coordinate_probe + confidence_probe
                return value, {
                    "coordinates": ca,
                    "plddt": output.plddt,
                    "coordinate_probe": coordinate_probe,
                    "confidence_probe": confidence_probe,
                }

        loss = model.build_loss(
            loss=Probe(),
            features=features,
            recycling_steps=1,
            sampling_steps=2,
            stop_grad_conf_coords=False,
        )
        key = jax.random.key(args.worker_seed)

        def objective(logits, model_loss, model_key):
            return model_loss(jax.nn.softmax(logits, axis=-1), key=model_key)

        forward = eqx.filter_jit(objective)
        backward = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
        logits = jnp.zeros((2, 20)).at[jnp.arange(2), jnp.array([0, 7])].set(4.0)
        # Same key/input on every call, including finite differences.
        baseline, _ = measured("forward", lambda: forward(logits, loss, key))
        (value, aux), grad = measured("backward", lambda: backward(logits, loss, key))
        (repeat_value, repeat_aux), repeat_grad = measured(
            "backward_repeat", lambda: backward(logits, loss, key)
        )
        leaves = jax.tree.leaves(
            (baseline, value, aux, grad, repeat_value, repeat_aux, repeat_grad)
        )
        if not all(np.isfinite(np.asarray(v)).all() for v in leaves):
            raise ValueError("Nonfinite forward or backward output")
        np.savez_compressed(
            root / "arrays.npz",
            gradient=np.asarray(grad),
            repeat_gradient=np.asarray(repeat_grad),
            coordinates=np.asarray(aux["coordinates"]),
            plddt=np.asarray(aux["plddt"]),
        )
        direction = (
            np.random.default_rng(args.worker_seed)
            .normal(size=(2, 20))
            .astype(np.float32)
        )
        direction -= direction.mean(axis=-1, keepdims=True)
        direction /= np.linalg.norm(direction)
        direction = jnp.asarray(direction)
        ad = float(jnp.sum(grad * direction))
        comparisons = []
        for eps in EPSILONS:
            plus, minus = logits + eps * direction, logits - eps * direction
            if not all(
                np.array_equal(np.asarray(jnp.argmax(x, axis=-1)), [0, 7])
                for x in (plus, minus)
            ):
                raise ValueError(
                    "Finite difference changed discrete residue identities"
                )
            vp, _ = measured(f"forward_plus_{eps}", lambda: forward(plus, loss, key))
            vm, _ = measured(f"forward_minus_{eps}", lambda: forward(minus, loss, key))
            fd = (float(vp) - float(vm)) / (2 * eps)
            if not np.isfinite(fd):
                raise ValueError("Nonfinite finite difference")
            comparisons.append(
                {
                    "epsilon": eps,
                    "autodiff": ad,
                    "finite_difference": fd,
                    "loss_plus": float(vp),
                    "loss_minus": float(vm),
                    "absolute_error": abs(ad - fd),
                    "matches": bool(np.isclose(ad, fd, rtol=0.05, atol=0.001)),
                }
            )
        checks = {
            "finite": True,
            "nonzero_gradient": bool(np.linalg.norm(np.asarray(grad)) > 0),
            "forward_backward_value_agree": bool(
                np.isclose(float(baseline), float(value), rtol=1e-5, atol=1e-6)
            ),
            "repeat_gradient_agree": bool(
                np.allclose(grad, repeat_grad, rtol=1e-5, atol=1e-6)
            ),
            "repeat_value_agree": bool(
                np.isclose(float(value), float(repeat_value), rtol=1e-5, atol=1e-6)
            ),
            "directional_derivative_agree": sum(v["matches"] for v in comparisons) >= 2,
        }
        report.update(
            checks=checks,
            passed=all(checks.values()),
            loss=float(value),
            forward_loss=float(baseline),
            repeat_loss=float(repeat_value),
            forward_backward_loss_difference=abs(float(baseline) - float(value)),
            repeat_gradient_max_absolute_difference=float(
                np.max(np.abs(np.asarray(grad) - np.asarray(repeat_grad)))
            ),
            gradient_norm=float(np.linalg.norm(np.asarray(grad))),
            finite_differences=comparisons,
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        report.update(
            elapsed_seconds=time.monotonic() - start,
            memory_at_completion=memory(devices),
        )
        save(root / "summary.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="Distinct CUDA indices; one independent seed per device",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without loading models or writing output",
    )
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    devices = args.devices.split(",")
    if (
        not devices
        or any(not d.isdecimal() for d in devices)
        or len(set(map(int, devices))) != len(devices)
    ):
        parser.error("--devices requires distinct nonnegative CUDA indices")
    if args.worker_seed is not None:
        if len(devices) != 1 or args.output is None or args.worker_seed < 0:
            parser.error(
                "worker requires one device, an output directory, and nonnegative seed"
            )
        return run_worker(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (
        args.output or REPO / "results" / f"opendde_toy_backward_{stamp}_{os.getpid()}"
    ).resolve()
    jobs = []
    for seed, device in enumerate(devices):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--devices",
            device,
            "--worker-seed",
            str(seed),
            "--output",
            str(root / f"seed{seed}"),
        ]
        jobs.append({"seed": seed, "device": device, "command": command})
    print(
        f"Output: {root}\nFull OpenDDE on fixed toy input: {len(jobs)} independent workers",
        flush=True,
    )
    print(
        "Per worker: 2 backward calls + 7 forward-only calls; 2 sampling steps, 1 recycle.",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", jobs)
    processes = []
    statuses = []
    try:
        for job in jobs:
            log = (root / "logs" / f"seed{job['seed']}.log").open("w")
            try:
                proc = subprocess.Popen(
                    job["command"], stdout=log, stderr=subprocess.STDOUT, cwd=REPO
                )
            finally:
                log.close()
            processes.append((job, proc))
            print(
                f"Started seed{job['seed']} on GPU {job['device']} (PID {proc.pid})",
                flush=True,
            )
        for job, proc in processes:
            code = proc.wait()
            statuses.append(
                {"seed": job["seed"], "device": job["device"], "exit_code": code}
            )
            print(f"Finished seed{job['seed']}: exit {code}", flush=True)
    finally:
        for job, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for job, proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if not any(s["seed"] == job["seed"] for s in statuses):
                statuses.append(
                    {
                        "seed": job["seed"],
                        "device": job["device"],
                        "exit_code": proc.returncode,
                    }
                )
        save(root / "status.json", statuses)
    passed = len(statuses) == len(jobs) and all(s["exit_code"] == 0 for s in statuses)
    save(
        root / "summary.json",
        {
            "passed": passed,
            "workers": statuses,
            "scope": "fixed toy full-model derivative; no production memory or structure quality claim",
        },
    )
    print(f"{'PASS' if passed else 'FAIL'}: review {root / 'summary.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
