"""CPU-only attention numerics using fixed arrays, with no model checkpoints.

Compare JAX eager/JIT and PyTorch against an independent NumPy FP64 analytic
reference. Mixed BF16 observations are descriptive, not correctness verdicts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

EPSILONS = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001)
TOLERANCES = {
    "fp64": {"rtol": 1e-10, "atol": 1e-12},
    "fp32": {"rtol": 1e-4, "atol": 1e-6},
}


def fixture(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, 5)) / 2
    weights = [rng.normal(size=(5, width)) / 2 for width in (3, 3, 2)]
    directions = rng.normal(size=(3, *x.shape))
    directions /= np.linalg.norm(directions, axis=(1, 2), keepdims=True)
    return x, weights, directions


def numpy_reference(x, weights):
    """Explicit chain rule for mean(square(softmax(Q K.T / sqrt(d)) V))."""
    wq, wk, wv = weights
    q, k, v = x @ wq, x @ wk, x @ wv
    scale = np.sqrt(q.shape[-1])
    scores = q @ k.T / scale
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    output = p @ v
    loss = np.mean(output**2)
    dout = 2 * output / output.size
    dp, dv = dout @ v.T, p.T @ dout
    dscores = p * (dp - np.sum(dp * p, axis=-1, keepdims=True))
    dq, dk = dscores @ k / scale, dscores.T @ q / scale
    dx = dq @ wq.T + dk @ wk.T + dv @ wv.T
    return float(loss), output, dx


def difference(actual, reference):
    actual, reference = (
        np.asarray(actual, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    return {
        "max_absolute_error": float(np.max(np.abs(actual - reference))),
        "relative_l2_error": float(
            np.linalg.norm(actual - reference)
            / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
        ),
    }


def run(output_dir):
    # Set before importing either backend. This script is a standalone process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import torch

    jax.config.update("jax_enable_x64", True)
    torch.set_num_threads(1)
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("This experiment requires CPU only")
    output_dir.mkdir(parents=True, exist_ok=False)
    x_np, weights_np, directions = fixture()
    ref_loss, ref_output, ref_grad = numpy_reference(x_np, weights_np)
    np.savez_compressed(
        output_dir / "fixture.npz",
        x=x_np,
        wq=weights_np[0],
        wk=weights_np[1],
        wv=weights_np[2],
        directions=directions,
        reference_gradient=ref_grad,
    )
    config = {
        "scope": "Fixed nonbiological arrays; no model, checkpoint, or sequence input",
        "backend": "CPU",
        "versions": {
            "jax": jax.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "function": "mean(square(softmax((X Wq / sqrt(3)) (X Wk).T) (X Wv)))",
        "fixture_seed": 0,
        "fixture_sha256": hashlib.sha256(
            b"".join(a.tobytes() for a in [x_np, *weights_np, directions])
        ).hexdigest(),
        "epsilons": EPSILONS,
        "smooth_reference_tolerances": TOLERANCES,
        "tolerance_scope": "Fixed small smooth synthetic function only; not full-model criteria",
        "mixed_bf16_protocol": "FP32 projections/scaling, then Q/K/V cast to BF16; "
        "score matmul output BF16; softmax explicitly FP32 then cast to BF16; "
        "attention-value matmul output BF16; final loss FP32. Backend-specific "
        "internal accumulation is not controlled.",
        "finite_difference_interpretation": "Nominal-step central differences; inspect the "
        "step-size curve. Smaller steps can amplify rounding. No range-overlap gate.",
        "full_model_gradient_validation_status": "not_assessed",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    observations, checks, artifacts = [], [], {}
    for precision in ("fp64", "fp32", "mixed_bf16"):
        jd = jnp.float64 if precision == "fp64" else jnp.float32
        td = torch.float64 if precision == "fp64" else torch.float32
        mixed = precision == "mixed_bf16"
        x = jnp.asarray(x_np, dtype=jd)
        jw = tuple(jnp.asarray(w, dtype=jd) for w in weights_np)

        def objective(a, w):
            q, k, v = (a @ matrix for matrix in w)
            q = q / jnp.asarray(np.sqrt(q.shape[-1]), dtype=a.dtype)
            if any(z.dtype != a.dtype for z in (q, k, v)):
                raise TypeError("Projection/scaling changed the declared precision")
            if mixed:
                q, k, v = (z.astype(jnp.bfloat16) for z in (q, k, v))
            scores = q @ k.T
            p = jax.nn.softmax(scores.astype(jd), axis=-1)
            if mixed:
                p = p.astype(jnp.bfloat16)
            out = (p @ v).astype(jd)
            return jnp.mean(out**2), out

        tx = torch.tensor(x_np, dtype=td, requires_grad=True)
        tw = tuple(torch.tensor(w, dtype=td) for w in weights_np)

        def torch_objective(a):
            q, k, v = (a @ matrix for matrix in tw)
            q = q / np.sqrt(q.shape[-1])
            if mixed:
                q, k, v = (z.to(torch.bfloat16) for z in (q, k, v))
            p = torch.softmax((q @ k.T).to(td), dim=-1)
            if mixed:
                p = p.to(torch.bfloat16)
            out = (p @ v).to(td)
            return (out**2).mean(), out

        torch_loss, torch_out = torch_objective(tx)
        torch_loss.backward()
        torch_grad = tx.grad.detach().numpy().copy()
        torch_out_np = torch_out.detach().numpy()
        torch_loss_value = float(torch_loss.detach())
        artifacts[f"{precision}_torch_gradient"] = torch_grad
        if precision in TOLERANCES:
            for name, actual, expected in (
                ("loss", torch_loss_value, ref_loss),
                ("output", torch_out_np, ref_output),
                ("gradient", torch_grad, ref_grad),
            ):
                checks.append(
                    {
                        "name": f"{precision}/torch/{name}/numpy_analytic",
                        "passed": bool(
                            np.allclose(actual, expected, **TOLERANCES[precision])
                        ),
                        **difference(actual, expected),
                    }
                )
        mode_results = {}
        for mode in ("eager", "jit"):
            forward = jax.jit(objective) if mode == "jit" else objective
            backward = jax.value_and_grad(objective, has_aux=True)
            if mode == "jit":
                backward = jax.jit(backward)
            (loss, out), grad = backward(x, jw)
            (repeat_loss, _), repeat_grad = backward(x, jw)
            forward_loss, _ = forward(x, jw)
            grad_np, out_np = np.asarray(grad), np.asarray(out)
            mode_results[mode] = (float(loss), out_np, grad_np)
            artifacts[f"{precision}_{mode}_gradient"] = grad_np
            comparison = {
                "precision": precision,
                "mode": mode,
                "input_dtype": str(x.dtype),
                "output_dtype": str(out.dtype),
                "gradient_dtype": str(grad.dtype),
                "loss": float(loss),
                "torch_loss": torch_loss_value,
                "forward_backward_primal_gap": abs(float(forward_loss) - float(loss)),
                "repeated_loss_identical": float(loss) == float(repeat_loss),
                "repeated_gradient_identical": bool(
                    np.array_equal(grad_np, np.asarray(repeat_grad))
                ),
                "output_vs_torch": difference(out_np, torch_out_np),
                "gradient_vs_torch": difference(grad_np, torch_grad),
                "gradient_vs_unquantized_reference": difference(grad_np, ref_grad),
                "finite_differences": [],
            }
            if precision in TOLERANCES:
                for name, actual, expected in (
                    ("loss", float(loss), ref_loss),
                    ("output", out_np, ref_output),
                    ("gradient", grad_np, ref_grad),
                ):
                    checks.append(
                        {
                            "name": f"{precision}/{mode}/{name}/numpy_analytic",
                            "passed": bool(
                                np.allclose(actual, expected, **TOLERANCES[precision])
                            ),
                            **difference(actual, expected),
                        }
                    )
            for direction_index, direction in enumerate(directions):
                d = jnp.asarray(direction, dtype=jd)
                ad = float(
                    np.sum(grad_np.astype(np.float64) * np.asarray(d, dtype=np.float64))
                )
                torch_ad = float(
                    np.sum(
                        torch_grad.astype(np.float64) * np.asarray(d, dtype=np.float64)
                    )
                )
                for eps in EPSILONS:
                    plus, minus = (
                        float(forward(x + eps * d, jw)[0]),
                        float(forward(x - eps * d, jw)[0]),
                    )
                    with torch.no_grad():
                        dt = torch.tensor(direction, dtype=td)
                        tp = float(torch_objective(tx + eps * dt)[0])
                        tm = float(torch_objective(tx - eps * dt)[0])
                    comparison["finite_differences"].append(
                        {
                            "direction": direction_index,
                            "epsilon": eps,
                            "jax_autodiff": ad,
                            "torch_autodiff": torch_ad,
                            "jax_finite_difference": (plus - minus) / (2 * eps),
                            "torch_finite_difference": (tp - tm) / (2 * eps),
                            "jax_absolute_gap": abs((plus - minus) / (2 * eps) - ad),
                        }
                    )
            observations.append(comparison)
            print(
                f"{precision}/{mode}: gradient relative error vs PyTorch "
                f"{comparison['gradient_vs_torch']['relative_l2_error']:.6g}",
                flush=True,
            )
        eager, compiled = mode_results["eager"], mode_results["jit"]
        observations[-1]["jit_vs_eager"] = {
            "loss_absolute_gap": abs(eager[0] - compiled[0]),
            "output": difference(compiled[1], eager[1]),
            "gradient": difference(compiled[2], eager[2]),
        }
    finite = all(np.isfinite(array).all() for array in artifacts.values())
    checks.append({"name": "all_saved_gradients_finite", "passed": bool(finite)})
    passed = all(check["passed"] for check in checks)
    np.savez_compressed(output_dir / "gradients.npz", **artifacts)
    report = {
        "schema_version": 1,
        "experiment_completed": True,
        "smooth_synthetic_reference_checks_passed": passed,
        "mixed_bf16_correctness_status": "not_assessed",
        "full_model_gradient_validation_status": "not_assessed",
        "checks": checks,
        "observations": observations,
        "interpretation": "Smooth FP64/FP32 checks apply only to this fixed synthetic "
        "attention function. Mixed BF16 and finite differences are descriptive. "
        "This experiment neither loads nor validates a biological model.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Smooth synthetic reference checks: {'passed' if passed else 'failed'}")
    print(f"Report: {output_dir / 'summary.json'}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output
        or Path(__file__).resolve().parents[1]
        / "results"
        / f"synthetic_attention_{stamp}_{os.getpid()}"
    )
    return run(output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
