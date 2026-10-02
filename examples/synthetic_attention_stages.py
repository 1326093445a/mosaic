"""Stage-level CPU/GPU attention diagnostics on seeded nonbiological arrays.

No checkpoints, sequences, or shared model kernels are loaded or modified.
Observations localize numerical differences; they are not model validation.
"""

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
import time
import traceback
from pathlib import Path

import numpy as np

from synthetic_attention_numerics import (
    TOLERANCES,
    difference,
    fixture,
    numpy_reference,
)

VARIANTS = {
    "fp32": frozenset(),
    "qkv_rounding_only": frozenset({"qkv"}),
    "score_matmul_only": frozenset({"scores"}),
    "probability_rounding_only": frozenset({"probabilities"}),
    "output_matmul_only": frozenset({"output"}),
    "original_mixed_bf16": frozenset({"qkv", "scores", "probabilities", "output"}),
}


def compare(a, b):
    return {"identical": bool(np.array_equal(a, b)), **difference(a, b)}


SIZES = {"small": (4, 5, 3, 2), "medium": (64, 32, 16, 16), "large": (256, 64, 32, 32)}


def numeric_fixture(seed, size):
    if size == "small":
        return fixture(seed)
    n, width, head, value = SIZES[size]
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, width)) / 2
    weights = [
        rng.normal(size=(width, w)) / np.sqrt(width) for w in (head, head, value)
    ]
    return x, weights, None


def run(
    root,
    *,
    backend="cpu",
    device="0",
    seed=0,
    size="small",
    precision_mode="native",
    repeats=3,
):
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if backend == "cpu" else str(device)
    os.environ["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import torch

    jax.config.update("jax_enable_x64", True)
    torch.set_num_threads(1)
    expected = "cpu" if backend == "cpu" else "gpu"
    if len(jax.devices()) != 1 or any(d.platform != expected for d in jax.devices()):
        raise RuntimeError(f"Expected exactly one {expected} JAX device")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is required; CPU-only PyTorch cannot run this GPU comparison"
        )
    torch_device = "cpu" if backend == "cpu" else "cuda:0"
    if precision_mode == "strict":
        jax.config.update("jax_default_matmul_precision", "highest")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()

    def memory():
        try:
            stats = jax.devices()[0].memory_stats()
            jm = {"supported": stats is not None, "stats": stats}
        except Exception as exc:
            jm = {"supported": False, "error": str(exc)}
        tm = {"supported": backend == "cuda"}
        if backend == "cuda":
            tm.update(
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
        return {"jax": jm, "torch": tm}

    def measured(label, fn, engine="jax"):
        event = {
            "label": label,
            "utc": datetime.now(timezone.utc).isoformat(),
            "memory_before": memory(),
        }
        began = time.monotonic()
        print(f"START {label}", flush=True)
        try:
            result = fn()
            if engine == "jax":
                jax.block_until_ready(result)
            elif engine == "torch" and backend == "cuda":
                torch.cuda.synchronize()
            event["status"] = "ok"
            return result
        except Exception as exc:
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event.update(seconds=time.monotonic() - began, memory_after=memory())
            with (root / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
            print(
                f"END {label}: {event['status']} ({event['seconds']:.3f}s)", flush=True
            )

    x_np, weights_np, _ = numeric_fixture(seed, size)
    x = jnp.asarray(x_np, dtype=jnp.float32)
    jw = tuple(jnp.asarray(w, dtype=jnp.float32) for w in weights_np)
    tw = tuple(
        torch.tensor(w, dtype=torch.float32, device=torch_device) for w in weights_np
    )
    np.savez_compressed(
        root / "fixture.npz",
        x=x_np,
        wq=weights_np[0],
        wk=weights_np[1],
        wv=weights_np[2],
    )
    config = {
        "scope": "Seeded synthetic attention only; no biological model",
        "backend": backend,
        "jax_device": str(jax.devices()[0]),
        "jax_device_kind": jax.devices()[0].device_kind,
        "torch_device": torch_device,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_name": torch.cuda.get_device_name(0)
        if backend == "cuda"
        else "CPU",
        "seed": seed,
        "size": size,
        "dimensions": SIZES[size],
        "repeats": repeats,
        "precision_mode": precision_mode,
        "jax_matmul_precision": jax.config.jax_default_matmul_precision,
        "torch_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "NVIDIA_TF32_OVERRIDE",
                "JAX_DEFAULT_MATMUL_PRECISION",
            )
        },
        "memory_note": "Per-process cumulative framework allocation counters, not incremental backward memory; CPU JAX counters may be unsupported.",
        "versions": {
            "jax": jax.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
        },
        "variants": {name: sorted(flags) for name, flags in VARIANTS.items()},
        "stage_definitions": {
            "qkv": "Round projected/scaled Q, K, V to BF16. Other stages remain FP32 unless selected.",
            "scores": "Cast Q/K matmul operands to BF16 and produce BF16 scores; softmax remains FP32.",
            "probabilities": "Round FP32 softmax probabilities to BF16.",
            "output": "Cast probability/value matmul operands to BF16 and produce BF16 output; loss remains FP32.",
        },
        "precision_note": "Matmul variants necessarily change operand and result precision together. "
        "They are not isolated changes to hardware accumulation precision.",
        "instrumentation_note": "Intermediate gradients are measured by independent zero-valued "
        "additive probes at each stage. Additional outputs/probes can alter compilation. "
        "Their effect on the original output-only objective is measured explicitly.",
        "stage_screening_tolerance": TOLERANCES["fp32"],
        "stage_screening_note": "Exploratory screening across seeded synthetic sizes; used to distinguish small FP32 differences in this "
        "fixture from larger differences; not a BF16 correctness threshold.",
        "full_model_gradient_validation_status": "not_assessed",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, arrays, checks = [], {}, []
    for name, flags in VARIANTS.items():

        def jax_graph(a, weights, offsets=None):
            stages = {}

            def stage(label, value):
                if offsets is not None:
                    value = value + offsets[label]
                stages[label] = value
                return value

            q = stage("q_projection", a @ weights[0])
            k = stage("k_projection", a @ weights[1])
            v = stage("v_projection", a @ weights[2])
            q = stage("q_scaled", q / jnp.asarray(np.sqrt(q.shape[-1]), dtype=a.dtype))
            if q.dtype != a.dtype:
                raise TypeError("Unexpected scaling promotion")
            boundary = jnp.bfloat16 if "qkv" in flags else jnp.float32
            q = stage("q_boundary", q.astype(boundary))
            k = stage("k_boundary", k.astype(boundary))
            v = stage("v_boundary", v.astype(boundary))
            score_dtype = jnp.bfloat16 if "scores" in flags else jnp.float32
            qs = stage("q_score_operand", q.astype(score_dtype))
            ks = stage("k_score_operand", k.astype(score_dtype))
            scores = stage("scores", qs @ ks.T)
            p = stage(
                "softmax_fp32", jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
            )
            p = stage(
                "probability_boundary",
                p.astype(jnp.bfloat16 if "probabilities" in flags else jnp.float32),
            )
            out_dtype = jnp.bfloat16 if "output" in flags else jnp.float32
            po = stage("probability_output_operand", p.astype(out_dtype))
            vo = stage("value_output_operand", v.astype(out_dtype))
            out = stage("output_product", po @ vo)
            out = stage("output_fp32", out.astype(jnp.float32))
            return jnp.mean(out**2), stages

        def original(a, weights):
            loss, stages = jax_graph(a, weights)
            return loss, stages["output_fp32"]

        def torch_graph(a):
            stages = {}

            def stage(label, value):
                value.retain_grad()
                stages[label] = value
                return value

            q = stage("q_projection", a @ tw[0])
            k = stage("k_projection", a @ tw[1])
            v = stage("v_projection", a @ tw[2])
            q = stage("q_scaled", q / float(np.sqrt(q.shape[-1])))
            boundary = torch.bfloat16 if "qkv" in flags else torch.float32
            q = stage("q_boundary", q.to(boundary))
            k = stage("k_boundary", k.to(boundary))
            v = stage("v_boundary", v.to(boundary))
            score_dtype = torch.bfloat16 if "scores" in flags else torch.float32
            qs = stage("q_score_operand", q.to(score_dtype))
            ks = stage("k_score_operand", k.to(score_dtype))
            scores = stage("scores", qs @ ks.T)
            p = stage("softmax_fp32", torch.softmax(scores.float(), dim=-1))
            p = stage(
                "probability_boundary",
                p.to(torch.bfloat16 if "probabilities" in flags else torch.float32),
            )
            out_dtype = torch.bfloat16 if "output" in flags else torch.float32
            po = stage("probability_output_operand", p.to(out_dtype))
            vo = stage("value_output_operand", v.to(out_dtype))
            out = stage("output_product", po @ vo)
            out = stage("output_fp32", out.float())
            return (out**2).mean(), stages

        tx = torch.tensor(
            x_np, dtype=torch.float32, device=torch_device, requires_grad=True
        )

        def torch_evaluate():
            value, stages = torch_graph(tx)
            value.backward()
            return value, stages

        tloss, tstage = measured(f"{name}/torch", torch_evaluate, "torch")
        tgrad = tx.grad.detach().cpu().numpy()
        tvalues = {k: v.detach().float().cpu().numpy() for k, v in tstage.items()}
        tgrads = {k: v.grad.detach().float().cpu().numpy() for k, v in tstage.items()}
        arrays[f"{name}/torch/input_gradient"] = tgrad
        for label in tvalues:
            arrays[f"{name}/torch/values/{label}"] = tvalues[label]
            arrays[f"{name}/torch/stage_gradients/{label}"] = tgrads[label]
        if name == "fp32":
            rl, ro, rg = numpy_reference(x_np, weights_np)
            for label, actual, ref in (
                ("loss", float(tloss.detach()), rl),
                ("output", tvalues["output_fp32"], ro),
                ("gradient", tgrad, rg),
            ):
                checks.append(
                    {
                        "name": f"fp32/torch/{label}/numpy_reference",
                        "passed": bool(np.allclose(actual, ref, **TOLERANCES["fp32"])),
                    }
                )
        baselines, instrumented = {}, {}
        for mode in ("eager", "jit"):
            backward = jax.value_and_grad(original, has_aux=True)
            if mode == "jit":
                backward = jax.jit(backward)
            (loss, out), grad = measured(
                f"{name}/{mode}/baseline", lambda: backward(x, jw)
            )
            repeat_records = []
            for repeat_index in range(1, repeats):
                (repeat_loss, repeat_out), repeat_grad = measured(
                    f"{name}/{mode}/repeat{repeat_index}", lambda: backward(x, jw)
                )
                repeat_records.append(
                    {
                        "repeat_index": repeat_index,
                        "loss_absolute_gap": abs(float(repeat_loss) - float(loss)),
                        "output": compare(np.asarray(repeat_out), np.asarray(out)),
                        "gradient": compare(np.asarray(repeat_grad), np.asarray(grad)),
                    }
                )
            baselines[mode] = (float(loss), np.asarray(out), np.asarray(grad))
            _, template = jax_graph(x, jw)
            zeros = {k: jnp.zeros_like(v) for k, v in template.items()}

            def observed(a, offsets, weights):
                return jax_graph(a, weights, offsets)

            observed_backward = jax.value_and_grad(
                observed, argnums=(0, 1), has_aux=True
            )
            if mode == "jit":
                observed_backward = jax.jit(observed_backward)
                ir_dir = root / "compiled_ir"
                ir_dir.mkdir(exist_ok=True)

                def export_ir():
                    (ir_dir / f"{name}_original_backward.hlo.txt").write_text(
                        backward.lower(x, jw).compile().as_text()
                    )
                    (ir_dir / f"{name}_instrumented_backward.hlo.txt").write_text(
                        observed_backward.lower(x, zeros, jw).compile().as_text()
                    )

                measured(f"{name}/{mode}/compile_ir", export_ir, "host")
            (observed_loss, stages), (observed_grad, stage_grads) = measured(
                f"{name}/{mode}/instrumented", lambda: observed_backward(x, zeros, jw)
            )
            instrumented[mode] = (stages, stage_grads)
            arrays[f"{name}/{mode}/input_gradient"] = np.asarray(grad)
            stage_rows = []
            for label, value in stages.items():
                value_np = np.asarray(value, dtype=np.float32)
                sg = np.asarray(stage_grads[label], dtype=np.float32)
                arrays[f"{name}/{mode}/values/{label}"] = value_np
                arrays[f"{name}/{mode}/stage_gradients/{label}"] = sg
                row = {
                    "stage": label,
                    "jax_dtype": str(value.dtype),
                    "torch_dtype": str(tstage[label].dtype),
                    "values_vs_torch": compare(value_np, tvalues[label]),
                    "stage_gradient_vs_torch": compare(sg, tgrads[label]),
                }
                if mode == "jit":
                    ev, eg = instrumented["eager"]
                    row["values_vs_eager"] = compare(
                        value_np, np.asarray(ev[label], dtype=np.float32)
                    )
                    row["stage_gradient_vs_eager"] = compare(
                        sg, np.asarray(eg[label], dtype=np.float32)
                    )
                stage_rows.append(row)
            result = {
                "variant": name,
                "mode": mode,
                "loss": float(loss),
                "loss_vs_torch_absolute_gap": abs(float(loss) - float(tloss.detach())),
                "output_vs_torch": compare(np.asarray(out), tvalues["output_fp32"]),
                "input_gradient_vs_torch": compare(np.asarray(grad), tgrad),
                "repeat_output_identical": all(
                    r["output"]["identical"] for r in repeat_records
                ),
                "repeat_gradient_identical": all(
                    r["gradient"]["identical"] for r in repeat_records
                ),
                "repeats": repeat_records,
                "instrumentation_effect": {
                    "loss_absolute_gap": abs(float(observed_loss) - float(loss)),
                    "output": compare(
                        np.asarray(stages["output_fp32"]), np.asarray(out)
                    ),
                    "input_gradient": compare(
                        np.asarray(observed_grad), np.asarray(grad)
                    ),
                },
                "stages": stage_rows,
            }
            if mode == "jit":
                result["original_jit_vs_eager"] = {
                    "output": compare(np.asarray(out), baselines["eager"][1]),
                    "input_gradient": compare(np.asarray(grad), baselines["eager"][2]),
                }
                # Use dependency order from the eager Python graph, not sorted pytree order.
                result["first_observed_value_stage_outside_fp32_tolerance"] = next(
                    (
                        label
                        for label in template
                        if not np.allclose(
                            np.asarray(stages[label], dtype=np.float32),
                            np.asarray(
                                instrumented["eager"][0][label], dtype=np.float32
                            ),
                            **TOLERANCES["fp32"],
                        )
                    ),
                    None,
                )
            if name == "fp32":
                rl, ro, rg = numpy_reference(x_np, weights_np)
                for label, actual, ref in (
                    ("loss", float(loss), rl),
                    ("output", np.asarray(out), ro),
                    ("gradient", np.asarray(grad), rg),
                ):
                    checks.append(
                        {
                            "name": f"fp32/{mode}/{label}/numpy_reference",
                            "passed": bool(
                                np.allclose(actual, ref, **TOLERANCES["fp32"])
                            ),
                        }
                    )
            rows.append(result)
            with (root / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                f"{name}/{mode}: input gradient relative difference vs PyTorch "
                f"{result['input_gradient_vs_torch']['relative_l2_error']:.6g}; "
                f"instrumentation effect {result['instrumentation_effect']['input_gradient']['relative_l2_error']:.6g}",
                flush=True,
            )
    checks.append(
        {
            "name": "all_saved_arrays_finite",
            "passed": all(bool(np.isfinite(a).all()) for a in arrays.values()),
        }
    )
    checks.append(
        {
            "name": "all_original_repeats_identical",
            "passed": all(
                r["repeat_output_identical"] and r["repeat_gradient_identical"]
                for r in rows
            ),
        }
    )
    np.savez_compressed(root / "arrays.npz", **arrays)
    report = {
        "experiment_completed": True,
        "elapsed_seconds": time.monotonic() - start,
        "memory_at_completion": memory(),
        "synthetic_control_checks_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "observations": rows,
        "mixed_bf16_correctness_status": "not_assessed",
        "full_model_gradient_validation_status": "not_assessed",
        "interpretation": "First observed divergence belongs to the instrumented graph; "
        "consult instrumentation_effect before attributing it to the original compiled graph. "
        "Single-stage variants identify sensitivity, not unique causation.",
    }
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Report: {root / 'summary.json'}")
    return 0 if report["synthetic_control_checks_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=tuple(SIZES), default="small")
    parser.add_argument(
        "--precision-mode", choices=("strict", "native"), default="native"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.seed < 0 or args.repeats < 2:
        parser.error("seed must be nonnegative and repeats must be at least two")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = (
        args.output
        or Path(__file__).resolve().parents[1]
        / "results"
        / f"synthetic_attention_stages_{stamp}_{os.getpid()}"
    )
    root = root.resolve()
    if root.exists():
        parser.error(f"Output already exists; choose a new directory: {root}")
    try:
        return run(
            root,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
            size=args.size,
            precision_mode=args.precision_mode,
            repeats=args.repeats,
        )
    except Exception as exc:
        # Preserve exception details even when setup fails before model-free evaluation.
        if root.exists():
            (root / "failure.json").write_text(
                json.dumps(
                    {
                        "experiment_completed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
