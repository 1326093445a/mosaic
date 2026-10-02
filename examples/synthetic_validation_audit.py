"""Independent artifact checks for the synthetic attention pipeline only.

No biological inputs, checkpoint loading, or shared model code. Exact equality
is a repeatability observation, never a certificate of full-model correctness.
"""

import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from synthetic_attention_numerics import TOLERANCES, difference, numpy_reference
from synthetic_attention_stages import numeric_fixture


def fixture_digest(values):
    return hashlib.sha256(b"".join(a.tobytes() for a in values)).hexdigest()


def comparison(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(
            "Comparison requires identical shapes and dtypes; broadcasting is forbidden"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison requires finite arrays")
    return {"identical": bool(np.array_equal(a, b)), **difference(a, b)}


def audit(root, statuses):
    """Recompute reference and repeat checks from saved, uninstrumented arrays."""
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    integrity, numerical, within, across, forward_paths = [], [], [], [], []
    same_device, cross_device = {}, {}
    signatures = []

    def check(name, passed):
        integrity.append({"check": name, "passed": bool(passed)})

    def valid_array(label, value, shape):
        valid = (
            value.shape == shape
            and value.dtype == np.dtype("float32")
            and np.isfinite(value).all()
        )
        check(f"{label}/shape_dtype_finite", valid)
        return valid

    def record_numerical(job, mode, evaluation, quantity, actual, expected):
        numerical.append(
            {
                "worker": job["name"],
                "precision_mode": job["precision_mode"],
                "mode": mode,
                "evaluation": evaluation,
                "quantity": quantity,
                "passed": bool(np.allclose(actual, expected, **TOLERANCES["fp32"])),
                **difference(actual, expected),
            }
        )

    check(
        "planned_worker_names",
        [j["name"] for j in plan["jobs"]] == [j["name"] for j in statuses],
    )
    for job in statuses:
        name = job["name"]
        folder = root / "workers" / name
        check(f"{name}/completed", job["state"] in ("complete", "checks_failed"))
        try:
            config = json.loads((folder / "config.json").read_text())
            report = json.loads((folder / "summary.json").read_text())
            check(f"{name}/report_completed", report["experiment_completed"])
            check(f"{name}/fixed_device", job.get("assigned_device") == job["device"])
            check(
                f"{name}/visible_device",
                config["environment"]["CUDA_VISIBLE_DEVICES"]
                == ("" if plan["jax_backend"] == "cpu" else job["device"]),
            )
            check(
                f"{name}/requested_config",
                all(config[k] == job[k] for k in ("seed", "size", "precision_mode"))
                and config["repeats"] == plan["repeats"]
                and set(config["variants"]) == set(plan["variants"]),
            )
            check(
                f"{name}/backend",
                config["backend"] == plan["jax_backend"]
                and config["torch_backend"] == plan["torch_backend"],
            )
            if job["precision_mode"] == "strict":
                check(
                    f"{name}/strict_policy",
                    config["jax_matmul_precision"] == "highest"
                    and config["torch_matmul_precision"] == "highest"
                    and not config["torch_allow_tf32"],
                )
            signature = json.dumps(
                {
                    k: config[k]
                    for k in (
                        "versions",
                        "jax_device_kind",
                        "torch_device_name",
                        "backend",
                        "torch_backend",
                    )
                },
                sort_keys=True,
            )
            # Include effective cache/environment settings without mixing CUDA ordinals
            # into cross-device matching or strict/native precision policies together.
            signature += json.dumps(
                {
                    "cache": config.get("jax_compilation_cache"),
                    "environment": {
                        k: v
                        for k, v in config["environment"].items()
                        if k != "CUDA_VISIBLE_DEVICES"
                    },
                },
                sort_keys=True,
            )
            signatures.append(signature)
            with np.load(folder / "fixture.npz", allow_pickle=False) as data:
                inputs = [data[k] for k in ("x", "wq", "wk", "wv")]
            x, weights, _ = numeric_fixture(job["seed"], job["size"])
            expected = [x, *weights]
            check(
                f"{name}/seeded_fixture",
                all(
                    np.array_equal(a, b) for a, b in zip(inputs, expected, strict=True)
                ),
            )
            digest = fixture_digest(inputs)
            check(f"{name}/fixture_hash", digest == config["fixture_sha256"])
            # Compare to the independent FP64 analytic reference at actual FP32 inputs.
            exact_inputs = [a.astype(np.float32).astype(np.float64) for a in inputs]
            ref_loss, ref_out, ref_grad = numpy_reference(
                exact_inputs[0], exact_inputs[1:]
            )
            reference = {
                "original_loss": np.asarray(ref_loss),
                "original_output": ref_out,
                "input_gradient": ref_grad,
            }
            rows = report["observations"]
            check(
                f"{name}/observation_coverage",
                len(rows) == len(plan["variants"]) * 2
                and {(r["variant"], r["mode"]) for r in rows}
                == {(v, m) for v in plan["variants"] for m in ("eager", "jit")},
            )
            with np.load(folder / "arrays.npz", allow_pickle=False) as data:
                for variant in plan["variants"]:
                    for mode in ("torch", "eager", "jit"):
                        prefix = f"{variant}/{mode}"
                        originals = {k: data[f"{prefix}/{k}"] for k in reference}
                        valid = [
                            valid_array(
                                f"{name}/{prefix}/{key}", value, reference[key].shape
                            )
                            for key, value in originals.items()
                        ]
                        if not all(valid):
                            continue
                        for key, value in originals.items():
                            if variant == "fp32":
                                record_numerical(
                                    job, mode, "baseline", key, value, reference[key]
                                )
                        # Only the smooth FP32 control gets a consistency gate.
                        # BF16 materialization/fusion can affect what is rounded.
                        if variant == "fp32":
                            check(
                                f"{name}/{prefix}/loss_output_consistency",
                                np.allclose(
                                    originals["original_loss"],
                                    np.mean(
                                        originals["original_output"].astype(np.float64)
                                        ** 2
                                    ),
                                    rtol=2e-6,
                                    atol=1e-9,
                                ),
                            )
                        if mode != "torch":
                            observation = next(
                                r
                                for r in rows
                                if r["variant"] == variant and r["mode"] == mode
                            )
                            check(
                                f"{name}/{prefix}/reported_loss",
                                float(originals["original_loss"])
                                == observation["loss"],
                            )
                            check(
                                f"{name}/{prefix}/repeat_coverage",
                                len(observation["repeats"]) == plan["repeats"] - 1,
                            )
                            for repeat in range(1, plan["repeats"]):
                                for key, value in originals.items():
                                    actual = data[f"{prefix}/repeat{repeat}/{key}"]
                                    if not valid_array(
                                        f"{name}/{prefix}/repeat{repeat}/{key}",
                                        actual,
                                        reference[key].shape,
                                    ):
                                        continue
                                    if variant == "fp32":
                                        record_numerical(
                                            job,
                                            mode,
                                            f"repeat{repeat}",
                                            key,
                                            actual,
                                            reference[key],
                                        )
                                    within.append(
                                        {
                                            "worker": name,
                                            "variant": variant,
                                            "mode": mode,
                                            "repeat": repeat,
                                            "quantity": key,
                                            **comparison(actual, value),
                                        }
                                    )
                            for key in ("original_loss", "original_output"):
                                actual = data[f"{prefix}/forward_only/{key}"]
                                if not valid_array(
                                    f"{name}/{prefix}/forward_only/{key}",
                                    actual,
                                    reference[key].shape,
                                ):
                                    continue
                                forward_paths.append(
                                    {
                                        "worker": name,
                                        "variant": variant,
                                        "mode": mode,
                                        "quantity": key,
                                        **comparison(actual, originals[key]),
                                    }
                                )
                                if variant == "fp32":
                                    record_numerical(
                                        job,
                                        mode,
                                        "forward_only",
                                        key,
                                        actual,
                                        reference[key],
                                    )
                                    record_numerical(
                                        job,
                                        mode,
                                        "forward_vs_backward",
                                        key,
                                        actual,
                                        originals[key],
                                    )
                        group = (
                            job["seed"],
                            job["size"],
                            job["precision_mode"],
                            variant,
                            mode,
                        )
                        for kind, cache, group_key in (
                            (
                                "same_device_fresh_process",
                                same_device,
                                (*group, job["device"]),
                            ),
                            ("cross_device_first_round", cross_device, group),
                        ):
                            if kind == "cross_device_first_round" and job["round"] != 0:
                                continue
                            if group_key not in cache:
                                cache[group_key] = (job, digest, signature, originals)
                                continue
                            previous, prev_digest, prev_signature, baseline = cache[
                                group_key
                            ]
                            matched = (
                                digest == prev_digest and signature == prev_signature
                            )
                            check(
                                f"{name}/{prefix}/{kind}/matched_fixture_environment",
                                matched,
                            )
                            for key, value in originals.items():
                                across.append(
                                    {
                                        "comparison": kind,
                                        "reference_worker": previous["name"],
                                        "worker": name,
                                        "reference_device": previous["device"],
                                        "device": job["device"],
                                        "variant": variant,
                                        "mode": mode,
                                        "quantity": key,
                                        "matched_fixture_environment": matched,
                                        **comparison(value, baseline[key]),
                                    }
                                )
                events = [
                    json.loads(line)
                    for line in (folder / "events.jsonl").read_text().splitlines()
                ]
                expected_labels = {f"{v}/torch" for v in plan["variants"]}
                for v in plan["variants"]:
                    for mode in ("eager", "jit"):
                        expected_labels.update(
                            {
                                f"{v}/{mode}/baseline",
                                f"{v}/{mode}/instrumented",
                                f"{v}/{mode}/forward_only",
                            }
                        )
                        expected_labels.update(
                            f"{v}/{mode}/repeat{i}" for i in range(1, plan["repeats"])
                        )
                    expected_labels.add(f"{v}/jit/compile_ir")
                    check(
                        f"{name}/{v}/compiled_ir_present",
                        all(
                            (folder / "compiled_ir" / f"{v}_{label}_backward.hlo.txt")
                            .stat()
                            .st_size
                            > 0
                            for label in ("original", "instrumented")
                        ),
                    )
                check(
                    f"{name}/events",
                    len(events) == len(expected_labels)
                    and {e["label"] for e in events} == expected_labels
                    and all(e["status"] == "ok" for e in events),
                )
        except Exception as exc:
            integrity.append(
                {
                    "check": f"{name}/artifacts_readable",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    check(
        "fixed_software_and_hardware_kind",
        bool(signatures) and len(set(signatures)) == 1,
    )
    strict = [row for row in numerical if row["precision_mode"] == "strict"]
    integration_ok = bool(integrity) and all(c["passed"] for c in integrity)
    same = [r for r in across if r["comparison"] == "same_device_fresh_process"]
    repeat_status = "not_assessed"
    if integration_ok and same and within:
        repeat_status = (
            "identical"
            if all(r["identical"] for r in same + within)
            else "variation_observed"
        )
    numerical_status = "not_assessed"
    if integration_ok and strict:
        numerical_status = "passed" if all(r["passed"] for r in strict) else "failed"
    return {
        "schema_version": 2,
        "scope": "End-to-end synthetic pipeline, not a biological or full-model validation",
        "axes": {
            "strict_fp32_numerical_controls": {
                "status": numerical_status,
                "tolerances": TOLERANCES["fp32"],
                "checked_quantities": len(strict),
            },
            "repeatability": {
                "status": repeat_status,
                "within_process_rows": len(within),
                "same_device_fresh_process_rows": len(same),
                "fresh_process_nonidentical_rows": sum(
                    not r["identical"] for r in same
                ),
                "cross_device_rows": len(across) - len(same),
            },
            "synthetic_pipeline_integrity": {
                "status": "passed" if integration_ok else "failed"
            },
            "full_model_correctness": {
                "status": "not_assessed",
                "reason": "No model, checkpoint, biological input, or full-model memory measurement is included.",
            },
        },
        "interpretation": "BF16 and native-policy gaps are observations, not backward-bug diagnoses. Repeat variation is reported separately and does not by itself fail strict numerical controls. Exact agreement does not establish correctness.",
        "integrity_checks": integrity,
        "numerical_checks": numerical,
        "within_process": within,
        "between_processes": across,
        "forward_path_comparisons": forward_paths,
    }


def verify_archive(root, archive):
    """Verify actual archived bytes and exact file coverage, without extraction."""
    root, archive = Path(root), Path(archive)
    manifest = (root / "checksums.sha256").read_text()
    expected = {
        f"{root.name}/{name}": digest
        for digest, name in (line.split("  ", 1) for line in manifest.splitlines())
    }
    expected[f"{root.name}/checksums.sha256"] = hashlib.sha256(
        manifest.encode()
    ).hexdigest()
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.isdir():
                continue
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in seen
            ):
                raise ValueError(f"Unexpected archive entry: {member.name}")
            stream = handle.extractfile(member)
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != expected[member.name]
            ):
                raise ValueError(f"Archive checksum mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive is missing manifest entries")
    return {"archive_verified": True, "verified_files_including_manifest": len(seen)}
