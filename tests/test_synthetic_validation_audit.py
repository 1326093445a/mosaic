"""Negative controls for synthetic evidence, without importing a GPU framework."""

import importlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest


@pytest.fixture
def audit_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("synthetic_validation_audit")


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(audit_module, tmp_path):
    """Small valid artifact bundle for adversarial audit tests, not a GPU mock."""
    x, weights, _ = audit_module.numeric_fixture(0, "small")
    inputs = [x, *weights]
    actual = [a.astype(np.float32).astype(np.float64) for a in inputs]
    loss, out, grad = audit_module.numpy_reference(actual[0], actual[1:])
    arrays = {}
    for mode in ("torch", "eager", "jit"):
        original = {
            "original_loss": np.asarray(loss, dtype=np.float32),
            "original_output": out.astype(np.float32),
            "input_gradient": grad.astype(np.float32),
        }
        arrays.update({f"fp32/{mode}/{k}": v for k, v in original.items()})
        if mode != "torch":
            arrays.update({f"fp32/{mode}/repeat1/{k}": v for k, v in original.items()})
            arrays.update(
                {
                    f"fp32/{mode}/forward_only/{k}": v
                    for k, v in original.items()
                    if k != "input_gradient"
                }
            )
    statuses = []
    for index in range(2):
        name = f"round{index}"
        job = dict(
            name=name,
            seed=0,
            size="small",
            precision_mode="strict",
            round=index,
            assigned_device="cpu",
            device="cpu",
            state="complete",
        )
        statuses.append(job)
        folder = tmp_path / "workers" / name
        folder.mkdir(parents=True)
        save(
            folder / "config.json",
            dict(
                seed=0,
                size="small",
                precision_mode="strict",
                repeats=2,
                variants={"fp32": []},
                environment={"CUDA_VISIBLE_DEVICES": ""},
                backend="cpu",
                torch_backend="cpu",
                jax_matmul_precision="highest",
                torch_matmul_precision="highest",
                torch_allow_tf32=False,
                versions={"jax": "fixture"},
                jax_device_kind="CPU",
                torch_device_name="CPU",
                fixture_sha256=audit_module.fixture_digest(inputs),
            ),
        )
        save(
            folder / "summary.json",
            {
                "experiment_completed": True,
                "observations": [
                    {
                        "variant": "fp32",
                        "mode": mode,
                        "loss": float(np.float32(loss)),
                        "repeats": [{}],
                    }
                    for mode in ("eager", "jit")
                ],
            },
        )
        np.savez(
            folder / "fixture.npz", x=x, wq=weights[0], wk=weights[1], wv=weights[2]
        )
        np.savez(folder / "arrays.npz", **arrays)
        labels = ["fp32/torch", "fp32/jit/compile_ir"] + [
            f"fp32/{mode}/{event}"
            for mode in ("eager", "jit")
            for event in ("baseline", "repeat1", "instrumented", "forward_only")
        ]
        (folder / "events.jsonl").write_text(
            "\n".join(json.dumps({"label": label, "status": "ok"}) for label in labels)
        )
        (folder / "compiled_ir").mkdir()
        for kind in ("original", "instrumented"):
            (folder / "compiled_ir" / f"fp32_{kind}_backward.hlo.txt").write_text(
                "synthetic test placeholder"
            )
    save(
        tmp_path / "plan.json",
        {
            "jobs": statuses,
            "repeats": 2,
            "variants": ["fp32"],
            "jax_backend": "cpu",
            "torch_backend": "cpu",
        },
    )
    return tmp_path, statuses


def test_independent_axes_never_claim_full_model_validation(audit_module, bundle):
    axes = audit_module.audit(*bundle)["axes"]
    assert axes["strict_fp32_numerical_controls"]["status"] == "passed"
    assert axes["synthetic_pipeline_integrity"]["status"] == "passed"
    assert axes["repeatability"]["status"] == "identical"
    assert axes["full_model_correctness"]["status"] == "not_assessed"


def test_wrong_gradient_is_detected_despite_successful_worker(audit_module, bundle):
    root, _ = bundle
    path = root / "workers/round0/arrays.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays["fp32/jit/input_gradient"] *= 2
    np.savez(path, **arrays)
    axes = audit_module.audit(*bundle)["axes"]
    assert axes["strict_fp32_numerical_controls"]["status"] == "failed"
    assert axes["repeatability"]["status"] == "variation_observed"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_array",
        "wrong_device",
        "changed_version",
        "missing_event",
        "changed_fixture",
        "wrong_reported_loss",
    ],
)
def test_broken_evidence_cannot_pass_integration(audit_module, bundle, damage):
    root, jobs = bundle
    folder = root / "workers/round1"
    if damage == "missing_array":
        (folder / "arrays.npz").unlink()
    elif damage == "wrong_device":
        jobs[1]["device"] = "7"
    elif damage == "changed_version":
        path = folder / "config.json"
        value = json.loads(path.read_text())
        value["versions"]["jax"] = "changed"
        save(path, value)
    elif damage == "missing_event":
        (folder / "events.jsonl").write_text("")
    elif damage == "changed_fixture":
        path = folder / "fixture.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
        arrays["x"] += 1
        np.savez(path, **arrays)
    elif damage == "wrong_reported_loss":
        path = folder / "summary.json"
        value = json.loads(path.read_text())
        value["observations"][0]["loss"] += 10
        save(path, value)
    axes = audit_module.audit(root, jobs)["axes"]
    assert axes["synthetic_pipeline_integrity"]["status"] == "failed"
    assert axes["strict_fp32_numerical_controls"]["status"] == "not_assessed"
    assert axes["repeatability"]["status"] == "not_assessed"


def test_archive_verifier_reads_bytes_not_just_manifest(audit_module, tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("original")
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (root / "checksums.sha256").write_text(f"{digest}  payload.txt\n")
    archive = tmp_path / "bundle.tar.gz"

    def pack():
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(root, arcname=root.name)

    pack()
    assert (
        audit_module.verify_archive(root, archive)["verified_files_including_manifest"]
        == 2
    )
    payload.write_text("corrupted")
    pack()
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_module.verify_archive(root, archive)


def test_same_device_and_cross_device_comparisons_are_separate(
    audit_module, bundle, tmp_path
):
    import shutil

    original, jobs = bundle
    root = tmp_path / "two_device_schema"
    statuses = []
    for device in ("0", "1"):
        for job in jobs:
            current = {
                **job,
                "name": f"device{device}_{job['name']}",
                "assigned_device": device,
                "device": device,
            }
            statuses.append(current)
            destination = root / "workers" / current["name"]
            shutil.copytree(original / "workers" / job["name"], destination)
            path = destination / "config.json"
            config = json.loads(path.read_text())
            config["environment"]["CUDA_VISIBLE_DEVICES"] = device
            config["backend"] = "cuda"
            save(path, config)
    plan = json.loads((original / "plan.json").read_text())
    plan.update(jobs=statuses, jax_backend="cuda")
    save(root / "plan.json", plan)
    report = audit_module.audit(root, statuses)
    assert report["axes"]["synthetic_pipeline_integrity"]["status"] == "passed"
    rows = report["between_processes"]
    same = [r for r in rows if r["comparison"] == "same_device_fresh_process"]
    cross = [r for r in rows if r["comparison"] == "cross_device_first_round"]
    assert len(same) == 18
    assert len(cross) == 9
    assert all(r["device"] == r["reference_device"] for r in same)
    assert all(r["device"] != r["reference_device"] for r in cross)


def rewrite_arrays(root, transform):
    path = root / "workers/round0/arrays.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    transform(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    "quantity", ["input_gradient", "original_output", "original_loss"]
)
def test_every_fp32_repeat_must_pass_reference(audit_module, bundle, quantity):
    def damage(arrays):
        arrays[f"fp32/jit/repeat1/{quantity}"] *= 100

    rewrite_arrays(bundle[0], damage)
    report = audit_module.audit(*bundle)
    assert report["axes"]["strict_fp32_numerical_controls"]["status"] == "failed"
    assert any(
        not row["passed"]
        and row["evaluation"] == "repeat1"
        and row["quantity"] == quantity
        for row in report["numerical_checks"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "broadcast_row",
        "integer",
        "float64",
        "nan",
        "inf",
        "scalar_shape",
        "output_shape",
    ],
)
def test_malformed_repeats_fail_before_comparison(audit_module, bundle, damage):
    def corrupt(arrays):
        key = "fp32/jit/repeat1/input_gradient"
        if damage == "broadcast_row":
            arrays[key] = arrays[key][:1]
        elif damage == "integer":
            arrays[key] = arrays[key].astype(np.int32)
        elif damage == "float64":
            arrays[key] = arrays[key].astype(np.float64)
        elif damage in ("nan", "inf"):
            arrays[key][0, 0] = float(damage)
        elif damage == "scalar_shape":
            key = "fp32/jit/repeat1/original_loss"
            arrays[key] = arrays[key].reshape(1)
        else:
            key = "fp32/jit/repeat1/original_output"
            arrays[key] = arrays[key][:1]

    rewrite_arrays(bundle[0], corrupt)
    report = audit_module.audit(*bundle)
    assert report["axes"]["synthetic_pipeline_integrity"]["status"] == "failed"
    assert report["axes"]["strict_fp32_numerical_controls"]["status"] == "not_assessed"
    # Error reports must remain serializable, even for NaN/Inf corruption.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("damage", ["incorrect_output", "missing_forward"])
def test_forward_only_path_is_required_and_checked(audit_module, bundle, damage):
    def corrupt(arrays):
        key = "fp32/jit/forward_only/original_output"
        if damage == "missing_forward":
            del arrays[key]
        else:
            arrays[key] *= 100

    rewrite_arrays(bundle[0], corrupt)
    report = audit_module.audit(*bundle)
    if damage == "missing_forward":
        assert report["axes"]["synthetic_pipeline_integrity"]["status"] == "failed"
    else:
        assert report["axes"]["strict_fp32_numerical_controls"]["status"] == "failed"
        assert any(
            not r["passed"] and r["evaluation"] == "forward_only"
            for r in report["numerical_checks"]
        )


@pytest.mark.parametrize(
    "fault", ["repeat_numerical_failure", "worker_timeout", "corrupt_archive"]
)
def test_runner_failure_propagation_preserves_records(
    audit_module, bundle, monkeypatch, tmp_path, fault
):
    """Real child processes and packaging; fixture workers isolate runner behavior."""
    from types import SimpleNamespace
    import sys

    runner = importlib.import_module("synthetic_attention_cluster")
    root, _ = bundle
    if fault == "repeat_numerical_failure":

        def corrupt(arrays):
            arrays["fp32/jit/repeat1/input_gradient"] *= 100

        rewrite_arrays(root, corrupt)
    monkeypatch.setattr(
        runner, "preflight_source", lambda *args: "print('fixture-worker preflight')"
    )
    # Aggregation is independently exercised by real numerical CPU runs. Here we
    # isolate the real audit, subprocess timeout handling, and final exit policy.
    monkeypatch.setattr(runner, "collect", lambda *args: {})

    def fixture_command(job, output, device, args):
        source = root / "workers" / f"round{job['round']}"
        dest = output / "workers" / job["name"]
        code = "import shutil,sys; shutil.copytree(sys.argv[1], sys.argv[2])"
        if fault == "worker_timeout":
            code = "import time; print('worker started', flush=True); time.sleep(30)"
        return [sys.executable, "-u", "-c", code, str(source), str(dest)]

    monkeypatch.setattr(runner, "command", fixture_command)
    if fault == "corrupt_archive":
        real_verify = audit_module.verify_archive

        def corrupt_then_verify(root, archive):
            archive.write_bytes(b"corrupted gzip data")
            return real_verify(root, archive)

        monkeypatch.setattr(audit_module, "verify_archive", corrupt_then_verify)
    args = SimpleNamespace(
        cpu=True,
        devices="0",
        seeds="0",
        sizes="small",
        precision_modes="strict",
        rounds=2,
        repeats=2,
        torch_backend="cpu",
        fixed_devices=True,
        validation_suite=True,
        variants="fp32",
        output=tmp_path / "run",
        dry_run=False,
        no_archive=False,
        worker_timeout_minutes=0.001 if fault == "worker_timeout" else 1,
        telemetry_seconds=15,
    )
    devices, jobs = runner.make_plan(args)
    assert runner.run(args, devices, jobs) == 1
    summary = json.loads((args.output / "summary.json").read_text())
    validation = json.loads((args.output / "validation.json").read_text())
    completion = json.loads(args.output.with_name("run.completion.json").read_text())
    assert completion["exit_code"] == 1
    assert (args.output / "status.json").exists()
    assert (args.output / "checksums.sha256").exists()
    assert args.output.with_name("run.tar.gz").exists()
    if fault == "repeat_numerical_failure":
        assert summary["batch_completed"]
        assert not summary["synthetic_suite_passed"]
        assert (
            validation["axes"]["strict_fp32_numerical_controls"]["status"] == "failed"
        )
        assert completion["archive_verification_status"] == "passed"
    elif fault == "worker_timeout":
        assert summary["worker_state_counts"] == {"timeout": 2}
        assert not summary["batch_completed"]
        assert completion["archive_verification_status"] == "passed"
    else:
        assert summary[
            "synthetic_suite_passed"
        ]  # Numerical success cannot hide packaging failure.
        assert completion["archive_verification_status"] == "failed"
        assert "Archive creation/verification failed" in completion["failure"]
