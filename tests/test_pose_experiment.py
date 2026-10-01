"""Model-free diagnostics and cluster orchestration checks."""

import csv
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mosaic.search import SearchConfig


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("p17_pose_diagnostics"), importlib.import_module(
        "p17_pose_experiment"
    )


def references():
    binder = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [3.8, 3.8, 0.0], [3.8, 3.8, 3.8]]
    )
    return binder, binder + [7.0, 0.0, 0.0]


def test_geometry_and_degeneracy(modules):
    diag, _ = modules
    result = diag.geometry_checks(*references())
    assert result["binder_shift"]["binder_pose_rmsd_A"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="degenerate"):
        diag.geometry_checks(np.zeros((4, 3)), references()[1])


@pytest.mark.parametrize("precision", ["tensorfloat32", "float32"])
@pytest.mark.parametrize("compiled", [False, True])
def test_real_reference_pose_precision_and_gradient(modules, precision, compiled):
    """Exercise realistic coordinate scales on whichever JAX backend is selected.

    Run with JAX_PLATFORMS=cuda to catch reduced-precision GPU regressions;
    small synthetic point sets alone did not catch the original failure.
    """
    import gemmi
    import jax
    import jax.numpy as jnp
    from mosaic.losses.structure_prediction import BinderPoseRMSD

    diag, _ = modules
    model = gemmi.read_structure(
        str(Path(__file__).resolve().parents[1] / "P17_JN1.pdb")
    )[0]
    binder, target = [
        np.array([list(residue["CA"][0].pos) for residue in model[chain]])
        for chain in ("B", "T")
    ]
    loss_fn = BinderPoseRMSD(binder, target, rmsd_tolerance=0.0)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    reference = np.concatenate([binder, target])

    def evaluate(shift):
        # Translate binder in the reference frame, then rotate the entire
        # complex. Target alignment must recover the imposed shift exactly.
        coords = jnp.asarray(reference).at[: len(binder), 0].add(shift)
        coords = (
            jnp.matmul(
                coords, jnp.asarray(rotation), precision=jax.lax.Precision.HIGHEST
            )
            + 7.0
        )
        return loss_fn(
            jnp.zeros((len(binder), 20)),
            SimpleNamespace(
                backbone_coordinates=jnp.repeat(coords[:, None], 4, axis=1)
            ),
            key=None,
        )

    with jax.default_matmul_precision(precision):
        controls = diag.geometry_checks(binder, target)
        fn = jax.value_and_grad(evaluate, has_aux=True)
        if compiled:
            fn = jax.jit(fn)
        (value, aux), gradient = fn(4.0)
        # The local precision scope must not leak into the surrounding model.
        assert jax.config.jax_default_matmul_precision == precision
    assert max(c["numpy_jax_max_error_A"] for c in controls.values()) < 1e-3
    assert float(value) == pytest.approx(4.0, abs=1e-3)
    assert float(aux["pose_target_fit_rmsd"]) < 1e-3
    assert float(gradient) == pytest.approx(1.0, abs=1e-3)


def test_reference_audit_records_insertions_and_rejects_missing_ca(modules):
    import gemmi

    diag, _ = modules
    model = gemmi.Model("1")
    for name, coords in zip(("B", "T"), references()):
        chain = gemmi.Chain(name)
        for i, coord in enumerate(coords):
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.seqid = gemmi.SeqId(i + 1, "A" if i == 2 else " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.pos = gemmi.Position(*coord)
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    report = diag.audit_reference(model, ("B", "T"), ("AAAA", "AAAA"), references())
    assert report["chains"][0]["mapping"][2]["insertion_code"] == "A"
    del model["B"][0][0]
    with pytest.raises(ValueError, match="missing/alternate CA"):
        diag.audit_reference(model, ("B", "T"), ("AAAA", "AAAA"), references())


@pytest.mark.parametrize(
    "target_fit,influence,passed",
    [(0.0, True, True), (16.0, True, False), (0.0, False, False)],
)
@pytest.mark.parametrize("geometry_passed", [True, False])
def test_diagnostic_gates_target_fit_and_effective_proposals(
    modules, tmp_path, target_fit, influence, passed, geometry_passed
):
    diag, _ = modules
    (tmp_path / "tables").mkdir()
    (tmp_path / "confidence").mkdir()
    wt = np.zeros(4, dtype=int)
    on = np.zeros((4, 20))
    on[0, 1] = -1.0
    off = np.zeros((4, 20))
    off[1, 1] = -1.0
    if not influence:
        off = on.copy()
    metrics = dict(binder_pose_rmsd=2.0, pose_target_fit_rmsd=target_fit)
    calls = []

    def predict(sequence, seed):
        calls.append((tuple(sequence), seed))
        binder, target = references()
        return (
            np.zeros((8, 8)),
            np.concatenate([binder + [2.0, 0.0, 0.0], target]),
            0.8,
            {"passed": geometry_passed},
        )

    report = diag.run_diagnostic(
        root=tmp_path,
        wt=wt,
        mask=np.array([True, True, False, False]),
        config=SearchConfig(),
        gradient_on=lambda seq: (2.0, on, metrics),
        gradient_off=lambda seq: (0.0, off, metrics),
        predict=predict,
        repeat_predict=predict,
        references=references(),
        seeds=[0, 1],
        pae_cutoff=12.0,
        distance_cutoff=12.0,
        max_target_rmsd=3.0,
        min_proposal_tv=1e-4,
        repeat_factor=3.0,
    )
    assert report["passed"] is (passed and geometry_passed)
    assert report["checks"]["predicted_backbone_plausible"] is geometry_passed
    assert report["full_gradient_calls"] == 3
    assert report["full_prediction_calls"] == len(calls) == (8 if influence else 6)
    assert (tmp_path / "tables/diagnostic_proposals.csv").exists()
    assert (
        sum(row["label"] == "WT_repeat" for row in report["forward_observations"]) == 2
    )


def test_entropy_normalization_can_erase_gradient_magnitude_change(modules):
    diag, _ = modules
    gradient = np.array([[0.0, -1.0, -2.0], [0.0, -0.5, -0.25]])
    effect, *_ = diag.proposal_comparison(
        np.zeros(2, dtype=int),
        np.ones(2, dtype=bool),
        SearchConfig(alphabet_size=3),
        [gradient, gradient, 100.0 * gradient],
    )
    assert effect["cdr_gradient_difference_norm"] > 1
    assert effect["proposal_tv"] < 1e-12


def test_plan_has_matched_arms_and_heldout_seeds(modules, tmp_path):
    _, launcher = modules
    args = SimpleNamespace(
        output_dir=tmp_path,
        devices=list("01234567"),
        search_seeds=[0, 1],
        sampling_steps=8,
        opendde_dtype="bf16",
        max_score_calls=32,
        max_gradient_calls=32,
        max_proposals=320,
        weight_pose=1.0,
        pose_margin=3.0,
        max_target_rmsd=3.0,
        min_proposal_tv=1e-4,
        repeat_factor=3.0,
    )
    plan = launcher.build_plan(args)
    assert (
        len(plan["diagnostic"]) == 2
        and len(plan["search"]) == len(plan["heldout"]) == 8
    )
    for jobs in plan.values():
        for job in jobs:
            command = job["command"]
            assert command[command.index("--opendde-dtype") + 1] == "bf16"
    for job in plan["search"]:
        command = job["command"]
        assert command[
            command.index("--selection-seeds") + 1 : command.index("--selection-seeds")
            + 3
        ] == ["0", "1"]
        assert ("--retention-pose-margin" in command) == job["name"].startswith(
            ("C_", "D_")
        )
    for job in plan["heldout"]:
        command = job["command"]
        assert command[command.index("--seeds") + 1 : command.index("--seeds") + 4] == [
            "101",
            "102",
            "103",
        ]


def test_dry_run_creates_no_outputs_or_patches(modules, tmp_path, monkeypatch, capsys):
    _, launcher = modules

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run started a process")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    root = tmp_path / "out"
    launcher.main(["--dry-run", "--output-dir", str(root), "--devices", "0,1"])
    assert not root.exists()
    assert "heldout" in capsys.readouterr().out


def test_worker_batches_do_not_overlap_devices_and_record_exit_codes(modules, tmp_path):
    _, launcher = modules
    (tmp_path / "logs").mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, pathlib, time\n"
        "root = pathlib.Path(__file__).parent\n"
        "lock = root / ('device_' + os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "with lock.open('x'): time.sleep(0.05)\n"
        "lock.unlink()\n"
    )
    jobs = [dict(name=str(i), command=[sys.executable, str(worker)]) for i in range(4)]
    launcher.run_stage("test", jobs, ["0", "1"], tmp_path)
    rows = list(csv.reader((tmp_path / "status.tsv").open(), delimiter="\t"))
    assert len(rows) == 4 and all(row[-1] == "0" for row in rows)
    assert [row[2] for row in rows] == ["0", "1", "0", "1"]


def test_failed_worker_prevents_next_batch(modules, tmp_path):
    _, launcher = modules
    (tmp_path / "logs").mkdir()
    jobs = [
        dict(name="fail", command=[sys.executable, "-c", "raise SystemExit(7)"]),
        dict(name="never", command=[sys.executable, "-c", "raise SystemExit(0)"]),
    ]
    with pytest.raises(RuntimeError, match="worker failed"):
        launcher.run_stage("test", jobs, ["0"], tmp_path)
    assert not (tmp_path / "logs/test_never.log").exists()
    assert (tmp_path / "status.tsv").read_text().strip().endswith("7")


def test_gate_rejects_exit_zero_without_valid_report(modules, tmp_path):
    _, launcher = modules
    for seed in (0, 1):
        root = tmp_path / "diagnostic" / f"model_seed{seed}"
        root.mkdir(parents=True)
        (root / "diagnostic.json").write_text(
            json.dumps(dict(schema_version=1, passed=True))
        )
    with pytest.raises(RuntimeError, match="gate failed"):
        launcher.check_gate(tmp_path)


def test_main_diagnostic_failure_prevents_search(modules, tmp_path, monkeypatch):
    _, launcher = modules
    stages = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)

    def stage(name, *args):
        stages.append(name)
        raise RuntimeError("diagnostic worker failed")

    monkeypatch.setattr(launcher, "run_stage", stage)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="diagnostic"):
        launcher.main(["--output-dir", str(root), "--devices", "0,1"])
    assert stages == ["diagnostic"]
    report = json.loads((root / "experiment.json").read_text())
    assert report["completed"] is False and report["failed_stage"] == "diagnostic"


def test_heldout_summary_uses_common_wt_ceiling_and_preserves_raw_scores(
    modules, tmp_path
):
    _, launcher = modules
    (tmp_path / "heldout/tables").mkdir(parents=True)
    (tmp_path / "tables").mkdir()
    rows = []
    for candidate, poses, scores in [
        (0, [2.0, 4.0, 3.0], [0.1, 0.2, 0.3]),
        (1, [3.0, 8.0, 5.0], [0.8, 0.9, 0.7]),
    ]:
        for seed, pose, score in zip([101, 102, 103], poses, scores):
            rows.append(
                dict(
                    candidate_id=candidate,
                    selection_seed=seed,
                    ipsae_min=score,
                    binder_pose_rmsd_A=pose,
                    target_aligned_rmsd_A=1.0,
                    binder_internal_rmsd_A=2.0,
                )
            )
    with (tmp_path / "heldout/tables/predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "heldout/tables/source_runs.csv").write_text(
        "candidate_id,source_run\n1,B_guidance_seed0\n1,D_both_seed0\n"
    )
    launcher.summarize_heldout(tmp_path, 3.0)
    summary = list(csv.DictReader((tmp_path / "tables/heldout_candidates.csv").open()))
    assert all(float(r["common_reporting_ceiling_A"]) == 7.0 for r in summary)
    assert summary[0]["pose_feasible_all_seeds"] == "True"
    assert summary[1]["pose_feasible_all_seeds"] == "False"
    assert float(summary[1]["mean_ipsae_min"]) == pytest.approx(0.8)
    assert float(summary[1]["worst_ipsae_min"]) == pytest.approx(0.7)
    assert float(summary[1]["pose_feasible_seed_fraction"]) == pytest.approx(2 / 3)
    assert summary[1]["source_runs"] == "B_guidance_seed0;D_both_seed0"


def test_search_summary_uses_one_reporting_ceiling_across_arms(modules, tmp_path):
    _, launcher = modules
    (tmp_path / "tables").mkdir()
    for index, wt in enumerate([2.0, 4.0]):
        directory = tmp_path / "search" / f"arm{index}"
        (directory / "tables").mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps(
                dict(
                    config=dict(seed=0),
                    arguments=dict(weight_pose=1.0, retention_pose_margin=3.0),
                )
            )
        )
        (directory / "summary.json").write_text(
            json.dumps(
                dict(
                    retention_pose_ceiling_A=wt + 3,
                    best_score=0.7,
                    best_constraint_violation=0.0,
                    full_prediction_calls=4,
                    full_gradient_calls=1,
                    stats=dict(score_calls=2),
                )
            )
        )
        (directory / "tables/candidates.csv").write_text(
            f"is_wt,worst_pose_rmsd_A\nTrue,{wt}\nFalse,7\n"
        )
    launcher.summarize_search(tmp_path, 3.0)
    rows = list(csv.DictReader((tmp_path / "tables/search_runs.csv").open()))
    assert all(float(r["common_reporting_ceiling_A"]) == 6.0 for r in rows)
    assert all(
        float(r["evaluated_feasible_fraction_including_WT"]) == 0.5 for r in rows
    )
    assert [float(r["calibrated_ceiling_A"]) for r in rows] == [5.0, 7.0]
