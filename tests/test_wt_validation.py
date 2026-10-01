"""Model-free, independent geometry and WT-only control-plan regressions."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return tuple(
        importlib.import_module(name)
        for name in (
            "p17_structure_audit",
            "p17_wt_validation",
            "p17_pose_diagnostics",
            "p17_pose_experiment",
        )
    )


@pytest.fixture
def reference(modules):
    audit, runner, _, _ = modules
    _, residues, bb, asym, idx = runner.load_reference(runner.REPO / "P17_JN1.pdb")
    n = len(bb)
    coords = bb.reshape(-1, 3)
    names = np.tile(audit.BACKBONE_NAMES, n)
    tokens = np.repeat(np.arange(n), 4)
    atom37 = np.zeros((n, 37, 3))
    mask37 = np.zeros((n, 37))
    atom37[:, audit.BACKBONE_SLOTS] = bb
    mask37[:, audit.BACKBONE_SLOTS] = 1
    return coords, tokens, names, np.ones(len(coords)), asym, idx, atom37, mask37


def test_reference_mapping_is_order_independent_and_ignores_padding(modules, reference):
    audit, *_ = modules
    coords, tokens, names, mask, *rest = reference
    rng = np.random.default_rng(4)
    permutation = rng.permutation(len(coords))
    result = audit.audit_raw_mapping(
        np.concatenate([coords[permutation], [[np.nan] * 3]]),
        np.append(tokens[permutation], 99999),
        np.append(names[permutation], ""),
        np.append(mask[permutation], 0),
        *rest,
    )
    assert result["passed"] and result["mapping_agrees"]
    assert result["mapping_max_error_A"] == 0


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "nonfinite", "bad_raw", "bad_mapping"]
)
def test_mapping_audit_localizes_invalid_coordinates(modules, reference, failure):
    audit, *_ = modules
    coords, tokens, names, mask, asym, idx, atom37, mask37 = reference
    if failure == "missing":
        mask[0] = 0
    elif failure == "duplicate":
        names[2] = "N"
    elif failure == "nonfinite":
        coords[0, 0] = np.nan
    elif failure == "bad_raw":
        coords *= 4
        atom37 *= 4
    else:
        atom37[:, 1, 0] += 10
    report = audit.audit_raw_mapping(
        coords, tokens, names, mask, asym, idx, atom37, mask37
    )
    assert not report["passed"]
    if failure == "bad_raw":
        assert report["mapping_agrees"]
        assert not report["raw_passed"]
    if failure == "bad_mapping":
        assert report["raw_passed"]
        assert not report["mapping_agrees"]
    # Invalid coordinates must still be serializable as evidence.
    json.dumps(report, allow_nan=False)


def test_chain_boundaries_are_not_bonded_and_internal_geometry_is_checked(
    modules, reference
):
    audit, *_ = modules
    coords, tokens, names, mask, asym, idx, atom37, mask37 = reference
    bb = atom37[:, audit.BACKBONE_SLOTS].copy()
    bb[asym == 1] += 1000
    report = audit.backbone_geometry(bb, np.ones(bb.shape[:2]), asym, idx)
    assert report["passed"]
    bb[:, 0] += [10, 0, 0]
    report = audit.backbone_geometry(bb, np.ones(bb.shape[:2]), asym, idx)
    assert not report["passed"]
    assert all(
        c["distances"]["adjacent_CA"]["violations"] == 0 for c in report["chains"]
    )


def test_tv_above_possible_range_is_inconclusive(modules):
    _, _, diag, _ = modules
    for noise, required in (
        (0.38453196647718824, 1.1535958994315647),
        (0.556383345, 1.669150035),
    ):
        threshold, resolvable, influence, status = diag.assess_influence(
            dict(proposal_tv=0.95, repeat_proposal_tv=noise), 1e-4, 3.0
        )
        assert threshold == pytest.approx(required)
        assert not resolvable and influence is None
        assert status == "inconclusive_repeat_variability"
    assert (
        diag.assess_influence(dict(proposal_tv=0.9, repeat_proposal_tv=0.01), 1e-4, 3)[
            2
        ]
        is True
    )
    assert (
        diag.assess_influence(dict(proposal_tv=0.02, repeat_proposal_tv=0.01), 1e-4, 3)[
            2
        ]
        is False
    )


def test_wt_plan_matches_budgets_without_search(modules, tmp_path):
    _, runner, _, _ = modules
    args = SimpleNamespace(
        paths=list(runner.PATHS),
        steps=[8, 64],
        seeds=[0, 1],
        recycles=4,
        opendde_dtype="bf16",
        reference=runner.REPO / "P17_JN1.pdb",
        output_dir=tmp_path,
    )
    plan = runner.build_plan(args)
    assert len(plan) == 12 and len({job["name"] for job in plan}) == 12
    for job in plan:
        cmd = job["command"]
        assert cmd[cmd.index("--recycles") + 1] == "4"
        assert cmd[cmd.index("--opendde-dtype") + 1] == "bf16"
        assert "--worker" in cmd
        assert "search" not in " ".join(cmd)


def test_dry_run_creates_nothing(modules, tmp_path):
    _, runner, _, _ = modules
    output = tmp_path / "never_created"
    runner.main(["--dry-run", "--output-dir", str(output)])
    assert not output.exists()


def test_old_gate_reports_cannot_bypass_new_geometry_checks(modules, tmp_path):
    _, _, _, launcher = modules
    for seed in (0, 1):
        path = tmp_path / "diagnostic" / f"model_seed{seed}"
        path.mkdir(parents=True)
        checks = {
            name: True
            for name in (
                "completed",
                "same_coordinate_reporting",
                "interpretable_target_fit",
                "proposal_influence",
                "paired_pose_consistent",
            )
        }
        (path / "diagnostic.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    passed=True,
                    checks=checks,
                    reference_audit={"geometry": {"identity": {}}},
                )
            )
        )
    with pytest.raises(RuntimeError, match="gate failed"):
        launcher.check_gate(tmp_path)


def test_collection_distinguishes_geometry_failure_from_execution_failure(
    modules, tmp_path
):
    _, runner, _, _ = modules
    (tmp_path / "tables").mkdir()
    path = tmp_path / "workers" / "invalid"
    path.mkdir(parents=True)
    report = dict(
        repeat=0,
        passed=False,
        mapping_agrees=True,
        raw_geometry={
            "chains": [
                {
                    "distances": {
                        "adjacent_CA": {"median_A": 16.0},
                        "N_CA": {"median_A": 18.0},
                    }
                }
            ]
        },
    )
    (path / "summary.json").write_text(json.dumps(dict(reports=[report])))
    runner.collect_results(tmp_path, [dict(name="invalid"), dict(name="crashed")])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["completed"] and not summary["geometry_passed"]
    assert summary["rows"][0]["completed"] and not summary["rows"][0]["geometry_passed"]
    assert not summary["rows"][1]["completed"]


def test_raw_cif_roundtrip_preserves_named_backbone(modules, reference, tmp_path):
    import gemmi

    audit, runner, *_ = modules
    coords, tokens, names, mask, asym, idx, *_ = reference
    encoded = np.zeros((len(names), 4, 64))
    for i, name in enumerate(names):
        for j, char in enumerate(str(name).ljust(4)):
            encoded[i, j, ord(char) - 32] = 1
    np.testing.assert_array_equal(audit.decode_atom_names(encoded), names)
    arrays = dict(
        ref_atom_name_chars=encoded,
        atom_to_token_idx=tokens,
        ref_mask=mask,
        asym_id=asym,
        residue_index=idx,
    )
    _, residue_names, *_ = runner.load_reference(runner.REPO / "P17_JN1.pdb")
    path = tmp_path / "raw.cif"
    runner.write_raw_cif(path, coords, arrays, residue_names)
    structure = gemmi.read_structure(str(path))[0]
    assert [chain.name for chain in structure] == ["B", "T"]
    actual = np.array([list(a.pos) for c in structure for r in c for a in r])
    np.testing.assert_allclose(actual, coords, rtol=0, atol=1e-5)


def test_native_rollout_seed_is_explicit_and_prepared_batch_is_immutable(modules):
    import torch

    _, runner, *_ = modules
    original = {
        "input_feature_dict": {
            "inference_seed": torch.tensor(0),
            "ref_pos": torch.zeros(2, 3),
        }
    }
    first = runner.native_prediction_input(original, 1)
    repeat = runner.native_prediction_input(original, 1)
    assert first["input_feature_dict"]["inference_seed"].item() == 1
    assert repeat["input_feature_dict"]["inference_seed"].item() == 1
    first["input_feature_dict"]["ref_pos"][0, 0] = 100
    assert repeat["input_feature_dict"]["ref_pos"][0, 0] == 0
    assert original["input_feature_dict"]["inference_seed"].item() == 0
    assert original["input_feature_dict"]["ref_pos"][0, 0] == 0
