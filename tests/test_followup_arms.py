"""The four follow-up arms: decoy, budget/exploration, pose-zero, alpha5.

Each arm exists to answer a specific objection to the 2026-10-05 results, so
the tests check that the arm actually differs from the baseline in the one way
it claims to and in no other way. Also covers the decoy's fold confound, which
is what would make the negative control pass vacuously.

No model, no GPU.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import p17_alpha_recovery as driver  # noqa: E402
from p17_alpha_reference import (  # noqa: E402
    epitope_scrambled_target,
    real_decoy_target,
    shuffled_target,
)


def _args(stage, *extra, out="/tmp/x"):
    return driver.build_parser().parse_args(
        [stage, "--output-dir", out, "--devices", "0", *extra]
    )


# Decoy target construction ---------------------------------------------------


def test_shuffle_preserves_length_and_composition():
    """Matching both removes length and amino-acid frequency as explanations."""
    target = "ACDEFGHIKLMNPQRSTVWY" * 9 + "ACDE"
    decoy, info = shuffled_target(target, seed=0)
    assert len(decoy) == len(target)
    assert sorted(decoy) == sorted(target)
    assert info["composition_matched"] is True
    assert info["length"] == len(target)
    assert 0.0 <= info["identity_to_real_target"] <= 1.0


def test_shuffle_is_deterministic_and_seed_dependent():
    target = "ACDEFGHIKLMNPQRSTVWY" * 9
    assert shuffled_target(target, 0)[0] == shuffled_target(target, 0)[0]
    assert shuffled_target(target, 0)[0] != shuffled_target(target, 1)[0]


def test_shuffle_actually_moves_most_residues():
    """A decoy that is nearly the real target is not a negative control."""
    target = "ACDEFGHIKLMNPQRSTVWY" * 9
    _, info = shuffled_target(target, seed=0)
    assert info["identity_to_real_target"] < 0.25


# Arm construction ------------------------------------------------------------


def test_decoy_arm_passes_the_decoy_sequence_and_nothing_else_differs():
    """The negative control must differ from the real arm in one input."""
    args = _args("decoy")
    args._decoy_sequence = "W" * 184
    jobs = driver.search_jobs(Path("/tmp/x"), args.policies, [0], None, 5, args, None)
    command = jobs[0]["command"]
    assert command[command.index("--target-sequence") + 1] == "W" * 184
    # Same entry point, same policy default, no start override.
    assert command[command.index("--policy") + 1] == "population"
    assert "--start-sequence" not in command


def test_pose_zero_arm_sets_the_weight_and_changes_nothing_else():
    zero = _args("jn1", "--weight-pose", "0")
    base = _args("jn1")
    z = driver.search_jobs(Path("/tmp/x"), zero.policies, [0], None, 5, zero, None)[0]
    b = driver.search_jobs(Path("/tmp/x"), base.policies, [0], None, 5, base, None)[0]
    assert z["command"][z["command"].index("--weight-pose") + 1] == "0.0"
    assert "--weight-pose" not in b["command"], "baseline leaves the default alone"
    stripped = [t for t in z["command"] if t not in ("--weight-pose", "0.0")]
    assert stripped == b["command"], "pose weight is the only difference"


def test_budget_arm_raises_ceilings_and_warms_the_chain():
    """Section 20.8: all eight runs stopped at 32 scored sequences."""
    args = _args(
        "jn1",
        "--max-score-calls", "300",
        "--target-entropy", "0.8",
        "--acceptance-temperature", "0.05",
    )
    command = driver.search_jobs(
        Path("/tmp/x"), args.policies, [0], None, 5, args, None
    )[0]["command"]
    assert command[command.index("--max-score-calls") + 1] == "300"
    assert command[command.index("--target-entropy") + 1] == "0.8"
    assert command[command.index("--acceptance-temperature") + 1] == "0.05"


def test_exploration_knobs_are_omitted_when_not_requested():
    """Unset knobs must not silently pin the search CLI's own defaults."""
    args = _args("jn1")
    command = driver.search_jobs(
        Path("/tmp/x"), args.policies, [0], None, 5, args, None
    )[0]["command"]
    for flag in ("--target-entropy", "--acceptance-temperature", "--width",
                 "--weight-pose", "--target-sequence"):
        assert flag not in command


def test_saliency_is_requested_only_when_asked():
    on = _args("jn1", "--save-saliency")
    off = _args("jn1")
    for args, expected in ((on, True), (off, False)):
        command = driver.search_jobs(
            Path("/tmp/x"), args.policies, [0], None, 5, args, None
        )[0]["command"]
        assert ("--save-saliency" in command) is expected


# The epitope-scrambled decoy -------------------------------------------------


def test_epitope_scramble_changes_only_the_epitope_and_keeps_composition():
    target = "ACDEFGHIKLMNPQRST"
    epitope = [2, 5, 9, 13]
    decoy, info = epitope_scrambled_target(target, epitope, seed=0)
    assert len(decoy) == len(target)
    assert sorted(decoy) == sorted(target), "composition is preserved"
    outside = [i for i in range(len(target)) if i not in epitope]
    assert all(decoy[i] == target[i] for i in outside), "only the epitope moves"
    assert sorted(decoy[i] for i in epitope) == sorted(target[i] for i in epitope)
    assert info["kind"] == "epitope_scrambled"
    assert info["epitope_size"] == 4
    assert info["epitope_positions"] == epitope


def test_epitope_scramble_records_positions_it_left_alone_by_chance():
    """A permutation can fix a position, and the arm's provenance must say so."""
    decoy, info = epitope_scrambled_target("AAAACDEFGH", [0, 1, 2, 3], seed=0)
    assert decoy == "AAAACDEFGH"
    assert info["epitope_residues_unchanged_by_chance"] == 4
    assert info["identity_to_real_target"] == 1.0


def test_epitope_scramble_refuses_indices_it_cannot_trust():
    for bad in ([5], [0, 99], [-1, 2], [3, 3]):
        with pytest.raises(ValueError):
            epitope_scrambled_target("ACDEFG", bad, seed=0)


def test_reference_epitope_matches_the_interface_the_search_itself_uses():
    """Same file, chains and 8 A CA-CA cutoff as the search, not a constant."""
    from p17_hallucination_search import CONTACT_DISTANCE, load_structure
    from p17_alpha_reference import contact_epitope

    model, _, target = load_structure()
    from p17_hallucination_search import reference_binder_target_ca

    binder_ca, target_ca = reference_binder_target_ca(model)
    expected = contact_epitope(binder_ca, target_ca, CONTACT_DISTANCE)

    sequence, epitope = driver.reference_epitope(None, None, None)
    assert sequence == target
    assert epitope.tolist() == expected.tolist()
    assert epitope.size == 16 and len(sequence) == 184


def test_decoy_arm_defaults_to_the_epitope_scramble_not_the_shuffle(
    tmp_path, monkeypatch
):
    """The shuffle does not fold, so it must not be what a bare run picks."""
    _, calls = _decoy_run(tmp_path, monkeypatch, target_plddt=0.88)
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["decoy"]["kind"] == "epitope_scrambled"
    assert report["decoy"]["epitope_size"] == 16
    assert report["decoy"]["identity_to_real_target"] > 0.9, "only the epitope moved"
    decoy = calls["fold_check_command"][
        calls["fold_check_command"].index("--target-sequence") + 1
    ]
    assert len(decoy) == 184


def test_decoy_mode_shuffled_is_still_reachable_for_comparison(
    tmp_path, monkeypatch
):
    _, _ = _decoy_run(
        tmp_path, monkeypatch, target_plddt=0.88, extra=["--decoy-mode", "shuffled"]
    )
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["decoy"]["kind"] == "shuffled"


# The real decoy --------------------------------------------------------------


REPO = Path(__file__).resolve().parent.parent


def test_real_decoy_is_trimmed_to_the_reference_length_and_records_the_trim():
    target = driver.reference_target_sequence(None, None)
    decoy, info = real_decoy_target(REPO / "IL7RA.cif", "A", len(target))
    assert len(decoy) == len(target) == 184
    assert info["kind"] == "real_trimmed"
    assert info["source_length"] == 198
    assert info["trimmed_from_n_terminus"] + info["trimmed_from_c_terminus"] == 14
    assert info["kept_author_residues"] == [41, 224]
    assert not set(decoy) - set("ACDEFGHIKLMNPQRSTVWY")
    identity = sum(a == b for a, b in zip(decoy, target)) / len(target)
    assert identity < 0.2, "a decoy must not resemble the real target"


def test_real_decoy_refuses_a_chain_too_short_to_trim():
    with pytest.raises(ValueError, match="trimming cannot lengthen"):
        real_decoy_target(REPO / "P17_JN1.pdb", "B", 184)


def test_real_decoy_names_the_available_chains_when_asked_for_a_missing_one():
    with pytest.raises(ValueError, match="not in"):
        real_decoy_target(REPO / "IL7RA.cif", "Z", 184)


def test_decoy_arm_refuses_both_decoy_sources_at_once(tmp_path):
    args = driver.build_parser().parse_args(
        ["decoy", "--output-dir", str(tmp_path), "--devices", "0",
         "--decoy-sequence", "A" * 184,
         "--decoy-structure", str(REPO / "IL7RA.cif")]
    )
    with pytest.raises(ValueError, match="either --decoy-sequence or"):
        driver.decoy(args)


# The decoy's fold confound ---------------------------------------------------


def _decoy_run(
    tmp_path, monkeypatch, target_plddt, force=False, extra=(), target_fit=17.4
):
    """`target_plddt` is the gate; `target_fit` is recorded but not gating.

    A decoy's fit to the *real* target's coordinates is large whether or not
    the decoy folds, so the fixture keeps it at the measured shuffled-decoy
    value and varies only the intrinsic measure.
    """
    calls = {"searches": 0, "fold_check_command": None}

    def fake_workers(jobs, devices, log_dir):
        if jobs[0]["name"] == "fold_check":
            calls["fold_check_command"] = list(jobs[0]["command"])
            run = tmp_path / "fold_check"
            (run / "tables").mkdir(parents=True)
            (run / "confidence/candidate_00000").mkdir(parents=True)
            rows = []
            for seed in (0, 1):
                arrays = run / f"confidence/candidate_00000/seed_{seed}.npz"
                np.savez(
                    arrays,
                    plddt=np.concatenate(
                        [np.full(123, 0.85), np.full(184, target_plddt)]
                    ),
                    asym_id=np.concatenate(
                        [np.zeros(123, int), np.ones(184, int)]
                    ),
                )
                rows.append(
                    dict(candidate_id="0", selection_seed=seed, is_wt="True",
                         ipsae_min=0.02, binder_pose_rmsd_A=40.0,
                         target_aligned_rmsd_A=target_fit,
                         binder_internal_rmsd_A=2.0,
                         confidence_file=arrays.relative_to(run).as_posix())
                )
            with (run / "tables/predictions.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        else:
            calls["searches"] += 1
        return [0] * len(jobs)

    monkeypatch.setattr(driver, "run_workers", fake_workers)
    monkeypatch.setattr(driver, "run_heldout", lambda *a, **k: "heldout")
    monkeypatch.setattr(driver, "write_jn1_table", lambda *a, **k: None)
    argv = ["decoy", "--output-dir", str(tmp_path), "--devices", "0", *extra]
    if force:
        argv.append("--force")
    return driver.decoy(driver.build_parser().parse_args(argv)), calls


def test_fold_check_scores_the_decoy_against_the_reference_it_came_from(
    tmp_path, monkeypatch
):
    """The regression that broke the first cluster run of this arm.

    The decoy's length is taken from one reference and then validated against
    whichever reference the fold-check command names. When those disagreed the
    search script's length guard refused the run with exit 2, and the arm died
    before scoring anything. The default decoy is built from JN.1, so the
    fold check must not name a `--complex` at all.
    """
    _, calls = _decoy_run(tmp_path, monkeypatch, target_plddt=0.91)
    command = calls["fold_check_command"]
    assert "--complex" not in command, "JN.1 is the search script's own default"
    decoy = command[command.index("--target-sequence") + 1]
    assert len(decoy) == len(driver.reference_target_sequence(None, None))


def test_fold_check_follows_an_explicit_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "reference_target_sequence", lambda *a, **k: "M" * 195
    )
    _, calls = _decoy_run(
        tmp_path, monkeypatch, target_plddt=0.91,
        extra=["--complex", str(driver.ALPHA_COMPLEX_PDB)],
    )
    command = calls["fold_check_command"]
    assert command.count("--complex") == 1
    assert command[command.index("--complex") + 1] == str(driver.ALPHA_COMPLEX_PDB)
    assert len(command[command.index("--target-sequence") + 1]) == 195


def test_decoy_refuses_to_search_when_its_target_does_not_fold(tmp_path, monkeypatch):
    """The confound that would make this control pass vacuously.

    A shuffled target has no native fold. If the predictor cannot place it,
    low interface confidence follows from that rather than from absent
    complementarity, and the arm proves nothing.
    """
    with pytest.raises(RuntimeError, match="uninterpretable"):
        _decoy_run(tmp_path, monkeypatch, target_plddt=0.41)
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is False
    assert report["wt_metrics"]["mean_target_plddt"] == pytest.approx(0.41)


def test_the_fold_gate_reads_plddt_and_not_fit_to_the_real_target(
    tmp_path, monkeypatch
):
    """Why the gate changed on 2026-10-05.

    `target_aligned_rmsd_A` superimposes the predicted decoy on the *real*
    target's coordinates, so a correctly folded unrelated protein scores as
    badly on it as an unfoldable scramble. Gating on it would reject every
    usable decoy. The decoy chain's own pLDDT is intrinsic, so a folded decoy
    passes with its fit to the real target left as large as ever.
    """
    code, calls = _decoy_run(
        tmp_path, monkeypatch, target_plddt=0.91, target_fit=28.0
    )
    assert code == 0 and calls["searches"] == 1
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is True
    assert report["wt_metrics"]["mean_target_fit_A"] == pytest.approx(28.0)


def test_a_fold_check_without_confidence_arrays_is_fatal(tmp_path, monkeypatch):
    """Silently skipping the gate would make the whole arm unreadable."""
    monkeypatch.setattr(driver, "run_heldout", lambda *a, **k: "heldout")
    monkeypatch.setattr(driver, "write_jn1_table", lambda *a, **k: None)

    def fake_workers(jobs, devices, log_dir):
        run = tmp_path / "fold_check"
        (run / "tables").mkdir(parents=True)
        rows = [dict(candidate_id="0", selection_seed=0, ipsae_min=0.02,
                     binder_pose_rmsd_A=40.0, target_aligned_rmsd_A=17.4,
                     confidence_file="confidence/missing.npz")]
        with (run / "tables/predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return [0] * len(jobs)

    monkeypatch.setattr(driver, "run_workers", fake_workers)
    args = driver.build_parser().parse_args(
        ["decoy", "--output-dir", str(tmp_path), "--devices", "0", "--force"]
    )
    with pytest.raises(RuntimeError, match="no confidence arrays"):
        driver.decoy(args)


def test_decoy_proceeds_when_its_target_places_consistently(tmp_path, monkeypatch):
    code, calls = _decoy_run(tmp_path, monkeypatch, target_plddt=0.91)
    assert code == 0
    assert calls["searches"] == 1
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is True
    assert report["decoy"]["kind"] == "epitope_scrambled", "the default decoy"
    assert len(report["decoy_sequence"]) == report["decoy"]["length"] == 184


def test_force_records_the_failed_fold_check_rather_than_hiding_it(
    tmp_path, monkeypatch
):
    code, calls = _decoy_run(tmp_path, monkeypatch, target_plddt=0.41, force=True)
    assert code == 0 and calls["searches"] == 1
    report = json.loads((tmp_path / "fold_check.json").read_text())
    assert report["interpretable"] is False, "the failure stays on record"


def test_supplied_decoy_must_match_the_reference_target_length(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "reference_target_sequence", lambda *a: "M" * 184)
    args = driver.build_parser().parse_args(
        ["decoy", "--output-dir", str(tmp_path), "--devices", "0",
         "--decoy-sequence", "ACDEF"]
    )
    with pytest.raises(ValueError, match="lengths must match"):
        driver.decoy(args)


# Saliency persistence --------------------------------------------------------


def test_saliency_callback_receives_every_feasible_move(monkeypatch):
    """The matrix is computed to build the proposal distribution and otherwise
    discarded, so the callback is the only chance to record it."""
    from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search

    wt = np.zeros(6, dtype=np.int32)
    mask = np.array([True, True, True, False, False, False])
    captured = []

    def gradient_fn(seq):
        g = np.zeros((6, 20), dtype=np.float64)
        g[0, 1] = -1.0
        return float(seq.sum()), g

    def confidence_fn(seq):
        return ConfidenceScore(value=float(seq.sum()) * 0.01)

    run_gradient_search(
        wt=wt, designable_mask=mask,
        gradient_fn=gradient_fn, confidence_fn=confidence_fn,
        config=SearchConfig(policy="population", width=2, edit_budget=2,
                            max_score_calls=4, max_gradient_calls=4,
                            max_proposals=8, seed=0),
        on_saliency=lambda **kw: captured.append(kw),
    )
    assert captured, "the hook fired"
    first = captured[0]
    assert len(first["moves"]) == len(first["deltas"])
    # Only designable positions may appear.
    assert {m[0] for m in first["moves"]} <= {0, 1, 2}
    assert np.all(np.isfinite(first["deltas"]))


def test_search_without_the_hook_behaves_identically():
    """Instrumentation must not change the trajectory."""
    from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search

    def build():
        wt = np.zeros(6, dtype=np.int32)
        mask = np.array([True, True, True, False, False, False])
        return wt, mask

    def gradient_fn(seq):
        g = np.zeros((6, 20), dtype=np.float64)
        g[0, 1] = -1.0
        return float(seq.sum()), g

    def confidence_fn(seq):
        return ConfidenceScore(value=float(seq.sum()) * 0.01)

    cfg = dict(policy="population", width=2, edit_budget=2, max_score_calls=4,
               max_gradient_calls=4, max_proposals=8, seed=0)
    results = []
    for hook in (None, lambda **kw: None):
        wt, mask = build()
        results.append(
            run_gradient_search(
                wt=wt, designable_mask=mask, gradient_fn=gradient_fn,
                confidence_fn=confidence_fn, config=SearchConfig(**cfg),
                on_saliency=hook,
            ).best.score
        )
    assert results[0] == results[1]


# The launcher -----------------------------------------------------------------


FOLLOWUPS = REPO / "examples/run_p17_followups.sh"


def _dry_run(*args):
    import os
    import subprocess

    return subprocess.run(
        ["bash", str(FOLLOWUPS), "--dry-run", *map(str, args)],
        cwd=REPO, env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )


def test_launcher_defaults_the_decoy_arm_to_the_epitope_scramble():
    """Neither decoy that fails its own gate may come back by omission.

    Measured mean target pLDDT: epitope-scrambled 0.838 against the real
    target's 0.890, whole-chain shuffle 0.405, IL7RA trimmed 0.425. A launcher
    that quietly picked one of the last two would void the arm, and would now
    do it partway through the run rather than before it started.
    """
    result = _dry_run("--arms", "decoy")
    assert result.returncode == 0, result.stderr
    assert "--decoy-mode epitope" in result.stdout
    assert "--decoy-structure" not in result.stdout
    assert "IL7RA" not in result.stdout
    assert not list(REPO.glob("results/p17_followups_*/decoy")), "dry run made nothing"


def test_launcher_still_allows_an_explicit_real_protein_decoy():
    result = _dry_run("--arms", "decoy", "--decoy-structure", REPO / "IL7RA.cif")
    assert result.returncode == 0, result.stderr
    assert "--decoy-structure" in result.stdout and "IL7RA.cif" in result.stdout
    assert "--decoy-mode" not in result.stdout


def _dry_run_with_stub_smi(tmp_path, *args, busy_pids="", absent=False):
    """Dry-run the launcher against a stubbed nvidia-smi.

    The occupancy check's behaviour is how it reads that output, not whether
    this host happens to have eight free devices.
    """
    import os
    import subprocess

    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    script = directory / "nvidia-smi"
    body = (
        '      echo "No devices were found"; exit 6 ;;\n'
        if absent
        else f'      printf "%s" {busy_pids!r}; exit 0 ;;\n'
    )
    script.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        "    --query-compute-apps=pid)\n" + body + "  esac\n"
        "done\nexit 0\n"
    )
    script.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{directory}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(FOLLOWUPS), "--arms", "decoy", *map(str, args)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )


def test_launcher_refuses_to_launch_onto_a_busy_device(tmp_path):
    """Workers preallocate, so a busy device kills them rather than queueing.

    Added 2026-10-05: this launcher had no occupancy check, which is how a
    second run onto the same node would have failed.
    """
    result = _dry_run_with_stub_smi(
        tmp_path, "--devices", "0", busy_pids="4242\n"
    )
    assert result.returncode == 2
    assert "already hold a process" in result.stderr
    assert "Applied five OpenDDE patches" not in result.stdout


def test_launcher_allows_a_busy_device_only_when_told_to(tmp_path):
    result = _dry_run_with_stub_smi(
        tmp_path, "--devices", "0", "--allow-busy-gpus", "--dry-run",
        busy_pids="4242\n",
    )
    assert result.returncode == 0, result.stderr
    assert "already hold a process" in result.stderr


def test_launcher_treats_an_absent_device_as_fatal_not_busy(tmp_path):
    """--allow-busy-gpus must not cover a device that is not there.

    nvidia-smi prints to stdout and exits nonzero for an absent device, so
    counting its output as a process reports every absent device as busy.
    """
    result = _dry_run_with_stub_smi(
        tmp_path, "--devices", "0,1", "--allow-busy-gpus", absent=True
    )
    assert result.returncode == 2
    assert "do not exist on this host" in result.stderr
    assert "already hold a process" not in result.stderr


def test_launcher_reports_free_devices_before_launching(tmp_path):
    result = _dry_run_with_stub_smi(tmp_path, "--devices", "0,1,2", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "all 3 requested devices exist and are free" in result.stdout


def test_launcher_refuses_a_decoy_structure_that_is_not_there(tmp_path):
    result = _dry_run("--arms", "decoy", "--decoy-structure", tmp_path / "nope.cif")
    assert result.returncode == 2
    assert "Decoy structure not found" in result.stderr
