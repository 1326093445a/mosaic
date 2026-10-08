"""End-to-end wiring for the recovery pipelines, through the real contracts.

Motivation: two bugs in this pipeline reached the cluster because the tests
checked *interfaces* (the flags exist, the globs resolve) and not *contracts*
(what the downstream script does with what the upstream script wrote). Both
surfaced only after a search had already run.

So these drive the real `load_candidates`, `select_shard`, `merge_shard_tables`
and table writers over archives shaped exactly as `p17_confidence_search.py`
writes them. Only the GPU prediction is stubbed. No model, no checkpoint.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import p17_alpha_recovery as driver  # noqa: E402
from p17_rescore_winners import (  # noqa: E402
    load_candidates,
    merge_shard_tables,
    select_shard,
)

BINDER_LEN = 123
# Designable positions as the real mask produces them (CDR1 26-33, CDR2 51-58,
# CDR3 97-109, 1-indexed -> 0-indexed).
DESIGNABLE = sorted(
    set(range(25, 33)) | set(range(50, 58)) | set(range(96, 109))
)
REFERENCE_BINDER = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(BINDER_LEN))
TARGET_JN1 = "M" * 184
TARGET_ALPHA = "M" * 195


def mutate(sequence, positions, replacement="W"):
    residues = list(sequence)
    for position in positions:
        residues[position] = "Y" if residues[position] == replacement else replacement
    return "".join(residues)


def write_search_run(root, name, *, start, winner, target, budget, reference=None,
                     budget_anchor=None):
    """A search run directory exactly as `p17_confidence_search.py` leaves one.

    `budget_anchor` is "reference" for a run launched by the continuous stage
    under `--budget-anchor reference`, which starts at a seed but counts its
    edits from the reference. Left None, the archive looks like every run
    written before that flag existed and the two anchors coincide.
    """
    run = root / "search" / name
    (run / "tables").mkdir(parents=True)
    anchor_extras = {}
    if budget_anchor is not None:
        anchor_extras = dict(
            budget_anchor=budget_anchor,
            budget_anchor_sequence=(
                (reference or start) if budget_anchor == "reference" else start
            ),
        )
    (run / "config.json").write_text(
        json.dumps(
            dict(
                output_layout_version=2,
                binder_sequence=start,
                reference_binder_sequence=reference if reference else start,
                start_differs_from_reference=start != (reference or start),
                **anchor_extras,
                target_sequence=target,
                checkpoint="opendde_abag.pt",
                recycling_steps=4,
                designable_positions_0idx=DESIGNABLE,
                config=dict(edit_budget=budget),
                arguments=dict(
                    sampling_steps=64,
                    pae_cutoff=12.0,
                    distance_cutoff=12.0,
                    selection_seeds=[0, 1],
                ),
            ),
            indent=2,
        )
    )
    (run / "summary.json").write_text(
        json.dumps(dict(best_sequence=winner, best_id=3, best_score=0.25))
    )
    return run


def test_load_candidates_reads_a_run_that_started_from_the_reference(tmp_path):
    """The JN.1 shape: the search starts at WT, so start == reference."""
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[:5])
    write_search_run(
        tmp_path, "population_seed0",
        start=REFERENCE_BINDER, winner=winner, target=TARGET_JN1, budget=5,
    )
    candidates, links, baseline = load_candidates(tmp_path / "search")
    assert [c["sequence"] for c in candidates] == [REFERENCE_BINDER, winner]
    assert candidates[0]["is_wt"] is True
    assert links[0]["candidate_id"] == 1
    assert baseline["reference_binder_sequence"] == REFERENCE_BINDER


def test_load_candidates_reads_a_run_that_started_from_a_damaged_sequence(tmp_path):
    """The Alpha shape: edits are measured from the damaged start, not the
    reference, which is what makes the recovery control's archive valid."""
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    winner = mutate(damaged, DESIGNABLE[2:4])
    write_search_run(
        tmp_path, "population_seed0",
        start=damaged, winner=winner, target=TARGET_ALPHA, budget=2,
        reference=REFERENCE_BINDER,
    )
    candidates, _, baseline = load_candidates(tmp_path / "search")
    assert candidates[0]["sequence"] == damaged, "candidate 0 is the run's start"
    assert baseline["reference_binder_sequence"] == REFERENCE_BINDER
    assert baseline["start_differs_from_reference"] is True


def test_load_candidates_rejects_a_winner_outside_the_recorded_budget(tmp_path):
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    too_far = mutate(damaged, DESIGNABLE[2:9])
    write_search_run(
        tmp_path, "population_seed0",
        start=damaged, winner=too_far, target=TARGET_ALPHA, budget=2,
        reference=REFERENCE_BINDER,
    )
    with pytest.raises(ValueError, match="over the recorded edit_budget"):
        load_candidates(tmp_path / "search")


def test_load_candidates_counts_the_budget_from_the_recorded_anchor(tmp_path):
    """The continuous-stage shape, and the bug that killed every held-out shard
    of the bindcraft cell on 2026-10-08.

    Under `--budget-anchor reference` the search starts at a seed but counts
    edits from the reference. A winner at the budget from the reference can sit
    one edit FURTHER from the seed, because reverting the seed's own edit is
    itself a difference from the seed. Counting against the start therefore
    rejected a run that never broke its budget.
    """
    seed = mutate(REFERENCE_BINDER, DESIGNABLE[:1])
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[1:6])
    assert sum(a != b for a, b in zip(REFERENCE_BINDER, winner)) == 5, "at budget"
    assert sum(a != b for a, b in zip(seed, winner)) == 6, "over budget from the seed"

    write_search_run(
        tmp_path, "population_seed0",
        start=seed, winner=winner, target=TARGET_JN1, budget=5,
        reference=REFERENCE_BINDER, budget_anchor="reference",
    )
    candidates, links, baseline = load_candidates(tmp_path / "search")
    assert [c["sequence"] for c in candidates] == [seed, winner]
    assert candidates[0]["sequence"] == seed, "candidate 0 stays the run's start"
    assert baseline["budget_anchor_sequence"] == REFERENCE_BINDER
    assert links[0]["candidate_id"] == 1


def test_load_candidates_still_rejects_a_winner_over_the_anchored_budget(tmp_path):
    """The anchor relaxes where the budget is measured from, not how big it is."""
    seed = mutate(REFERENCE_BINDER, DESIGNABLE[:1])
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[1:8])
    assert sum(a != b for a, b in zip(REFERENCE_BINDER, winner)) == 7
    write_search_run(
        tmp_path, "population_seed0",
        start=seed, winner=winner, target=TARGET_JN1, budget=5,
        reference=REFERENCE_BINDER, budget_anchor="reference",
    )
    with pytest.raises(ValueError, match="over the recorded edit_budget"):
        load_candidates(tmp_path / "search")


def test_load_candidates_rejects_anchored_drift_outside_the_designable_mask(tmp_path):
    """The mask check must cover drift from the start as well as from the
    anchor, so swapping the anchor cannot smuggle a fixed-position change in."""
    seed = mutate(REFERENCE_BINDER, DESIGNABLE[:1])
    winner = mutate(seed, [0])  # position 0 is not designable
    write_search_run(
        tmp_path, "population_seed0",
        start=seed, winner=winner, target=TARGET_JN1, budget=5,
        reference=REFERENCE_BINDER, budget_anchor="reference",
    )
    with pytest.raises(ValueError, match="outside the designable mask"):
        load_candidates(tmp_path / "search")


def test_load_candidates_rejects_runs_against_different_targets(tmp_path):
    """Mixing an Alpha run and a JN.1 run in one batch must not merge."""
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[:3])
    write_search_run(
        tmp_path, "a", start=REFERENCE_BINDER, winner=winner,
        target=TARGET_JN1, budget=5,
    )
    write_search_run(
        tmp_path, "b", start=REFERENCE_BINDER, winner=winner,
        target=TARGET_ALPHA, budget=5,
    )
    with pytest.raises(ValueError, match="inconsistent batch target_sequence"):
        load_candidates(tmp_path / "search")


def test_shared_winners_across_runs_become_one_candidate_with_both_sources(tmp_path):
    """Four seeds converging on one sequence is one prediction, not four."""
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[:4])
    for name in ("population_seed0", "population_seed1", "population_seed2"):
        write_search_run(
            tmp_path, name, start=REFERENCE_BINDER, winner=winner,
            target=TARGET_JN1, budget=5,
        )
    candidates, links, _ = load_candidates(tmp_path / "search")
    assert len(candidates) == 2, "WT plus one distinct winner"
    assert len(links) == 3, "but three source runs preserved"
    assert {link["candidate_id"] for link in links} == {1}


def _full_pipeline(tmp_path, *, target, start, reference, budget, n_shards):
    """Search archives -> sharded prediction -> merge -> table, for real.

    Stands in for the GPU stage only: each shard writes the prediction CSV and
    artifacts that `merge_shard_tables` requires, including the files it checks
    for existence.
    """
    winners = [mutate(start, DESIGNABLE[i : i + 2]) for i in (2, 4, 6)]
    for index, winner in enumerate(winners):
        write_search_run(
            tmp_path, f"population_seed{index}", start=start, winner=winner,
            target=target, budget=budget, reference=reference,
        )
    candidates, links, _ = load_candidates(tmp_path / "search")

    stage = tmp_path / "heldout"
    total = 0
    for shard in range(n_shards):
        assigned = select_shard(candidates, n_shards, shard)
        directory = stage / f"shard_{shard}"
        (directory / "tables").mkdir(parents=True)
        rows = []
        for candidate in assigned:
            for seed in driver.HELDOUT_SEEDS:
                stem = f"candidate_{candidate['candidate_id']:05d}/seed_{seed}"
                for suffix in ("pdb", "cif", "npz"):
                    path = directory / f"{suffix}/{stem}.{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
                # WT scores badly; winners better, and closer in pose.
                is_wt = candidate["candidate_id"] == 0
                rows.append(
                    dict(
                        candidate_id=candidate["candidate_id"],
                        selection_seed=seed,
                        ipsae_min=0.01 if is_wt else 0.30,
                        binder_pose_rmsd_A=40.0 if is_wt else 25.0,
                        target_aligned_rmsd_A=1.85,
                        binder_internal_rmsd_A=2.0,
                        structure_file=f"pdb/{stem}.pdb",
                        cif_file=f"cif/{stem}.cif",
                        confidence_file=f"npz/{stem}.npz",
                    )
                )
        total += len(rows)
        for filename, records in (
            ("candidates.csv", assigned),
            ("source_runs.csv", [link for link in links
                                 if link["candidate_id"] in
                                 {c["candidate_id"] for c in assigned}]),
            ("predictions.csv", rows),
        ):
            if not records:
                records = []
            fieldnames = (
                list(records[0]) if records
                else {"candidates.csv": ["candidate_id", "sequence", "is_wt"],
                      "source_runs.csv": ["candidate_id", "source_run",
                                          "source_candidate_id", "original_score"],
                      "predictions.csv": list(rows[0])}[filename]
            )
            with (directory / "tables" / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        (directory / "summary.json").write_text(
            json.dumps(dict(completed=True, sequences=len(assigned),
                            prediction_calls=len(rows)))
        )
    return candidates, total


def test_jn1_pipeline_runs_end_to_end_and_reports_gains_over_wt(tmp_path):
    candidates, total = _full_pipeline(
        tmp_path, target=TARGET_JN1, start=REFERENCE_BINDER,
        reference=REFERENCE_BINDER, budget=5, n_shards=4,
    )
    assert total == len(candidates) * len(driver.HELDOUT_SEEDS)

    driver.write_jn1_table(tmp_path)
    rows = list(csv.DictReader((tmp_path / "tables/jn1_recovery.csv").open()))
    assert len(rows) == len(candidates)
    wt = next(r for r in rows if r["is_wt"] == "True")
    winner = next(r for r in rows if r["is_wt"] == "False")
    assert float(wt["ipsae_gain_over_wt"]) == pytest.approx(0.0)
    assert float(winner["ipsae_gain_over_wt"]) == pytest.approx(0.29)
    assert float(winner["pose_change_vs_wt_A"]) == pytest.approx(-15.0)
    assert int(winner["heldout_seeds"]) == len(driver.HELDOUT_SEEDS)
    context = json.loads((tmp_path / "tables/jn1_context.json").read_text())
    assert context["heldout_seeds"] == list(driver.HELDOUT_SEEDS)


def test_alpha_pipeline_runs_end_to_end_from_a_damaged_start(tmp_path):
    """The case that failed on the cluster, now driven all the way through."""
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    candidates, _ = _full_pipeline(
        tmp_path, target=TARGET_ALPHA, start=damaged,
        reference=REFERENCE_BINDER, budget=2, n_shards=4,
    )
    calibration = dict(
        reference=dict(mean_ipsae=0.80, worst_pose_rmsd_A=2.6),
        rungs=[dict(n_edits=2, scored=True, mean_ipsae=0.00)],
        recommended_rung=2,
    )
    driver.write_recovery_table(tmp_path, calibration, 2, 2)
    rows = list(csv.DictReader((tmp_path / "tables/recovery.csv").open()))
    assert len(rows) == len(candidates)
    wt = next(r for r in rows if r["candidate_id"] == "0")
    winner = next(r for r in rows if r["candidate_id"] != "0")
    # Span is 0.80 - 0.00; the damaged start recovers none of it by definition.
    assert float(wt["ipsae_recovery_fraction"]) == pytest.approx(0.01 / 0.80)
    assert float(winner["ipsae_recovery_fraction"]) == pytest.approx(0.30 / 0.80)


def test_every_candidate_lands_in_exactly_one_shard(tmp_path):
    """A candidate scored twice or never would corrupt the held-out table."""
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[:3])
    write_search_run(
        tmp_path, "population_seed0", start=REFERENCE_BINDER, winner=winner,
        target=TARGET_JN1, budget=5,
    )
    candidates, _, _ = load_candidates(tmp_path / "search")
    for n_shards in (1, 2, 4, 8):
        assigned = [
            c["candidate_id"]
            for shard in range(n_shards)
            for c in select_shard(candidates, n_shards, shard)
        ]
        assert sorted(assigned) == [c["candidate_id"] for c in candidates]


def test_more_shards_than_candidates_leaves_empty_shards(tmp_path):
    """Why shards 5-7 reported ok on 2026-10-04: they had nothing to do.

    An `ok` shard is not evidence that anything was scored.
    """
    winner = mutate(REFERENCE_BINDER, DESIGNABLE[:3])
    write_search_run(
        tmp_path, "population_seed0", start=REFERENCE_BINDER, winner=winner,
        target=TARGET_JN1, budget=5,
    )
    candidates, _, _ = load_candidates(tmp_path / "search")
    empty = [s for s in range(8) if not select_shard(candidates, 8, s)]
    assert len(empty) == 8 - len(candidates)


def test_merge_refuses_a_shard_whose_artifacts_are_missing(tmp_path):
    _full_pipeline(
        tmp_path, target=TARGET_JN1, start=REFERENCE_BINDER,
        reference=REFERENCE_BINDER, budget=5, n_shards=2,
    )
    for path in (tmp_path / "heldout/shard_0/pdb").rglob("*.pdb"):
        path.unlink()
    with pytest.raises(ValueError, match="missing prediction artifact"):
        merge_shard_tables(tmp_path / "heldout", 2)


# The reference-consistency guard: both bug layers from 2026-10-04 ------------

from p17_rescore_winners import check_reference_consistency  # noqa: E402


def _archive(start, reference, target, **extra):
    record = dict(
        binder_sequence=start,
        reference_binder_sequence=reference,
        start_differs_from_reference=start != reference,
        target_sequence=target,
    )
    record.update(extra)
    return record


def test_guard_accepts_a_jn1_run_against_the_jn1_reference():
    check_reference_consistency(
        REFERENCE_BINDER,
        TARGET_JN1,
        _archive(REFERENCE_BINDER, REFERENCE_BINDER, TARGET_JN1),
        Path("P17_JN1.pdb"),
        "B",
        "T",
    )


def test_guard_rejects_alpha_runs_against_the_jn1_reference():
    """Layer one: the failure the cluster hit. Targets differ, 195 vs 184."""
    with pytest.raises(ValueError, match="target differs from archived"):
        check_reference_consistency(
            REFERENCE_BINDER,
            TARGET_JN1,
            _archive(REFERENCE_BINDER, REFERENCE_BINDER, TARGET_ALPHA),
            Path("P17_JN1.pdb"),
            "B",
            "T",
        )


def test_guard_accepts_a_damaged_start_against_its_own_reference():
    """Layer two: comparing the damaged start to the reference refused every
    recovery run, even once the right --complex was passed."""
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    check_reference_consistency(
        REFERENCE_BINDER,
        TARGET_ALPHA,
        _archive(damaged, REFERENCE_BINDER, TARGET_ALPHA),
        Path("P17_Alpha.pdb"),
        "B",
        "A",
    )


def test_guard_still_rejects_a_damaged_run_against_the_wrong_reference():
    """Relaxing the binder check must not relax the target check."""
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    with pytest.raises(ValueError, match="target differs"):
        check_reference_consistency(
            REFERENCE_BINDER,
            TARGET_JN1,
            _archive(damaged, REFERENCE_BINDER, TARGET_ALPHA),
            Path("P17_JN1.pdb"),
            "B",
            "T",
        )


def test_guard_rejects_a_binder_that_is_not_the_recorded_reference():
    """A different binder means different reference CA coordinates."""
    other = mutate(REFERENCE_BINDER, [5])  # framework, outside the mask
    with pytest.raises(ValueError, match="binder differs from archived"):
        check_reference_consistency(
            other,
            TARGET_ALPHA,
            _archive(REFERENCE_BINDER, REFERENCE_BINDER, TARGET_ALPHA),
            Path("P17_Alpha.pdb"),
            "B",
            "A",
        )


def test_guard_handles_the_archive_already_on_the_cluster():
    """The in-flight run predates `reference_binder_sequence`.

    Its config records `start_differs_from_reference: true`, so the binder
    identity check is skipped and length is still enforced -- which is what
    makes that run retryable without re-searching.
    """
    damaged = mutate(REFERENCE_BINDER, DESIGNABLE[:2])
    legacy = dict(
        binder_sequence=damaged,
        start_differs_from_reference=True,
        target_sequence=TARGET_ALPHA,
    )
    check_reference_consistency(
        REFERENCE_BINDER, TARGET_ALPHA, legacy, Path("P17_Alpha.pdb"), "B", "A"
    )

    with pytest.raises(ValueError, match="different-length chains"):
        check_reference_consistency(
            REFERENCE_BINDER[:-1],
            TARGET_ALPHA,
            legacy,
            Path("P17_Alpha.pdb"),
            "B",
            "A",
        )


def test_guard_keeps_the_strict_check_for_legacy_jn1_archives():
    """Archives with neither field and no damaged start stay strictly checked."""
    legacy = dict(binder_sequence=REFERENCE_BINDER, target_sequence=TARGET_JN1)
    check_reference_consistency(
        REFERENCE_BINDER, TARGET_JN1, legacy, Path("P17_JN1.pdb"), "B", "T"
    )
    with pytest.raises(ValueError, match="binder differs"):
        check_reference_consistency(
            mutate(REFERENCE_BINDER, [5]),
            TARGET_JN1,
            legacy,
            Path("P17_JN1.pdb"),
            "B",
            "T",
        )


# The decoy archive: the third layer, found on 2026-10-05 ---------------------


# The shared JN.1 fixture is a run of one residue, which cannot be permuted
# into anything different. A decoy needs a target with some variety in it.
TARGET_VARIED = "".join(
    "ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(len(TARGET_JN1))
)


def _decoy_archive(real, decoy, binder=None):
    binder = binder or REFERENCE_BINDER
    return _archive(
        binder, binder, decoy,
        decoy_target=dict(
            real_target_sequence=real,
            decoy_target_sequence=decoy,
            identity_to_real_target=0.9239,
        ),
    )


def _scramble(sequence, positions):
    chars = list(sequence)
    rotated = [chars[p] for p in positions][1:] + [chars[positions[0]]]
    for position, residue in zip(positions, rotated):
        chars[position] = residue
    return "".join(chars)


def test_guard_returns_the_real_target_for_an_ordinary_archive():
    """The return value is what gets folded, so it has to be the right one."""
    scored = check_reference_consistency(
        REFERENCE_BINDER,
        TARGET_JN1,
        _archive(REFERENCE_BINDER, REFERENCE_BINDER, TARGET_JN1),
        Path("P17_JN1.pdb"), "B", "T",
    )
    assert scored == TARGET_JN1


def test_guard_accepts_a_decoy_archive_and_returns_the_decoy():
    """The negative control's archived target differs by design.

    The gate refused this arm outright on 2026-10-05. Relaxing it without
    returning the decoy would have been worse: every decoy winner would have
    been rescored against the real JN.1 target, reporting numbers for a
    complex the search never evaluated.
    """
    decoy = _scramble(TARGET_VARIED, [10, 20, 30, 40])
    assert decoy != TARGET_VARIED and len(decoy) == len(TARGET_VARIED)
    assert sorted(decoy) == sorted(TARGET_VARIED), "composition is preserved"
    scored = check_reference_consistency(
        REFERENCE_BINDER,
        TARGET_VARIED,
        _decoy_archive(TARGET_VARIED, decoy),
        Path("P17_JN1.pdb"), "B", "T",
    )
    assert scored == decoy, "the decoy is what the search scored"


def test_guard_still_catches_the_wrong_reference_under_a_decoy_archive():
    """A decoy must not become a licence to accept any reference at all."""
    decoy = _scramble(TARGET_VARIED, [10, 20, 30, 40])
    with pytest.raises(ValueError, match="real target this decoy run replaced"):
        check_reference_consistency(
            REFERENCE_BINDER,
            TARGET_ALPHA,
            _decoy_archive(TARGET_VARIED, decoy),
            Path("P17_Alpha.pdb"), "B", "A",
        )


def test_guard_rejects_an_archive_whose_decoy_record_disagrees_with_itself():
    decoy = _scramble(TARGET_VARIED, [10, 20, 30, 40])
    other = _scramble(TARGET_VARIED, [5, 15, 25, 35])
    assert decoy != other
    archive = _decoy_archive(TARGET_VARIED, decoy)
    archive["target_sequence"] = other
    with pytest.raises(ValueError, match="internally inconsistent"):
        check_reference_consistency(
            REFERENCE_BINDER, TARGET_VARIED, archive,
            Path("P17_JN1.pdb"), "B", "T",
        )


def test_guard_rejects_a_decoy_of_a_different_length_than_the_reference():
    """The reference coordinates are reused, so the lengths must match."""
    archive = _decoy_archive(TARGET_JN1, TARGET_JN1[:-3])
    with pytest.raises(ValueError, match="lengths differ"):
        check_reference_consistency(
            REFERENCE_BINDER, TARGET_JN1, archive,
            Path("P17_JN1.pdb"), "B", "T",
        )


def test_guard_still_checks_the_binder_under_a_decoy_archive():
    decoy = _scramble(TARGET_VARIED, [10, 20, 30, 40])
    archive = _decoy_archive(
        TARGET_VARIED, decoy, binder="A" * len(REFERENCE_BINDER)
    )
    with pytest.raises(ValueError, match="binder differs from archived"):
        check_reference_consistency(
            REFERENCE_BINDER, TARGET_VARIED, archive,
            Path("P17_JN1.pdb"), "B", "T",
        )


# The folded target and the exported target must be one sequence -------------


def test_the_rescorer_folds_and_exports_the_same_target():
    """The second half of the decoy bug, found 2026-10-05.

    `check_reference_consistency` returns the sequence to fold, and the first
    fix used it for the model features but kept passing the *reference's*
    target to `save_prediction`. For the negative control those differ, so
    every shard with candidates died on save_prediction's own invariant --
    which did its job, but only after the arm had run. Asserted structurally
    because the failure is a name mismatch between two call sites, and no
    cheap runtime test reaches both.
    """
    import ast

    source = (REPO / "examples/p17_rescore_winners.py").read_text()
    tree = ast.parse(source)

    folded, exported, returned = [], [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "TargetChain":
            assert node.args and isinstance(node.args[0], ast.Name), (
                "the folded target should be a plain name, not an expression"
            )
            folded.append(node.args[0].id)
        if isinstance(func, ast.Attribute) and func.attr == "save_prediction":
            assert len(node.args) >= 4, "save_prediction takes the target 4th"
            assert isinstance(node.args[3], ast.Name)
            exported.append(node.args[3].id)
        if isinstance(func, ast.Name) and func.id == "check_reference_consistency":
            parent = next(
                (
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.Assign) and n.value is node
                ),
                None,
            )
            if parent and isinstance(parent.targets[0], ast.Name):
                returned.append(parent.targets[0].id)

    assert folded, "no TargetChain call found"
    assert exported, "no save_prediction call found"
    assert returned, "the guard's return value should be bound to a name"
    assert set(folded) == set(exported), (
        f"folded {folded} but exported {exported}; a decoy archive would "
        "export predictions labelled with the wrong target"
    )
    assert set(folded) == set(returned), (
        "both should use what check_reference_consistency returned, which is "
        "the decoy for a negative-control archive"
    )


def test_the_reference_target_is_not_reusable_after_the_guard():
    """Keeping it in scope is what allowed the wrong one to be exported."""
    import re

    lines = (REPO / "examples/p17_rescore_winners.py").read_text().splitlines()
    # `reference_target_chain` is a different name, so match on word bounds.
    name = re.compile(r"\breference_target\b")
    deletions = [i for i, line in enumerate(lines) if line.strip() == "del reference_target"]
    assert len(deletions) == 1, "the reference target should be released once"
    after = [
        line for line in lines[deletions[0] + 1:]
        if name.search(line) and not line.lstrip().startswith("#")
    ]
    assert after == [], (
        f"reference_target is still reachable after the guard: {after}"
    )
