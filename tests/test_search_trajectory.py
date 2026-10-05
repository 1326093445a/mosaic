"""Trajectory reconstruction from a completed run's event log.

The measured cases at the bottom are pinned to the archived Alpha run, because
the point of this module is reading a specific finished experiment correctly
and a synthetic log cannot catch a misread of the real schema.
"""

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

import p17_search_trajectory as traj  # noqa: E402
from mosaic.common import TOKENS  # noqa: E402

ALPHA_RUN = REPO / "results/p17_alpha_recovery_20261004_234512_3428883/search"
needs_archive = pytest.mark.skipif(
    not ALPHA_RUN.is_dir(), reason="archived Alpha run not present"
)


def tokens(sequence):
    return [TOKENS.index(aa) for aa in sequence]


def write_log(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def evaluation(candidate_id, sequence, score):
    return dict(
        event="evaluation", candidate_id=candidate_id,
        sequence=tokens(sequence), score=score,
    )


def proposal(parent_id, candidate_id, position, to_residue, *, reverted=-1,
             predicted=-0.1, accepted=True, **extra):
    return dict(
        event="proposal", parent_id=parent_id, candidate_id=candidate_id,
        move=[position, TOKENS.index(to_residue), reverted],
        predicted_loss_delta=predicted, accepted=accepted,
        gradient_calls=1, score_calls=1, attempt=0, duplicate=False,
        cache_hit=False, feasible_moves=10, entropy=0.6,
        proposal_temperature=0.03, **extra,
    )


# Reading the log ------------------------------------------------------------


def test_a_truncated_final_line_is_dropped_not_fatal(tmp_path):
    """A killed run leaves a partial line; the rest stays usable."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "initialization"}\n{"event": "prop')
    assert [e["event"] for e in traj.load_events(path)] == ["initialization"]


def test_a_malformed_line_before_the_end_is_a_corrupt_log(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\nnot json\n{"event": "b"}\n')
    with pytest.raises(ValueError, match="line 2"):
        traj.load_events(path)


# Naming the move ------------------------------------------------------------


def test_the_from_residue_comes_from_the_parent_not_the_start(tmp_path):
    """A second edit at an already-edited position replaced the first edit."""
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "AAA", 0.1),
        evaluation(1, "ACA", 0.2),
        proposal(1, 2, 1, "W"),
        evaluation(2, "AWA", 0.3),
    ])
    rows = traj.trajectory(traj.load_events(log))
    assert len(rows) == 1
    assert rows[0]["from_residue"] == "C", "the parent's residue, not the start's"
    assert rows[0]["to_residue"] == "W"
    assert rows[0]["position_1idx"] == 2


def test_the_paired_reversion_is_another_position_not_this_move(tmp_path):
    """`_moves()` pays for an edit by restoring a DIFFERENT position.

    Reading the third element as "this move is a reversion" inverts the
    meaning, and at the edit cap nearly every proposal carries one.
    """
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "AAA", 0.1),
        proposal(0, 1, 2, "W", reverted=0),
        evaluation(1, "AAW", 0.2),
    ])
    row = traj.trajectory(traj.load_events(log))[0]
    assert row["paired_reversion_0idx"] == 0
    assert row["paid_for_by_reversion"] is True
    assert row["position_0idx"] == 2, "the move's own position is unchanged"


def test_restores_reference_is_unknown_without_a_reference(tmp_path):
    """Pre-2026-10-05 runs recorded reference_binder_sequence: null."""
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "AAA", 0.1), proposal(0, 1, 1, "W"),
        evaluation(1, "AWA", 0.2),
    ])
    rows = traj.trajectory(traj.load_events(log))
    assert rows[0]["restores_reference"] is None
    assert rows[0]["reference_residue"] == ""


def test_restores_reference_marks_the_crystal_residue_only(tmp_path):
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "ACA", 0.1),
        proposal(0, 1, 1, "G"),
        evaluation(1, "AGA", 0.8),
        proposal(0, 2, 1, "W"),
        evaluation(2, "AWA", 0.2),
    ])
    rows = traj.trajectory(traj.load_events(log), reference_sequence="AGA")
    assert [r["restores_reference"] for r in rows] == [True, False]
    assert rows[0]["reference_residue"] == "G"


# Realized outcome -----------------------------------------------------------


def test_realized_delta_is_child_minus_parent_and_none_when_unscored(tmp_path):
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "AAA", 0.10),
        proposal(0, 1, 1, "W"),
        evaluation(1, "AWA", 0.85),
        proposal(0, 2, 1, "C", accepted=False),
    ])
    rows = traj.trajectory(traj.load_events(log))
    assert rows[0]["realized_score_delta"] == pytest.approx(0.75)
    assert rows[1]["realized_score_delta"] is None, "never folded, so unknown"
    assert rows[1]["sign_agrees"] is None


def test_sign_agreement_counts_only_measurable_proposals(tmp_path):
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "AAA", 0.10),
        # predicted improvement, realized improvement: agrees
        proposal(0, 1, 1, "W", predicted=-0.2),
        evaluation(1, "AWA", 0.85),
        # predicted improvement, realized loss: disagrees
        proposal(0, 2, 1, "C", predicted=-0.2),
        evaluation(2, "ACA", 0.01),
        # never scored: not counted either way
        proposal(0, 3, 1, "Y", predicted=-0.2, accepted=False),
    ])
    rows = traj.trajectory(traj.load_events(log))
    assert traj.sign_agreement(rows) == dict(counted=2, agreed=1, rate=0.5)


def test_sign_agreement_on_a_log_with_nothing_measurable(tmp_path):
    log = write_log(tmp_path / "events.jsonl", [proposal(0, 1, 1, "W")])
    assert traj.sign_agreement(traj.trajectory(traj.load_events(log)))["rate"] is None


# Per-position summary -------------------------------------------------------


def test_position_history_ranks_by_visits_and_pools_residues(tmp_path):
    log = write_log(tmp_path / "events.jsonl", [
        evaluation(0, "ACA", 0.1),
        proposal(0, 1, 1, "G"), evaluation(1, "AGA", 0.8),
        proposal(0, 2, 1, "W", accepted=False), evaluation(2, "AWA", 0.2),
        proposal(0, 3, 0, "M"), evaluation(3, "MCA", 0.3),
    ])
    rows = traj.trajectory(traj.load_events(log), reference_sequence="AGA")
    history = traj.position_history(rows)
    assert [h["position_1idx"] for h in history] == [2, 1]
    busiest = history[0]
    assert busiest["proposed"] == 2 and busiest["accepted"] == 1
    assert busiest["residues_tried"] == "GW"
    assert busiest["residues_accepted"] == "G"
    assert busiest["restores_reference"] == 1
    assert busiest["best_predicted_loss_delta"] == pytest.approx(-0.1)


def test_find_runs_accepts_a_run_a_parent_and_a_search_directory(tmp_path):
    single = tmp_path / "one"
    write_log(single / "logs/events.jsonl", [proposal(0, 1, 0, "W")])
    assert traj.find_runs(single) == [single]
    assert traj.find_runs(tmp_path) == [single]

    nested = tmp_path / "run/search/population_seed0"
    write_log(nested / "logs/events.jsonl", [proposal(0, 1, 0, "W")])
    assert traj.find_runs(tmp_path / "run") == [nested]


# The archived Alpha run -----------------------------------------------------


@needs_archive
def test_the_search_did_propose_reverting_the_damage_at_position_100():
    """Corrects the record: the gradient did find the crystal residue.

    Measured 2026-10-05 on population_seed0: position 100 proposed G three
    times, accepted every time, realized score deltas +0.1271, +0.7559 and
    +0.7543. §20.4's reading of the winner diffs as "no reversion" was wrong
    for this run -- its winner differs from the crystal only at 104 and 106.
    """
    from p17_alpha_reference import load_complex

    reference = load_complex(REPO / "P17_Alpha.pdb", "B", "A")["binder_seq"]
    assert reference[99] == "G", "position 100 (1-indexed) is the damaged site"

    rows = traj.trajectory(
        traj.load_events(ALPHA_RUN / "population_seed0/logs/events.jsonl"),
        reference,
    )
    restoring = [r for r in rows if r["restores_reference"]]
    assert len(restoring) == 3
    assert all(r["position_1idx"] == 100 for r in restoring)
    assert all(r["to_residue"] == "G" for r in restoring)
    assert all(r["accepted"] for r in restoring)
    assert max(r["realized_score_delta"] for r in restoring) > 0.75


@needs_archive
def test_only_one_of_four_runs_ever_proposed_the_crystal_residue_there():
    """Position 100 was the most-visited site in every run, G proposed in one.

    This is the finding that makes the Alpha control readable: the gradient
    located the damaged position reliably and named the right residue there
    only once in four runs.
    """
    from p17_alpha_reference import load_complex

    reference = load_complex(REPO / "P17_Alpha.pdb", "B", "A")["binder_seq"]
    proposed_g = {}
    for run in sorted(ALPHA_RUN.glob("*")):
        log = run / "logs/events.jsonl"
        if not log.is_file():
            continue
        rows = traj.trajectory(traj.load_events(log), reference)
        at_100 = [r for r in rows if r["position_1idx"] == 100]
        assert at_100, f"{run.name} never touched position 100"
        proposed_g[run.name] = sum(1 for r in at_100 if r["to_residue"] == "G")
    assert len(proposed_g) == 4
    assert sum(1 for count in proposed_g.values() if count) == 1
    assert proposed_g["population_seed0"] == 3


@needs_archive
def test_102_g_to_f_was_a_single_proposal_not_a_converged_discovery():
    """Measured: independent_seed0, one proposal, accepted, realized +0.6896."""
    rows = traj.trajectory(
        traj.load_events(ALPHA_RUN / "independent_seed0/logs/events.jsonl")
    )
    at_102 = [r for r in rows if r["position_1idx"] == 102]
    assert len(at_102) == 1
    assert at_102[0]["from_residue"] == "G" and at_102[0]["to_residue"] == "F"
    assert at_102[0]["accepted"]
    assert at_102[0]["realized_score_delta"] == pytest.approx(0.6896, abs=1e-3)
