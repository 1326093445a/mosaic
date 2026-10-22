"""CPU-only checks of joint retention and endpoint selection."""

from dataclasses import replace

import numpy as np
import pytest

from mosaic.search import (
    Candidate,
    ConfidenceScore,
    SearchConfig,
    _pareto_population,
    candidate_selection_key,
    pareto_front,
    run_gradient_search,
)


def config(**kwargs):
    return replace(
        SearchConfig(
            policy="population", width=2, alphabet_size=2, edit_budget=2,
            pareto_metric="pose", pareto_ceiling=3.0,
        ),
        **kwargs,
    )


def candidate(i, score, pose, violation=0.0):
    return Candidate(i, (i,), score, {"pose": pose}, violation)


def test_front_and_final_choice_distinguish_tradeoffs_constraints_and_ties():
    candidates = [
        candidate(0, 0.5, 8),
        candidate(1, 0.9, 10),
        candidate(2, 0.8, 3),
        candidate(3, 0.6, 1),
        candidate(4, 0.7, 4),
        candidate(5, 0.8, 3),
        candidate(6, 1.0, 0.5, violation=1),
    ]
    front = pareto_front(candidates, "pose")
    assert [c.id for c in front] == [3, 2, 1]
    assert max(candidates, key=lambda c: candidate_selection_key(c, config())).id == 2
    fallback = config(pareto_ceiling=0.5)
    assert max(candidates, key=lambda c: candidate_selection_key(c, fallback)).id == 3
    assert pareto_front([], "pose") == []


def test_crowding_preserves_extremes_and_is_invariant_to_units():
    front = [candidate(i, score, pose) for i, (score, pose) in enumerate(
        [(0.1, 1), (0.2, 2), (0.3, 3), (0.8, 4), (0.9, 10)]
    )]
    chosen = _pareto_population(front, "pose", 3)
    assert {c.id for c in chosen} == {0, 3, 4}
    rescaled = [replace(c, metrics={"pose": c.metrics["pose"] * 1000}) for c in front]
    assert [c.id for c in _pareto_population(rescaled, "pose", 3)] == [c.id for c in chosen]
    assert [c.id for c in _pareto_population(front[:1], "pose", 3)] == [0, 0, 0]


@pytest.mark.parametrize("proposed_pose,expected_admission", [(1.0, True), (12.0, False)])
def test_search_retains_useful_pose_tradeoff_but_rejects_dominated_proposal(
    proposed_pose, expected_admission
):
    values = {
        (0, 0, 0): (0.8, 8.0),
        (1, 0, 0): (0.9, 10.0),
        (0, 0, 1): (0.7, proposed_pose),
    }
    calls = []

    def score(sequence):
        calls.append(tuple(sequence))
        confidence, pose = values[tuple(sequence)]
        return ConfidenceScore(confidence, {"pose": pose})

    kwargs = dict(
        wt=np.zeros(3, dtype=int),
        designable_mask=np.ones(3, dtype=bool),
        initial_sequences=np.array([[0, 0, 0], [1, 0, 0]]),
        gradient_fn=lambda _: (0.0, np.array([[0, 1], [0, 1], [0, -1]])),
        confidence_fn=score,
    )
    events = []
    settings = config(max_proposals=1, max_score_calls=3, target_entropy=1e-6,
                      acceptance_temperature=0)
    result = run_gradient_search(**kwargs, config=settings, on_event=events.append)
    assert len(calls) == result.stats["score_calls"] == 3
    assert result.stats["gradient_calls"] == 1
    proposal = next(e for e in events if e["event"] == "proposal")
    assert proposal["accepted"] == expected_admission
    assert proposal["competitor_id"] is None
    assert result.best.id == (2 if expected_admission else 0)
    assert (2 in {c.id for c in result.active}) == expected_admission
    assert {c.id for c in result.pareto_front} == ({0, 1, 2} if expected_admission else {0, 1})
    baseline = run_gradient_search(
        **kwargs, config=replace(settings, pareto_metric=None, pareto_ceiling=None)
    )
    assert {c.id for c in baseline.active} == {0, 1}
    assert baseline.best.id == 1


@pytest.mark.parametrize("arity", [1, 2])
def test_pareto_search_preserves_fixed_positions_budget_and_reproducibility(arity):
    settings = config(width=4, move_arity=arity, max_score_calls=10,
                      max_proposals=40, seed=17)

    def run():
        return run_gradient_search(
            wt=np.zeros(4, dtype=int),
            designable_mask=np.array([True, False, True, True]),
            initial_sequences=np.repeat([[1, 0, 0, 0]], settings.width, axis=0),
            gradient_fn=lambda s: (-float(s.sum()), np.tile([0.0, -1.0], (4, 1))),
            confidence_fn=lambda s: ConfidenceScore(
                float(s @ [1, 0, 3, 2]), {"pose": float(10 + s @ [2, 0, -3, 1])}
            ),
            config=settings,
        )

    result, repeated = run(), run()
    assert result.evaluated == repeated.evaluated
    assert result.active == repeated.active
    assert result.best == repeated.best
    assert all(c.sequence[1] == 0 for c in result.evaluated)
    assert all(np.count_nonzero(c.sequence) <= 2 for c in result.evaluated)
    assert result.stats["score_calls"] == len(result.evaluated) <= 10


@pytest.mark.parametrize("overrides", [
    {"policy": "independent"}, {"width": 1}, {"pareto_ceiling": None},
    {"pareto_metric": None}, {"pareto_metric": ""}, {"pareto_ceiling": np.inf},
])
def test_pareto_configuration_rejects_incomplete_or_incompatible_settings(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


@pytest.mark.parametrize("metrics", [{}, {"pose": np.nan}])
def test_missing_or_nonfinite_pareto_metric_fails(metrics):
    with pytest.raises(ValueError, match="missing Pareto metric|nonfinite"):
        run_gradient_search(
            wt=np.array([0]), designable_mask=np.array([True]),
            gradient_fn=lambda _: pytest.fail("must fail before gradients"),
            confidence_fn=lambda _: ConfidenceScore(1.0, metrics),
            config=config(),
        )
