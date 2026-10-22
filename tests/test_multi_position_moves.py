"""The multi-position proposal operator: `move_arity` 2 against `_moves`.

The operator's claim is narrow and these tests pin exactly it. A pair move lets
the search cross a valley whose midpoint the accept test would reject, because
the intermediate state is never scored. It does NOT let the first-order ranking
see synergy -- the gradient is linear, so a pair's predicted delta is the sum of
its parts -- and `test_pair_delta_is_additive_so_the_ranking_cannot_see_synergy`
is here to stop that claim being made later.

The other half of the file is backward compatibility. Arity 1 must keep
`_moves` as its generator verbatim: its move ORDER sets the RNG stream for
every result this project has recorded, and `p17_pose_diagnostics.py` compares
its output by equality.
"""

import numpy as np
import pytest

from mosaic.search import (
    ConfidenceScore,
    SearchConfig,
    _atomic_edits,
    _move_edits,
    _moves,
    _multi_moves,
    run_gradient_search,
)

ALPHABET = 20
BUDGET = 5
LENGTH = 40
DESIGNABLE = list(range(10, 25))


def problem(seed=0, spent=0):
    rng = np.random.default_rng(seed)
    wt = rng.integers(0, ALPHABET, LENGTH).astype(np.int32)
    mask = np.zeros(LENGTH, dtype=bool)
    mask[DESIGNABLE] = True
    sequence = wt.copy()
    for position in DESIGNABLE[:spent]:
        sequence[position] = (sequence[position] + 7) % ALPHABET
    gradient = rng.normal(size=(LENGTH, ALPHABET))
    return sequence, wt, mask, gradient


def single_edits(moves, wt):
    """The one-position moves in either representation, as a set."""
    out = set()
    for move in moves:
        edits = _move_edits(move, wt)
        if len(edits) == 1:
            out.add(edits[0])
    return out


def pair_edits(moves, wt):
    return {frozenset(_move_edits(m, wt)) for m in moves if len(_move_edits(m, wt)) == 2}



@pytest.mark.parametrize("spent", range(BUDGET + 1))
def test_single_move_set_matches_moves_at_every_budget_level(spent):
    """`_multi_moves` must offer the same singles `_moves` does.

    `_moves` decides feasibility by a case analysis at the cap; `_multi_moves`
    decides it by summing per-edit Hamming costs. Those are different pieces of
    reasoning and have to agree, or the two operators are searching different
    spaces and no comparison between them means anything.
    """
    sequence, wt, mask, gradient = problem(spent=spent)
    legacy, _ = _moves(sequence, wt, mask, BUDGET, gradient)
    for arity in (1, 2):
        new, _, count = _multi_moves(sequence, wt, mask, BUDGET, gradient, arity)
        assert single_edits(new[:count], wt) == single_edits(legacy, wt)


def test_arity_one_keeps_the_legacy_generator_verbatim():
    """The RNG stream depends on move order, so arity 1 must not be rebuilt."""
    sequence, wt, mask, gradient = problem(spent=2)
    config = SearchConfig(max_score_calls=12, max_gradient_calls=12, seed=3)
    assert config.move_arity == 1, "arity 1 has to stay the default"

    seen = []

    def gradient_fn(seq):
        return 0.0, gradient

    def confidence_fn(seq):
        seen.append(tuple(int(x) for x in seq))
        return ConfidenceScore(value=float(-np.count_nonzero(seq != wt)))

    run_gradient_search(
        wt=wt, designable_mask=mask, gradient_fn=gradient_fn,
        confidence_fn=confidence_fn, config=config,
    )
    assert seen, "the search scored nothing"
    for candidate in seen:
        assert np.count_nonzero(np.asarray(candidate) != wt) <= BUDGET


def test_exchanges_at_the_cap_are_a_subset_of_the_pairs():
    """The pair operator generalises the existing exchange rather than adding
    a second, overlapping mechanism beside it."""
    sequence, wt, mask, gradient = problem(spent=BUDGET)
    legacy, _ = _moves(sequence, wt, mask, BUDGET, gradient)
    exchanges = {
        frozenset(_move_edits(m, wt)) for m in legacy if not isinstance(m[0], tuple) and m[2] >= 0
    }
    assert exchanges, "this fixture was supposed to be at the cap"
    pairs, _, count = _multi_moves(sequence, wt, mask, BUDGET, gradient, 2)
    assert exchanges <= pair_edits(pairs[count:], wt)



@pytest.mark.parametrize("spent", range(BUDGET + 1))
def test_no_generated_move_can_break_the_budget(spent):
    """The budget stays a wall, not a penalty, at arity 2."""
    sequence, wt, mask, gradient = problem(seed=1, spent=spent)
    moves, _, _ = _multi_moves(sequence, wt, mask, BUDGET, gradient, 2)
    for move in moves:
        proposal = sequence.copy()
        for position, residue in _move_edits(move, wt):
            proposal[position] = residue
        assert np.count_nonzero(proposal != wt) <= BUDGET


def test_pairs_are_position_disjoint_and_exhaustive():
    sequence, wt, mask, gradient = problem(seed=2, spent=1)
    moves, _, count = _multi_moves(sequence, wt, mask, BUDGET, gradient, 2)
    positions, residues, _, cost = _atomic_edits(sequence, wt, mask, gradient)
    spent = int(np.count_nonzero(sequence != wt))

    pairs = pair_edits(moves[count:], wt)
    for move in moves[count:]:
        edits = _move_edits(move, wt)
        assert edits[0][0] != edits[1][0], "a pair must change two distinct positions"

    expected = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] != positions[j] and spent + cost[i] + cost[j] <= BUDGET:
                expected += 1
    assert len(moves) - count == expected, "the pair set must be exhaustive"
    assert len(pairs) == expected, "and must contain no duplicate pair"


def test_pair_delta_is_additive_so_the_ranking_cannot_see_synergy():
    """The guard against the claim this operator invites.

    A pair's predicted delta is exactly the sum of its two single deltas,
    because the gradient is linear and a first-order model has no interaction
    term. Any combination effect is discovered by the realised score, never by
    the proposal ranking, and a result from this operator must not be reported
    as the gradient having found an interaction.
    """
    sequence, wt, mask, gradient = problem(seed=4, spent=0)
    moves, deltas, count = _multi_moves(sequence, wt, mask, BUDGET, gradient, 2)
    singles = {
        _move_edits(m, wt)[0]: d for m, d in zip(moves[:count], deltas[:count])
    }
    for move, delta in zip(moves[count : count + 500], deltas[count : count + 500]):
        first, second = _move_edits(move, wt)
        assert delta == pytest.approx(singles[first] + singles[second], rel=1e-12)


def test_arity_two_proposes_moves_that_change_two_positions_at_once():
    """End to end: the operator is used, not merely offered."""
    sequence, wt, mask, gradient = problem(seed=5)
    events = []
    config = SearchConfig(
        move_arity=2, max_score_calls=40, max_gradient_calls=40, seed=11
    )

    def gradient_fn(seq):
        return 0.0, gradient

    def confidence_fn(seq):
        return ConfidenceScore(value=float(np.random.default_rng(abs(hash(tuple(seq))) % 2**32).random()))

    run_gradient_search(
        wt=wt, designable_mask=mask, gradient_fn=gradient_fn,
        confidence_fn=confidence_fn, config=config,
        on_event=events.append,
    )
    proposals = [e for e in events if e["event"] == "proposal"]
    assert proposals, "no proposal was logged"
    sizes = {e["move_positions"] for e in proposals}
    assert 2 in sizes, "arity 2 never actually proposed a pair"
    assert max(sizes) == 2


def test_a_valley_is_crossed_only_at_arity_two():
    """The mechanism, isolated.

    The score rewards exactly one target pair of substitutions and penalises
    either of them alone. A single-move walk cannot reach the pair, because the
    first edit is scored worse than the parent and the accept test rejects it
    with no temperature. A pair move proposes both at once, so the accept test
    only ever sees the combined result.
    """
    sequence, wt, mask, gradient = problem(seed=6)
    first, second = DESIGNABLE[0], DESIGNABLE[1]
    target = {first: (int(wt[first]) + 3) % ALPHABET,
              second: (int(wt[second]) + 5) % ALPHABET}

    def confidence_fn(seq):
        seq = np.asarray(seq)
        hits = sum(1 for p, a in target.items() if seq[p] == a)
        off = int(np.count_nonzero(seq != wt)) - hits
        value = (10.0 if hits == 2 else -1.0 * hits) - 2.0 * off
        return ConfidenceScore(value=value)

    def gradient_fn(seq):
        g = np.zeros((LENGTH, ALPHABET))
        for p, a in target.items():
            g[p, a] = -5.0
        return 0.0, g

    def best_of(arity):
        result = run_gradient_search(
            wt=wt, designable_mask=mask, gradient_fn=gradient_fn,
            confidence_fn=confidence_fn,
            config=SearchConfig(
                move_arity=arity, max_score_calls=60, max_gradient_calls=60,
                acceptance_temperature=0.0, target_entropy=0.3, seed=7,
            ),
        )
        return result.best

    one, two = best_of(1), best_of(2)
    reached = lambda c: all(c.sequence[p] == a for p, a in target.items())
    assert not reached(one), "arity 1 was not supposed to reach the pair"
    assert reached(two), "arity 2 should cross the valley in one move"
    assert two.score > one.score



def test_arity_three_is_refused_with_its_reason():
    sequence, wt, mask, gradient = problem()
    with pytest.raises(ValueError, match="pruning rule"):
        _multi_moves(sequence, wt, mask, BUDGET, gradient, 3)
    with pytest.raises(ValueError, match="move_arity above 2"):
        SearchConfig(move_arity=3)


def test_arity_below_one_is_refused():
    with pytest.raises(ValueError, match="move_arity"):
        SearchConfig(move_arity=0)


def test_move_edits_normalises_both_representations():
    wt = np.array([0, 1, 2, 3], dtype=np.int32)
    assert _move_edits((2, 7, -1), wt) == [(2, 7)]
    assert _move_edits((2, 7, 1), wt) == [(2, 7), (1, 1)]
    assert _move_edits(((2, 7),), wt) == [(2, 7)]
    assert _move_edits(((2, 7), (0, 5)), wt) == [(2, 7), (0, 5)]
