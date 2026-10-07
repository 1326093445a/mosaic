"""Behavioral checks for the continuous seeding stage's discrete hand-off.

Two things in `examples/p17_continuous_seed.py` decide what the relaxed
optimizer actually hands the discrete search, and neither was covered:
`project_to_budget`, which truncates the relaxed optimum to the edit budget,
and the initialization, which decides whether the optimizer starts at the
parent or de novo. Both are pure array code, so they need no model.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

# The real designable geometry: CDR_RESIDUE_INDICES_1IDX, 29 positions.
N_DESIGNABLE = 29
BUDGET = 5


@pytest.fixture(scope="module")
def seeder():
    spec = importlib.util.spec_from_file_location(
        "p17_continuous_seed_test", REPO / "examples" / "p17_continuous_seed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relaxed_at_parent(parent, n_tokens=20):
    """A relaxed optimum that agrees with the parent everywhere."""
    relaxed = np.zeros((len(parent), n_tokens), dtype=np.float32)
    relaxed[np.arange(len(parent)), parent] = 1.0
    return relaxed


# --- project_to_budget -----------------------------------------------------


def test_projection_never_exceeds_the_budget_however_many_edits_are_wanted(seeder):
    parent = np.zeros(12, dtype=np.int32)
    designable = np.arange(10)
    # Every designable position prefers a non-parent residue by a wide margin.
    relaxed = relaxed_at_parent(parent)
    for pos in designable:
        relaxed[pos] = 0.0
        relaxed[pos, 1] = 1.0
    tokens, margins, kept = seeder.project_to_budget(relaxed, parent, designable, 3)
    assert len(margins) == 10, "all ten substitutions should be candidates"
    assert len(kept) == 3
    assert int((tokens != parent).sum()) == 3


def test_projection_keeps_the_highest_margin_substitutions(seeder):
    parent = np.zeros(6, dtype=np.int32)
    designable = np.arange(6)
    relaxed = np.zeros((6, 20), dtype=np.float32)
    # Position p prefers residue 1 with margin (p + 1) / 10.
    for pos in range(6):
        relaxed[pos, 0] = 0.0
        relaxed[pos, 1] = (pos + 1) / 10.0
    tokens, _, kept = seeder.project_to_budget(relaxed, parent, designable, 2)
    kept_positions = sorted(pos for _, pos, _ in kept)
    assert kept_positions == [4, 5], "the two widest margins are positions 4 and 5"
    assert tokens[4] == 1 and tokens[5] == 1
    assert int((tokens != parent).sum()) == 2


def test_projection_ranks_by_margin_not_by_absolute_probability(seeder):
    """A saturated position the optimizer barely prefers must lose to a
    middling position it prefers strongly. The margins are probability
    differences, so absolute mass is not the ranking."""
    parent = np.zeros(2, dtype=np.int32)
    designable = np.arange(2)
    relaxed = np.zeros((2, 20), dtype=np.float32)
    relaxed[0, 0], relaxed[0, 1] = 0.45, 0.50   # high mass, margin 0.05
    relaxed[1, 0], relaxed[1, 1] = 0.05, 0.40   # lower mass, margin 0.35
    _, _, kept = seeder.project_to_budget(relaxed, parent, designable, 1)
    assert [pos for _, pos, _ in kept] == [1]


def test_projection_skips_positions_the_optimizer_left_at_the_parent(seeder):
    parent = np.array([3, 3, 3, 3], dtype=np.int32)
    designable = np.arange(4)
    relaxed = relaxed_at_parent(parent)
    relaxed[2] = 0.0
    relaxed[2, 7] = 1.0
    tokens, margins, kept = seeder.project_to_budget(
        relaxed, parent, designable, BUDGET
    )
    assert len(margins) == 1, "only one position wants to move"
    assert len(kept) == 1
    assert tokens[2] == 7
    assert np.array_equal(np.delete(tokens, 2), np.delete(parent, 2))


def test_projection_leaves_non_designable_positions_at_the_parent(seeder):
    parent = np.zeros(8, dtype=np.int32)
    designable = np.array([1, 3])
    relaxed = np.zeros((8, 20), dtype=np.float32)
    relaxed[:, 5] = 1.0  # every position, designable or not, prefers residue 5
    tokens, _, _ = seeder.project_to_budget(relaxed, parent, designable, BUDGET)
    assert tokens[1] == 5 and tokens[3] == 5
    untouched = [i for i in range(8) if i not in (1, 3)]
    assert np.array_equal(tokens[untouched], parent[untouched])


def test_zero_budget_returns_the_parent_unchanged(seeder):
    parent = np.zeros(5, dtype=np.int32)
    designable = np.arange(5)
    relaxed = np.zeros((5, 20), dtype=np.float32)
    relaxed[:, 9] = 1.0
    tokens, margins, kept = seeder.project_to_budget(relaxed, parent, designable, 0)
    assert kept == []
    assert len(margins) == 5, "candidates are still reported for the audit trail"
    assert np.array_equal(tokens, parent)


def test_projection_does_not_mutate_the_parent_array(seeder):
    parent = np.zeros(4, dtype=np.int32)
    before = parent.copy()
    relaxed = np.zeros((4, 20), dtype=np.float32)
    relaxed[:, 2] = 1.0
    seeder.project_to_budget(relaxed, parent, np.arange(4), BUDGET)
    assert np.array_equal(parent, before)


# --- anchored_logits -------------------------------------------------------


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def edit_budget_expectation(probs, parent_rows):
    """`EditBudget`'s own quantity: expected substitutions from the parent."""
    p_parent = probs[np.arange(len(parent_rows)), parent_rows]
    return float((1.0 - p_parent).sum())


def test_anchored_start_lands_inside_the_edit_budget(seeder):
    """The regression this initialization fixes.

    `EditBudget` is a soft hinge, `5.0 * relu(E(s) - budget)`. A noise start
    puts about 1/20 of the mass on the parent at each of the 29 designable
    positions, so `E` opens near 27.6 and the hinge opens near 113 -- larger
    than every other term in the composite objective combined. Anchoring at
    the parent opens inside the budget, so the hinge is inactive and the first
    steps optimize the readouts instead of walking back to the parent.
    """
    rng = np.random.default_rng(0)
    parent = rng.integers(0, 20, size=120).astype(np.int32)
    designable = np.arange(N_DESIGNABLE)
    noise = 0.01 * rng.standard_normal((N_DESIGNABLE, 20))
    parent_rows = parent[designable]

    def hinge(probs):
        return 5.0 * max(0.0, edit_budget_expectation(probs, parent_rows) - BUDGET)

    anchored = softmax(seeder.anchored_logits(parent, designable, 5.0, noise))
    assert edit_budget_expectation(anchored, parent_rows) < BUDGET
    assert hinge(anchored) == 0.0

    noise_only = softmax(noise)
    assert 25 < edit_budget_expectation(noise_only, parent_rows) < 30
    assert 100 < hinge(noise_only) < 125


def test_anchored_start_concentrates_on_the_parent_at_the_expected_mass(seeder):
    parent = np.arange(N_DESIGNABLE, dtype=np.int32) % 20
    designable = np.arange(N_DESIGNABLE)
    noise = np.zeros((N_DESIGNABLE, 20))
    for scale, expected in ((5.0, 0.886), (6.0, 0.955)):
        probs = softmax(seeder.anchored_logits(parent, designable, scale, noise))
        p_parent = probs[np.arange(N_DESIGNABLE), parent[designable]]
        assert np.allclose(p_parent, expected, atol=5e-3)
        assert p_parent.min() < 1.0, "the start must stay differentiable, not a vertex"


def test_anchored_start_keeps_the_parent_as_argmax_everywhere(seeder):
    rng = np.random.default_rng(1)
    parent = rng.integers(0, 20, size=60).astype(np.int32)
    designable = np.arange(N_DESIGNABLE)
    noise = 0.01 * rng.standard_normal((N_DESIGNABLE, 20))
    logits = seeder.anchored_logits(parent, designable, 5.0, noise)
    assert np.array_equal(logits.argmax(-1), parent[designable])


def test_anchored_logits_rejects_mismatched_noise(seeder):
    parent = np.zeros(40, dtype=np.int32)
    with pytest.raises(ValueError, match="expected"):
        seeder.anchored_logits(parent, np.arange(10), 5.0, np.zeros((9, 20)))


# --- single budget accounting ----------------------------------------------
#
# The seeding stage and the search each enforce `edit_budget`, so unless they
# share an anchor a 5-edit seed followed by a 5-edit search drifts 10 from the
# parent. `--budget-anchor reference` in `examples/p17_confidence_search.py`
# shares it by passing the parent as `wt` and the seed as `initial_sequences`.
# These checks exercise that wiring against `mosaic.search` directly, since the
# harness itself needs a model.


def run_anchored(wt, initial, budget, steps=400):
    from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search

    config = SearchConfig(
        policy="population",
        edit_budget=budget,
        max_score_calls=steps,
        max_gradient_calls=steps,
        max_proposals=steps * 4,
        seed=0,
    )
    width = config.width
    alphabet = config.alphabet_size
    seen = []

    def gradient(sequence):
        seen.append(np.array(sequence))
        # Reward moving away from the parent, so the budget is the only brake.
        g = -np.tile(np.arange(alphabet, dtype=float), (len(wt), 1))
        return -float(np.count_nonzero(sequence != wt)), g

    def confidence(sequence):
        seen.append(np.array(sequence))
        return ConfidenceScore(float(np.count_nonzero(sequence != wt)))

    result = run_gradient_search(
        wt=wt,
        designable_mask=np.ones(len(wt), dtype=bool),
        config=config,
        initial_sequences=np.repeat(np.asarray(initial)[None], width, axis=0),
        gradient_fn=gradient,
        confidence_fn=confidence,
    )
    return result, seen


def test_anchoring_to_the_parent_shares_one_budget_with_the_seeding_stage():
    """A seed that already spent 3 of 5 edits leaves the search 2, not 5."""
    parent = np.zeros(10, dtype=int)
    seed = parent.copy()
    seed[:3] = 1  # three edits already spent by the producer
    result, seen = run_anchored(parent, seed, budget=5)

    drift = [int(np.count_nonzero(s != parent)) for s in seen]
    assert max(drift) <= 5, "no candidate may exceed the shared budget"
    assert int(np.count_nonzero(np.array(result.best.sequence) != parent)) <= 5


def test_anchoring_to_the_seed_instead_permits_double_the_drift():
    """The behavior `--budget-anchor start` keeps, shown as the contrast: with
    the seed as `wt` the search spends a second full budget on top of the
    producer's, reaching 8 edits from the parent."""
    parent = np.zeros(10, dtype=int)
    seed = parent.copy()
    seed[:3] = 1
    result, seen = run_anchored(seed, seed, budget=5)

    drift = [int(np.count_nonzero(s != parent)) for s in seen]
    assert max(drift) > 5, "this is the double-count the shared anchor prevents"
    assert max(drift) <= 8, "three producer edits plus five search edits"


def test_a_seed_over_the_shared_budget_is_rejected_rather_than_silently_capped():
    parent = np.zeros(10, dtype=int)
    seed = parent.copy()
    seed[:6] = 1  # six edits against a budget of five
    with pytest.raises(ValueError, match="edit budget"):
        run_anchored(parent, seed, budget=5)


# --- the optimizers themselves ---------------------------------------------
#
# The 2026-10-06 seeding cells completed all 125 optimization steps and then
# died unpacking the optimizer's return value, so the stage has never produced
# a sequence. These run all three optimizers end to end on a CPU-sized loss
# with the anchored start, which covers everything in the seeding stage except
# loading OpenDDE: the init, each optimizer's own return convention, the
# `relaxed_array` guard, and the projection onto the budget.


@pytest.fixture(scope="module")
def toy_loss():
    """A real `EditBudget` hinge plus a quadratic pull toward a target."""
    import jax.numpy as jnp
    from mosaic.common import LossTerm
    from mosaic.losses.transformations import EditBudget

    rng = np.random.default_rng(7)
    parent = rng.integers(0, 20, size=N_DESIGNABLE).astype(np.int32)
    s_ref = np.zeros((N_DESIGNABLE, 20), dtype=np.float32)
    s_ref[np.arange(N_DESIGNABLE), parent] = 1.0
    target = np.asarray(rng.random((N_DESIGNABLE, 20)), dtype=np.float32)

    class Quadratic(LossTerm):
        def __call__(self, seq, *, key=None):
            value = ((seq - jnp.asarray(target)) ** 2).sum()
            return value, {"quadratic": value}

    loss = Quadratic() + 5.0 * EditBudget(
        s_ref=s_ref,
        designable=np.ones(N_DESIGNABLE, dtype=bool),
        budget=float(BUDGET),
    )
    return loss, parent, s_ref


@pytest.mark.parametrize("method", ["apgm", "bindcraft", "colabdesign"])
def test_every_optimizer_returns_a_usable_relaxed_sequence(seeder, toy_loss, method):
    import jax

    loss, parent, _ = toy_loss
    designable = np.arange(N_DESIGNABLE)
    key = jax.random.key(0)
    init_key, run_key = jax.random.split(key)
    noise = 0.01 * np.asarray(jax.random.normal(init_key, (N_DESIGNABLE, 20)))
    x0 = jax.numpy.asarray(
        seeder.anchored_logits(parent, designable, 5.0, noise)
    )

    from mosaic.optimizers import bindcraft_design, colabdesign_stage, simplex_APGM

    if method == "bindcraft":
        raw = bindcraft_design(loss_function=loss, x=x0, lr=0.1, key=run_key)
    elif method == "colabdesign":
        raw = colabdesign_stage(
            loss_function=loss, x=x0, n_steps=10, soft_start=0.0, soft_end=1.0,
            temp_start=1.0, temp_end=0.01, hard=False, lr=0.1, key=run_key,
        )
    else:
        raw = simplex_APGM(
            loss_function=loss, x=jax.nn.softmax(x0, -1), n_steps=10,
            stepsize=0.1, momentum=0.9, key=run_key,
        )

    relaxed = np.asarray(seeder.relaxed_array(raw))
    assert relaxed.shape == (N_DESIGNABLE, 20), (
        f"{method} returned {relaxed.shape}; the return convention changed"
    )
    assert np.isfinite(relaxed).all(), f"{method} produced non-finite values"

    # The hand-off: project onto the budget and check it is a usable sequence.
    tokens, margins, kept = seeder.project_to_budget(
        relaxed, parent, designable, BUDGET
    )
    assert len(kept) <= BUDGET
    assert int((tokens != parent).sum()) <= BUDGET
    assert tokens.dtype == np.int32
    assert (tokens >= 0).all() and (tokens < 20).all()


def test_relaxed_array_handles_both_return_conventions(seeder):
    """The exact crash of 2026-10-06: a bare array unpacked as a 2-tuple."""
    bare = np.zeros((4, 20))
    pair = (np.ones((4, 20)), np.zeros((4, 20)))
    assert seeder.relaxed_array(bare) is bare
    assert seeder.relaxed_array(pair) is pair[0]
