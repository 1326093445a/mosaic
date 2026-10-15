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
    """What a high init scale bought, kept for the record.

    `EditBudget` is a soft hinge, `5.0 * relu(E(s) - budget)`. A noise start
    puts about 1/20 of the mass on the parent at each of the 29 designable
    positions, so `E` opens near 27.6 and the hinge opens near 113 -- larger
    than every other term in the composite objective combined. Anchoring at
    scale 5.0 opens inside the budget, so the hinge is inactive at step 0.

    This is still true and is why 5.0 was chosen. What it bought was a start
    where the HINGE is inactive; what it cost was never measured until
    2026-10-13, when the hinge turned out to be the thing that decides whether
    the relaxation leaves the parent at all. See
    `test_the_hinge_alone_decides_whether_the_relaxation_leaves_the_parent`.
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


# --- the 2026-10-13 defaults -----------------------------------------------
#
# Seven replicates of three methods returned 0, 0, 0, 0, 1, 1 and 2 edits on a
# budget of 5, and at 0 the argmax was still the parent at all 29 designable
# positions. These pin each of the four causes separately, because none of the
# four fixes is sufficient alone and a regression in any one of them would look
# like the same silent null.


def seeder_defaults(seeder):
    """The seeder's EFFECTIVE defaults, from an actual parse.

    Not `{a.dest: a.default for a in parser._actions}`, which is what this used
    to do and which is wrong for every `--flag` / `--no-flag` pair: the two
    actions share one `dest`, and the `store_false` half carries an implicit
    `default=True`, so introspection reports whichever came last. That silently
    reported `logit_norm=True` after the default had been changed to False, and
    the assertion built on it passed while pinning nothing.
    """
    import argparse, contextlib, io
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, argv=None):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(SystemExit):
                seeder.main(["--method", "apgm", "--out", "/dev/null"])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert "parser" in captured, (
        "main() did not reach parse_args; it may have been restructured"
    )
    return vars(real(captured["parser"], ["--method", "apgm", "--out", "/dev/null"]))


def test_the_default_init_scale_matches_colabdesigns_own(seeder):
    """Pinned so a change is deliberate, NOT because 1.0 is shown to be better.

    ColabDesign's wildtype start is `np.eye(20)[aatype]` -- value 1, no scale --
    and the former default was 5.0. The measured effect of the difference is
    small: on the CPU toy loss it changes only how many steps a logit-space
    method needs once the hinge is off (see
    `test_the_init_scale_sets_the_speed_not_the_direction`), and a cluster probe
    on 2026-10-13 (old code, hinge on, n=1 per cell) moved colabdesign from 0
    edits to 1-5 and left bindcraft at 0-2. An earlier version of this docstring
    attributed the null to a 148:1 gradient-weight ratio from the softmax chain
    rule; that was cited from `biohub_optimizer`, which the seeder does not use,
    and the toy data do not support it.
    """
    assert seeder_defaults(seeder)["init_logit_scale"] == 1.0


def test_the_hinge_is_off_during_the_relaxation_by_default(seeder):
    """`project_to_budget` enforces the budget exactly; the hinge does not need to.

    The one default with direct evidence behind it. See
    `test_the_hinge_alone_decides_whether_the_relaxation_leaves_the_parent`.
    """
    assert seeder_defaults(seeder)["weight_edit_budget"] == 0.0


def test_the_step_default_is_past_the_regime_the_paper_calls_unconverged(seeder):
    assert seeder_defaults(seeder)["steps"] >= 200


def test_the_straight_through_stage_is_not_run_at_a_saturated_temperature(seeder):
    """`hard_temp` is the one shaping change kept on by default, and why.

    At temp 1e-2 the estimator's gradient is a saturated softmax's Jacobian, and
    ColabDesign's `lr_scale = (1 - soft) + soft*temp` shrinks the step by
    another factor of temp, so the one stage that shows the frozen predictor a
    valid one-hot could not move at all. That is a fact about the code rather
    than a measured effect; the toy loss cannot test the reason it matters here,
    which is that OpenDDE was trained on one-hot sequences.
    """
    assert seeder_defaults(seeder)["hard_temp"] == 1.0


def test_the_unvalidated_shaping_options_are_off_by_default(seeder):
    """Both shipped ON on 2026-10-13, and both were wrong to.

    `logit_norm`: on a uniform start the auto-calibrated gamma is 0 and every
    update is multiplied back to exactly zero with a live gradient, which
    `project_to_budget` cannot show because it only takes an argmax. Beyond
    that it is simply unvalidated on the real loss, and it couples the init
    scale to a standing constraint on every later iterate. An earlier version
    of this docstring also claimed gamma caps per-residue confidence at about
    p=0.125; that was wrong, and
    `test_gamma_bounds_the_aggregate_spread_not_each_row` records why.

    `stochastic_hard`: the argmax is what ColabDesign does, and sampling did not
    help on the toy.

    Only `--weight-edit-budget` has direct evidence, so only it is on.
    """
    d = seeder_defaults(seeder)
    assert d["logit_norm"] is False
    assert d["stochastic_hard"] is False
    assert d["weight_edit_budget"] == 0.0


def test_the_discrete_search_keeps_the_hinge_at_full_weight():
    """The seeder turning the hinge off must not reach the discrete search.

    There the hinge is inert below the cap anyway (at a one-hot `E(s)` equals
    Hamming exactly), so its weight costs nothing and changing it would move a
    result this project has thirty sections of history on.
    """
    import sys
    sys.path.insert(0, str(REPO / "examples"))
    import inspect
    from p17_hallucination_search import WEIGHT_EDIT_BUDGET, build_composite_losses

    default = inspect.signature(build_composite_losses).parameters[
        "edit_budget_weight"
    ].default
    assert default == WEIGHT_EDIT_BUDGET == 5.0


def test_steps_reaches_bindcrafts_four_stages():
    """`--steps` was silently dropped for bindcraft; every run was 125 steps.

    The four stage lengths are hardcoded as 50:25:45:5, and the seeder called
    `bindcraft_design` without overriding any of them, so the flag it passed
    for the other two methods did nothing here.
    """
    import inspect
    from mosaic.optimizers import bindcraft_design

    assert "n_steps" in inspect.signature(bindcraft_design).parameters


@pytest.mark.parametrize("total,expected_sum", [(125, 125), (600, 600), (37, 37)])
def test_bindcraft_rescaling_preserves_the_total_exactly(total, expected_sum):
    """Rounding four proportions must not lose or invent steps."""
    counts = (50, 25, 45, 5)
    scaled = [max(1, int(round(c * total / sum(counts)))) for c in counts]
    drift = total - sum(scaled)
    if drift:
        longest = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[longest] = max(1, scaled[longest] + drift)
    assert sum(scaled) == expected_sum
    assert all(n >= 1 for n in scaled), "no stage may be scaled out of existence"


def test_bindcraft_refuses_a_total_it_cannot_split():
    from mosaic.optimizers import bindcraft_design

    with pytest.raises(ValueError, match="four stages"):
        bindcraft_design(loss_function=None, x=np.zeros((4, 20)), n_steps=3)


def test_logit_normalisation_is_the_identity_at_the_calibrated_scale():
    """Gamma is calibrated to the opening spread so step 0 does not move.

    A fixed gamma of 1.0 instead rescales the gap to the block's own standard
    deviation, which for a parent-anchored start AMPLIFIES it -- a gap of 1.0
    over 29x20 becomes about 4.6 -- re-saturating what this prevents.
    """
    import jax.numpy as jnp
    from mosaic.optimizers import _layer_norm_logits

    rng = np.random.default_rng(3)
    parent = rng.integers(0, 20, size=N_DESIGNABLE)
    x = np.zeros((N_DESIGNABLE, 20), dtype=np.float32)
    x[np.arange(N_DESIGNABLE), parent] = 1.0
    x = x - x.mean(-1, keepdims=True)

    gamma = float(np.sqrt(((x - x.mean()) ** 2).mean()))
    identity = np.asarray(_layer_norm_logits(jnp.asarray(x), gamma))
    assert np.allclose(identity, x, atol=1e-5), "calibrated gamma must not move x"

    amplified = np.asarray(_layer_norm_logits(jnp.asarray(x), 1.0))
    gap_before = float(x.max(-1).mean() - x.min(-1).mean())
    gap_after = float(amplified.max(-1).mean() - amplified.min(-1).mean())
    assert gap_after > 3 * gap_before, (
        f"gamma=1.0 was supposed to amplify the gap; {gap_before:.2f} -> {gap_after:.2f}"
    )


def test_logit_normalisation_bounds_the_spread_however_far_it_drifts():
    import jax.numpy as jnp
    from mosaic.optimizers import _layer_norm_logits

    for drift in (1.0, 10.0, 1000.0):
        x = drift * np.asarray(
            np.random.default_rng(0).standard_normal((N_DESIGNABLE, 20)),
            dtype=np.float32,
        )
        out = np.asarray(_layer_norm_logits(jnp.asarray(x), 0.5))
        assert abs(float(out.mean())) < 1e-3
        assert abs(float(out.std()) - 0.5) < 1e-3, "gamma must set the spread"


def test_stochastic_hard_draws_a_one_hot_that_is_not_always_the_argmax():
    """Fast SeqProp's sampler, not ColabDesign's argmax.

    The straight-through gradient is the softmax Jacobian either way, so this
    changes only which vertex the predictor is shown.
    """
    import jax
    import jax.numpy as jnp
    from mosaic.optimizers import _colabdesign_transform

    z = jnp.asarray(
        np.random.default_rng(5).standard_normal((N_DESIGNABLE, 20)), dtype=jnp.float32
    )
    one = jnp.asarray(1.0)
    deterministic = _colabdesign_transform(z, one, one, one, None)
    argmax_pick = tuple(np.asarray(deterministic.argmax(-1)))
    picks = {
        tuple(
            np.asarray(
                _colabdesign_transform(z, one, one, one, jax.random.key(k)).argmax(-1)
            )
        )
        for k in range(12)
    }
    assert len(picks) > 1, "sampling produced the same draw every time"
    assert picks != {argmax_pick}, "sampling only ever reproduced the argmax"
    # Whatever produced it, the predictor must still be shown a one-hot: that
    # is the entire reason for the hard branch.
    for name, seq in (("argmax", deterministic),
                      ("sampled", _colabdesign_transform(
                          z, one, one, one, jax.random.key(0)))):
        for row in np.asarray(seq):
            assert np.isclose(row.sum(), 1.0) and np.isclose(row.max(), 1.0), (
                f"{name} branch did not produce a one-hot"
            )


# --- behaviour, through the code `main()` actually runs --------------------
#
# Measured 2026-10-13 on this toy loss (a quadratic pull to a random target plus
# a real `EditBudget` hinge), 29 designable positions, budget 5, through
# `run_relaxation` at its PRODUCTION optimizer settings. "wanted" is the number
# of positions whose preferred residue differs from the parent --
# `candidate_substitutions_considered` in the seed JSON -- and is the useful
# readout, because `edits` is capped at the budget and hides everything above it.
#
#   24 steps        scale 1.0            scale 5.0
#     hinge 5.0     0 wanted (all)       0 wanted (all)
#     hinge 0.0     21 / 20 / 28         0 wanted (ALL THREE)
#
#   125 steps       scale 1.0            scale 5.0
#     hinge 5.0     0 / 5 / 4            0 / 4 / 9
#     hinge 0.0     21 / 26 / 28         0 / 27 / 28
#                   (apgm / bindcraft / colabdesign)
#
# Two things follow, and an earlier version of these tests got both wrong by
# passing apgm settings of its own (0.1/0.9/1.0) while main() passed
# 0.05/0.5/1.2:
#
#   - The hinge is NECESSARY but NOT SUFFICIENT. Switching it off moves all
#     three methods only when the init scale is also 1.0.
#   - At scale 5.0 and 24 steps NOTHING moves in either arm, and simplex_APGM
#     never moves at scale 5.0 at all. So a zero in both arms does not implicate
#     the loss; it is also what these settings produce on a loss with no model
#     in it.

@pytest.fixture(scope="module")
def toy_problem():
    """A loss factory so the hinge weight can vary with everything else fixed."""
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

    def make_loss(hinge_weight):
        if hinge_weight == 0.0:
            return Quadratic()
        return Quadratic() + hinge_weight * EditBudget(
            s_ref=s_ref, designable=np.ones(N_DESIGNABLE, dtype=bool),
            budget=float(BUDGET),
        )

    return make_loss, parent


def relax(seeder, toy_problem, method, *, steps, scale, hinge):
    """Edits and wanted-substitutions from `run_relaxation` at the shipped options."""
    import contextlib
    import io

    import jax
    import jax.numpy as jnp

    make_loss, parent = toy_problem
    init_key, run_key = jax.random.split(jax.random.key(0))
    noise = 0.01 * np.asarray(jax.random.normal(init_key, (N_DESIGNABLE, 20)))
    x0 = jnp.asarray(
        seeder.anchored_logits(parent, np.arange(N_DESIGNABLE), scale, noise)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        relaxed = np.asarray(seeder.run_relaxation(
            method, make_loss(hinge), x0, run_key, steps=steps, lr=0.1,
            logit_norm=False, stochastic_hard=False, hard_temp=1.0,
            hard_fraction=0.25,
        ))
    tokens, margins, _ = seeder.project_to_budget(
        relaxed, parent, np.arange(N_DESIGNABLE), BUDGET
    )
    return int((tokens != parent).sum()), len(margins)


# apgm moves inside 24 steps; the logit-space methods need the longer budget the
# real run gives them, so the contrast is measured where both arms can move.
_STEPS = {"apgm": 24, "bindcraft": 125, "colabdesign": 125}


@pytest.mark.parametrize("method", ["apgm", "bindcraft", "colabdesign"])
def test_the_hinge_decides_the_outcome_at_the_shipped_init_scale(
    seeder, toy_problem, method
):
    """The factor with direct evidence, at the scale the seeder actually ships.

    `EditBudget` is `w * relu(E(s) - budget)` with `E(s) = sum_pos (1 - p_parent)`,
    the EXPECTED Hamming distance. At a one-hot that equals Hamming exactly, so
    for the discrete search it is inert below the cap. On a relaxed sequence its
    only route to satisfaction is raising p_parent, so the relaxation is held at
    the parent. `project_to_budget` enforces the budget exactly downstream,
    which is why the hinge can be switched off here without losing the cap.

    Measured at `scale=1.0`, which is `--init-logit-scale`'s default. It does
    NOT hold at every scale -- see
    `test_the_hinge_is_necessary_but_not_sufficient`, which pins the case where
    switching the hinge off changes nothing.
    """
    steps = _STEPS[method]
    _, wanted_on = relax(seeder, toy_problem, method, steps=steps, scale=1.0, hinge=5.0)
    edits_off, wanted_off = relax(
        seeder, toy_problem, method, steps=steps, scale=1.0, hinge=0.0
    )
    assert wanted_on <= BUDGET, (
        f"{method}: hinge ON still wanted {wanted_on} substitutions"
    )
    assert wanted_off >= 3 * BUDGET, (
        f"{method}: hinge OFF wanted only {wanted_off}; expected the relaxation to "
        "want far more than the budget allows, which is the truncation regime"
    )
    assert edits_off == BUDGET, "and the projection must still fill the budget exactly"


@pytest.mark.parametrize("method", ["apgm", "bindcraft", "colabdesign"])
def test_the_hinge_is_necessary_but_not_sufficient(seeder, toy_problem, method):
    """The counterexample, pinned so the hinge claim cannot be over-read.

    At init scale 5.0 on a 24-step budget, switching the hinge off moves NONE of
    the three methods. So "zero substitutions wanted" is not diagnostic of the
    loss or of the hinge on its own -- it is also what these settings produce on
    a toy loss containing no model at all. §31.10 of docs/P17_JN1.md used to
    read a zero in both arms as implicating OpenDDE's gradient; this is why it
    no longer does.
    """
    _, wanted_on = relax(seeder, toy_problem, method, steps=24, scale=5.0, hinge=5.0)
    _, wanted_off = relax(seeder, toy_problem, method, steps=24, scale=5.0, hinge=0.0)
    assert wanted_on == 0 and wanted_off == 0, (
        f"{method}: expected both arms dead at scale 5.0 / 24 steps, got "
        f"{wanted_on} and {wanted_off}; the counterexample has moved and "
        "§31.10's interpretation needs rechecking"
    )


def test_simplex_apgm_never_leaves_a_scale_five_start(seeder, toy_problem):
    """For apgm the init scale is absolute, not a speed effect.

    An earlier version of this file claimed the scale "sets the speed, not the
    direction" for every method. That was measured with apgm settings the tests
    supplied themselves (stepsize 0.1, momentum 0.9, scale 1.0); at the
    production 0.05/0.5/1.2 it is false for apgm, which wants zero
    substitutions from a scale-5.0 start however long it runs.
    """
    for steps in (24, 125):
        _, wanted = relax(seeder, toy_problem, "apgm", steps=steps, scale=5.0,
                          hinge=0.0)
        assert wanted == 0, f"apgm wanted {wanted} at scale 5.0 after {steps} steps"
    _, wanted_low = relax(seeder, toy_problem, "apgm", steps=24, scale=1.0, hinge=0.0)
    assert wanted_low >= 3 * BUDGET, (
        f"apgm wanted only {wanted_low} at scale 1.0; the contrast with scale 5.0 "
        "is the whole point of this test"
    )


@pytest.mark.parametrize("method", ["bindcraft", "colabdesign"])
def test_the_init_scale_sets_the_speed_for_the_logit_space_methods(
    seeder, toy_problem, method
):
    """For the two logit-space methods only, and only with the hinge off.

    The preferred residue's logit has to cross the gap the parent's scale
    opened, so a short budget separates the two scales and a long one does not.
    Note the scope: this says nothing about the trajectories being equivalent,
    only that both reach a comparable COUNT of wanted substitutions by 125
    steps, and it does not hold for apgm at all.
    """
    _, fast = relax(seeder, toy_problem, method, steps=24, scale=1.0, hinge=0.0)
    _, slow = relax(seeder, toy_problem, method, steps=24, scale=5.0, hinge=0.0)
    assert fast > slow, (
        f"{method}: scale 1.0 wanted {fast} and scale 5.0 wanted {slow} at 24 steps; "
        "the lower scale was supposed to flip first"
    )
    _, long_low = relax(seeder, toy_problem, method, steps=125, scale=1.0, hinge=0.0)
    _, long_high = relax(seeder, toy_problem, method, steps=125, scale=5.0, hinge=0.0)
    assert min(long_low, long_high) >= 3 * BUDGET, (
        f"{method}: at 125 steps scale 1.0 wanted {long_low} and scale 5.0 wanted "
        f"{long_high}; both were supposed to have crossed by then"
    )


# Shorter than the contrast tests above: at scale 1.0 with the hinge off, the
# logit-space methods have already moved by 40 steps.
_SHIPPED_STEPS = {"apgm": 24, "bindcraft": 40, "colabdesign": 40}


@pytest.mark.parametrize("method", ["apgm", "bindcraft", "colabdesign"])
def test_the_shipped_defaults_spend_the_whole_budget(seeder, toy_problem, method):
    """Hinge off, scale 1.0, the options on: the configuration `main()` ships."""
    edits, wanted = relax(
        seeder, toy_problem, method, steps=_SHIPPED_STEPS[method], scale=1.0,
        hinge=0.0,
    )
    assert edits == BUDGET, f"{method} spent only {edits} of {BUDGET} edits"
    assert wanted > BUDGET, f"{method} wanted only {wanted}"


def test_run_relaxation_refuses_an_unknown_method(seeder, toy_problem):
    make_loss, parent = toy_problem
    import jax
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="unknown method"):
        seeder.run_relaxation(
            "adam", make_loss(0.0), jnp.zeros((N_DESIGNABLE, 20)), jax.random.key(0),
            steps=4, lr=0.1, logit_norm=False, stochastic_hard=False,
            hard_temp=1.0, hard_fraction=0.25,
        )


def test_a_degenerate_start_is_refused_rather_than_frozen():
    """The uniform-start freeze, which silently did nothing with a live gradient."""
    import jax.numpy as jnp
    from mosaic.optimizers import _calibrate_logit_norm_scale, _layer_norm_logits

    for name, x in (("exactly uniform", jnp.zeros((5, 20))),
                    ("uniform nonzero", jnp.full((5, 20), 3.0)),
                    ("0.01 * normal (the --init noise start)",
                     jnp.full((5, 20), 0.0).at[0, 0].set(0.001))):
        with pytest.raises(ValueError, match="could not be calibrated"):
            _calibrate_logit_norm_scale(x)

    # and what it was doing instead: wiping a real update to exactly zero
    stepped = jnp.zeros((5, 20)).at[2, 3].set(0.27)
    assert float(jnp.abs(_layer_norm_logits(stepped, 0.0)).max()) == 0.0

    # a start with a real spread still calibrates
    anchored = jnp.zeros((5, 20)).at[:, 0].set(1.0)
    anchored = anchored - anchored.mean(-1, keepdims=True)
    gamma = _calibrate_logit_norm_scale(anchored)
    assert 0.2 < gamma < 0.25


def test_gamma_bounds_the_aggregate_spread_not_each_row():
    """What normalisation actually constrains, after I got this wrong.

    `_layer_norm_logits` normalises over the WHOLE (N, 20) block, so gamma fixes
    the GLOBAL standard deviation. An earlier version of this test evaluated a
    (1, 20) array and concluded that gamma 0.218 "caps the softmax at about
    p=0.125 per residue, so the relaxation can never become confident". That is
    false for the shape the seeder actually uses: concentrating the spread in
    one row of 29 reaches p=0.91 at the same gamma, because a single entry can
    exceed the global deviation by up to about sqrt(29 * 20) = 24.

    What gamma does bound is the aggregate: every row cannot be peaked at once,
    because that would require a global deviation larger than gamma. So a small
    gamma rations total confidence across positions rather than forbidding it
    anywhere.
    """
    import jax
    import jax.numpy as jnp
    from mosaic.optimizers import _layer_norm_logits

    gamma = 0.218

    def max_p(x):
        z = _layer_norm_logits(jnp.asarray(x, dtype=jnp.float32), gamma)
        return float(jax.nn.softmax(z, -1).max())

    every_row = np.zeros((N_DESIGNABLE, 20), dtype=np.float32)
    every_row[np.arange(N_DESIGNABLE), 0] = 1.0
    one_row = np.zeros((N_DESIGNABLE, 20), dtype=np.float32)
    one_row[0, 0] = 1.0

    assert max_p(every_row) < 0.15, (
        "with every row peaked the global deviation is spread thin, so no row "
        "gets far from uniform"
    )
    assert max_p(one_row) > 0.8, (
        "but concentrating in one row reaches high confidence at the SAME gamma "
        "-- which is why the per-row cap claim was wrong"
    )
    # and the global deviation is gamma in both cases, which is the invariant
    for x in (every_row, one_row):
        z = _layer_norm_logits(jnp.asarray(x, dtype=jnp.float32), gamma)
        assert abs(float(jnp.std(z)) - gamma) < 1e-3


@pytest.mark.parametrize("n_steps", [4, 5, 6, 7, 9, 13, 125, 600])
def test_bindcraft_executes_exactly_the_requested_number_of_steps(n_steps):
    """The budget overshoot, measured on the real function rather than a copy.

    Four equal stages at `n_steps=6` ran 7 steps: the old rescaling dumped the
    rounding remainder on one stage and clamped it at a floor of 1, and the
    clamp could swallow part of the correction. `trajectory_fn` fires once per
    step, so this counts what actually executed -- a Python counter inside the
    loss would not, because the loss is traced.
    """
    import jax
    import jax.numpy as jnp
    from mosaic.common import LossTerm
    from mosaic.optimizers import bindcraft_design

    class Quad(LossTerm):
        def __call__(self, seq, *, key=None):
            return (seq ** 2).sum(), {}

    equal = dict(logits_iters=(1, 1), soft_iters=1, hard_iters=1)
    for kwargs in ({}, equal):
        _, trajectory = bindcraft_design(
            loss_function=Quad(), x=jnp.zeros((3, 20)), lr=0.1,
            key=jax.random.key(0), n_steps=n_steps,
            trajectory_fn=lambda aux, x: 1, **kwargs,
        )
        assert len(trajectory) == n_steps, (
            f"requested {n_steps} steps with {kwargs or 'the published schedule'}, "
            f"executed {len(trajectory)}"
        )


def test_apportionment_is_exact_and_preserves_a_published_schedule():
    import itertools

    from mosaic.optimizers import _apportion_stages

    assert _apportion_stages((50, 25, 45, 5), 125) == [50, 25, 45, 5], (
        "a schedule passed at its own total must come back unchanged"
    )
    assert sum(_apportion_stages((1, 1, 1, 1), 6)) == 6, "the n=6 overshoot"
    for counts in itertools.product((1, 2, 5, 9, 50), repeat=4):
        for n in (4, 5, 6, 7, 11, 37, 125, 600):
            scaled = _apportion_stages(counts, n)
            assert sum(scaled) == n, (counts, n, scaled)
            assert min(scaled) >= 1, (counts, n, scaled)
    with pytest.raises(ValueError, match="cannot cover|cannot give"):
        _apportion_stages((1, 1, 1, 1), 3)


def test_the_seeder_does_not_pay_for_a_best_iterate_it_discards(
    seeder, toy_problem, monkeypatch
):
    """`relaxed_array` takes result[0], so APGM scoring is spend for nothing.

    126 evaluations on the probe's 125-step runs, 378 across its three APGM
    arms, for a value never read.
    """
    import contextlib
    import io

    import jax
    import jax.numpy as jnp

    import mosaic.optimizers as optimizers

    make_loss, parent = toy_problem
    init_key, run_key = jax.random.split(jax.random.key(0))
    noise = 0.01 * np.asarray(jax.random.normal(init_key, (N_DESIGNABLE, 20)))
    x0 = jnp.asarray(
        seeder.anchored_logits(parent, np.arange(N_DESIGNABLE), 1.0, noise)
    )
    calls = []
    real = optimizers._eval_loss

    def counting(loss_function, x, key):
        calls.append(1)
        return real(loss_function, x, key)

    monkeypatch.setattr(optimizers, "_eval_loss", counting)
    with contextlib.redirect_stdout(io.StringIO()):
        seeder.run_relaxation(
            "apgm", make_loss(0.0), x0, run_key, steps=5, lr=0.1,
            logit_norm=False, stochastic_hard=False, hard_temp=1.0,
            hard_fraction=0.25,
        )
    assert not calls, f"the seeder paid for {len(calls)} discarded evaluations"


# --- relaxed_concentration: separating "converged" from "never moved" ------
#
# `project_to_budget` reports an argmax, and on the shipped anchored start the
# parent is already the argmax at all 29 designable positions before the first
# step. So `candidate_substitutions_considered == 0` has two causes with
# opposite meanings, and until 2026-10-14 the seed JSON recorded nothing that
# told them apart -- which is exactly the question the real-loss run of that
# date left open for `apgm` (section 32).


def test_concentration_separates_a_converged_relaxation_from_an_untouched_one(seeder):
    parent = np.arange(N_DESIGNABLE, dtype=np.int32) % 20

    converged = relaxed_at_parent(parent)
    converged = 0.98 * converged + 0.02 / 20  # not a vertex, but decided
    moved = seeder.relaxed_concentration(converged, parent)

    untouched = softmax(seeder.anchored_logits(
        parent, np.arange(N_DESIGNABLE), 1.0, np.zeros((N_DESIGNABLE, 20))))
    still = seeder.relaxed_concentration(untouched, parent)

    # Both report 0 wanted substitutions: the parent is the argmax in both.
    assert (converged.argmax(-1) == parent).all()
    assert (untouched.argmax(-1) == parent).all()
    # Only the concentration separates them.
    assert moved["relaxed_peak_probability_mean"] > 0.9
    assert still["relaxed_peak_probability_mean"] == pytest.approx(0.125, abs=5e-3)


def test_the_untouched_peak_is_the_init_scales_own_parent_mass(seeder):
    """The gate's 0.3 cutoff has to sit above every scale this stage ships.

    `--init-logit-scale 1.0` is the default and puts 0.125 on the parent. The
    cutoff is only meaningful if that number is a property of the start rather
    than an assertion in a docstring, so it is measured here -- and at 5.0,
    the former default, an untouched start would read 0.886 and the gate would
    call it converged. That is correct: at scale 5.0 the start IS decided.
    """
    parent = np.arange(N_DESIGNABLE, dtype=np.int32) % 20
    designable = np.arange(N_DESIGNABLE)
    for scale, expected in ((1.0, 0.125), (5.0, 0.886)):
        probs = softmax(seeder.anchored_logits(
            parent, designable, scale, np.zeros((N_DESIGNABLE, 20))))
        c = seeder.relaxed_concentration(probs, parent)
        assert c["relaxed_peak_probability_mean"] == pytest.approx(expected, abs=5e-3)
        assert c["relaxed_parent_probability_mean"] == pytest.approx(expected, abs=5e-3)


def test_concentration_reports_the_parent_mass_separately_from_the_peak(seeder):
    """A relaxation can be decided and not be at the parent; both are recorded."""
    parent = np.zeros(N_DESIGNABLE, dtype=np.int32)
    relaxed = np.full((N_DESIGNABLE, 20), 0.01 / 19, dtype=np.float64)
    relaxed[:, 7] = 0.99          # decided, and not on the parent
    c = seeder.relaxed_concentration(relaxed, parent)
    assert c["relaxed_peak_probability_mean"] == pytest.approx(0.99, abs=1e-6)
    assert c["relaxed_parent_probability_mean"] == pytest.approx(0.01 / 19, abs=1e-6)


def test_concentration_flags_rows_that_are_not_probabilities(seeder):
    """The numbers only mean what they say if the rows sum to one.

    All three optimizers are supposed to return probabilities through
    `run_relaxation` -- `bindcraft_design` softmaxes, `simplex_APGM` stays on
    the simplex, and `run_relaxation` softmaxes `colabdesign_stage` itself --
    but `colabdesign_stage` is logits-in/logits-out, so one dropped softmax
    would turn every margin and every concentration above into a logit
    difference while still printing as a probability.
    """
    parent = np.zeros(N_DESIGNABLE, dtype=np.int32)
    logits = np.linspace(-4, 4, N_DESIGNABLE * 20).reshape(N_DESIGNABLE, 20)
    assert seeder.relaxed_concentration(
        logits, parent)["relaxed_row_sum_max_deviation"] > 1.0
    assert seeder.relaxed_concentration(
        softmax(logits), parent)["relaxed_row_sum_max_deviation"] < 1e-6


def test_the_minimum_peak_is_reported_not_only_the_mean(seeder):
    """A mean hides one undecided position among 28 decided ones."""
    parent = np.zeros(N_DESIGNABLE, dtype=np.int32)
    relaxed = np.full((N_DESIGNABLE, 20), 0.01 / 19, dtype=np.float64)
    relaxed[:, 3] = 0.99
    relaxed[0] = 1.0 / 20                      # one row still uniform
    c = seeder.relaxed_concentration(relaxed, parent)
    assert c["relaxed_peak_probability_mean"] > 0.95
    assert c["relaxed_peak_probability_min"] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("method", ["apgm", "bindcraft", "colabdesign"])
def test_run_relaxation_returns_probabilities_for_every_method(
    seeder, toy_problem, method
):
    """What the concentration fields and the margins are denominated in.

    `colabdesign_stage` is logits-in/logits-out by design and `run_relaxation`
    softmaxes it on the way out; `bindcraft_design` softmaxes internally and
    `simplex_APGM` never leaves the simplex. The seed JSON's margins and its
    peak/parent probabilities are read as probabilities across all three, so
    that has to hold for the function `main()` actually calls, not for the
    optimizers called directly -- the older test above calls them directly and
    so would not notice a dropped softmax here.
    """
    import contextlib
    import io

    import jax
    import jax.numpy as jnp

    make_loss, parent = toy_problem
    init_key, run_key = jax.random.split(jax.random.key(0))
    noise = 0.01 * np.asarray(jax.random.normal(init_key, (N_DESIGNABLE, 20)))
    x0 = jnp.asarray(
        seeder.anchored_logits(parent, np.arange(N_DESIGNABLE), 1.0, noise)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        relaxed = np.asarray(seeder.run_relaxation(
            method, make_loss(0.0), x0, run_key, steps=24, lr=0.1,
            logit_norm=False, stochastic_hard=False, hard_temp=1.0,
            hard_fraction=0.25,
        ))
    c = seeder.relaxed_concentration(relaxed, parent)
    assert c["relaxed_row_sum_max_deviation"] < 1e-3, (
        f"{method} did not come back in probability space: rows deviate by "
        f"{c['relaxed_row_sum_max_deviation']}"
    )
    assert 0.0 < c["relaxed_peak_probability_min"] <= 1.0
    assert 0.0 <= c["relaxed_parent_probability_mean"] <= 1.0


def test_main_actually_records_the_fields_the_gate_reads():
    """The write itself, which no behavioural test can reach.

    `main()` loads OpenDDE, so the JSON it writes is only ever produced on the
    cluster. Deleting `designable_positions` and `**concentration` from that
    one `json.dumps` call passed every test in this file and every test of the
    launcher: the gate would then read a *fresh* run as a pre-2026-10-14 one
    and say so, which is loud but only to a reader who notices the date. This
    is the structural check instead -- the record's keys, read off the source.
    """
    import ast

    tree = ast.parse((REPO / "examples" / "p17_continuous_seed.py").read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    dumps = [n for n in ast.walk(main)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "dumps"]
    assert len(dumps) == 1, "main() no longer has exactly one json.dumps"
    record = next(a for a in dumps[0].args if isinstance(a, ast.Call))
    keywords = {k.arg for k in record.keywords}
    assert "designable_positions" in keywords, (
        "the gate reads this to know what denominator the wanted counts are "
        "against; without it a changed CDR mask is invisible"
    )
    assert None in keywords, (
        "`**concentration` is gone: 0 wanted substitutions is then "
        "unattributable between a converged relaxation and a frozen one, "
        "which is what section 32 could not resolve for apgm"
    )


# --- projection_tie_at_cut: the ranking that is not a ranking ---------------
#
# §32.5. `project_to_budget`'s docstring claimed the kept edits are the ones
# the relaxation pushed hardest for "not an arbitrary subset". Under a tie that
# is false: the tuple sort falls through to position index, descending. The
# 2026-10-15 `apgm` noise probe hit it exactly -- 26 wanted at margins
# 1.000-1.000, five handed over, all of them 105-109.


def test_a_saturated_relaxation_hands_over_the_highest_numbered_positions(seeder):
    """The defect itself, pinned so a tie-break change is a visible decision."""
    cdr = [26, 27, 28, 29, 30, 31, 32, 33, 51, 52, 53, 54, 55, 56, 57, 58,
           97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    designable = np.array([p - 1 for p in cdr])
    rng = np.random.default_rng(0)
    parent = rng.integers(0, 20, size=123).astype(np.int32)
    relaxed = np.zeros((123, 20))
    relaxed[np.arange(123), parent] = 1.0
    for pos in designable:                      # a vertex, on a non-parent residue
        relaxed[pos] = 0.0
        relaxed[pos, (parent[pos] + 1) % 20] = 1.0

    _tokens, margins, kept = seeder.project_to_budget(
        relaxed, parent, designable, 5)
    assert len(margins) == len(cdr), "a vertex everywhere wants every position"
    assert {round(m, 9) for m, _, _ in margins} == {1.0}
    assert sorted(p + 1 for _, p, _ in kept) == cdr[-5:] == [105, 106, 107, 108, 109]


def test_the_tie_is_measured_rather_than_left_silent(seeder):
    tied = [(1.0, p, 3) for p in range(26)]
    report = seeder.projection_tie_at_cut(tied, 5)
    assert report["projection_tied_group"] == 26
    assert report["projection_tied_discarded"] == 21
    assert report["projection_margin_gap_at_cut"] == 0.0


def test_distinct_margins_report_no_tie(seeder):
    """Today's `bindcraft` and `colabdesign` shape: 15 wanted, margins apart."""
    distinct = [(0.70 + 0.01 * i, i, 3) for i in range(15)]
    report = seeder.projection_tie_at_cut(distinct, 5)
    assert report["projection_tied_discarded"] == 0
    assert report["projection_tied_group"] == 1
    assert report["projection_margin_gap_at_cut"] == pytest.approx(0.01)


def test_nothing_discarded_means_no_tie_to_report(seeder):
    """Fewer wanted than the budget: the cut does not exist."""
    report = seeder.projection_tie_at_cut([(0.9, 1, 3), (0.8, 2, 3)], 5)
    assert report["projection_tied_discarded"] == 0
    assert report["projection_margin_gap_at_cut"] is None
    assert seeder.projection_tie_at_cut([(0.9, 1, 3)], 0)[
        "projection_margin_gap_at_cut"] is None


def test_a_near_tie_is_a_number_not_a_verdict(seeder):
    """The gap is reported; no threshold decides what counts as "near".

    A cut 1e-9 apart is as arbitrary as an exact tie in practice, but picking
    the tolerance that says so is a judgement this code does not get to make
    on the reader's behalf.
    """
    near = [(1.0, 0, 3), (1.0, 1, 3), (1.0, 2, 3), (1.0, 3, 3),
            (1.0, 4, 3), (1.0 - 1e-9, 5, 3)]
    report = seeder.projection_tie_at_cut(near, 5)
    assert report["projection_tied_discarded"] == 0, "not an exact tie"
    assert 0.0 < report["projection_margin_gap_at_cut"] < 1e-5


def test_the_tie_fields_reach_the_record(seeder):
    """Same structural check as the concentration fields: `main()` writes them."""
    import ast

    tree = ast.parse((REPO / "examples" / "p17_continuous_seed.py").read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    dumps = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "dumps"]
    record = next(a for a in dumps[0].args if isinstance(a, ast.Call))
    starred = [k for k in record.keywords if k.arg is None]
    names = {k.value.id for k in starred if isinstance(k.value, ast.Name)}
    # A subset check, not equality: this exists to catch a group being
    # DELETED, and equality would make every future addition fail a test that
    # gains no safety from it.
    required = {"concentration", "tie", "objective"}
    assert required <= names, (
        f"main() no longer records {sorted(required - names)}"
    )


def test_main_uses_no_name_it_has_not_bound():
    """A NameError in `main()` costs the whole run, after it has finished.

    `main()` loads OpenDDE, so nothing in this file executes it: the §32.5
    diagnostics were added with a call to `loss_function`, a name that exists
    in `run_relaxation` and not in `main`, where the loss is
    `variable_only_loss`. That is a crash at the *write*, after 600 steps and
    two GPU-hours, and `relaxed_loss`'s own try/except could not have caught
    it -- the guard wraps the body, not the call.

    So: every name `main()` reads must be one it binds, imports, takes as an
    argument, or finds at module level.
    """
    import ast
    import builtins

    source = (REPO / "examples" / "p17_continuous_seed.py").read_text()
    tree = ast.parse(source)
    module_level = {n.id for node in tree.body for n in ast.walk(node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    module_level |= {n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_level |= {(a.asname or a.name).split(".")[0]
                             for a in node.names}

    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    bound = {a.arg for a in main.args.args} | {
        a.arg for a in main.args.kwonlyargs}
    for node in ast.walk(main):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                               ast.DictComp)):
            for gen in node.generators:
                for n in ast.walk(gen.target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            bound |= {a.arg for a in node.args.args}

    read = {n.id for n in ast.walk(main)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unbound = read - bound - module_level - set(dir(builtins))
    assert not unbound, f"main() reads names it never binds: {sorted(unbound)}"


def test_the_objective_is_evaluated_at_both_endpoints_under_one_key(seeder, toy_problem):
    """Descent, not a re-draw: the same key at the start and at the end."""
    import jax
    import jax.numpy as jnp

    make_loss, parent = toy_problem
    loss = make_loss(0.0)
    designable = np.arange(N_DESIGNABLE)
    x0 = jnp.asarray(seeder.anchored_logits(
        parent, designable, 1.0, np.zeros((N_DESIGNABLE, 20))))
    # A deliberately DIFFERENT point: the parent one-hot against the
    # near-uniform start. Which of the two the toy prefers is not the subject
    # here -- on this loss the descent comes out negative -- only that both
    # were evaluated and that the three recorded numbers agree.
    relaxed = relaxed_at_parent(parent).astype(np.float64)
    report = seeder.relaxed_loss(loss, x0, relaxed, jax.random.key(0))
    assert report["loss_eval_error"] is None
    assert report["loss_at_init"] is not None
    assert report["loss_at_relaxed"] is not None
    assert report["loss_descent"] == pytest.approx(
        report["loss_at_init"] - report["loss_at_relaxed"], abs=1e-6)


def test_a_failing_loss_does_not_destroy_a_finished_run(seeder):
    """Two GPU-hours of completed work must survive a broken diagnostic."""
    class Exploding:
        def __call__(self, seq, *, key=None):
            raise RuntimeError("boom")

    import jax
    report = seeder.relaxed_loss(
        Exploding(), np.zeros((N_DESIGNABLE, 20)),
        np.full((N_DESIGNABLE, 20), 1.0 / 20), jax.random.key(0))
    assert report["loss_at_relaxed"] is None
    assert report["loss_descent"] is None
    assert "boom" in report["loss_eval_error"]


def test_a_nonfinite_loss_is_recorded_as_missing_not_as_a_number(seeder):
    """`nan` descent would read as a measurement. It is not one."""
    class NotANumber:
        def __call__(self, seq, *, key=None):
            return float("nan"), {}

    import jax
    report = seeder.relaxed_loss(
        NotANumber(), np.zeros((N_DESIGNABLE, 20)),
        np.full((N_DESIGNABLE, 20), 1.0 / 20), jax.random.key(0))
    assert report["loss_at_init"] is None
    assert report["loss_at_relaxed"] is None
    assert report["loss_descent"] is None
    assert report["loss_eval_error"] is None, "it evaluated; it was not finite"
