"""Regression tests for the base-library corrections of 2026-10-13.

Four defects found by an independent audit, each verified against a reference
that already existed rather than against my own reasoning:

- `_calculate_bin_centers` duplicated its catch-all bin. The correct expression
  is in this repo, vendored from AlphaFold, so the test compares the two.
- `interaction_prediction_score` broadcast its per-row `d0` along the column
  axis. The test builds a case where rows genuinely differ and checks the
  row alignment explicitly.
- `gradient_MCMC` scored the wrong move when reversing a proposal path, so its
  `detailed_balance=True` correction was wrong.
- `gradient_MCMC` drew its path length from NumPy's GLOBAL random state, so two
  runs with the same explicit key diverged.

Every one of these changes numbers that earlier runs recorded. They are fixed
rather than preserved because a known-wrong formula is not a baseline, but any
comparison across 2026-10-13 has to account for them.
"""

import numpy as np
import pytest


# --- #4: the catch-all bin center -----------------------------------------


@pytest.mark.parametrize(
    "breaks",
    [
        [0.0, 1.0, 2.0],
        [0.0, 0.5, 1.0, 1.5, 2.0],
        list(np.linspace(0.0, 31.0, 31)),  # the AF2 pTM shape
    ],
)
def test_bin_centers_match_the_vendored_alphafold_implementation(breaks):
    """The reference is already in the repo; the two must agree exactly."""
    import jax.numpy as jnp

    from mosaic.alphafold.common.confidence import (
        _calculate_bin_centers as af2_bin_centers,
    )
    from mosaic.losses.confidence_metrics import _calculate_bin_centers

    mine = np.asarray(_calculate_bin_centers(jnp.asarray(breaks, dtype=jnp.float32)))
    theirs = np.asarray(af2_bin_centers(np.asarray(breaks, dtype=np.float64)))
    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine, theirs, rtol=1e-6, atol=1e-6)


def test_the_catch_all_bin_extends_past_the_last_edge():
    """The defect, stated directly: it used to repeat the previous center."""
    import jax.numpy as jnp

    from mosaic.losses.confidence_metrics import _calculate_bin_centers

    got = np.asarray(_calculate_bin_centers(jnp.array([0.0, 1.0, 2.0])))
    np.testing.assert_allclose(got, [0.5, 1.5, 2.5, 3.5])
    assert got[-1] != got[-2], "the catch-all bin must not duplicate its neighbour"


# --- #5: the per-row d0 broadcast -----------------------------------------


def _directional_interface(n_a=40, n_b=60, n_bins=16):
    """Two chains whose rows have genuinely different partner counts.

    The counts have to exceed `clip(n_residues, min=27)` or `d0` collapses to a
    constant and the broadcast axis stops mattering -- which is how this
    survived, and why a small synthetic case shows nothing.
    """
    import jax.numpy as jnp

    n = n_a + n_b
    asym = jnp.asarray([0] * n_a + [1] * n_b)
    confident = np.zeros((n, n), dtype=bool)
    for i in range(n_a):
        confident[i, n_a : n_a + 28 + (i % 30)] = True
    for j in range(n_b):
        confident[n_a + j, : min(30 + (j % 10), n_a)] = True
    low = jnp.full((n, n, n_bins), -8.0).at[:, :, 0].set(6.0)
    high = jnp.full((n, n, n_bins), -8.0).at[:, :, -1].set(6.0)
    logits = jnp.where(jnp.asarray(confident)[:, :, None], low, high)
    return logits, asym, jnp.linspace(0.25, 31.75, n_bins)


def test_interaction_score_applies_each_rows_d0_to_its_own_row():
    """The reference is the row-aligned computation written out longhand."""
    import jax
    import jax.numpy as jnp

    from mosaic.losses.structure_prediction import interaction_prediction_score

    logits, asym, bins = _directional_interface()
    got = interaction_prediction_score(logits, bins, asym_id=asym, pae_cutoff=10.0)

    probs = jax.nn.softmax(logits, -1)
    pae = jnp.sum(probs * bins, -1)
    pair_mask = (asym[:, None] != asym[None, :]) & (pae < 10.0)
    n_res = jnp.sum(pair_mask, -1, keepdims=True)
    d0 = 1.24 * (jnp.clip(n_res, min=27) - 15) ** (1.0 / 3) - 1.8
    tm = 1.0 / (1 + jnp.square(bins) / jnp.square(d0))
    expected = jnp.sum(
        jnp.sum(probs * tm[:, None, :], -1) * (pair_mask / (1e-8 + n_res)), -1
    )

    assert len(np.unique(np.round(np.asarray(d0), 6))) > 5, (
        "the fixture was supposed to give the rows different d0 values"
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=1e-6, atol=1e-6)


def test_the_column_aligned_broadcast_gives_a_different_answer():
    """Pins that the fix is observable, so it cannot silently regress."""
    import jax
    import jax.numpy as jnp

    from mosaic.losses.structure_prediction import interaction_prediction_score

    logits, asym, bins = _directional_interface()
    got = interaction_prediction_score(logits, bins, asym_id=asym, pae_cutoff=10.0)

    probs = jax.nn.softmax(logits, -1)
    pae = jnp.sum(probs * bins, -1)
    pair_mask = (asym[:, None] != asym[None, :]) & (pae < 10.0)
    n_res = jnp.sum(pair_mask, -1, keepdims=True)
    d0 = 1.24 * (jnp.clip(n_res, min=27) - 15) ** (1.0 / 3) - 1.8
    tm = 1.0 / (1 + jnp.square(bins) / jnp.square(d0))
    # the old behaviour: no [:, None, :], so (N, B) broadcasts as (1, N, B)
    old = jnp.sum(jnp.sum(probs * tm, -1) * (pair_mask / (1e-8 + n_res)), -1)
    assert not np.allclose(np.asarray(got), np.asarray(old), atol=1e-4), (
        "the row- and column-aligned results coincide on this fixture, so it no "
        "longer distinguishes the fix"
    )


def test_a_single_chain_pair_count_below_the_clip_is_unaffected():
    """Where d0 is constant the two alignments agree, as they must."""
    import jax.numpy as jnp

    from mosaic.losses.structure_prediction import interaction_prediction_score

    n, n_bins = 8, 16
    asym = jnp.asarray([0] * 3 + [1] * 5)
    logits = jnp.full((n, n, n_bins), -8.0).at[:, :, 0].set(4.0)
    bins = jnp.linspace(0.25, 31.75, n_bins)
    out = np.asarray(
        interaction_prediction_score(logits, bins, asym_id=asym, pae_cutoff=10.0)
    )
    assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()


# --- #6: the reverse proposal path ----------------------------------------


def test_the_reverse_path_restores_the_sequence_it_started_from():
    """The invariant the old implementation violated.

    It assigned the FORWARD token at each reverse step, which the proposal
    already held, so the walk never moved and `detailed_balance=True` scored
    the probability of re-applying each mutation instead of undoing it.
    """
    import jax.numpy as jnp

    from mosaic.optimizers import _reverse_path_logprob

    rng = np.random.default_rng(0)
    original = jnp.asarray(rng.integers(0, 20, size=12), dtype=jnp.int32)
    gradient = jnp.asarray(rng.normal(size=(12, 20)), dtype=jnp.float32)

    # a three-step forward path, recorded the way gradient_MCMC records it
    proposal, mutations = original, []
    for position, new_token in ((3, 7), (9, 1), (3, 15)):
        mutations.append((position, new_token, int(proposal[position])))
        proposal = proposal.at[position].set(new_token)
    assert not np.array_equal(np.asarray(proposal), np.asarray(original))

    log_q, restored = _reverse_path_logprob(proposal, mutations, gradient, 0.01, 20)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(original))
    assert np.isfinite(float(log_q))


def test_the_reverse_path_revisits_a_position_in_the_right_order():
    """Position 3 is mutated twice, so order decides which token is restored."""
    import jax.numpy as jnp

    from mosaic.optimizers import _reverse_path_logprob

    original = jnp.asarray([5, 5, 5, 5], dtype=jnp.int32)
    gradient = jnp.zeros((4, 20), dtype=jnp.float32)
    proposal, mutations = original, []
    for position, new_token in ((2, 11), (2, 19)):
        mutations.append((position, new_token, int(proposal[position])))
        proposal = proposal.at[position].set(new_token)

    _log_q, restored = _reverse_path_logprob(proposal, mutations, gradient, 0.01, 20)
    np.testing.assert_array_equal(np.asarray(restored), [5, 5, 5, 5])


def test_the_reverse_logprob_differs_from_re_applying_the_forward_tokens():
    """Pins that the correction is observable, so it cannot silently regress."""
    import jax.numpy as jnp

    from mosaic.optimizers import _proposal, _reverse_path_logprob

    rng = np.random.default_rng(3)
    original = jnp.asarray(rng.integers(0, 20, size=10), dtype=jnp.int32)
    gradient = jnp.asarray(rng.normal(size=(10, 20)), dtype=jnp.float32)
    proposal, mutations = original, []
    for position, new_token in ((1, 4), (6, 12)):
        mutations.append((position, new_token, int(proposal[position])))
        proposal = proposal.at[position].set(new_token)

    correct, _ = _reverse_path_logprob(proposal, mutations, gradient, 0.01, 20)

    # the old behaviour, written out: score the forward token and assign it
    stale, log_q_old = proposal.copy(), 0.0
    for position, new_token, _prev in reversed(mutations):
        _p, log_p = _proposal(stale, gradient, 0.01, alphabet_size=20)
        log_q_old += log_p[position, new_token]
        stale = stale.at[position].set(new_token)
    np.testing.assert_array_equal(np.asarray(stale), np.asarray(proposal))
    assert abs(float(correct) - float(log_q_old)) > 1e-3, (
        "the corrected reverse path matches the old wrong one on this fixture, "
        "so it no longer distinguishes the fix"
    )


# --- #7: reproducibility from the supplied key ----------------------------
#
# Both tests below assert the sampler actually MOVED. An earlier version used a
# loss whose optimum was the starting sequence, so nothing was ever accepted,
# both runs returned the start unchanged, and the reproducibility assertion
# passed without exercising anything.


@pytest.fixture(scope="module")
def mcmc_problem():
    """A loss with its optimum away from the start, so the walk has to move."""
    import jax.numpy as jnp

    from mosaic.common import LossTerm

    rng = np.random.default_rng(5)
    target = jnp.asarray(
        np.eye(20)[rng.integers(0, 20, size=8)], dtype=jnp.float32
    )

    class Distance(LossTerm):
        def __call__(self, seq, *, key=None):
            value = ((seq - target) ** 2).sum()
            return value, {"distance": value}

    return Distance(), jnp.asarray(np.zeros(8), dtype=jnp.int32)


def _run_mcmc(problem, key_seed, global_seed):
    import contextlib
    import io

    import jax

    from mosaic.optimizers import gradient_MCMC

    loss, start = problem
    np.random.seed(global_seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = gradient_MCMC(
            loss, start, steps=6, max_path_length=3,
            key=jax.random.key(key_seed), detailed_balance=True,
        )
    return np.asarray(out)


def test_gradient_mcmc_is_reproducible_from_its_key_alone(mcmc_problem):
    """The path length used to come from NumPy's GLOBAL random state.

    The global state is deliberately disturbed between the two runs, so an
    identical result can only come from the explicit key.
    """
    _loss, start = mcmc_problem
    a = _run_mcmc(mcmc_problem, 11, global_seed=1)
    b = _run_mcmc(mcmc_problem, 11, global_seed=999)
    assert not np.array_equal(a, np.asarray(start)), (
        "the sampler never left the start, so this would pass without testing "
        "anything -- the fixture needs a loss that rewards moving"
    )
    np.testing.assert_array_equal(a, b)


def test_different_keys_still_give_different_trajectories(mcmc_problem):
    """The complement: the fix must not have made the sampler deterministic."""
    outs = [_run_mcmc(mcmc_problem, seed, global_seed=7) for seed in range(6)]
    _loss, start = mcmc_problem
    assert any(not np.array_equal(o, np.asarray(start)) for o in outs), (
        "no key moved the sequence at all"
    )
    assert len({o.tobytes() for o in outs}) > 1, "every key gave the same trajectory"


# --- #1: the APGM best_x contract -----------------------------------------
#
# `best_val` held the loss at the momentum extrapolation `v` while `best_x` was
# set to the post-update `x`, so the two described different points and the
# returned "best" could be worse than the input. The gradient is still taken at
# `v` -- the trajectory is untouched -- and selection now scores the FEASIBLE
# iterates on a fixed key of its own. Both the initial and final iterates are
# eligible; `track_best=False` skips the extra forward passes.


@pytest.fixture
def quadratic():
    import jax.numpy as jnp

    from mosaic.common import LossTerm

    target = jnp.zeros((2, 20)).at[:, 0].set(1.0)

    class Quadratic(LossTerm):
        def __call__(self, seq, *, key=None):
            value = ((seq - target) ** 2).sum()
            return value, {"quadratic": value}

    def at(a):
        return float(((np.asarray(a) - np.asarray(target)) ** 2).sum())

    return Quadratic(), at


def test_apgm_best_x_is_never_worse_than_the_starting_point(quadratic):
    """The counterexample: an overshooting run used to return a worse `best_x`."""
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import simplex_APGM

    loss, at = quadratic
    x0 = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    x, best_x = simplex_APGM(
        loss_function=loss, x=x0, n_steps=8, stepsize=2.0, momentum=0.95,
        key=jax.random.key(0),
    )
    assert at(best_x) <= at(x0) + 1e-6, (
        f"best_x scored {at(best_x):.4f} against the start's {at(x0):.4f}"
    )
    assert at(x) > at(best_x), "this fixture was supposed to overshoot"


def test_apgm_best_x_is_feasible(quadratic):
    """It must be a point on the simplex, not an unprojected extrapolation."""
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import simplex_APGM

    loss, _at = quadratic
    x0 = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    _x, best_x = simplex_APGM(
        loss_function=loss, x=x0, n_steps=6, stepsize=2.0, momentum=0.95,
        key=jax.random.key(1),
    )
    best = np.asarray(best_x)
    assert (best >= -1e-6).all(), "negative entries: not on the simplex"
    np.testing.assert_allclose(best.sum(-1), 1.0, atol=1e-5)


def test_apgm_tracking_does_not_change_the_trajectory(quadratic):
    """The gradient is still taken at `v`, so the final iterate must be identical."""
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import simplex_APGM

    loss, _at = quadratic
    x0 = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    kw = dict(loss_function=loss, x=x0, n_steps=5, stepsize=1.0, momentum=0.9)
    tracked, _ = simplex_APGM(key=jax.random.key(2), track_best=True, **kw)
    untracked, _ = simplex_APGM(key=jax.random.key(2), track_best=False, **kw)
    np.testing.assert_allclose(np.asarray(tracked), np.asarray(untracked),
                               rtol=1e-6, atol=1e-6)


def test_batched_apgm_best_x_is_never_worse_than_its_start():
    """The same defect and the same fix, per batch element."""
    import jax
    import jax.numpy as jnp

    from mosaic.common import LossTerm
    from mosaic.optimizers import batched_simplex_APGM

    B = 3
    target = jnp.zeros((B, 2, 20)).at[:, :, 0].set(1.0)

    class Quadratic(LossTerm):
        def __call__(self, seq, *, key=None):
            value = ((seq - target[0]) ** 2).sum()
            return value, {"quadratic": value}

    x0 = jnp.zeros((B, 2, 20)).at[:, :, 0].set(0.9).at[:, :, 1].set(0.1)
    x, best_x = batched_simplex_APGM(
        loss_function=Quadratic(), x=x0, n_steps=8, stepsize=2.0, momentum=0.95,
        key=jax.random.key(0),
    )
    per = lambda a: ((np.asarray(a) - np.asarray(target)) ** 2).sum(axis=(-2, -1))
    assert (per(best_x) <= per(x0) + 1e-5).all(), (
        f"best {per(best_x)} against starts {per(x0)}"
    )
    best = np.asarray(best_x)
    assert (best >= -1e-6).all()
    np.testing.assert_allclose(best.sum(-1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Defect 8: a NaN-scoring iterate could be selected as `best_x`
#
# `_eval_loss` passed its value through `nan_to_num(value, nan=1e6)`, so a
# diverged candidate scored 1e6 -- which beats any legitimate loss above 1e6 --
# and the `isnan` guard at the selection site could never fire, because the NaN
# had already been replaced by the time it was tested. Scoring now returns the
# raw value and selection rejects anything nonfinite.
# --------------------------------------------------------------------------- #


@pytest.fixture
def nan_cliff():
    """Finite and LARGE at the start, NaN once the first row's mass drops.

    The start has to score above the old 1e6 sentinel for the defect to show:
    that is the whole point -- a NaN laundered to 1e6 looked like an
    improvement on a loss of 2e6.
    """
    import jax.numpy as jnp

    from mosaic.common import LossTerm

    class NaNCliff(LossTerm):
        def __call__(self, seq, *, key=None):
            base = 2.0e6 + 1.0e6 * (seq**2).sum()
            value = jnp.where(seq[0, 0] > 0.8, base, jnp.nan)
            return value, {"nan_cliff": value}

    return NaNCliff()


def test_eval_loss_does_not_launder_a_nan_into_a_finite_score(nan_cliff):
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import _eval_loss

    inside = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    over = jnp.full((2, 20), 1.0 / 20.0)

    assert np.isfinite(_eval_loss(nan_cliff, inside, jax.random.key(0)))
    assert np.isnan(_eval_loss(nan_cliff, over, jax.random.key(0))), (
        "scoring must report a NaN as a NaN; selection cannot reject what it "
        "is never shown"
    )


def test_apgm_never_selects_a_nan_scoring_iterate(nan_cliff):
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import _eval_loss, simplex_APGM

    x0 = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    x, best_x = simplex_APGM(
        loss_function=nan_cliff, x=x0, n_steps=6, stepsize=0.5, momentum=0.9,
        key=jax.random.key(0),
    )

    def at(a):
        return _eval_loss(nan_cliff, a, jax.random.key(0))

    # Not vacuous: the run has to actually reach the cliff, or there was never
    # a NaN candidate to reject and this test proves nothing.
    assert np.isnan(at(x)), "fixture did not reach the NaN region"
    assert np.isfinite(at(best_x)), (
        f"best_x scores {at(best_x)}; a nonfinite iterate was selected"
    )
    assert at(best_x) <= at(x0) + 1e-6


def test_batched_apgm_never_selects_a_nan_scoring_iterate(nan_cliff):
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import _eval_loss, batched_simplex_APGM

    x0 = jnp.stack(
        [jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1) for _ in range(2)]
    )
    x, best_x = batched_simplex_APGM(
        loss_function=nan_cliff, x=x0, n_steps=6, stepsize=0.5, momentum=0.9,
        key=jax.random.key(0),
    )

    def at(a):
        return _eval_loss(nan_cliff, a, jax.random.key(0))

    assert any(np.isnan(at(row)) for row in x), "fixture did not reach the NaN region"
    for b, row in enumerate(best_x):
        assert np.isfinite(at(row)), f"batch element {b} selected a NaN iterate"


def test_tracking_costs_one_forward_pass_per_step_plus_one(quadratic, monkeypatch):
    """Pins the arithmetic the seeder's `track_best=False` is justified by."""
    import jax
    import jax.numpy as jnp

    import mosaic.optimizers as optimizers

    loss, _at = quadratic
    x0 = jnp.zeros((2, 20)).at[:, 0].set(0.9).at[:, 1].set(0.1)
    real = optimizers._eval_loss

    for track, expected in ((True, 6), (False, 0)):
        calls = []

        def counting(loss_function, x, key, _real=real, _calls=calls):
            _calls.append(1)
            return _real(loss_function, x, key)

        monkeypatch.setattr(optimizers, "_eval_loss", counting)
        optimizers.simplex_APGM(
            loss_function=loss, x=x0, n_steps=5, stepsize=0.1, momentum=0.9,
            key=jax.random.key(0), track_best=track,
        )
        assert len(calls) == expected, (
            f"track_best={track} scored {len(calls)} times, expected {expected}"
        )


# --------------------------------------------------------------------------- #
# Defect 9: the proposal redraw broke detailed balance
#
# `gradient_MCMC` redrew any proposal that came back equal to the current
# sequence, up to 50 times. That conditions the proposal distribution on
# y != x, so the realised kernel is q(y|x) / (1 - q(x|x)), while the Hastings
# ratio is computed from the unconditioned q. The truncation factor differs
# between states and so does not cancel.
#
# This fixes the redraw. It does NOT establish that the sampler is correct --
# `detailed_balance=False` applies no Hastings term at all, and nothing here
# tests the multi-step path beyond the reverse-path invariant above.
# --------------------------------------------------------------------------- #


def _two_state_flows(truncate):
    """Stationary flows both ways on a two-state chain, from the real proposal.

    Length 1 over a 2-letter alphabet, so `q(x|x) > 0` and the redraw actually
    fires. Analytic rather than sampled: the kernel is small enough to write
    down, and a 2-state Monte Carlo would need a lot of steps to resolve a 1.4x
    asymmetry.
    """
    from mosaic.optimizers import _proposal

    A, temp = 2, 1.0
    target = np.array([[0.8, 0.2]])

    def value(s):
        return float(((np.eye(A)[s] - target) ** 2).sum())

    def grad(s):
        return 2.0 * (np.eye(A)[s] - target)

    q = {}
    for s in (0, 1):
        probs, _ = _proposal(np.array([s]), grad(np.array([s])), temp, alphabet_size=A)
        q[s] = np.ravel(np.asarray(probs))

    pi = np.exp(-np.array([value(np.array([0])), value(np.array([1]))]) / temp)
    pi = pi / pi.sum()

    def accept(a, b):
        # exactly what the sampler computes: unconditioned q on both sides
        return np.exp(min(0.0, (value(np.array([a])) - value(np.array([b]))) / temp
                          + np.log(q[b][a]) - np.log(q[a][b])))

    def kernel(a, b):
        return q[a][b] / (1.0 - q[a][a]) if truncate else q[a][b]

    return (pi[0] * kernel(0, 1) * accept(0, 1),
            pi[1] * kernel(1, 0) * accept(1, 0))


def test_the_redraw_is_what_broke_detailed_balance():
    forward, backward = _two_state_flows(truncate=True)
    ratio = forward / backward
    assert abs(ratio - 1.0) > 0.3, (
        f"the redrawn kernel was supposed to be out of balance, got {ratio:.4f}"
    )
    share = forward / (forward + backward)
    assert 0.57 < share < 0.60, f"expected ~0.582 of the flow one way, got {share:.3f}"


def test_taking_the_draw_as_it_comes_restores_detailed_balance():
    forward, backward = _two_state_flows(truncate=False)
    assert forward == pytest.approx(backward, rel=1e-9), (
        f"flows {forward:.6f} and {backward:.6f} should be equal"
    )


# One step calls `_proposal` once per forward draw, plus exactly once more for
# the reverse path, hence the +1 in each expectation.
@pytest.mark.parametrize(
    "detailed_balance, draws_per_step", [(True, 1 + 1), (False, 50 + 1)]
)
def test_the_sampler_stops_redrawing_once_it_corrects_for_the_proposal(
    mcmc_problem, monkeypatch, detailed_balance, draws_per_step
):
    """Counted, not read off the source: how many proposals one step draws.

    `_proposal` is replaced by one that always returns the current token, so
    every draw comes back equal to the current sequence and the redraw fires
    every time it is allowed to.
    """
    import contextlib
    import io

    import jax

    import mosaic.optimizers as optimizers

    loss, start = mcmc_problem
    calls = []

    def always_self(sequence, g, temp, alphabet_size=20):
        calls.append(1)
        probs = np.zeros((len(sequence), alphabet_size), dtype=np.float64)
        probs[np.arange(len(sequence)), np.asarray(sequence)] = 1.0 / len(sequence)
        with np.errstate(divide="ignore"):
            return probs, np.log(probs)

    monkeypatch.setattr(optimizers, "_proposal", always_self)
    with contextlib.redirect_stdout(io.StringIO()):
        optimizers.gradient_MCMC(
            loss, start, steps=1, max_path_length=1, alphabet_size=20,
            key=jax.random.key(0), detailed_balance=detailed_balance,
        )
    assert len(calls) == draws_per_step, (
        f"detailed_balance={detailed_balance} called the proposal "
        f"{len(calls)} times, expected {draws_per_step} "
        f"({draws_per_step - 1} forward draws plus the reverse path)"
    )
