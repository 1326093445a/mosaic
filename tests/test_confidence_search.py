"""Weight-free behavioral checks for constrained, confidence-driven search."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mosaic.search import (
    ConfidenceScore,
    SearchConfig,
    _moves,
    _proposal_distribution,
    run_gradient_search,
)


def run_toy(
    config, *, confidence=None, gradient=None, initial=None, wt=None, mask=None
):
    wt = np.zeros(3, dtype=int) if wt is None else wt
    mask = np.ones(len(wt), dtype=bool) if mask is None else mask
    events = []
    result = run_gradient_search(
        wt=wt,
        designable_mask=mask,
        config=config,
        initial_sequences=initial,
        gradient_fn=gradient
        or (
            lambda s: (
                -float(s.sum()),
                -np.tile(np.arange(config.alphabet_size, dtype=float), (len(wt), 1)),
            )
        ),
        confidence_fn=confidence or (lambda s: ConfidenceScore(float(s.sum()))),
        on_event=events.append,
    )
    return result, events


@pytest.mark.parametrize("policy", ["independent", "population"])
def test_full_confidence_overrules_favorable_cheap_proposals(policy):
    config = SearchConfig(
        policy=policy,
        width=1,
        alphabet_size=2,
        edit_budget=1,
        max_score_calls=2,
        acceptance_temperature=0,
    )
    result, events = run_toy(
        config, wt=np.array([0]), confidence=lambda s: ConfidenceScore(-float(s.sum()))
    )
    assert result.active[0].sequence == (0,)
    assert result.best.sequence == (0,)
    proposal = next(e for e in events if e["event"] == "proposal")
    assert proposal["predicted_loss_delta"] < 0
    assert not proposal["accepted"]
    assert result.stats["score_calls"] == 2


@pytest.mark.parametrize("policy", ["independent", "population"])
def test_full_confidence_can_accept_a_worse_cheap_proposal(policy):
    config = SearchConfig(
        policy=policy,
        width=1,
        alphabet_size=2,
        edit_budget=1,
        max_score_calls=2,
        acceptance_temperature=0,
    )
    result, events = run_toy(
        config,
        wt=np.array([0]),
        gradient=lambda s: (float(s.sum()), np.array([[0.0, 1.0]])),
    )
    assert result.active[0].sequence == (1,)
    assert (
        next(e for e in events if e["event"] == "proposal")["predicted_loss_delta"] > 0
    )


def test_population_competes_locally_while_independent_competes_with_parent():
    config = SearchConfig(
        width=2,
        alphabet_size=2,
        edit_budget=1,
        max_score_calls=3,
        max_proposals=1,
        target_entropy=0.0001,
        acceptance_temperature=0,
    )
    initial = np.array([[1, 0, 0], [0, 1, 0]])
    scores = {(1, 0, 0): 0.9, (0, 1, 0): 0.1, (0, 0, 0): 0.5}

    def gradient(s):
        return 0.0, np.array([[-100.0, 0.0], [0.0, 100.0], [0.0, 100.0]])

    kwargs = dict(
        initial=initial,
        gradient=gradient,
        confidence=lambda s: ConfidenceScore(scores[tuple(s)]),
    )
    independent, _ = run_toy(config, **kwargs)
    population, events = run_toy(replace(config, policy="population"), **kwargs)
    assert [c.sequence for c in independent.active] == [(1, 0, 0), (0, 1, 0)]
    assert [c.sequence for c in population.active] == [(1, 0, 0), (0, 0, 0)]
    proposal = next(e for e in events if e["event"] == "proposal")
    assert proposal["parent_slot"] == 0
    assert proposal["competitor_slot"] == 1


@pytest.mark.parametrize("policy", ["independent", "population"])
def test_every_evaluated_candidate_respects_mask_and_wt_distance(policy):
    wt = np.array([2, 0, 1, 0])
    mask = np.array([False, True, False, True])
    config = SearchConfig(
        policy=policy,
        width=3,
        alphabet_size=3,
        edit_budget=1,
        max_score_calls=20,
        max_proposals=80,
        acceptance_temperature=1.0,
    )
    result, events = run_toy(config, wt=wt, mask=mask)
    for candidate in result.evaluated:
        sequence = np.array(candidate.sequence)
        np.testing.assert_array_equal(sequence[~mask], wt[~mask])
        assert np.count_nonzero(sequence != wt) <= 1
    assert len(result.evaluated) > 1
    assert result.stats["score_calls"] == len(result.evaluated)
    assert result.stats["cache_hits"] > 0
    assert any(e["event"] == "proposal" and e["move"][2] >= 0 for e in events)


def test_at_cap_supports_reversion_replacement_and_position_exchange():
    seq, wt = np.array([1, 0, 0]), np.zeros(3, dtype=int)
    moves, _ = _moves(seq, wt, np.array([True, True, False]), 1, np.zeros((3, 3)))
    outcomes = set()
    for pos, aa, reverted in moves:
        child = seq.copy()
        child[pos] = aa
        if reverted >= 0:
            child[reverted] = wt[reverted]
        assert np.count_nonzero(child != wt) <= 1
        outcomes.add(tuple(child))
    assert outcomes == {(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0)}


def test_population_keeps_best_active_even_when_worse_moves_are_likely():
    config = SearchConfig(
        policy="population",
        width=3,
        alphabet_size=3,
        edit_budget=2,
        max_proposals=100,
        acceptance_temperature=1000.0,
    )
    result, events = run_toy(
        config, confidence=lambda s: ConfidenceScore(float(s[0] * 3 + s[1] - s[2]))
    )
    scores = {c.id: c.score for c in result.evaluated}
    best = -np.inf
    for event in events:
        if event["event"] == "proposal":
            current = max(scores[i] for i in event["active_ids"])
            assert current >= best
            best = current
    assert max(c.score for c in result.active) == result.best.score


def test_entropy_control_is_gradient_scale_invariant_and_handles_ties():
    deltas = np.array([-3.0, -1.0, 0.0, 2.0])
    probabilities, entropy, temperature = _proposal_distribution(deltas, 0.65)
    scaled, scaled_entropy, scaled_temperature = _proposal_distribution(
        deltas * 1000, 0.65
    )
    np.testing.assert_allclose(probabilities, scaled)
    assert entropy == pytest.approx(0.65)
    assert scaled_entropy == pytest.approx(entropy)
    assert scaled_temperature == pytest.approx(temperature * 1000)
    probs, actual, _ = _proposal_distribution(np.array([0.0, 0.0, 1.0]), 0.1)
    np.testing.assert_allclose(probs, [0.5, 0.5, 0.0])
    assert actual > 0.1  # Report unattainable target honestly.
    probs, actual, _ = _proposal_distribution(np.ones(4), 0.6)
    np.testing.assert_allclose(probs, 0.25)
    assert actual == 1


def test_reproducible_search_and_no_global_numpy_rng_dependence():
    config = SearchConfig(alphabet_size=3, edit_budget=2, max_proposals=35)
    first, events1 = run_toy(config)
    np.random.seed(857)
    second, events2 = run_toy(config)
    assert first.evaluated == second.evaluated
    assert first.active == second.active

    # Timings naturally differ; proposal decisions must not.
    def decisions(events):
        return [
            {k: v for k, v in e.items() if k != "elapsed_seconds"}
            for e in events
            if e["event"] == "proposal"
        ]

    assert decisions(events1) == decisions(events2)


def test_model_call_limits_count_initialization_and_cached_revisits():
    config = SearchConfig(width=1, alphabet_size=2, edit_budget=1, max_score_calls=2)
    result, _ = run_toy(config, wt=np.array([0]))
    assert result.stop_reason == "score_budget"
    assert result.stats["score_calls"] == 2
    assert result.stats["gradient_calls"] == 1
    result, _ = run_toy(
        replace(config, max_score_calls=10, max_gradient_calls=1), wt=np.array([0])
    )
    assert result.stop_reason == "gradient_budget"
    assert result.stats["gradient_calls"] == 1
    assert result.stats["score_calls"] == 2


@pytest.mark.parametrize(
    "mask,budget", [(np.ones(3, dtype=bool), 0), (np.zeros(3, dtype=bool), 3)]
)
def test_no_feasible_moves_returns_initial_candidate(mask, budget):
    result, _ = run_toy(SearchConfig(edit_budget=budget), mask=mask)
    assert result.stop_reason == "no_feasible_moves"
    assert result.stats["score_calls"] == 1


@pytest.mark.parametrize(
    "initial,mask,budget",
    [
        ([[1, 0, 0]], [False, True, True], 1),
        ([[1, 1, 0]], [True, True, True], 1),
        ([[0.5, 0, 0]], [True, True, True], 1),
        ([[20, 0, 0]], [True, True, True], 1),
    ],
)
def test_invalid_initial_states_fail_before_model_calls(initial, mask, budget):
    def unexpected(_):
        pytest.fail("invalid input triggered a model call")

    with pytest.raises(ValueError):
        run_toy(
            SearchConfig(width=1, edit_budget=budget),
            initial=np.array(initial),
            mask=np.array(mask),
            confidence=unexpected,
            gradient=unexpected,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_confidence_fails_loudly(bad_value):
    with pytest.raises(ValueError, match="nonfinite"):
        run_toy(SearchConfig(), confidence=lambda s: ConfidenceScore(bad_value))


def test_nonfinite_gradients_are_not_silently_sanitized():
    with pytest.raises(ValueError, match="nonfinite"):
        run_toy(
            SearchConfig(alphabet_size=2),
            gradient=lambda s: (0.0, np.full((3, 2), np.nan)),
        )


@pytest.fixture
def runner(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "p17_confidence_runner_test", examples / "p17_confidence_search.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_preserves_pae_directionality_and_distance_is_diagnostic_only(runner):
    # A perfect binder->target direction cannot hide a poor reverse direction.
    pae = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    ca = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    near = runner.confidence_metrics(pae, ca, 1, 12.0, 12.0)
    far = runner.confidence_metrics(pae, ca * 100, 1, 12.0, 12.0)
    assert near["bt_ipsae"] == 1.0
    assert near["tb_ipsae"] == pytest.approx(1 / 82)
    assert near["ipsae_min"] == near["tb_ipsae"] == far["ipsae_min"]
    assert near["interface_binder_count"] == 1
    assert far["interface_binder_count"] == 0
    no_interface = runner.confidence_metrics(np.full((3, 3), 20.0), ca, 1, 12.0, 12.0)
    assert no_interface["ipsae_min"] == 0.0


def test_runner_rejects_invalid_args_before_loading_models_or_creating_output(
    runner, tmp_path
):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--policy",
                "population",
                "--output-dir",
                str(output),
                "--selection-seeds",
                "0",
                "0",
            ]
        )
    assert exc.value.code == 2
    assert not output.exists()


def test_selection_averages_seed_minima_not_best_seed_or_minimum_of_means(runner):
    calls = []

    def predict(sequence, seed):
        calls.append((tuple(sequence), seed))
        pae = np.array([[0.0, 0.0], [9.0, 0.0]])
        return (pae if seed == 7 else pae.T), np.zeros((2, 3)), 0.5

    score = runner.score_prediction_samples(predict, np.array([0]), [7, 9], 12.0, 12.0)
    assert calls == [((0,), 7), ((0,), 9)]
    assert score.value == pytest.approx(1 / 82)
    assert score.metrics["bt_ipsae"] > score.value
    assert score.metrics["tb_ipsae"] > score.value
    assert score.metrics["seed_7_bt_ipsae"] == 1.0
    assert score.metrics["seed_9_tb_ipsae"] == 1.0


def test_device_memory_stats_reports_real_or_explicitly_unsupported(runner):
    # No model loading required -- this only exercises jax.devices() itself,
    # so it runs the same on CPU-only CI as it does on the GPU cluster node.
    stats = runner.device_memory_stats()
    assert stats, "expected at least one device"
    for device_name, device_stats in stats.items():
        assert isinstance(device_name, str) and device_name
        assert isinstance(device_stats, dict)
        assert "supported" in device_stats
        if device_stats["supported"]:
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                assert key in device_stats
                assert isinstance(device_stats[key], int)
                assert device_stats[key] >= 0
        else:
            assert "note" in device_stats


def test_device_memory_stats_peak_is_monotonic_across_calls(runner):
    # peak_bytes_in_use is a running maximum since process start (no per-call
    # reset exists) -- confirm it behaves that way rather than silently
    # assuming it, since this is the exact caveat callers must know about
    # before reading memory.jsonl.
    import jax

    if not jax.devices()[0].memory_stats():
        pytest.skip("backend does not report memory stats")
    before = runner.device_memory_stats()
    jax.block_until_ready(jax.numpy.ones((256, 256)) @ jax.numpy.ones((256, 256)))
    after = runner.device_memory_stats()
    for name in before:
        if before[name]["supported"] and after[name]["supported"]:
            assert after[name]["peak_bytes_in_use"] >= before[name]["peak_bytes_in_use"]


@pytest.mark.parametrize("weights", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
def test_each_confidence_component_contributes_a_sequence_gradient(runner, weights):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp

    terms = runner.build_confidence_terms(*weights)
    bins = jnp.array([0.5, 8.0, 20.0])

    def loss(sequence):
        # Toy differentiable predictor: sequence changes low-error PAE logits.
        logits = jnp.broadcast_to(
            jnp.stack([sequence[0, 0], jnp.array(0.0), jnp.array(0.0)]), (4, 4, 3)
        )
        output = SimpleNamespace(
            pae_logits=logits,
            pae_bins=bins,
            pae=jax.nn.softmax(logits) @ bins,
            full_sequence=jnp.zeros((4, 20)),
        )
        return terms(sequence=sequence, output=output, key=jax.random.key(0))[0]

    sequence = jnp.zeros((2, 20))
    gradient = jax.grad(loss)(sequence)
    assert np.all(np.isfinite(gradient))
    assert float(gradient[0, 0]) < 0  # More low-PAE weight lowers every objective.


def test_full_confidence_terms_reach_the_shared_proposal_gradient(runner, monkeypatch):
    from types import SimpleNamespace
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mosaic.common import LossTerm
    from mosaic.optimizers import _ranking_leaf
    import p17_hallucination_search as original

    class ZeroLoss(LossTerm):
        def __call__(self, sequence, key, **kwargs):
            return jnp.sum(sequence) * 0.0, {}

    class ToyFullLoss(LossTerm):
        structural_loss: object

        def __call__(self, sequence, key):
            bins = jnp.array([0.5, 8.0, 20.0])
            logits = jnp.broadcast_to(
                jnp.stack([sequence[0, 0], jnp.array(0.0), jnp.array(0.0)]), (4, 4, 3)
            )
            output = SimpleNamespace(
                pae_logits=logits,
                pae_bins=bins,
                pae=jax.nn.softmax(logits) @ bins,
                full_sequence=jnp.zeros((4, 20)),
            )
            return self.structural_loss(sequence=sequence, output=output, key=key)

    class ToyModel:
        def __init__(self):
            self.calls = []

        def build_loss(self, **kwargs):
            self.calls.append(kwargs)
            return ToyFullLoss(kwargs["loss"])

        def build_distogram_only_loss(self, **kwargs):
            return ZeroLoss()

    # Isolate the newly wired confidence signal while retaining the real loss
    # composition, gradient clipping, edit penalty and JAX differentiation.
    for name in (
        "BinderTargetContact",
        "BinderPoseRMSD",
        "BinderPoseDistogramDrift",
        "Ablang2PseudoLikelihood",
    ):
        monkeypatch.setattr(original, name, lambda *a, **kw: ZeroLoss())
    model = ToyModel()
    kwargs = dict(
        opendde=model,
        features=None,
        ablang2_model=None,
        ablang2_tokenizer=None,
        reference_distances=np.zeros((2, 2)),
        reference_binder_ca=np.zeros((2, 3)),
        reference_target_ca=np.zeros((2, 3)),
        binder_seq="AR",
        designable_idx=np.array([0, 1]),
        epitope_idx=np.array([0]),
        edit_budget=2,
        stop_grad_ablang2=False,
        opendde_path="full",
        pose_tolerance=0.0,
        opendde_sampling_steps=8,
        opendde_num_samples=1,
    )
    terms = runner.build_confidence_terms(0.025, 0.05, 0.025)
    enabled, _ = original.build_composite_losses(**kwargs, confidence_loss=terms)
    legacy, _ = original.build_composite_losses(**kwargs)
    sequence = jax.nn.one_hot(jnp.array([0, 1]), 20)
    (value, aux), gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(enabled, has_aux=True)
    )(sequence, key=jax.random.key(0))
    (_, _), old_gradient = eqx.filter_value_and_grad(legacy, has_aux=True)(
        sequence, key=jax.random.key(0)
    )
    assert np.isfinite(value)
    assert not np.allclose(gradient, old_gradient)
    for name in ("iptm", "bt_pae", "tb_pae", "pTMEnergy"):
        assert _ranking_leaf(aux, name) is not None
    assert model.calls[0]["sampling_steps"] == 8
    with pytest.raises(ValueError, match="requires"):
        original.build_composite_losses(
            **{**kwargs, "opendde_path": "distogram"}, confidence_loss=terms
        )
    proxy, _ = original.build_composite_losses(
        **{**kwargs, "opendde_path": "distogram"}
    )
    assert np.isfinite(proxy(sequence, key=jax.random.key(0))[0])


def test_runner_defaults_to_full_guidance_with_recordable_weights(runner):
    args = runner.build_parser().parse_args(
        ["--policy", "population", "--output-dir", "unused"]
    )
    assert args.proposal_path == "full"
    assert (args.weight_iptm, args.weight_interface_pae, args.weight_ptm_energy) == (
        0.025,
        0.05,
        0.025,
    )
    assert args.pose_rmsd_tolerance == 0.0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--weight-iptm", "-1"),
        ("--weight-interface-pae", "nan"),
        ("--weight-ptm-energy", "inf"),
        ("--pose-rmsd-tolerance", "-1"),
    ],
)
def test_invalid_full_objective_settings_fail_before_output_creation(
    runner, tmp_path, flag, value
):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        runner.main(
            ["--policy", "population", "--output-dir", str(output), flag, value]
        )
    assert not output.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_memory_record_covers_input_failure_and_success(runner, monkeypatch, fail):
    records = []
    monkeypatch.setattr(
        runner, "device_memory_stats", lambda: {"cpu": {"supported": False}}
    )

    def log(kind, before, after, **fields):
        records.append(dict(kind=kind, **fields))

    def call():
        with runner.record_memory_call(log, "gradient", np.array([1, 2]), seed=7):
            if fail:
                raise RuntimeError("input allocation failed")

    if fail:
        with pytest.raises(RuntimeError, match="input allocation"):
            call()
    else:
        call()
    assert len(records) == 1
    assert records[0]["sequence"] == [1, 2]
    assert records[0]["seed"] == 7
    assert records[0]["status"] == ("error" if fail else "ok")
    assert (records[0]["error"] is not None) == fail


def test_memory_logging_failure_does_not_replace_model_error(runner, monkeypatch):
    monkeypatch.setattr(runner, "device_memory_stats", lambda: {})

    def broken_log(*args, **kwargs):
        raise OSError("disk full")

    with pytest.raises(RuntimeError, match="original model failure"):
        with runner.record_memory_call(broken_log, "gradient", np.array([0])):
            raise RuntimeError("original model failure")
