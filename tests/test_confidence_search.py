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


@pytest.fixture
def export_prediction():
    from mosaic.losses.structure_prediction import StructureModelOutput

    coords = np.arange(4 * 37 * 3, dtype=np.float32).reshape(4, 37, 3) / 10
    mask = np.zeros((4, 37), dtype=np.float32)
    mask[:, :3] = 1
    mask[1, 32] = 1  # ARG CZ: export includes side chains, not just CA atoms.
    return StructureModelOutput(
        distogram_logits=np.zeros((4, 4, 2)),
        distogram_bins=np.array([1.0, 2.0]),
        plddt=np.array([0.7, 0.8, 0.9, 0.6]),
        pae=np.arange(16, dtype=np.float32).reshape(4, 4),
        pae_logits=np.zeros((4, 4, 2)),
        pae_bins=np.array([1.0, 2.0]),
        structure_coordinates=coords.reshape(-1, 3),
        backbone_coordinates=coords[:, [0, 1, 2, 4]],
        full_sequence=np.eye(20)[[0, 1, 7, 2]],  # AR + GN in mosaic ordering.
        asym_id=np.array([0, 0, 1, 1]),
        residue_idx=np.array([1, 2, 1, 2]),
        atom37_coords=coords,
        atom37_mask=mask,
    )


def test_exact_scoring_structures_and_confidence_are_exported(
    runner, export_prediction, tmp_path
):
    import csv
    import equinox as eqx
    import gemmi
    import jax

    # Exercise the JIT boundary used in the real runner without loading weights.
    compact = eqx.filter_jit(runner.compact_prediction)(export_prediction)
    output = jax.tree.map(np.asarray, compact)
    assert output.pae_logits is None and output.distogram_logits is None
    artifacts = runner.SearchOutputs(tmp_path)
    assert artifacts.candidate_id([0, 1]) == 0
    assert artifacts.candidate_id([0, 1]) == 0
    assert artifacts.candidate_id([0, 2]) == 1
    metrics = runner.confidence_metrics(
        output.pae, output.backbone_coordinates[:, 1], 2, 12.0, 12.0
    ) | {"iptm": 0.8}
    for seed in [3, 9]:
        row = artifacts.save_prediction(0, seed, "AR", "GN", output, metrics)
        st = gemmi.read_structure(str(tmp_path / row["structure_file"]))
        assert [chain.name for chain in st[0]] == ["A", "B"]
        cif = gemmi.read_structure(str(tmp_path / row["cif_file"]))
        assert [chain.name for chain in cif[0]] == ["A", "B"]
        np.testing.assert_allclose(
            cif[0]["A"][1]["CZ"][0].pos.tolist(),
            output.atom37_coords[1, 32],
            atol=0.001,
        )
        assert [[res.name for res in chain] for chain in st[0]] == [
            ["ALA", "ARG"],
            ["GLY", "ASN"],
        ]
        assert [res.seqid.num for res in st[0]["B"]] == [1, 2]
        assert st[0]["A"][1]["CZ"][0].b_iso == pytest.approx(80.0)
        np.testing.assert_allclose(
            st[0]["A"][1]["CZ"][0].pos.tolist(), output.atom37_coords[1, 32], atol=0.001
        )
        with np.load(tmp_path / row["confidence_file"]) as arrays:
            np.testing.assert_array_equal(arrays["pae"], output.pae)
            np.testing.assert_array_equal(arrays["atom37_coords"], output.atom37_coords)
            assert arrays["sequence"].item() == "ARGN"
    rows = list(csv.DictReader((tmp_path / "tables/predictions.csv").open()))
    assert [r["selection_seed"] for r in rows] == ["3", "9"]
    assert all(float(r["ipsae_min"]) == metrics["ipsae_min"] for r in rows)
    assert all(r["binder_chain"] == "A" and r["target_chain"] == "B" for r in rows)
    assert artifacts.copy_best(0, [3, 9]) == [
        "best/seed_3.pdb",
        "best/seed_3.cif",
        "best/seed_9.pdb",
        "best/seed_9.cif",
    ]
    assert (tmp_path / "best/seed_3.pdb").read_bytes() == (
        tmp_path / rows[0]["structure_file"]
    ).read_bytes()
    with pytest.raises(FileExistsError):
        artifacts.save_prediction(0, 3, "AR", "GN", output, metrics)


@pytest.mark.parametrize("failure", ["sequence", "nonfinite", "ca_mismatch"])
def test_export_rejects_misidentified_or_invalid_structures(
    runner, export_prediction, tmp_path, failure
):
    import equinox as eqx

    output = export_prediction
    binder = "AR"
    if failure == "sequence":
        binder = "AA"
    elif failure == "nonfinite":
        output = eqx.tree_at(lambda p: p.plddt, output, np.full(4, np.nan))
    else:
        output = eqx.tree_at(
            lambda p: p.backbone_coordinates, output, output.backbone_coordinates + 1
        )
    artifacts = runner.SearchOutputs(tmp_path)
    with pytest.raises(ValueError):
        artifacts.save_prediction(0, 0, binder, "GN", output, {"iptm": 0.8})
    assert not (tmp_path / "tables/predictions.csv").exists()
    assert not list((tmp_path / "structures").rglob("*.pdb"))


@pytest.mark.parametrize("retention", [False, True])
def test_runner_writes_complete_layout_without_extra_predictions(
    runner, export_prediction, tmp_path, monkeypatch, retention
):
    import csv
    import json
    import equinox as eqx
    import jax.numpy as jnp
    import mosaic.models.opendde as opendde_module
    import mosaic.losses.ablang2 as ablang_module
    import p17_hallucination_search as original

    class ToyModel(eqx.Module):
        output: object

        def binder_features(self, *args, **kwargs):
            return None, None

        def model_output(self, *, PSSM, **kwargs):
            return eqx.tree_at(
                lambda p: p.full_sequence,
                self.output,
                jnp.concatenate([PSSM, jnp.asarray(self.output.full_sequence[2:])]),
            )

    def toy_loss(sequence, *, key):
        return -jnp.sum(sequence * jnp.arange(20)), {}

    monkeypatch.setattr(
        opendde_module, "OpenDDEModelAbag", lambda: ToyModel(export_prediction)
    )
    monkeypatch.setattr(ablang_module, "load_ablang2", lambda: (None, None))
    monkeypatch.setattr(original, "load_structure", lambda: (None, "AR", "GN"))
    monkeypatch.setattr(original, "CDR_RESIDUE_INDICES_1IDX", [1, 2])
    monkeypatch.setattr(original, "HOTSPOT_TARGET_RESIDUE_INDICES_1IDX", [1])
    monkeypatch.setattr(
        original, "reference_binder_target_ca_distances", lambda _: np.zeros((2, 2))
    )
    monkeypatch.setattr(
        original,
        "reference_binder_target_ca",
        lambda _: (np.zeros((2, 3)), np.zeros((2, 3))),
    )
    monkeypatch.setattr(
        original, "build_composite_losses", lambda **kw: (toy_loss, None)
    )
    out = tmp_path / "run"
    runner.main(
        [
            "--policy",
            "population",
            "--output-dir",
            str(out),
            *(["--retention-pose-margin", "3"] if retention else []),
            "--width",
            "2",
            "--max-score-calls",
            "2",
            "--max-gradient-calls",
            "1",
            "--max-proposals",
            "2",
            "--selection-seeds",
            "0",
            "1",
        ]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["retention_pose_ceiling_A"] is not None) == retention
    assert summary["best_constraint_violation"] == 0
    assert summary["full_prediction_calls"] == 4
    assert summary["full_gradient_calls"] == 1
    assert len(list((out / "structures").rglob("*.pdb"))) == 4
    predictions = list(csv.DictReader((out / "tables/predictions.csv").open()))
    candidates = list(csv.DictReader((out / "tables/candidates.csv").open()))
    assert len(predictions) == 4 and len(candidates) == 2
    assert all(float(p["binder_pose_rmsd_A"]) >= 0 for p in predictions)
    assert len(list((out / "structures").rglob("*.cif"))) == 4
    for candidate in candidates:
        matching = [p for p in predictions if p["candidate_id"] == candidate["id"]]
        assert len(matching) == 2
        assert all(p["binder_sequence"] == candidate["sequence"] for p in matching)
        assert float(candidate["score"]) == pytest.approx(
            np.mean([float(p["ipsae_min"]) for p in matching])
        )
    memory = [
        json.loads(line)
        for line in (out / "logs/memory.jsonl").read_text().splitlines()
    ]
    assert sum(e["event"] == "confidence" for e in memory) == 4
    assert len(summary["best_structure_files"]) == 4
    assert all((out / path).exists() for path in summary["best_structure_files"])
    assert (out / "logs/events.jsonl").exists() and (out / "README.md").exists()


def test_pose_rmsd_removes_global_motion_but_preserves_binder_displacement(runner):
    from p17_search_outputs import pose_metrics

    binder = np.array([[3.0, 1.0, 0.0], [4.0, 0.0, 1.0], [3.0, 0.0, 2.0]])
    target = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    prediction = np.concatenate([binder, target]) @ rotation + np.array([7.0, 8.0, 9.0])
    metrics = pose_metrics(prediction, binder, target)
    assert all(v == pytest.approx(0.0, abs=1e-10) for v in metrics.values())
    prediction[:3] += np.array([0.0, 5.0, 0.0])
    metrics = pose_metrics(prediction, binder, target)
    assert metrics["binder_pose_rmsd_A"] == pytest.approx(5.0)
    assert metrics["binder_internal_rmsd_A"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["target_aligned_rmsd_A"] == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError):
        pose_metrics(prediction[:-1], binder, target)


def test_winner_loader_deduplicates_and_preserves_source_ids(runner, tmp_path):
    import json
    import tarfile
    from p17_rescore_winners import load_candidates

    batch = tmp_path / "batch"
    config = dict(
        binder_sequence="AR",
        target_sequence="GN",
        checkpoint="opendde_abag.pt",
        recycling_steps=4,
        designable_positions_0idx=[0],
        config={"edit_budget": 1},
        arguments=dict(
            sampling_steps=8, pae_cutoff=12.0, distance_cutoff=12.0, selection_seeds=[0]
        ),
    )
    for policy in ["independent", "population"]:
        directory = batch / "runs" / policy / "seed_0"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "summary.json").write_text(
            json.dumps(dict(best_sequence="VR", best_id=7, best_score=0.2))
        )
    candidates, links, _ = load_candidates(batch)
    assert [c["sequence"] for c in candidates] == ["AR", "VR"]
    assert [r["candidate_id"] for r in links] == [1, 1]
    assert all(r["source_candidate_id"] == 7 for r in links)
    archive = tmp_path / "batch.tar.gz"
    with tarfile.open(archive, "w:gz") as f:
        f.add(batch, arcname="batch")
    assert load_candidates(archive)[0] == candidates
    (directory / "summary.json").write_text(
        json.dumps(dict(best_sequence="VA", best_id=7, best_score=0.2))
    )
    with pytest.raises(ValueError, match="constraints"):
        load_candidates(batch)


def test_winner_rescoring_exports_scored_pose_without_optimization(
    runner, export_prediction, tmp_path, monkeypatch
):
    import csv
    import json
    import equinox as eqx
    import jax.numpy as jnp
    import mosaic.models.opendde as model_module
    import p17_hallucination_search as original
    import p17_rescore_winners as rescoring

    class ToyModel(eqx.Module):
        output: object

        def binder_features(self, *args, **kwargs):
            return None, None

        def model_output(self, *, PSSM, **kwargs):
            return eqx.tree_at(
                lambda p: p.full_sequence,
                self.output,
                jnp.concatenate([PSSM, jnp.asarray(self.output.full_sequence[2:])]),
            )

    baseline = dict(
        binder_sequence="AR",
        target_sequence="GN",
        recycling_steps=4,
        checkpoint="opendde_abag.pt",
        arguments=dict(
            sampling_steps=8, selection_seeds=[0], pae_cutoff=12.0, distance_cutoff=12.0
        ),
    )
    candidates = [
        dict(candidate_id=i, sequence=s, is_wt=i == 0)
        for i, s in enumerate(["AR", "VR"])
    ]
    links = [
        dict(
            candidate_id=1,
            source_run="population_seed0",
            source_candidate_id=7,
            original_score=0.2,
        )
    ]
    monkeypatch.setattr(
        rescoring, "load_candidates", lambda _: (candidates, links, baseline)
    )
    monkeypatch.setattr(
        model_module, "OpenDDEModelAbag", lambda: ToyModel(export_prediction)
    )
    monkeypatch.setattr(original, "load_structure", lambda: (None, "AR", "GN"))
    ca = export_prediction.backbone_coordinates[:, 1]
    monkeypatch.setattr(
        original, "reference_binder_target_ca", lambda _: (ca[:2], ca[2:])
    )
    reference = tmp_path / "ref.pdb"
    reference.write_text("HEADER    TEST\n")
    monkeypatch.setattr(original, "COMPLEX_PDB", reference)
    out = tmp_path / "rescore"
    rescoring.main(
        ["--input", str(tmp_path), "--output-dir", str(out), "--seeds", "0", "1"]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["prediction_calls"] == 4
    assert len(list((out / "structures").rglob("*.cif"))) == 4
    rows = list(csv.DictReader((out / "tables/predictions.csv").open()))
    assert len(rows) == 4
    assert all(
        float(row["binder_pose_rmsd_A"]) == pytest.approx(0.0, abs=1e-6) for row in rows
    )
    assert all((out / row["cif_file"]).exists() for row in rows)
    assert (out / "reference.pdb").read_bytes() == reference.read_bytes()


def test_rescore_shards_cover_each_candidate_once_and_keep_ids(runner):
    from p17_rescore_winners import select_shard

    candidates = [dict(candidate_id=i, sequence=str(i)) for i in range(9)]
    shards = [select_shard(candidates, 8, i) for i in range(8)]
    assert [r["candidate_id"] for r in shards[0]] == [0, 8]
    assert sorted(r["candidate_id"] for shard in shards for r in shard) == list(
        range(9)
    )
    assert select_shard(candidates, 10, 9) == []
    for count, index in [(0, 0), (8, -1), (8, 8)]:
        with pytest.raises(ValueError):
            select_shard(candidates, count, index)


def test_merge_pose_shards_rewrites_artifact_paths_and_checks_completion(
    runner, tmp_path
):
    import csv
    import json
    from p17_rescore_winners import merge_shard_tables

    for i in range(2):
        directory = tmp_path / f"shard_{i}"
        (directory / "tables").mkdir(parents=True)
        row = dict(
            candidate_id=i,
            selection_seed=0,
            structure_file="seed_0.pdb",
            cif_file="seed_0.cif",
            confidence_file="seed_0.npz",
        )
        for field in ("structure_file", "cif_file", "confidence_file"):
            (directory / row[field]).touch()
        tables = {
            "predictions.csv": [row],
            "candidates.csv": [dict(candidate_id=i, sequence=f"seq{i}")],
            "source_runs.csv": [dict(candidate_id=i, source_run=f"run{i}")],
        }
        for name, rows in tables.items():
            with (directory / "tables" / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (directory / "summary.json").write_text(
            json.dumps(dict(completed=True, sequences=1, prediction_calls=1))
        )
    missing = tmp_path / "shard_1/seed_0.cif"
    missing.unlink()
    with pytest.raises(ValueError, match="missing prediction artifact"):
        merge_shard_tables(tmp_path, 2)
    assert not (tmp_path / "tables").exists()
    missing.touch()
    merge_shard_tables(tmp_path, 2)
    rows = list(csv.DictReader((tmp_path / "tables/predictions.csv").open()))
    assert [row["candidate_id"] for row in rows] == ["0", "1"]
    assert [row["cif_file"] for row in rows] == [
        "shard_0/seed_0.cif",
        "shard_1/seed_0.cif",
    ]
    assert all((tmp_path / row["confidence_file"]).exists() for row in rows)
    assert json.loads((tmp_path / "summary.json").read_text())["prediction_calls"] == 2


@pytest.mark.parametrize("policy,width", [("independent", 1), ("population", 1), ("population", 2)])
def test_constraint_cannot_be_bypassed_by_temperature_or_duplicate_filling(policy, width):
    config = SearchConfig(policy=policy, width=width, alphabet_size=2,
                          max_score_calls=2, acceptance_temperature=1e9)
    result, events = run_toy(config, wt=np.array([0]), confidence=lambda seq: ConfidenceScore(
        float(seq[0]), constraint_violation=float(seq[0]),
    ))
    assert all(c.sequence == (0,) for c in result.active)
    assert result.best.sequence == (0,)
    assert not next(e for e in events if e["event"] == "proposal")["accepted"]


@pytest.mark.parametrize("policy", ["independent", "population"])
def test_lower_violation_beats_higher_confidence_in_acceptance_and_archive(policy):
    result, _ = run_toy(
        SearchConfig(policy=policy, width=1, alphabet_size=2, max_score_calls=2,
                     acceptance_temperature=0), wt=np.array([0]),
        confidence=lambda seq: ConfidenceScore(-10. * seq[0], constraint_violation=2. - seq[0]),
    )
    assert result.active[0].sequence == result.best.sequence == (1,)
    assert result.best.score == -10


@pytest.mark.parametrize("violation", [-1, float("nan"), float("inf")])
def test_invalid_constraint_rejected_before_search(violation):
    with pytest.raises(ValueError, match="constraint violation"):
        run_toy(SearchConfig(), confidence=lambda seq: ConfidenceScore(1., constraint_violation=violation))


def test_population_distance_ties_challenge_worse_constraint_first():
    from mosaic.search import Candidate, _competitor
    active = [Candidate(0, (1, 0), .1, {}, 0.), Candidate(1, (0, 1), .9, {}, 1.)]
    candidate = Candidate(2, (0, 0), .5, {}, 0.)
    assert _competitor(active, candidate, 0, "population") == 1


def test_pose_sampling_uses_worst_seed_without_extra_calls(runner):
    binder = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    target = binder + 5.
    calls = []
    def predict(seq, seed):
        calls.append(seed)
        return np.zeros((6, 6)), np.concatenate([binder + [seed, 0, 0], target]), .5
    score = runner.score_prediction_samples(predict, np.zeros(3), [1, 3], 12., 12.,
                                            references=(binder, target))
    assert calls == [1, 3]
    assert score.metrics["worst_pose_rmsd_A"] == pytest.approx(3.)
    assert score.metrics["binder_pose_rmsd_A"] == pytest.approx(2.)
    assert score.value == pytest.approx(1.)


def test_pose_toggle_changes_real_composite_gradient_and_preserves_aux_keys(runner, monkeypatch):
    from types import SimpleNamespace
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mosaic.common import LossTerm
    from mosaic.optimizers import _ranking_leaf
    import p17_hallucination_search as original

    reference = jnp.asarray(np.random.default_rng(2).normal(size=(8, 3)))
    class ZeroLoss(LossTerm):
        def __call__(self, sequence, key, **kwargs):
            return jnp.sum(sequence) * 0., {"key_marker": jax.random.uniform(key)}
    class ToyLoss(LossTerm):
        loss: object
        def __call__(self, sequence, key):
            ca = reference.at[:4, 0].add(2. + .2 * sequence[0, 0])
            output = SimpleNamespace(backbone_coordinates=jnp.repeat(ca[:, None], 4, axis=1))
            return self.loss(sequence=sequence, output=output, key=key)
    class Model:
        def build_loss(self, **kwargs):
            return ToyLoss(kwargs["loss"])
    for name in ("BinderTargetContact", "Ablang2PseudoLikelihood"):
        monkeypatch.setattr(original, name, lambda *a, **kw: ZeroLoss())
    kwargs = dict(opendde=Model(), features=None, ablang2_model=None, ablang2_tokenizer=None,
                  reference_distances=np.zeros((4, 4)), reference_binder_ca=reference[:4],
                  reference_target_ca=reference[4:], binder_seq="AAAA", designable_idx=np.arange(4),
                  epitope_idx=np.array([0]), edit_budget=5, stop_grad_ablang2=False,
                  opendde_path="full", pose_tolerance=0., opendde_sampling_steps=8,
                  opendde_num_samples=1, confidence_loss=ZeroLoss())
    evaluations = []
    for weight in (1., 0.):
        loss, _ = original.build_composite_losses(**kwargs, pose_weight=weight)
        evaluations.append(eqx.filter_value_and_grad(loss, has_aux=True)(
            jax.nn.one_hot(jnp.zeros(4, dtype=int), 20), key=jax.random.key(0)))
    ((value_on, aux_on), grad_on), ((value_off, aux_off), grad_off) = evaluations
    assert float(value_on) > float(value_off)
    assert np.linalg.norm(grad_on) > 0
    np.testing.assert_allclose(grad_off, 0., atol=1e-7)
    for key in ("binder_pose_rmsd", "pose_target_fit_rmsd", "key_marker"):
        assert float(_ranking_leaf(aux_on, key)) == pytest.approx(float(_ranking_leaf(aux_off, key)))
    # Full auxiliary trees (including downstream random-key markers) agree.
    for a, b in zip(jax.tree.leaves(aux_on), jax.tree.leaves(aux_off)):
        np.testing.assert_array_equal(a, b)
