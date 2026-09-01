"""Unit tests for the OpenDDE guidance/refold config wiring in
src/mosaic/workflows/boltzgen_vhh_guided.py (see
/home/yfeng17/.claude/plans/eager-jingling-pixel.md for the full feature).

OpenDDE is a selectable alternative to Boltz2 for both in-loop guidance
(build_opendde_guidance_loss) and post-refold ranking
(refold_pareto_with_opendde) -- these tests cover only the config-level
gating and mutual-exclusivity guard, which need no GPU, checkpoint, or
jopendde install. Everything that touches an actual OpenDDEModel is out of
scope here (see examples/opendde_smoke_test.py for that, meant to run on a
machine with jopendde installed).
"""
from pathlib import Path

import pytest

from mosaic.legacy.boltzgen_vhh_guided import (
    VHHDesignConfig,
    opendde_refold_ignores_template_config,
    run,
    uses_boltz2_guidance,
    uses_opendde_guidance,
)


def _minimal_cfg(**overrides) -> VHHDesignConfig:
    cfg = VHHDesignConfig(
        complex_cif_path=Path("/nonexistent/complex.cif"),
        binder_chain_id="H",
        target_chain_ids=["A"],
        cdr_residue_indices=[1],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_uses_opendde_guidance_false_by_default():
    cfg = _minimal_cfg()
    assert uses_opendde_guidance(cfg) is False


def test_uses_opendde_guidance_true_with_iptm_weight():
    cfg = _minimal_cfg(weight_opendde_iptm=0.5)
    assert uses_opendde_guidance(cfg) is True


def test_uses_opendde_guidance_true_with_contact_weight():
    cfg = _minimal_cfg(weight_opendde_contact=0.5)
    assert uses_opendde_guidance(cfg) is True


def test_run_rejects_both_boltz2_and_opendde_guidance_weights():
    # Regression guard: L_bind has exactly one anchor slot (see GuidanceLosses
    # docstring); setting both backends' weights is ambiguous about which one
    # should own it, so this must fail fast rather than silently pick one.
    cfg = _minimal_cfg(weight_boltz2_iptm=0.5, weight_opendde_iptm=0.5)
    with pytest.raises(ValueError, match="Boltz2.*OpenDDE|OpenDDE.*Boltz2"):
        run(cfg)


def test_mutual_exclusivity_guard_does_not_fire_for_opendde_alone():
    # Not run(cfg) here -- run() would proceed past this check into a real
    # BoltzGen load (network/checkpoint), which this test must not trigger.
    # The guard condition itself is the thing under test.
    cfg = _minimal_cfg(weight_opendde_iptm=0.5)
    assert not (uses_boltz2_guidance(cfg) and uses_opendde_guidance(cfg))


def test_mutual_exclusivity_guard_does_not_fire_for_boltz2_alone():
    cfg = _minimal_cfg(weight_boltz2_iptm=0.5)
    assert not (uses_boltz2_guidance(cfg) and uses_opendde_guidance(cfg))


def test_default_refold_backend_is_opendde():
    cfg = _minimal_cfg()
    assert cfg.refold_backend == "opendde"


def test_opendde_refold_ignores_template_config_true_by_default():
    # refold_backend defaults to "opendde" and refold_binder_template
    # defaults to True/"full" -- OpenDDE can't template, so the default
    # config implies templated refolding that silently never happens.
    cfg = _minimal_cfg()
    assert opendde_refold_ignores_template_config(cfg) is True


def test_opendde_refold_ignores_template_config_false_when_mode_none():
    cfg = _minimal_cfg(refold_binder_template_mode="none")
    assert opendde_refold_ignores_template_config(cfg) is False


def test_opendde_refold_ignores_template_config_false_when_template_off():
    cfg = _minimal_cfg(refold_binder_template=False)
    assert opendde_refold_ignores_template_config(cfg) is False


def test_run_rejects_invalid_refold_backend_before_model_loading():
    # Same fail-fast pattern as the ipSAE/mutual-exclusivity checks above:
    # validated in run() before cfg.output_dir.mkdir()/load_core_models(),
    # not only where refold_pareto_with_* would be dispatched -- the
    # nonexistent complex_cif_path here would otherwise fail with an
    # unrelated error only after paying for a real BoltzGen load.
    cfg = _minimal_cfg(refold_backend="not-a-real-backend")
    with pytest.raises(ValueError, match="refold-backend"):
        run(cfg)


def test_opendde_refold_ignores_template_config_false_for_boltz2_backend():
    cfg = _minimal_cfg(refold_backend="boltz2")
    assert opendde_refold_ignores_template_config(cfg) is False
