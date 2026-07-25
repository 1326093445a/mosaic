"""Unit tests for VHHDesignConfig validation in
src/mosaic/workflows/boltzgen_vhh_guided.py -- specifically that a positive
--weight-boltz2-ipsae is rejected outright rather than silently ignored.

ipSAE's PAE-cutoff masking makes it unsuitable as an in-loop gradient target
(see docs/guidance_implementation_todo.md); uses_boltz2_guidance() therefore
never triggers Boltz2 guidance on weight_boltz2_ipsae alone, which means the
ValueError inside build_boltz2_guidance_loss is unreachable when ipSAE is the
only positive Boltz2 weight. run() must reject it directly instead.

This module imports mosaic.workflows.boltzgen_vhh_guided, which pulls in
heavier dependencies than the plain boltzgen.py primitives -- no GPU or model
checkpoint is required, but the import itself is slower than
test_guidance_controller.py.
"""
from pathlib import Path

import pytest

from mosaic.workflows.boltzgen_vhh_guided import (
    VHHDesignConfig,
    guidance_anchor_is_empty_edit_budget,
    run,
    uses_boltz2_guidance,
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


def test_uses_boltz2_guidance_excludes_ipsae_alone():
    cfg = _minimal_cfg(weight_boltz2_ipsae=1.0)
    assert uses_boltz2_guidance(cfg) is False


def test_run_rejects_positive_ipsae_weight_even_when_only_boltz2_weight_set():
    # Regression test: uses_boltz2_guidance() excluding weight_boltz2_ipsae
    # previously meant build_boltz2_guidance_loss (and its ValueError guard)
    # was never called when ipSAE was the only positive Boltz2 weight, so
    # the weight was silently ignored instead of rejected. run() must catch
    # this before any model loading or file I/O -- the cif path here doesn't
    # even exist, so reaching past this check would fail with a different
    # error (file not found), not silently succeed.
    cfg = _minimal_cfg(weight_boltz2_ipsae=1.0)
    with pytest.raises(ValueError, match="weight-boltz2-ipsae"):
        run(cfg)


def test_run_rejects_positive_ipsae_weight_alongside_other_boltz2_weights():
    cfg = _minimal_cfg(weight_boltz2_ipsae=1.0, weight_boltz2_iptm=0.5)
    with pytest.raises(ValueError, match="weight-boltz2-ipsae"):
        run(cfg)


def test_guidance_anchor_not_empty_by_default():
    # Default weight_edit_budget is 5.00 (nonzero), so the promoted anchor
    # (no Boltz2/OpenDDE bind signal by default) is not a no-op.
    cfg = _minimal_cfg()
    assert guidance_anchor_is_empty_edit_budget(cfg) is False


def test_guidance_anchor_empty_when_edit_budget_zero_and_no_bind_signal():
    # Regression guard: with no Boltz2/OpenDDE bind signal, edit_loss is
    # promoted into the bind anchor slot (build_guidance_loss's fallback
    # branch). If weight_edit_budget is also 0, that anchor has an
    # all-zero gradient, and _compat_project(g_nat, g_bind=0) returns g_nat
    # completely unprojected (see guidance_anchor_is_empty_edit_budget's
    # docstring) -- guidance silently collapses to naturalness-only despite
    # the anchor/regularizer framing. This must be detectable, not silent.
    cfg = _minimal_cfg(weight_edit_budget=0.0, weight_ablang2=0.5)
    assert guidance_anchor_is_empty_edit_budget(cfg) is True


def test_guidance_anchor_not_empty_when_bind_signal_configured():
    # weight_edit_budget=0 alone doesn't matter if a real bind signal
    # (Boltz2/OpenDDE) is configured -- edit_loss is never promoted to
    # anchor in that case, it stays a regularizer projected against the
    # real bind_loss.
    cfg = _minimal_cfg(weight_edit_budget=0.0, weight_boltz2_iptm=0.5)
    assert guidance_anchor_is_empty_edit_budget(cfg) is False


def test_guidance_anchor_not_empty_when_guidance_skipped():
    # skip_guidance=True (v0) means guidance never runs at all -- the
    # promoted-anchor question doesn't apply.
    cfg = _minimal_cfg(weight_edit_budget=0.0, skip_guidance=True)
    assert guidance_anchor_is_empty_edit_budget(cfg) is False
