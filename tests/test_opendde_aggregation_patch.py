import importlib.util
from pathlib import Path

import pytest

from mosaic.opendde_numerics import aggregation_mode


def _patch_module():
    path = Path(__file__).resolve().parents[1] / "patches/patch_jopendde_aggregation.py"
    spec = importlib.util.spec_from_file_location("aggregation_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    patch = _patch_module()
    path = tmp_path / "transformer.py"
    path.write_text(
        "def aggregate(x_atom, atom_to_token_idx, n_token):\n" + patch.ORIGINAL
    )
    assert patch.patch_file(path)
    first = path.read_text()
    assert not patch.patch_file(path)
    assert path.read_text() == first
    for source in (
        "# unsupported implementation\n",
        first.replace("stable_segment_mean(x_atom", "other_mean(x_atom"),
    ):
        path.write_text(source)
        with pytest.raises(ValueError):
            patch.patch_file(path)
        assert path.read_text() == source


def test_aggregation_control_is_explicit(monkeypatch):
    monkeypatch.delenv("MOSAIC_OPENDDE_AGGREGATION", raising=False)
    assert aggregation_mode() == "stable"
    monkeypatch.setenv("MOSAIC_OPENDDE_AGGREGATION", "original")
    assert aggregation_mode() == "original"
    monkeypatch.setenv("MOSAIC_OPENDDE_AGGREGATION", "typo")
    with pytest.raises(ValueError, match="original or stable"):
        aggregation_mode()
