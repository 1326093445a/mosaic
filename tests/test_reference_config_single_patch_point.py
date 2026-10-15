"""`p17_reference_config` must be the ONE place the reference can be patched.

The constants moved into this module on 2026-10-14. The move itself was right;
the way consumers picked them up was not. A `from p17_reference_config import
COMPLEX_PDB` at module scope binds the value once, at import time, into the
consumer's own namespace -- so patching either the config module or one
consumer leaves every other consumer reading the real file, silently.

That is how `test_winner_rescoring_exports_scored_pose_without_optimization`
broke: the constant had been read via a FUNCTION-LOCAL import, which resolves
the attribute at call time and so saw the test's patch; hoisting it to module
scope did not. The rescorer copied the real P17_JN1.pdb over the fixture.

So this pins the contract rather than the symptom: consumers read the attribute
through the module at use time, and no consumer holds its own binding.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

NAMES = ("COMPLEX_PDB", "BINDER_CHAIN", "TARGET_CHAIN")

# Modules that take the shared JN.1 reference from the config module. Scripts
# with their own deliberate copies (p17_predict_hallucination_structures,
# p17_opendde_diffusion_steps_convergence_check,
# p17_hallucination_mcmc_with_full_opendde_rescoring) are NOT listed: they
# define their own constants, which is a separate question from this one.
CONSUMERS = ("p17_hallucination_search", "p17_rescore_winners")


def test_the_config_module_is_the_source():
    config = importlib.import_module("p17_reference_config")
    for name in NAMES:
        assert hasattr(config, name), name
    assert config.COMPLEX_PDB.name == "P17_JN1.pdb"


@pytest.mark.parametrize("module_name", CONSUMERS)
@pytest.mark.parametrize("name", NAMES)
def test_no_consumer_holds_its_own_binding(module_name, name):
    """A consumer with its own attribute has a `from … import`, and a patch to
    the config module will not reach it."""
    module = importlib.import_module(module_name)
    assert not hasattr(module, name), (
        f"{module_name} binds {name} itself, so patching "
        f"p17_reference_config.{name} will not reach it. Read it through the "
        f"module instead: `reference_config.{name}`."
    )


@pytest.mark.parametrize("module_name", CONSUMERS)
def test_each_consumer_imports_the_module_itself(module_name):
    module = importlib.import_module(module_name)
    assert getattr(module, "reference_config", None) is importlib.import_module(
        "p17_reference_config"
    ), f"{module_name} does not hold the config module under `reference_config`"


def test_patching_the_config_reaches_every_consumer(monkeypatch, tmp_path):
    """The property the single patch point is for."""
    config = importlib.import_module("p17_reference_config")
    fixture = tmp_path / "fixture.pdb"
    fixture.write_text("HEADER    TEST\n")
    monkeypatch.setattr(config, "COMPLEX_PDB", fixture)
    for module_name in CONSUMERS:
        module = importlib.import_module(module_name)
        assert module.reference_config.COMPLEX_PDB == fixture, module_name


def test_the_function_local_consumer_also_reads_through_the_module():
    """`p17_confidence_search` resolves it inside a function, which hid the
    problem there -- but it imported the names FROM p17_hallucination_search,
    so it would have broken the moment that module stopped holding them."""
    source = (REPO / "examples" / "p17_confidence_search.py").read_text()
    assert "import p17_reference_config as reference_config" in source
    assert "from p17_hallucination_search import BINDER_CHAIN" not in source
