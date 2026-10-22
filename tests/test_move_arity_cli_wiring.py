"""`--move-arity` from the command line to `SearchConfig`, and to the workers.

The operator itself is covered by test_multi_position_moves.py. What is
covered here is the plumbing, which is a separate failure mode: a flag that
parses, forwards and is then dropped produces a run that looks like a clean
null for the operator. The consolidated launcher's `arity2` cell is a
six-hour comparison resting on this path.

Both cases stop before any model loads.
"""

import argparse
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))



ALPHA = REPO / "P17_Alpha.pdb"


def base_argv(tmp_path, *extra):
    return [
        "--policy", "independent",
        "--output-dir", str(tmp_path / "run"),
        "--max-score-calls", "1",
        "--max-gradient-calls", "1",
        "--max-proposals", "1",
        *extra,
    ]


@pytest.mark.parametrize("arity,expected", [
    ("3", "move_arity above 2"),
    ("0", "move_arity"),
    ("-1", "move_arity"),
])
def test_an_unusable_arity_is_refused_by_searchconfig(tmp_path, capsys, arity, expected):
    """The refusal text comes from SearchConfig, which proves the value arrived.

    SearchConfig is built at the top of main(), before the checkpoint loads,
    so this reaches the validation without a GPU.
    """
    from p17_confidence_search import main

    with pytest.raises(SystemExit) as excinfo:
        main(base_argv(tmp_path, "--move-arity", arity))
    assert excinfo.value.code == 2
    assert expected in capsys.readouterr().err


def test_arity_one_is_the_default_so_recorded_results_stay_reproducible():
    """Every result this project holds was produced at arity 1."""
    from mosaic.search import SearchConfig
    from p17_confidence_search import build_parser

    args = build_parser().parse_args([
        "--policy", "independent", "--output-dir", "x",
    ])
    assert args.move_arity == 1
    assert SearchConfig().move_arity == 1



def fake_args(**overrides):
    """The attributes search_jobs() reads, with nothing else set."""
    fields = dict(
        move_arity=None, budget_anchor=None, start_sequence=None,
        weight_registry=None, registry_contact_distance=None,
        target_entropy=None, acceptance_temperature=None,
        max_score_calls=None, max_gradient_calls=None, max_proposals=None,
        steps=None, sampling_steps=None, opendde_dtype=None,
        aggregation=None, save_saliency=False, width=None, seed=None,
        proposal_path=None, weight_pose=None,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


def arity_in(command):
    if "--move-arity" not in command:
        return None
    return command[command.index("--move-arity") + 1]


@pytest.mark.parametrize("value,expected", [(2, "2"), (1, "1"), (None, None)])
def test_the_recovery_driver_forwards_arity_to_every_worker(tmp_path, value, expected):
    """A dropped flag here would run the baseline and label it `arity2`."""
    from p17_alpha_recovery import search_jobs

    jobs = search_jobs(
        root=tmp_path, policies=["population"], seeds=[0, 1],
        start_sequence=None, budget=5, args=fake_args(move_arity=value),
        complex_pdb=None,
    )
    assert jobs, "no search jobs were built"
    for job in jobs:
        assert arity_in(job["command"]) == expected
