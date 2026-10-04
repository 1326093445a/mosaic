"""Argument wiring for the P17+Alpha recovery control.

These run before any model loads, which is the point: a multi-day run must not
start from a misconfigured reference. Every case here is a misuse that would
otherwise produce a plausible-looking run against the wrong structure.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

ALPHA = REPO / "P17_Alpha.pdb"
pytestmark = pytest.mark.skipif(not ALPHA.is_file(), reason="P17_Alpha.pdb absent")


@pytest.fixture(scope="module")
def main():
    from p17_confidence_search import main as entry

    return entry


def _args(tmp_path, *extra):
    return [
        "--policy", "independent",
        "--output-dir", str(tmp_path / "run"),
        "--max-score-calls", "1",
        "--max-gradient-calls", "1",
        "--max-proposals", "1",
        *extra,
    ]


def _refused(main, capsys, argv, expected):
    """Assert the run was refused *for the stated reason*.

    `parser.error` exits 2 for any argument problem, so asserting only on
    SystemExit would pass on an unrelated failure -- an earlier revision of
    this file did exactly that and every case exited on a missing --policy
    without reaching the code under test.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert expected in message, message
    return message


def test_chain_flags_without_a_complex_are_refused(main, capsys, tmp_path):
    _refused(
        main,
        capsys,
        _args(tmp_path, "--binder-chain", "B"),
        "require --complex",
    )


def test_contact_epitope_without_a_complex_is_refused(main, capsys, tmp_path):
    """Contact mode needs a reference to derive contacts from."""
    _refused(
        main,
        capsys,
        _args(tmp_path, "--epitope-mode", "contact"),
        "requires --complex",
    )


def test_a_complex_cannot_reuse_the_jn1_hotspot_constants(main, capsys, tmp_path):
    """The five hotspots are JN.1 target positions; Alpha is 195 res at 334-528.

    Silently reusing them would anchor the contact loss on unrelated residues,
    so this must fail rather than default.
    """
    _refused(
        main,
        capsys,
        _args(tmp_path, "--complex", str(ALPHA), "--epitope-mode", "hotspots"),
        "do not transfer",
    )


def test_start_sequence_length_must_match_the_reference(main, capsys, tmp_path):
    _refused(
        main,
        capsys,
        _args(
            tmp_path,
            "--complex", str(ALPHA),
            "--epitope-mode", "contact",
            "--start-sequence", "ACDEF",
        ),
        "has 5 residues, reference binder has 123",
    )


def test_start_sequence_may_not_differ_outside_the_designable_mask(main, capsys, tmp_path):
    """A difference the search cannot undo would be permanent damage.

    Position 0 is framework, not CDR, so the search could never revert it and
    recovery would be impossible by construction.
    """
    from p17_alpha_reference import ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN, load_complex

    reference = load_complex(ALPHA, ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN)
    seq = list(reference["binder_seq"])
    seq[0] = "W" if seq[0] != "W" else "A"
    _refused(
        main,
        capsys,
        _args(
            tmp_path,
            "--complex", str(ALPHA),
            "--epitope-mode", "contact",
            "--start-sequence", "".join(seq),
        ),
        "outside the designable mask",
    )


def test_start_sequence_rejects_nonstandard_residues(main, capsys, tmp_path):
    from p17_alpha_reference import ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN, load_complex

    reference = load_complex(ALPHA, ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN)
    seq = list(reference["binder_seq"])
    seq[26] = "X"
    _refused(
        main,
        capsys,
        _args(
            tmp_path,
            "--complex", str(ALPHA),
            "--epitope-mode", "contact",
            "--start-sequence", "".join(seq),
        ),
        "non-standard residues",
    )
