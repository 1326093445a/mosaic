"""Reference loading and damage generation for the P17+Alpha recovery control.

The control's value depends on two things being exactly right before any GPU
time is spent: the reference correspondence (a dropped or misnumbered residue
shifts every position the loss compares) and the damaged starting point (fixed
and recorded before scoring, or the control becomes a search for an easy case).

Uses the real `P17_Alpha.pdb`. No model, no checkpoint, no prediction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from p17_alpha_reference import (  # noqa: E402
    ALPHA_BINDER_CHAIN,
    ALPHA_COMPLEX,
    ALPHA_TARGET_CHAIN,
    AMINO_ACIDS,
    ca_distance_matrix,
    ca_spacing_report,
    contact_epitope,
    damage_ladder,
    damage_sequence,
    load_complex,
)

JN1_COMPLEX = Path(__file__).resolve().parent.parent / "P17_JN1.pdb"
CDR_IDX = np.array(
    sorted(set(range(25, 33)) | set(range(50, 58)) | set(range(96, 109))), dtype=int
)


@pytest.fixture(scope="module")
def alpha():
    if not ALPHA_COMPLEX.is_file():
        pytest.skip(f"{ALPHA_COMPLEX} absent from this checkout")
    return load_complex(ALPHA_COMPLEX, ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN)


def test_alpha_loads_with_solvent_and_glycan_dropped(alpha):
    """Waters, EDO and the NAG/FUC chain must not reach the sequence."""
    assert len(alpha["binder_seq"]) == 123
    assert len(alpha["target_seq"]) == 195
    assert len(alpha["binder_ca"]) == 123
    assert len(alpha["target_ca"]) == 195
    # The crystal file interleaves solvent with protein in both chains, so a
    # naive residue walk would pick it up; these counts prove it was dropped.
    assert alpha["binder_dropped_residues"] > 0
    assert alpha["target_dropped_residues"] > 0
    assert set(alpha["binder_seq"]) <= set(AMINO_ACIDS)
    assert set(alpha["target_seq"]) <= set(AMINO_ACIDS)


def test_alpha_binder_matches_the_jn1_binder_exactly(alpha):
    """Why the existing CDR mask transfers with no renumbering.

    Both references hold the same unmodified P17. If this ever fails, the CDR
    indices in `p17_hallucination_search` cannot be reused for this control.
    """
    if not JN1_COMPLEX.is_file():
        pytest.skip("P17_JN1.pdb absent")
    jn1 = load_complex(JN1_COMPLEX, "B", "T")
    assert alpha["binder_seq"] == jn1["binder_seq"]
    assert len(jn1["target_seq"]) == 184, "and the targets differ, as expected"


def test_alpha_numbering_is_contiguous_and_recorded(alpha):
    assert alpha["binder_numbering"] == dict(first=1, last=123, count=123)
    assert alpha["target_numbering"] == dict(first=334, last=528, count=195)


def test_absent_chain_is_refused_by_name(alpha):
    with pytest.raises(ValueError, match="absent"):
        load_complex(ALPHA_COMPLEX, "B", "Z")


def test_glycan_only_chain_is_refused_rather_than_silently_empty():
    """Chain C codes to no amino acids; that must fail, not yield an empty seq."""
    with pytest.raises(ValueError, match="no amino acids"):
        load_complex(ALPHA_COMPLEX, "B", "C")


def test_reference_backbone_spacing_passes_the_sanity_bounds(alpha):
    for role in ("binder", "target"):
        report = ca_spacing_report(alpha[f"{role}_ca"], role)
        assert report["passed"], report
        assert 3.6 < report["median_A"] < 4.0


def test_contact_epitope_is_a_subset_of_the_target_and_nonempty(alpha):
    epitope = contact_epitope(alpha["binder_ca"], alpha["target_ca"], 8.0)
    assert epitope.size > 0
    assert epitope.max() < len(alpha["target_seq"])
    assert np.all(np.diff(epitope) > 0), "sorted and unique"
    # A tighter cutoff can only select a subset.
    tighter = contact_epitope(alpha["binder_ca"], alpha["target_ca"], 6.0)
    assert set(tighter.tolist()) <= set(epitope.tolist())


def test_contact_epitope_matches_an_independent_distance_calculation(alpha):
    distances = ca_distance_matrix(alpha["binder_ca"], alpha["target_ca"])
    expected = np.flatnonzero((distances <= 8.0).any(axis=0))
    assert np.array_equal(
        contact_epitope(alpha["binder_ca"], alpha["target_ca"], 8.0), expected
    )


def test_damage_is_deterministic_and_realizes_the_requested_edit_count(alpha):
    wt = alpha["binder_seq"]
    first, subs = damage_sequence(wt, CDR_IDX, 5, seed=0)
    again, subs_again = damage_sequence(wt, CDR_IDX, 5, seed=0)
    assert first == again and subs == subs_again
    assert sum(a != b for a, b in zip(wt, first)) == 5
    assert len(subs) == 5


def test_damage_only_touches_designable_positions(alpha):
    wt = alpha["binder_seq"]
    damaged, subs = damage_sequence(wt, CDR_IDX, 8, seed=3)
    changed = {i for i, (a, b) in enumerate(zip(wt, damaged)) if a != b}
    assert changed <= set(CDR_IDX.tolist())
    assert {s["position_0idx"] for s in subs} == changed
    for sub in subs:
        assert wt[sub["position_0idx"]] == sub["wt"]
        assert damaged[sub["position_0idx"]] == sub["damaged"]
        assert sub["wt"] != sub["damaged"], "an identity substitution is not an edit"


def test_different_seeds_give_different_damage(alpha):
    wt = alpha["binder_seq"]
    a, _ = damage_sequence(wt, CDR_IDX, 5, seed=0)
    b, _ = damage_sequence(wt, CDR_IDX, 5, seed=1)
    assert a != b


def test_damage_refuses_more_edits_than_designable_positions(alpha):
    with pytest.raises(ValueError, match="designable positions"):
        damage_sequence(alpha["binder_seq"], CDR_IDX, CDR_IDX.size + 1, seed=0)
    with pytest.raises(ValueError, match="at least 1"):
        damage_sequence(alpha["binder_seq"], CDR_IDX, 0, seed=0)


def test_ladder_is_reproducible_and_independent_across_rungs(alpha):
    """Adding a rung must not change the sequences already generated."""
    wt = alpha["binder_seq"]
    short = damage_ladder(wt, CDR_IDX, [2, 5], seed=7)
    long = damage_ladder(wt, CDR_IDX, [2, 5, 8, 12], seed=7)
    by_count = {r["n_edits"]: r["sequence"] for r in long}
    for rung in short:
        assert by_count[rung["n_edits"]] == rung["sequence"]
    assert [r["n_edits"] for r in long] == [2, 5, 8, 12]
    for rung in long:
        assert rung["hamming_from_wt"] == rung["n_edits"]
        assert sum(a != b for a, b in zip(wt, rung["sequence"])) == rung["n_edits"]


def test_ladder_deduplicates_and_sorts_requested_counts(alpha):
    rungs = damage_ladder(alpha["binder_seq"], CDR_IDX, [5, 2, 5], seed=7)
    assert [r["n_edits"] for r in rungs] == [2, 5]


def test_source_digest_is_recorded_for_provenance(alpha):
    assert len(alpha["source_sha256"]) == 64
    assert alpha["source_sha256"] == load_complex(
        ALPHA_COMPLEX, ALPHA_BINDER_CHAIN, ALPHA_TARGET_CHAIN
    )["source_sha256"]


def test_alternate_conformations_are_resolved_by_occupancy_and_recorded(alpha):
    """The gotcha specific to the crystal reference.

    11 residues are modeled twice. Occupancy must decide, because the altloc
    letter does not track it: chain A residue 375 has 'A' at 0.34 against 'B'
    at 0.66, so a letter-based convention would take the minor conformer.
    """
    records = alpha["target_altlocs_resolved"] + alpha["binder_altlocs_resolved"]
    assert len(records) == 11, records
    for record in records:
        assert record["chosen_occupancy"] == max(record["alternatives"].values())
        assert len(record["alternatives"]) > 1
    by_residue = {r["residue"]: r for r in alpha["target_altlocs_resolved"]}
    assert by_residue[375]["chosen_altloc"] == "B", "occupancy 0.66 beats A's 0.34"
    assert by_residue[345]["chosen_altloc"] == "A", "and A wins where it should"


def test_jn1_reference_has_no_alternate_conformations():
    """Which is why the JN.1 loader never had to make this choice."""
    if not JN1_COMPLEX.is_file():
        pytest.skip("P17_JN1.pdb absent")
    jn1 = load_complex(JN1_COMPLEX, "B", "T")
    assert jn1["binder_altlocs_resolved"] == []
    assert jn1["target_altlocs_resolved"] == []
