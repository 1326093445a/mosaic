"""Which directories the iRMSD audit accepts, and which candidate is the start.

Two layouts reach `p17_capri_audit.py` and `run_p17_consolidated.sh` passes
both: a screen with one subdirectory per cell (`explore/arity2/run/heldout`)
and a single run with `heldout/` at its own root, which is what
`run_p17_alpha_recovery.sh` writes. Handing the second one to the screen walk
found no cells and reported "no held-out structures", which is true of the
directory and false of the run.

The other half is the unmodified start. It has to be excluded from a cell
summary or it is averaged in as if it were a design, and the two drivers record
it differently: the JN.1 driver writes an `is_wt` column, the Alpha driver's
recovery.csv has no such column and relies on the convention that candidate 0
is the start.

No DockQ and no structures: these pin discovery, which is where both bugs were.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

import p17_capri_audit as audit  # noqa: E402


def shard(root, candidates=(0, 1), seeds=(101,)):
    """A shard directory shaped like the real ones, with empty PDBs."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "reference.pdb").write_text("")
    for cid in candidates:
        cand = root / "structures" / f"candidate_{cid:05d}"
        cand.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            (cand / f"seed_{seed}.pdb").write_text("")
    return root


def table(run_root, name, rows):
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    header = ",".join(rows[0])
    body = "\n".join(",".join(str(r[k]) for k in rows[0]) for r in rows)
    (run_root / "tables" / name).write_text(f"{header}\n{body}\n")


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

def test_a_screen_of_cells_is_discovered_per_cell(tmp_path):
    for cell in ("base", "arity2"):
        shard(tmp_path / cell / "run" / "heldout" / "shard_0")
    assert [c for c, _ in audit.cell_runs(tmp_path)] == ["arity2", "base"]
    jobs = audit.discover(tmp_path, None)
    assert {j["cell"] for j in jobs} == {"base", "arity2"}
    assert len(jobs) == 4  # two cells x two candidates x one seed


def test_a_single_run_is_discovered_as_one_cell(tmp_path):
    """The alpha layout: heldout/ beside ladder/ calibrate/ search/."""
    run = tmp_path / "alpha"
    for name in ("ladder", "calibrate", "search"):
        (run / name).mkdir(parents=True)
    shard(run / "heldout" / "shard_0")
    assert audit.cell_runs(run) == [("alpha", run)]
    jobs = audit.discover(run, None)
    assert jobs and {j["cell"] for j in jobs} == {"alpha"}
    # and the model paths stay relative to what was passed in, which is what
    # main() writes into the CSV
    for job in jobs:
        job["model"].relative_to(run)


def test_a_single_run_wins_over_the_screen_walk(tmp_path):
    """A run that also happens to contain a cell-shaped child is still a run."""
    run = tmp_path / "alpha"
    shard(run / "heldout" / "shard_0")
    shard(run / "decoy" / "run" / "heldout" / "shard_0")
    assert audit.cell_runs(run) == [("alpha", run)]


def test_cells_filter_applies_to_both_layouts(tmp_path):
    for cell in ("apgm", "bindcraft"):
        shard(tmp_path / cell / "run" / "heldout" / "shard_0")
    jobs = audit.discover(tmp_path, {"bindcraft"})
    assert {j["cell"] for j in jobs} == {"bindcraft"}


def test_sibling_directories_without_heldout_are_ignored(tmp_path):
    """OUT_ROOT holds free/ norms/ alpha/ explore/ beside the seeded cells."""
    shard(tmp_path / "bindcraft" / "run" / "heldout" / "shard_0")
    (tmp_path / "free").mkdir()
    (tmp_path / "norms").mkdir()
    (tmp_path / "alpha" / "heldout" / "shard_0").mkdir(parents=True)
    (tmp_path / "explore" / "base" / "run" / "heldout").mkdir(parents=True)
    assert [c for c, _ in audit.cell_runs(tmp_path)] == ["bindcraft"]


def test_a_shard_without_a_reference_is_an_error_not_a_skip(tmp_path):
    root = tmp_path / "base" / "run" / "heldout" / "shard_0"
    shard(root)
    (root / "reference.pdb").unlink()
    with pytest.raises(FileNotFoundError, match="reference.pdb"):
        audit.discover(tmp_path, None)


# --------------------------------------------------------------------------
# the unmodified start
# --------------------------------------------------------------------------

def test_is_wt_column_is_authoritative_when_present(tmp_path):
    run = tmp_path / "base" / "run"
    shard(run / "heldout" / "shard_0", candidates=(0, 1, 2))
    table(run, "jn1_recovery.csv", [
        dict(candidate_id=0, is_wt="True"),
        dict(candidate_id=1, is_wt="False"),
        dict(candidate_id=2, is_wt="True"),
    ])
    assert audit.wt_candidate_ids(run) == {0, 2}
    jobs = audit.discover(tmp_path, None)
    assert {j["candidate_id"] for j in jobs if j["is_wt"]} == {0, 2}


def test_a_table_without_is_wt_falls_back_to_candidate_zero(tmp_path):
    """The Alpha driver's recovery.csv has no is_wt column.

    Returning an empty set here would fold the damaged start into the cell mean
    as if the search had produced it.
    """
    run = tmp_path / "alpha"
    shard(run / "heldout" / "shard_0", candidates=(0, 1))
    table(run, "recovery.csv", [
        dict(candidate_id=0, mean_ipsae=0.1),
        dict(candidate_id=1, mean_ipsae=0.5),
    ])
    assert audit.wt_candidate_ids(run) == {0}
    jobs = audit.discover(run, None)
    assert {j["candidate_id"] for j in jobs if j["is_wt"]} == {0}


def test_jn1_table_is_preferred_over_recovery_when_both_exist(tmp_path):
    run = tmp_path / "base" / "run"
    table(run, "jn1_recovery.csv", [dict(candidate_id=3, is_wt="True")])
    table(run, "recovery.csv", [dict(candidate_id=0, mean_ipsae=0.1)])
    assert audit.wt_candidate_ids(run) == {3}


def test_no_table_at_all_marks_nothing_as_the_start(tmp_path):
    run = tmp_path / "base" / "run"
    shard(run / "heldout" / "shard_0")
    assert audit.wt_candidate_ids(run) == set()


def test_per_candidate_irmsd_averages_the_correlated_seeds(tmp_path):
    rows = [
        dict(cell="base", candidate_id=1, iRMSD=6.0),
        dict(cell="base", candidate_id=1, iRMSD=14.0),
        dict(cell="base", candidate_id=2, iRMSD=9.0),
        dict(cell="arity2", candidate_id=1, iRMSD=3.0),
    ]
    byc = audit.per_candidate_irmsd(rows)
    assert byc[("base", 1)] == 10.0   # not two hits and not two misses: one, at 10
    assert byc[("base", 2)] == 9.0
    assert byc[("arity2", 1)] == 3.0
    inside = [k for k, v in byc.items() if v <= audit.IRMSD_THRESHOLD]
    assert sorted(inside) == [("arity2", 1), ("base", 1), ("base", 2)]


# --------------------------------------------------------------------------
# the chain mapping
# --------------------------------------------------------------------------

def pdb(path, chains):
    """A PDB holding only CA atoms, `chains` mapping letter -> residue count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines, serial = [], 1
    for chain, count in chains.items():
        for i in range(1, count + 1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain}{i:4d}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_mapping_pairs_by_length_not_by_chain_letter(tmp_path):
    """Models write A/B, the JN.1 reference writes B/T for the same complex."""
    model = pdb(tmp_path / "m.pdb", {"A": 123, "B": 184})
    native = pdb(tmp_path / "n.pdb", {"B": 123, "T": 184})
    mapping, designed, d_des, d_tgt = audit.derive_mapping(model, native)
    assert mapping == "AB:BT"
    assert (designed, d_des, d_tgt) == (123, 0, 0)


def test_a_native_with_extra_resolved_residues_is_scored_not_refused(tmp_path):
    """P17+Alpha: the crystal reference resolves more than the model folded.

    123/195 against 124/205. DockQ aligns the two and scores the pair; the
    equality check that used to live here refused the Alpha run outright,
    which is the one task in this project with a known answer.
    """
    model = pdb(tmp_path / "m.pdb", {"A": 123, "B": 195})
    native = pdb(tmp_path / "n.pdb", {"B": 124, "A": 205})
    mapping, designed, d_des, d_tgt = audit.derive_mapping(model, native)
    assert mapping == "AB:BA"
    assert (designed, d_des, d_tgt) == (123, 1, 10)


def test_an_ambiguous_pairing_is_refused_with_both_costs(tmp_path):
    """The hazard: a swap compares the designed chain to the target.

    Rank pairing provably minimises the paired difference for two sorted
    pairs, so the guard can only fire on equality, and equality means one
    file's chains are BOTH longer than the other's -- the native's shorter
    chain exceeds the model's longer one. Then length carries no information
    about which chain is the designed one and the swap scores just as well.
    """
    model = pdb(tmp_path / "m.pdb", {"A": 100, "B": 110})
    native = pdb(tmp_path / "n.pdb", {"B": 200, "T": 300})
    with pytest.raises(ValueError, match="pairing is ambiguous"):
        audit.derive_mapping(model, native)


def test_an_overlapping_pairing_is_always_accepted(tmp_path):
    """The complement of the guard, so its reach is pinned from both sides."""
    model = pdb(tmp_path / "m.pdb", {"A": 123, "B": 195})
    for native_chains in ({"B": 130, "A": 210},     # more of both, interleaved
                          {"B": 120, "A": 190},     # less of both
                          {"B": 123, "A": 400}):    # a much longer target
        native = pdb(tmp_path / "n.pdb", native_chains)
        mapping, designed, _, _ = audit.derive_mapping(model, native)
        assert mapping == "AB:BA" and designed == 123


def test_equal_length_chains_inside_one_file_are_refused(tmp_path):
    model = pdb(tmp_path / "m.pdb", {"A": 150, "B": 150})
    native = pdb(tmp_path / "n.pdb", {"B": 123, "T": 184})
    with pytest.raises(ValueError, match="same length"):
        audit.derive_mapping(model, native)
    with pytest.raises(ValueError, match="same length"):
        audit.derive_mapping(native, model)


def test_a_file_without_exactly_two_chains_is_refused(tmp_path):
    model = pdb(tmp_path / "m.pdb", {"A": 123, "B": 184, "C": 50})
    native = pdb(tmp_path / "n.pdb", {"B": 123, "T": 184})
    with pytest.raises(ValueError, match="expected 2"):
        audit.derive_mapping(model, native)
