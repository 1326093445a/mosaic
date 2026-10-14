"""Two checks that a passing suite did not previously cover.

`merge_shard_tables` accepted shards evaluated on different seed sets, marked
the merged stage complete, and produced a table whose rows are not comparable
with each other. File existence, duplicate detection and row counts all passed:
none of them says the shards belong to one evaluation.

And the bundled mmCIF parser referenced an undefined name, so every parse
raised `NameError` into the errors dict and returned no structure -- silently,
because `parse` catches it.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import p17_rescore_winners as rescore  # noqa: E402


def write_shard(root, index, seeds, candidate_ids=(0,), dtype="bf16", steps=64):
    directory = root / f"shard_{index}"
    (directory / "tables").mkdir(parents=True)
    # `seeds=None` stands for a shard written before the signature existed.
    effective = seeds if seeds is not None else [0]
    calls = len(candidate_ids) * len(effective)
    summary = dict(
        completed=True,
        sequences=len(candidate_ids),
        prediction_calls=calls,
        shard_index=index,
        num_shards=2,
    )
    if seeds is not None:
        summary["evaluation"] = dict(
            seeds=sorted(seeds), sampling_steps=steps, opendde_dtype=dtype
        )
    (directory / "summary.json").write_text(json.dumps(summary))

    rows = {
        "candidates.csv": [
            dict(candidate_id=c, sequence="QVQL") for c in candidate_ids
        ],
        "source_runs.csv": [
            dict(candidate_id=c, run=f"r{index}", cell="base")
            for c in candidate_ids
        ],
        "predictions.csv": [
            dict(candidate_id=c, selection_seed=s, ipsae_min=0.5,
                 structure_file=f"art/c{c}_s{s}.pdb",
                 cif_file=f"art/c{c}_s{s}.cif",
                 confidence_file=f"art/c{c}_s{s}.json")
            for c in candidate_ids for s in effective
        ],
    }
    # The merge relocates these, so they have to exist.
    (directory / "art").mkdir()
    for row in rows["predictions.csv"]:
        for field in ("structure_file", "cif_file", "confidence_file"):
            (directory / row[field]).write_text("")
    for name, table in rows.items():
        with (directory / "tables" / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return directory


def test_shards_evaluated_on_the_same_seeds_merge(tmp_path):
    write_shard(tmp_path, 0, [101, 102], candidate_ids=(0,))
    write_shard(tmp_path, 1, [101, 102], candidate_ids=(1,))
    rescore.merge_shard_tables(tmp_path, 2)
    merged = json.loads((tmp_path / "summary.json").read_text())
    assert merged["completed"] is True


def test_shards_evaluated_on_different_seeds_are_refused(tmp_path):
    write_shard(tmp_path, 0, [101, 102], candidate_ids=(0,))
    write_shard(tmp_path, 1, [201, 202], candidate_ids=(1,))
    with pytest.raises(ValueError) as caught:
        rescore.merge_shard_tables(tmp_path, 2)
    assert "not evaluated the same way" in str(caught.value)
    assert "101" in str(caught.value) and "201" in str(caught.value)


def test_a_differing_precision_is_also_refused(tmp_path):
    write_shard(tmp_path, 0, [101], candidate_ids=(0,), dtype="bf16")
    write_shard(tmp_path, 1, [101], candidate_ids=(1,), dtype="fp32")
    with pytest.raises(ValueError):
        rescore.merge_shard_tables(tmp_path, 2)


def test_shards_without_a_signature_warn_rather_than_fail(tmp_path, capsys):
    """Shards written before 2026-10-14 predate the field.

    Historical outputs stay readable; what changes is that the gap is named.
    """
    write_shard(tmp_path, 0, None, candidate_ids=(0,))
    write_shard(tmp_path, 1, None, candidate_ids=(1,))
    rescore.merge_shard_tables(tmp_path, 2)
    assert "no evaluation signature" in capsys.readouterr().err


def test_the_signature_is_order_insensitive_in_its_seeds():
    assert rescore.evaluation_signature([2, 0, 1], 64, "bf16") == (
        rescore.evaluation_signature([1, 2, 0], 64, "bf16")
    )


# --------------------------------------------------------------------------
# the bundled mmCIF parser
# --------------------------------------------------------------------------


def test_the_bundled_mmcif_parser_parses_the_ubiquitin_fixture():
    from mosaic.alphafold.data import mmcif_parsing

    fixture = REPO / "tests" / "data" / "1ubq.cif"
    result = mmcif_parsing.parse(
        file_id="1ubq", mmcif_string=fixture.read_text()
    )
    assert result.errors == {}, result.errors
    assert result.mmcif_object is not None
    # 1UBQ is a 76-residue single chain at 1.8 A.
    assert result.mmcif_object.chain_to_seqres["A"].startswith("MQIFVKTLTGK")
    assert len(result.mmcif_object.chain_to_seqres["A"]) == 76
    assert result.mmcif_object.header["resolution"] == pytest.approx(1.8)


def test_the_parser_maps_an_unknown_monomer_to_x():
    """The extended table is the point: it carries modified residues, and
    anything it does not know becomes 'X' rather than raising.
    """
    from Bio.Data.PDBData import protein_letters_3to1_extended

    assert protein_letters_3to1_extended.get("MSE") == "M"
    assert protein_letters_3to1_extended.get("NOTARESIDUE", "X") == "X"


# --------------------------------------------------------------------------
# Round 8: the signature has to describe what RAN, not what was requested.
# --------------------------------------------------------------------------


def test_inherited_and_explicit_settings_give_the_same_signature():
    """One shard supplying 64 explicitly and one inheriting 64 are the same
    evaluation. Reading `args.sampling_steps` made the inherited one `null`
    and the pair incompatible."""
    explicit = rescore.evaluation_signature([101, 102], 64, "bf16")
    inherited = rescore.evaluation_signature([102, 101], 64, "bf16")
    assert explicit == inherited


def test_different_inherited_settings_give_different_signatures():
    """And the other direction: two shards that both inherited, but different
    values, recorded `null` for both and merged as compatible."""
    a = rescore.evaluation_signature([101], 64, "bf16")
    b = rescore.evaluation_signature([101], 128, "bf16")
    assert a != b


def test_the_signature_refuses_unresolved_settings():
    with pytest.raises(ValueError) as caught:
        rescore.evaluation_signature([101], None, "bf16")
    assert "effective settings" in str(caught.value)
    with pytest.raises(ValueError):
        rescore.evaluation_signature([101], 64, None)


def test_shards_differing_only_in_step_count_are_refused(tmp_path):
    write_shard(tmp_path, 0, [101], candidate_ids=(0,), steps=64)
    write_shard(tmp_path, 1, [101], candidate_ids=(1,), steps=128)
    with pytest.raises(ValueError) as caught:
        rescore.merge_shard_tables(tmp_path, 2)
    assert "not evaluated the same way" in str(caught.value)


# --------------------------------------------------------------------------
# Round 9: tar member names are not filesystem paths, and matching seeds,
# steps and precision do not establish one evaluation.
# --------------------------------------------------------------------------


WT = "QVQLQESGGG"


def _run_payload(checkpoint="ckpt-a", recycle=3):
    """A search run's config.json, with the fields the loader cross-checks."""
    return dict(
        binder_sequence=WT,
        target_sequence="MFVFLVLLPL",
        designable_positions_0idx=list(range(len(WT))),
        config=dict(edit_budget=5),
        checkpoint=checkpoint,
        recycling_steps=recycle,
        arguments=dict(
            sampling_steps=64, opendde_dtype="bf16",
            pae_cutoff=12.0, distance_cutoff=8.0, selection_seeds=[0, 1],
            edit_budget=5, budget_anchor="start",
        ),
    )


def _write_run(root, relative, payload=None):
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(_run_payload()))
    (directory / "summary.json").write_text(
        json.dumps(
            payload
            or dict(best_sequence=WT, best_id=0, best_score=0.5, best_edits=0)
        )
    )
    return directory


def _tar(root, destination, prefix=""):
    import tarfile

    with tarfile.open(destination, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"{prefix}{path.relative_to(root)}")
    return destination


def test_a_directory_of_runs_loads(tmp_path):
    """The control: the unpacked form has always worked."""
    source = tmp_path / "unpacked"
    _write_run(source, "run")
    candidates, _links, original = rescore.load_candidates(source)
    assert [c["sequence"] for c in candidates] == [WT]
    assert original["checkpoint"] == "ckpt-a"


@pytest.mark.parametrize("layout, prefix", [
    ("run", ""),          # tar of a directory containing run/
    ("run", "./"),        # the same, written with a leading ./
    (".", ""),            # config.json at the archive root
])
def test_common_tar_layouts_load_like_the_directory(tmp_path, layout, prefix):
    source = tmp_path / f"src_{layout.strip('.') or 'root'}_{len(prefix)}"
    _write_run(source, layout)
    archive = _tar(source, tmp_path / f"a_{layout.strip('.') or 'root'}_{len(prefix)}.tar",
                   prefix=prefix)
    candidates, _links, original = rescore.load_candidates(archive)
    assert [c["sequence"] for c in candidates] == [WT]
    assert original["checkpoint"] == "ckpt-a"


def test_sibling_agrees_with_pathlib_for_ordinary_paths():
    assert rescore.sibling("run/config.json", "summary.json") == "run/summary.json"
    assert rescore.sibling("config.json", "summary.json") == "summary.json"
    assert rescore.sibling("a/b/config.json", "summary.json") == "a/b/summary.json"


def test_the_signature_carries_model_and_cutoff_identity():
    original = _run_payload(checkpoint="ckpt-a", recycle=3)
    signature = rescore.evaluation_signature([101], 64, "bf16", original)
    assert signature["checkpoint"] == "ckpt-a"
    assert signature["recycling_steps"] == 3
    assert signature["pae_cutoff"] == 12.0
    assert signature["scored_target_is_decoy"] is False


@pytest.mark.parametrize("field, value", [
    ("checkpoint", "ckpt-b"),
    ("recycling_steps", 10),
])
def test_shards_under_different_models_do_not_match(field, value):
    base = rescore.evaluation_signature([101], 64, "bf16", _run_payload())
    other_payload = _run_payload()
    other_payload[field] = value
    other = rescore.evaluation_signature([101], 64, "bf16", other_payload)
    assert base != other, f"{field} is not part of the evaluation identity"


def test_a_decoy_archive_does_not_match_a_real_one():
    real = rescore.evaluation_signature([101], 64, "bf16", _run_payload())
    decoy_payload = _run_payload()
    decoy_payload["decoy_target"] = dict(identity_to_real_target=0.21)
    decoy = rescore.evaluation_signature([101], 64, "bf16", decoy_payload)
    assert real["scored_target_is_decoy"] is False
    assert decoy["scored_target_is_decoy"] is True
    assert real != decoy
