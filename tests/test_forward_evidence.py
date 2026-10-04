"""The forward-evidence gate for the pose experiment.

The gate decides whether a saved forward control supports the configuration a
cluster run pins. These checks cover what the predicate must refuse: a short
or partially-failed archive, the wrong sampling budget, the wrong precision,
and unchecked mapping. Synthetic summaries only; nothing is executed.
"""

import hashlib
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from p17_forward_evidence import (  # noqa: E402
    assess,
    find_supporting_archive,
    reference_digest,
    row_steps,
)

REFERENCE = "a" * 64
OTHER_REFERENCE = "b" * 64


def jax_row(**overrides):
    row = dict(
        worker="mosaic_stable_steps64_seed0",
        path="mosaic",
        repeat=0,
        sampling_steps=64,
        seed=0,
        opendde_dtype="bf16",
        reference_sha256=REFERENCE,
        aggregation_mode="stable",
        completed=True,
        geometry_passed=True,
        mapping_agrees=True,
        note=None,
    )
    row.update(overrides)
    return row


def summary(rows, **overrides):
    value = dict(
        completed=all(r.get("completed") for r in rows),
        geometry_passed=all(r.get("geometry_passed") is True for r in rows),
        expected_rows=len(rows),
        observed_rows=len(rows),
        rows=rows,
    )
    value.update(overrides)
    return value


def test_accepts_a_matching_archive_with_nothing_unverified():
    reasons, matching, unverified = assess(
        summary([jax_row()]), 64, "bf16", REFERENCE
    )
    assert reasons == []
    assert matching == 1
    assert unverified == []


def test_rejects_an_archive_built_against_a_different_reference():
    """Evidence about another complex is worse than no evidence."""
    reasons, matching, _ = assess(
        summary([jax_row()]), 64, "bf16", OTHER_REFERENCE
    )
    assert matching == 0
    assert any("reference sha256" in r for r in reasons)


def test_a_recorded_digest_nobody_compared_is_unverified_not_passed():
    """Without a wanted digest the gate must not claim the reference matched."""
    reasons, matching, unverified = assess(summary([jax_row()]), 64, "bf16")
    assert reasons == []
    assert matching == 1
    assert any("not compared" in item for item in unverified)


def test_an_unrecorded_digest_stays_unverified_even_when_one_is_wanted():
    reasons, matching, unverified = assess(
        summary([jax_row(reference_sha256=None)]), 64, "bf16", REFERENCE
    )
    assert reasons == []
    assert matching == 1
    assert any("reference identity not recorded" in item for item in unverified)


def test_reference_digest_reads_a_file_and_tolerates_a_missing_one(tmp_path):
    present = tmp_path / "ref.pdb"
    present.write_bytes(b"ATOM\n")
    assert reference_digest(present) == hashlib.sha256(b"ATOM\n").hexdigest()
    # A missing reference must not crash the gate; it downgrades the check.
    assert reference_digest(tmp_path / "absent.pdb") is None


def test_rejects_the_wrong_sampling_budget():
    reasons, matching, _ = assess(summary([jax_row()]), 8, "bf16")
    assert matching == 0
    assert any("sampling steps 64" in r and "pins 8" in r for r in reasons)


def test_rejects_the_wrong_precision():
    reasons, matching, _ = assess(
        summary([jax_row(opendde_dtype="fp32")]), 64, "bf16"
    )
    assert matching == 0
    assert any("precision 'fp32'" in r for r in reasons)


def test_rejects_unchecked_mapping():
    """An absent mapping result is unchecked, not passed."""
    for value in (None, False, "yes"):
        reasons, matching, _ = assess(
            summary([jax_row(mapping_agrees=value)]), 64, "bf16"
        )
        assert matching == 0
        assert any("mapping_agrees" in r for r in reasons)


def test_rejects_an_archive_whose_rows_fall_short_of_its_plan():
    """The gap finding 2 left open: a worker that vanished from the table."""
    value = summary([jax_row()], expected_rows=4, observed_rows=4)
    reasons, _, _ = assess(value, 64, "bf16")
    assert any("row coverage" in r for r in reasons)


def test_rejects_an_incomplete_row_and_surfaces_its_note():
    rows = [jax_row(), jax_row(completed=False, note="expected 2 reports, found 0")]
    reasons, _, _ = assess(summary(rows), 64, "bf16")
    assert any("expected 2 reports, found 0" in r for r in reasons)


def test_native_rows_need_no_mapping_but_never_count_as_evidence():
    native = jax_row(
        worker="native_steps64_seed0",
        path="native",
        aggregation_mode=None,
        mapping_agrees=None,
    )
    reasons, matching, _ = assess(summary([native]), 64, "bf16")
    assert not any("mapping_agrees" in r for r in reasons)
    assert matching == 0
    assert any("no passing JAX row" in r for r in reasons)


def test_older_archive_is_usable_but_reports_what_it_cannot_check():
    """Steps come from the worker name; precision and reference are unknown."""
    old = dict(
        worker="mosaic_stable_steps64_seed0",
        repeat=0,
        aggregation_mode="stable",
        completed=True,
        geometry_passed=True,
        mapping_agrees=True,
    )
    reasons, matching, unverified = assess(
        dict(completed=True, geometry_passed=True, rows=[old]), 64, "bf16"
    )
    assert reasons == []
    assert matching == 1
    assert any("precision not recorded" in item for item in unverified)
    assert any("reference identity not recorded" in item for item in unverified)
    assert any("coverage" in item for item in unverified)


def test_row_steps_prefers_the_field_over_the_worker_name():
    assert row_steps({"sampling_steps": 8, "worker": "m_steps64_seed0"}) == 8
    assert row_steps({"worker": "mosaic_stable_steps64_seed0"}) == 64
    assert row_steps({"worker": "no_step_count"}) is None


def test_search_prefers_a_supporting_archive_and_reports_failures(tmp_path):
    good = tmp_path / "p17_forward_validation_20261003_000002"
    bad = tmp_path / "p17_forward_validation_20261003_000001"
    for directory, rows in (
        (good, [jax_row()]),
        (bad, [jax_row(sampling_steps=8)]),
    ):
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(summary(rows)))

    path, matching, unverified = find_supporting_archive(
        tmp_path, 64, "bf16", REFERENCE
    )
    assert path == good / "summary.json"
    assert matching == 1
    assert unverified == []


def test_search_returns_every_failure_when_nothing_supports_the_run(tmp_path):
    directory = tmp_path / "p17_forward_validation_20261003_000001"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(summary([jax_row(sampling_steps=8)]))
    )
    (tmp_path / "p17_wt_validation_broken").mkdir()
    (tmp_path / "p17_wt_validation_broken" / "summary.json").write_text("{not json")

    path, failures, _ = find_supporting_archive(tmp_path, 64, "bf16")
    assert path is None
    assert len(failures) == 2
    assert any("unreadable" in r for reasons in failures.values() for r in reasons)


def test_no_candidates_is_a_failure_not_a_pass(tmp_path):
    path, failures, _ = find_supporting_archive(tmp_path, 64, "bf16")
    assert path is None
    assert failures == {}
