"""Does a completed forward control support the configuration about to launch?

The pose experiment is gated on forward evidence from this checkout. Deciding
that is a predicate over saved summaries, so it lives here rather than inside a
shell heredoc: `tests/test_forward_evidence.py` exercises it directly.

Reads only root `summary.json` files. Loads no model and runs nothing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO / "P17_JN1.pdb"

SEARCH_GLOBS = (
    "p17_forward_validation_*/summary.json",
    "forward_p17_review*/*/summary.json",
    "p17_wt_validation_*/summary.json",
)


def row_steps(row):
    """Sampling steps for a row, from the field or the worker name.

    Archives written before the collector recorded `sampling_steps` encode it
    in the worker name as `_steps<N>_`. Returns None when neither source gives
    it, which the caller treats as unverified rather than matching.
    """
    value = row.get("sampling_steps")
    if isinstance(value, int):
        return value
    match = re.search(r"_steps(\d+)(?:_|$)", str(row.get("worker", "")))
    return int(match.group(1)) if match else None


def reference_digest(path):
    """SHA-256 of the reference PDB, or None when it cannot be read.

    Returning None rather than raising keeps a missing reference from turning
    the gate into a crash: the caller downgrades the identity check to
    unverified and says so.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def assess(summary, want_steps, want_dtype, want_reference=None):
    """Reasons this archive does not support the run, plus what went unchecked.

    A supporting archive needs at least one JAX prediction that completed,
    passed geometry, carries an explicit `mapping_agrees` of True, used stable
    aggregation, and ran at `want_steps`. An absent key is unchecked, not
    passed. Native rows have no production mapping step, so that key is not
    required there, and they never count as stable-aggregation evidence.

    Precision and reference identity are checked only when the archive records
    them; older archives do not, and those are returned as unverified rather
    than silently treated as matching.

    `want_reference` is the SHA-256 the run's own reference PDB hashes to. A
    row recording a *different* digest is rejected: it is evidence about some
    other complex, which is worse than no evidence. Passing None skips the
    comparison and reports it as unverified, so an archive cannot be accepted
    as reference-matched merely because it recorded some digest.

    Returns `(reasons, matching_rows, unverified)`. Empty `reasons` means the
    archive supports the run.
    """
    reasons, unverified = [], []
    rows = summary.get("rows") or []
    if summary.get("completed") is not True:
        reasons.append("run did not complete")
    if summary.get("geometry_passed") is not True:
        reasons.append("geometry did not pass")
    if not rows:
        reasons.append("no per-prediction rows")

    # Coverage: reject an archive whose own table is short of its plan. Absent
    # on older archives, where row-level checks are all that is available.
    expected_rows = summary.get("expected_rows")
    observed_rows = summary.get("observed_rows")
    if isinstance(expected_rows, int) and isinstance(observed_rows, int):
        if observed_rows != expected_rows or len(rows) != expected_rows:
            reasons.append(
                f"row coverage {len(rows)}/{observed_rows} against a planned "
                f"{expected_rows}"
            )
    else:
        unverified.append("planned-versus-present worker coverage")

    matching = 0
    for row in rows:
        worker = str(row.get("worker", "?"))
        native = str(row.get("path") or worker).startswith("native")
        if row.get("completed") is not True:
            note = row.get("note")
            reasons.append(
                f"{worker}: did not complete" + (f" ({note})" if note else "")
            )
            continue
        if row.get("geometry_passed") is not True:
            reasons.append(f"{worker}: geometry did not pass")
            continue
        if native:
            continue
        if row.get("mapping_agrees") is not True:
            reasons.append(
                f"{worker}: mapping_agrees is "
                f"{row.get('mapping_agrees')!r}, expected True"
            )
            continue
        if row.get("aggregation_mode") != "stable":
            continue
        steps = row_steps(row)
        if steps != want_steps:
            reasons.append(
                f"{worker}: sampling steps {steps!r}, this run pins {want_steps}"
            )
            continue
        dtype = row.get("opendde_dtype")
        if dtype is None:
            unverified.append(f"{worker}: precision not recorded")
        elif dtype != want_dtype:
            reasons.append(f"{worker}: precision {dtype!r}, this run pins {want_dtype}")
            continue
        digest = row.get("reference_sha256")
        if digest is None:
            unverified.append(f"{worker}: reference identity not recorded")
        elif want_reference is None:
            unverified.append(
                f"{worker}: reference identity recorded but not compared"
            )
        elif digest != want_reference:
            reasons.append(
                f"{worker}: reference sha256 {str(digest)[:12]}..., this run "
                f"uses {want_reference[:12]}..."
            )
            continue
        matching += 1

    if not matching:
        reasons.append(
            f"no passing JAX row with stable aggregation at {want_steps} steps, "
            "which this run pins"
        )
    return reasons, matching, unverified


def candidate_summaries(results_dir):
    paths = set()
    for pattern in SEARCH_GLOBS:
        paths |= set(Path(results_dir).glob(pattern))
    return sorted(paths)


def find_supporting_archive(results_dir, want_steps, want_dtype, want_reference=None):
    """Newest-first search for an archive that supports this configuration.

    Returns `(path, matching_rows, unverified)` on success, or
    `(None, failures, [])` where `failures` maps each candidate to its reasons.
    """
    failures = {}
    for path in reversed(candidate_summaries(results_dir)):
        try:
            summary = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            failures[path] = [f"unreadable: {type(exc).__name__}"]
            continue
        reasons, matching, unverified = assess(
            summary, want_steps, want_dtype, want_reference
        )
        if not reasons:
            return path, matching, unverified
        failures[path] = reasons
    return None, failures, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--aggregation", default="stable")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="reference complex this run will use; archives recording a "
        "different digest are rejected (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    want_reference = reference_digest(args.reference)
    if want_reference is None:
        print(
            f"Reference {args.reference} is unreadable, so archive reference "
            "identity cannot be compared and is reported as unverified.",
            file=sys.stderr,
        )

    if args.aggregation != "stable":
        print(
            f"Aggregation {args.aggregation!r} is a deliberate control; forward "
            "evidence is checked for the stable arm only.",
            file=sys.stderr,
        )

    path, detail, unverified = find_supporting_archive(
        args.results_dir, args.steps, args.dtype, want_reference
    )
    if path is not None:
        print(
            f"Forward controls: passing evidence at {path} "
            f"({detail} stable-aggregation JAX predictions at {args.steps} steps)"
        )
        for item in dict.fromkeys(unverified):
            print(f"  not verified from this archive: {item}")
        if unverified:
            print(
                "  These were not recorded by the archive, so the gate does not "
                "assert them. Rerun the forward control from the current "
                "checkout to have them checked."
            )
        return 0

    print(
        "No forward control in results/ supports this run's configuration.",
        file=sys.stderr,
    )
    for candidate, reasons in detail.items():
        shown = reasons[:4]
        print(f"  {candidate}:", file=sys.stderr)
        for reason in shown:
            print(f"    - {reason}", file=sys.stderr)
        if len(reasons) > len(shown):
            print(f"    - ... and {len(reasons) - len(shown)} more", file=sys.stderr)
    if not detail:
        print("  (no candidate summary.json found)", file=sys.stderr)
    print(
        "Run examples/run_p17_forward_validation.sh first, copy the reviewed "
        "archive\ninto this checkout, or pass --skip-forward-check to launch "
        "anyway.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
