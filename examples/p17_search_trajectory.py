"""Reconstruct a completed search's move trajectory from its event log.

Section 20.11 item 6 persists the full first-order delta matrix, but only for
runs launched after `--save-saliency` existed. For a run that finished before
then the event log is all there is -- and it does hold the chosen move, its
predicted first-order delta and the realized score, which is enough to ask how
the search reached a given substitution and whether it ever tried a particular
one.

Two quantities appear here and they are NOT the same objective, which is
exactly §19.6's point:

  predicted_loss_delta  first-order Taylor delta on the *proposal loss*, which
                        includes the pose hinge and the auxiliary weights.
                        Negative means the proposal expects an improvement.
  realized_score_delta  change in the *retention objective*, the mean
                        directional-min ipSAE across selection seeds, measured
                        by actually folding the candidate. Positive is better.

So an informative gradient makes `-predicted_loss_delta` and
`realized_score_delta` agree in sign more often than chance. They can disagree
without either being wrong, because they rank different things; a systematic
disagreement is the interesting case, and the sign-agreement rate below is a
cheap read on it rather than a test of the gradient's correctness.

Nothing here needs a GPU: it reads only what the run already wrote.
"""

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


def load_events(log_path):
    """Every JSON record in order, skipping a truncated final line.

    A run killed mid-write leaves a partial line; dropping only a trailing
    fragment keeps the rest usable, while a malformed line anywhere earlier is
    a corrupt log and raises.
    """
    records, lines = [], Path(log_path).read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise ValueError(f"{log_path} line {number} is not valid JSON")
    return records


def sequences_and_scores(events):
    """candidate_id -> (token sequence, score) from the evaluation records."""
    sequences, scores = {}, {}
    for event in events:
        if event.get("event") != "evaluation":
            continue
        candidate = int(event["candidate_id"])
        sequences[candidate] = [int(t) for t in event["sequence"]]
        scores[candidate] = float(event["score"])
    return sequences, scores


def trajectory(events, reference_sequence=None):
    """One row per proposal, with the move named in residue terms.

    `move` is [position_0idx, residue_token, reverted_position_0idx]. The
    third element does NOT mean the move is itself a reversion. `_moves()` in
    `src/mosaic/search.py:97` pairs a new edit with restoring some *other*
    position to the search's start sequence, because at the edit cap a new
    edit is only feasible if an existing one is given up. So it names the
    position paid for this move, and at the cap nearly every proposal has one.

    Note what "start" means for a recovery run: the search's own reference is
    the *damaged* sequence, so restoring a position to the start restores the
    damage. Whether a move goes toward the undamaged crystal is a different
    question, and `reference_sequence` is what answers it -- the runs launched
    before 2026-10-05 recorded `reference_binder_sequence: null`, so it has to
    be supplied rather than read back from the run.

    The *from* residue is read off the parent's own sequence rather than any
    reference, so a second edit at an already-edited position reports what it
    actually replaced.
    """
    from mosaic.common import TOKENS

    sequences, scores = sequences_and_scores(events)
    rows = []
    for event in events:
        if event.get("event") != "proposal":
            continue
        position, token, reverted = (int(v) for v in event["move"])
        parent = int(event["parent_id"])
        child = int(event["candidate_id"])
        parent_sequence = sequences.get(parent)
        from_residue = (
            TOKENS[parent_sequence[position]] if parent_sequence else ""
        )
        child_score = scores.get(child)
        parent_score = scores.get(parent)
        realized = (
            child_score - parent_score
            if child_score is not None and parent_score is not None
            else None
        )
        predicted = float(event["predicted_loss_delta"])
        rows.append(
            dict(
                gradient_calls=int(event.get("gradient_calls", -1)),
                score_calls=int(event.get("score_calls", -1)),
                attempt=int(event.get("attempt", -1)),
                parent_id=parent,
                candidate_id=child,
                position_0idx=position,
                position_1idx=position + 1,
                from_residue=from_residue,
                to_residue=TOKENS[token],
                # The other position given up to afford this edit, or -1.
                paired_reversion_0idx=reverted,
                paid_for_by_reversion=reverted >= 0,
                # Did this move put the undamaged crystal residue back?
                restores_reference=(
                    None if reference_sequence is None
                    else TOKENS[token] == reference_sequence[position]
                ),
                reference_residue=(
                    "" if reference_sequence is None
                    else reference_sequence[position]
                ),
                predicted_loss_delta=predicted,
                parent_score=parent_score,
                candidate_score=child_score,
                realized_score_delta=realized,
                # The two objectives differ, so this records agreement, not
                # correctness. See the module docstring.
                sign_agrees=(
                    None if realized is None
                    else (predicted < 0) == (realized > 0)
                ),
                accepted=bool(event.get("accepted", False)),
                duplicate=bool(event.get("duplicate", False)),
                cache_hit=bool(event.get("cache_hit", False)),
                feasible_moves=int(event.get("feasible_moves", -1)),
                entropy=float(event.get("entropy", float("nan"))),
                proposal_temperature=float(
                    event.get("proposal_temperature", float("nan"))
                ),
            )
        )
    return rows


def position_history(rows):
    """Per-position counts and deltas, most-proposed first.

    Answers "which positions did the search keep returning to", which is the
    question §8.3 asks of LaMBO-2's position selection.
    """
    grouped = {}
    for row in rows:
        entry = grouped.setdefault(
            row["position_0idx"],
            dict(
                position_0idx=row["position_0idx"],
                position_1idx=row["position_1idx"],
                proposed=0,
                accepted=0,
                restores_reference=0,
                residues_tried=set(),
                residues_accepted=set(),
                predicted_deltas=[],
                realized_deltas=[],
            ),
        )
        entry["proposed"] += 1
        entry["residues_tried"].add(row["to_residue"])
        entry["predicted_deltas"].append(row["predicted_loss_delta"])
        if row["restores_reference"]:
            entry["restores_reference"] += 1
        if row["accepted"]:
            entry["accepted"] += 1
            entry["residues_accepted"].add(row["to_residue"])
        if row["realized_score_delta"] is not None:
            entry["realized_deltas"].append(row["realized_score_delta"])

    def mean(values):
        return sum(values) / len(values) if values else None

    out = []
    for entry in grouped.values():
        out.append(
            dict(
                position_0idx=entry["position_0idx"],
                position_1idx=entry["position_1idx"],
                proposed=entry["proposed"],
                accepted=entry["accepted"],
                restores_reference=entry["restores_reference"],
                residues_tried="".join(sorted(entry["residues_tried"])),
                residues_accepted="".join(sorted(entry["residues_accepted"])),
                mean_predicted_loss_delta=mean(entry["predicted_deltas"]),
                best_predicted_loss_delta=min(entry["predicted_deltas"]),
                mean_realized_score_delta=mean(entry["realized_deltas"]),
            )
        )
    out.sort(key=lambda r: (-r["proposed"], r["position_0idx"]))
    return out


def sign_agreement(rows):
    """How often the proposal's expected direction matched the measured one.

    Only proposals whose parent and child were both scored can be counted; a
    duplicate or a cache hit has no fresh measurement.
    """
    judged = [r for r in rows if r["sign_agrees"] is not None]
    if not judged:
        return dict(counted=0, agreed=0, rate=None)
    agreed = sum(1 for r in judged if r["sign_agrees"])
    return dict(
        counted=len(judged),
        agreed=agreed,
        rate=agreed / len(judged),
    )


def trace_position(rows, position_1idx):
    """Every proposal that touched one position, in order."""
    return [r for r in rows if r["position_1idx"] == position_1idx]


def write_tables(output_dir, rows, positions):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("trajectory.csv", rows), ("position_history.csv", positions)):
        if not data:
            continue
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return [output_dir / "trajectory.csv", output_dir / "position_history.csv"]


def find_runs(root):
    """Search run directories under `root`, or `root` itself if it is one."""
    root = Path(root)
    if (root / "logs/events.jsonl").is_file():
        return [root]
    return sorted(
        path.parent.parent
        for path in root.glob("*/logs/events.jsonl")
    ) or sorted(
        path.parent.parent for path in root.glob("search/*/logs/events.jsonl")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="a search run directory, or one holding search/<name>/ runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write tables (default: <run>/tables/trajectory/)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="reference complex whose binder sequence the damage was applied "
        "to, e.g. P17_Alpha.pdb. Without it, whether a move restores the "
        "crystal residue cannot be told from a pre-2026-10-05 run.",
    )
    parser.add_argument(
        "--binder-chain", default="B",
        help="binder chain in --reference (default: %(default)s)",
    )
    parser.add_argument(
        "--position",
        type=int,
        action="append",
        default=[],
        help="1-indexed binder position to trace in detail; repeatable",
    )
    args = parser.parse_args(argv)

    runs = find_runs(args.run)
    if not runs:
        parser.error(f"no logs/events.jsonl under {args.run}")

    reference = None
    if args.reference is not None:
        import sys

        sys.path.insert(0, str(EXAMPLES))
        from p17_alpha_reference import load_complex

        reference = load_complex(args.reference, args.binder_chain, "A")[
            "binder_seq"
        ]

    for run in runs:
        events = load_events(run / "logs/events.jsonl")
        rows = trajectory(events, reference)
        positions = position_history(rows)
        agreement = sign_agreement(rows)
        destination = args.output_dir or (run / "tables/trajectory")
        if len(runs) > 1 and args.output_dir is not None:
            destination = Path(args.output_dir) / run.name
        write_tables(destination, rows, positions)

        print(f"\n=== {run.name} ===")
        accepted = sum(1 for r in rows if r["accepted"])
        print(
            f"  {len(rows)} proposals, {accepted} accepted, "
            f"{sum(1 for r in rows if r['paid_for_by_reversion'])} paid for by "
            f"a reversion elsewhere, "
            f"{sum(1 for r in rows if r['duplicate'])} duplicates"
        )
        if agreement["rate"] is not None:
            print(
                f"  predicted/realized sign agreement: {agreement['agreed']}"
                f"/{agreement['counted']} ({agreement['rate']:.0%}) -- "
                "different objectives, so this is agreement, not correctness"
            )
        if reference is not None:
            toward = [r for r in rows if r["restores_reference"]]
            print(
                f"  {len(toward)} proposals put a crystal residue back, "
                f"{sum(1 for r in toward if r['accepted'])} accepted"
            )
        print("  most-visited positions:")
        for entry in positions[:6]:
            print(
                f"    {entry['position_1idx']:>4}  proposed {entry['proposed']:>2}"
                f"  accepted {entry['accepted']:>2}"
                f"  tried [{entry['residues_tried']}]"
                f"  best Dloss {entry['best_predicted_loss_delta']:+.4f}"
            )
        for position in args.position:
            traced = trace_position(rows, position)
            print(f"  position {position}:")
            if not traced:
                print("    never proposed")
                continue
            for row in traced:
                realized = (
                    f"{row['realized_score_delta']:+.4f}"
                    if row["realized_score_delta"] is not None
                    else "   n/a"
                )
                print(
                    f"    grad {row['gradient_calls']:>3}  "
                    f"{row['from_residue']}->{row['to_residue']}  "
                    f"predicted {row['predicted_loss_delta']:+.4f}  "
                    f"realized {realized}  "
                    f"{'accepted' if row['accepted'] else 'rejected'}"
                    f"{'  [restores crystal residue]' if row['restores_reference'] else ''}"
                )
        print(f"  wrote {destination}/trajectory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
