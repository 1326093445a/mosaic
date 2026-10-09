"""Test B: what rank did the correct residue get, at the gradient calls that happened.

WHY THIS EXISTS. §25 localized the residual failure as registry at the correct
site, and the open question underneath it is whether the first-order ranking
ever surfaces the right substitution. The search throws away everything but the
sampled move, so `tables/saliency.csv` -- written only under `--save-saliency`
-- is the one record of what the ranking looked like. This reads it.

WHAT IT ANSWERS. For a run started from a damaged sequence whose undamaged
reference is known, at every damaged position: among all moves offered at that
position, what rank did the move back to the REFERENCE residue receive? Rank 1
is the most favourable first-order delta.

That discriminates between two different failures with different fixes:

    ranked high, never sampled   ->  a sampling problem. Top-k enumeration or a
                                     lower proposal temperature would fix it.
    ranked low                   ->  the first-order ranking does not contain
                                     the answer at these points, and a better
                                     sampler over the same ranking cannot help.

⚠️ WHAT IT CANNOT ANSWER, from §28.8. It reads ranks at the gradient calls that
ACTUALLY OCCURRED. It says nothing about whether identity information is
available elsewhere along a trajectory, continuous or otherwise. Calling it a
gate on "whether the identity information is in the gradient at all" was too
strong and was corrected in §28.8. It chooses between the two failures above;
it does not settle whether continuous exploration has value.

No GPU, no models, no checkpoints. Reads CSV and JSON.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_config(run_dir):
    """The run's own record of where it started and what the reference was."""
    path = Path(run_dir, "config.json")
    if not path.is_file():
        raise SystemExit(f"no config.json in {run_dir}")
    return json.loads(path.read_text())


def damaged_positions(start, reference):
    if len(start) != len(reference):
        raise SystemExit(
            f"start is {len(start)} aa and reference is {len(reference)} aa"
        )
    return {i: reference[i] for i in range(len(start)) if start[i] != reference[i]}


def collect(saliency_path, wanted):
    """Ranks for the move back to the reference residue, per damaged position.

    `wanted` maps 0-indexed position to the reference residue. A row counts when
    it offers that residue at that position; `offered` counts every row at that
    position, so a rank can be read against the size of the pool it came from.
    """
    hits = defaultdict(list)
    offered = defaultdict(int)
    pool_sizes = defaultdict(set)
    rows = 0
    with open(saliency_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"position_0idx", "to_residue", "rank", "gradient_call"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(
                f"{saliency_path} is missing columns {sorted(missing)}; it was "
                "probably written by a version predating --save-saliency's "
                "current schema"
            )
        for row in reader:
            rows += 1
            pos = int(row["position_0idx"])
            if pos not in wanted:
                continue
            offered[pos] += 1
            pool_sizes[pos].add(int(row["gradient_call"]))
            if row["to_residue"] == wanted[pos]:
                hits[pos].append(int(row["rank"]))
    return hits, offered, pool_sizes, rows


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path,
        help="a search run directory holding config.json and tables/saliency.csv",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--top", type=int, default=10,
        help="a rank at or below this counts as surfaced (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    config = read_config(args.run)
    start = config["binder_sequence"]
    reference = config.get("reference_binder_sequence") or start
    if start == reference:
        raise SystemExit(
            "this run started AT its reference, so no position is damaged and "
            "there is no correct residue to rank. Test B needs a run started "
            "from a damaged sequence -- the Alpha recovery ladder's rungs."
        )

    saliency = args.run / "tables" / "saliency.csv"
    if not saliency.is_file():
        raise SystemExit(
            f"no {saliency}; the run was launched without --save-saliency and "
            "the ranking cannot be reconstructed from the event log afterwards"
        )

    wanted = damaged_positions(start, reference)
    hits, offered, pools, rows = collect(saliency, wanted)

    print(f"run:        {args.run}")
    print(f"saliency:   {rows} rows")
    print(f"damaged at: {sorted(p + 1 for p in wanted)}  "
          f"({len(wanted)} positions)")
    print()
    print(f"  {'pos':>5}  {'want':>4}  {'offers':>6}  {'calls':>5}  "
          f"{'best':>5}  {'median':>6}  {'top' + str(args.top):>6}")

    per_position = {}
    surfaced_any = 0
    for pos in sorted(wanted):
        ranks = sorted(hits.get(pos, []))
        calls = len(pools.get(pos, ()))
        best = ranks[0] if ranks else None
        median = percentile(ranks, 0.5)
        in_top = sum(1 for r in ranks if r <= args.top)
        if best is not None and best <= args.top:
            surfaced_any += 1
        per_position[pos + 1] = dict(
            reference_residue=wanted[pos],
            offers=offered.get(pos, 0),
            gradient_calls_touching_position=calls,
            best_rank=best,
            median_rank=median,
            ranks_in_top_n=in_top,
            all_ranks=ranks,
        )
        print(f"  {pos + 1:>5}  {wanted[pos]:>4}  {offered.get(pos, 0):>6}  "
              f"{calls:>5}  {str(best):>5}  {str(median):>6}  {in_top:>6}")

    never_offered = [p + 1 for p in sorted(wanted) if not offered.get(p)]
    all_ranks = sorted(r for pos in hits for r in hits[pos])
    report = dict(
        run=str(args.run),
        start_sequence=start,
        reference_sequence=reference,
        damaged_positions_1idx=sorted(p + 1 for p in wanted),
        top_n=args.top,
        saliency_rows=rows,
        per_position=per_position,
        positions_whose_correct_residue_reached_top_n=surfaced_any,
        positions_never_offered=never_offered,
        overall=dict(
            best_rank=all_ranks[0] if all_ranks else None,
            median_rank=percentile(all_ranks, 0.5),
            p90_rank=percentile(all_ranks, 0.9),
            observations=len(all_ranks),
        ),
        interpretation=(
            "Rank of the move back to the reference residue, among the moves "
            "offered at that position, at the gradient calls that actually "
            "occurred. A high rank that was never sampled is a sampling "
            "problem; a low rank means the first-order ranking did not contain "
            "the answer at these points. Per §28.8 this cannot speak to "
            "information available elsewhere along a trajectory."
        ),
    )

    print()
    if never_offered:
        print(f"never offered a move at: {never_offered}")
    if all_ranks:
        print(f"over all damaged positions: best rank {all_ranks[0]}, "
              f"median {percentile(all_ranks, 0.5)}, "
              f"p90 {percentile(all_ranks, 0.9)}, n={len(all_ranks)}")
        print(f"correct residue reached the top {args.top} at "
              f"{surfaced_any} of {len(wanted)} damaged positions")
        if surfaced_any == 0:
            print()
            print("  READING: the first-order ranking never surfaced the correct")
            print("  residue at these points. A better sampler over the same")
            print("  ranking cannot recover it; the ranking itself is the limit.")
        elif surfaced_any == len(wanted):
            print()
            print("  READING: the ranking surfaced every correct residue. The")
            print("  information was present and the sampler did not take it,")
            print("  which points at top-k enumeration or a colder temperature.")
        else:
            print()
            print(f"  READING: mixed. {surfaced_any} of {len(wanted)} positions had")
            print("  the answer available in the top ranks and the rest did not,")
            print("  so neither a sampling fix nor a ranking fix covers the result")
            print("  on its own.")
    else:
        print("the reference residue was never offered at any damaged position")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
