"""CAPRI/DockQ audit of a screen's held-out structures.

Why this exists. The project's two readouts are both the frozen predictor's
own output: interface confidence (ipSAE) and target-aligned pose RMSD. Neither
says whether the designed chain reproduces any of the reference interface's
contacts. Pose RMSD in particular is a rigid-body quantity -- a chain can move
20 A closer to where the reference chain sits without forming one native
contact -- so a pose gain is not an interface gain until `fnat` and `irmsd`
say so. Section 27.4 made that measurement for the 2026-10-06 screen and found
all 165 structures CAPRI `incorrect`, but it was computed outside the
repository and the script was not kept. This is that measurement, kept.

Scores every `heldout/shard_*/structures/candidate_*/seed_*.pdb` against its
own shard's `reference.pdb`, which is the same native the run itself measured
against. Takes either a screen with one subdirectory per cell or a single run
directory with `heldout/` at its root; `run_p17_consolidated.sh` passes both. The chain mapping is derived from residue counts rather than
hardcoded, so a renamed chain is an error rather than a silent mismatch.

DockQ pins `numpy<2`, which the project environment cannot satisfy, so it is
invoked as a subprocess against its own interpreter. Install it isolated:

    uv tool install dockq

Reads structures, runs no model, writes a CSV.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics as st

# DockQ's own published classification thresholds.
CAPRI_BOUNDS = ((0.80, "high"), (0.49, "medium"), (0.23, "acceptable"))

# THIS PROJECT'S OWN BAR, which is not CAPRI's. Set by the user on 2026-10-07
# and reaffirmed on 2026-10-08: a candidate counts when its interface RMSD is
# at or under this, because the designed chain is allowed to shift relative to
# the original interface and reproducing exact native contacts is not the goal.
# fnat and DockQ are still reported and still do not gate. Anything written for
# a reader outside this project must label this threshold explicitly, since a
# docking audience will assume CAPRI's acceptable class (<= 4 A, DockQ >= 0.23)
# and read the same table to the opposite conclusion.
IRMSD_THRESHOLD = 10.0


def capri_class(dockq):
    for bound, name in CAPRI_BOUNDS:
        if dockq >= bound:
            return name
    return "incorrect"


def chain_ca_counts(path):
    """CA atoms per chain, which is the residue count for these outputs."""
    counts = defaultdict(int)
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                counts[line[21]] += 1
    return dict(counts)


def derive_mapping(model_path, native_path):
    """`MODELCHAINS:NATIVECHAINS` for DockQ, matched by residue count.

    Both files hold exactly two chains, a designed chain and a target. The
    files disagree on the chain *letters* -- models write A/B, the reference
    writes B/T -- so pairing them by name would compare the designed chain to
    the target. Pairing by length is what is left, and an equal-length pair
    inside either file raises rather than guesses.

    The counts need not match ACROSS the two files. A predicted complex is the
    construct that was folded; a crystal reference is whatever was resolved,
    and on P17+Alpha that is one extra binder residue and ten extra target
    residues (123/195 against 124/205). DockQ aligns the two itself and scores
    that pair without complaint. Requiring equality here refused the Alpha
    run outright, which is the one task in this project with a known answer.

    What is still refused is an AMBIGUOUS pairing: the rank pairing must be
    strictly better than its swap, measured in residues. That is the hazard
    worth erroring on, because the swap compares the designed chain to the
    target and still produces a number.
    """
    model = chain_ca_counts(model_path)
    native = chain_ca_counts(native_path)
    for label, path, chains in (("model", model_path, model),
                                ("native", native_path, native)):
        if len(chains) != 2:
            raise ValueError(f"{label} {path} has chains {chains}, expected 2")
        if len(set(chains.values())) != 2:
            raise ValueError(
                f"{label} {path} chains are the same length ({chains}); the "
                "mapping cannot be derived by length"
            )
    m_designed, m_target = sorted(model, key=lambda c: model[c])
    n_designed, n_target = sorted(native, key=lambda c: native[c])
    aligned = (abs(model[m_designed] - native[n_designed])
               + abs(model[m_target] - native[n_target]))
    swapped = (abs(model[m_designed] - native[n_target])
               + abs(model[m_target] - native[n_designed]))
    if aligned >= swapped:
        raise ValueError(
            f"chain pairing is ambiguous: model {model} against native "
            f"{native} costs {aligned} residues paired by rank and {swapped} "
            "swapped, so the designed chain cannot be told from the target by "
            "length"
        )
    return (f"{m_designed}{m_target}:{n_designed}{n_target}", model[m_designed],
            native[n_designed] - model[m_designed],
            native[n_target] - model[m_target])


def run_dockq(dockq_bin, model_path, native_path, mapping):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dockq.json"
        proc = subprocess.run(
            [dockq_bin, "--mapping", mapping, "--json", str(out),
             str(model_path), str(native_path)],
            capture_output=True, text=True,
        )
        if not out.exists():
            raise RuntimeError(
                f"DockQ wrote no JSON for {model_path}\n{proc.stderr[-800:]}"
            )
        payload = json.loads(out.read_text())
    # One interface, so the single `best_result` entry carries the metrics.
    (result,) = payload["best_result"].values()
    return result


def wt_candidate_ids(run_root):
    """Candidate ids the run itself marked as the unmodified start.

    The JN.1 driver writes `is_wt` into jn1_recovery.csv and that column is
    authoritative. The Alpha driver's recovery.csv does not carry it, so fall
    back to the convention that builds these candidate lists: candidate 0 is
    the unmodified start (p17_rescore_winners.py registers it as
    `is_wt=idx == 0`). Returning an empty set instead would quietly fold the
    damaged start into the cell mean as if it were a design.
    """
    for name in ("jn1_recovery.csv", "recovery.csv"):
        table = run_root / "tables" / name
        if not table.exists():
            continue
        with open(table) as fh:
            rows = list(csv.DictReader(fh))
        if rows and "is_wt" in rows[0]:
            return {int(r["candidate_id"]) for r in rows if r["is_wt"] == "True"}
        return {0}
    return set()


def cell_runs(screen_root):
    """`(cell name, run root)` for every cell under a screen.

    Two layouts exist in this project and both are passed to this script.
    A screen holds one subdirectory per cell, each with its own `run/`
    (`explore/arity2/run/heldout`, `<consolidated>/bindcraft/run/heldout`).
    A single run holds `heldout/` at its own root instead
    (`run_p17_alpha_recovery.sh` makes `ladder/ calibrate/ search/ heldout/`
    side by side). Taking a run root and finding no cells inside it used to
    report "no held-out structures", which is true of the directory and false
    of the run.
    """
    if (screen_root / "heldout").is_dir():
        return [(screen_root.name, screen_root)]
    return [
        (child.name, child / "run")
        for child in sorted(screen_root.iterdir())
        if (child / "run" / "heldout").is_dir()
    ]


def discover(screen_root, cells):
    """`(scoreable rows, unscoreable shards)` under a screen or a single run.

    A shard with no `reference.pdb` has no ground truth, so nothing inside it
    can be scored. That used to raise, which let one incomplete shard of eight
    block the audit of the other seven -- the 2026-10-12 consolidated run died
    on `alpha/heldout/shard_5` with seven good shards beside it. It is reported
    instead, with the number of structures stranded in it, because the failure
    the raise was guarding against is a count that silently shrinks: "16 of 40
    inside the bar" must never be able to mean "of the 40 that happened to
    finish writing".
    """
    found, incomplete = [], []
    for cell, run_root in cell_runs(screen_root):
        if cells and cell not in cells:
            continue
        wt_ids = wt_candidate_ids(run_root)
        for shard in sorted((run_root / "heldout").glob("shard_*")):
            native = shard / "reference.pdb"
            if not native.exists():
                stranded = len(list((shard / "structures").glob("candidate_*/seed_*.pdb")))
                incomplete.append(dict(
                    cell=cell, shard=shard.name, path=shard,
                    reason="no reference.pdb", stranded=stranded,
                ))
                continue
            for cand in sorted((shard / "structures").glob("candidate_*")):
                cid = int(cand.name.split("_")[-1])
                for model in sorted(cand.glob("seed_*.pdb")):
                    found.append(dict(
                        cell=cell, shard=shard.name, candidate_id=cid,
                        seed=model.stem.split("_")[-1], model=model,
                        native=native, is_wt=cid in wt_ids,
                    ))
    return found, incomplete


def per_candidate_irmsd(rows):
    """Mean iRMSD per candidate, which is the independent unit.

    The held-out structural seeds of one candidate are three predictions of the
    same sequence and are correlated; they are not three measurements. Counting
    structures against the bar inflates n threefold and lets one lucky seed of
    one candidate carry a cell.
    """
    byc = defaultdict(list)
    for r in rows:
        byc[(r.get("cell"), r.get("candidate_id"))].append(r["iRMSD"])
    return {k: st.mean(v) for k, v in byc.items()}


def summarize(rows, label, threshold=IRMSD_THRESHOLD):
    if not rows:
        print(f"{label:12s} no structures")
        return
    def agg(key, best=max):
        v = [r[key] for r in rows]
        return st.mean(v), st.median(v), best(v)
    dq = agg("DockQ")
    fn = agg("fnat")
    ir = agg("iRMSD", min)
    lr = agg("LRMSD", min)
    classes = defaultdict(int)
    for r in rows:
        classes[r["capri_class"]] += 1
    worst_first = ", ".join(
        f"{n} {c}" for c, n in sorted(classes.items(), key=lambda kv: -kv[1])
    )
    # The project's own bar, per candidate. Reported first because it is the
    # criterion this work is actually judged against; the CAPRI classes stay
    # because they carry information the bar does not -- candidates have
    # cleared 10 A with fnat 0.000, so the bar can be met by placement alone.
    cand = per_candidate_irmsd(rows)
    hits = sum(1 for v in cand.values() if v <= threshold)
    mean_c = st.mean(cand.values())
    sem_c = (st.stdev(cand.values()) / len(cand) ** 0.5) if len(cand) > 1 else 0.0
    print(f"{label:12s} iRMSD<={threshold:g}A {hits:3d}/{len(cand):<3d} candidates   "
          f"mean {mean_c:5.2f} +- {sem_c:4.2f}   best {min(cand.values()):5.2f}")
    print(f"{'':12s} n={len(rows):3d}  DockQ {dq[0]:.3f}/{dq[1]:.3f}/{dq[2]:.3f}  "
          f"fnat {fn[0]:.3f}/{fn[1]:.3f}/{fn[2]:.3f}  "
          f"iRMSD {ir[0]:5.2f}/{ir[1]:5.2f}/{ir[2]:5.2f}  "
          f"LRMSD {lr[0]:6.2f}/{lr[1]:6.2f}/{lr[2]:6.2f}  [{worst_first}]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("screen_root", type=Path,
                        help="a screen directory holding one subdirectory per "
                             "cell, or a single run directory holding heldout/")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default <screen_root>/capri.csv)")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="restrict to these cell names")
    parser.add_argument("--dockq-bin", default=None,
                        help="DockQ executable (default: the one on PATH)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--irmsd-threshold", type=float, default=IRMSD_THRESHOLD,
                        help="this project's acceptance bar in angstroms "
                             "(default: %(default)s). Candidates are counted "
                             "against it per candidate, not per structure.")
    parser.add_argument("--include-wt", action="store_true",
                        help="keep the unmodified start in the per-cell summary; "
                             "it is always written to the CSV")
    args = parser.parse_args(argv)

    dockq_bin = args.dockq_bin or shutil.which("DockQ") or str(
        Path.home() / ".local" / "bin" / "DockQ"
    )
    if not Path(dockq_bin).exists():
        parser.error(
            f"no DockQ executable at {dockq_bin}; install it isolated with "
            "`uv tool install dockq` (it pins numpy<2 and must not share this "
            "project's environment)"
        )

    jobs, incomplete = discover(
        args.screen_root, set(args.cells) if args.cells else None
    )
    if incomplete:
        # Before the numbers, never after. A reader who sees the counts first
        # has already formed the conclusion by the time they reach the caveat.
        total = sum(entry["stranded"] for entry in incomplete)
        print(
            f"WARNING: {len(incomplete)} shard(s) cannot be scored and are "
            f"excluded, stranding {total} structure(s). Every count below is "
            "over what remains, not over what was launched."
        )
        for entry in incomplete:
            print(f"  {entry['cell']}/{entry['shard']}: {entry['reason']} "
                  f"({entry['stranded']} structure(s) stranded)")
        print()
    if not jobs:
        parser.error(
            f"no held-out structures under {args.screen_root}; expected either "
            "<screen_root>/heldout/shard_* (a single run) or "
            "<screen_root>/<cell>/run/heldout/shard_* (a screen)"
        )
    mapping, designed_len, d_designed, d_target = derive_mapping(
        jobs[0]["model"], jobs[0]["native"]
    )
    print(f"{len(jobs)} structures, chain mapping {mapping} "
          f"(designed chain {designed_len} residues), {args.workers} workers",
          flush=True)
    if d_designed or d_target:
        # Not an error -- see derive_mapping -- but say it, because a large
        # difference means the reference resolves residues the model never had
        # and DockQ is scoring the intersection.
        print(f"  native has {d_designed:+d} designed and {d_target:+d} target "
              "residues relative to the model; DockQ aligns them and scores "
              "the common interface", flush=True)

    def score(job):
        result = run_dockq(dockq_bin, job["model"], job["native"], mapping)
        row = dict(job)
        row["model"] = str(job["model"].relative_to(args.screen_root))
        row["native"] = str(job["native"].relative_to(args.screen_root))
        row.update({k: result[k] for k in
                    ("DockQ", "fnat", "fnonnat", "iRMSD", "LRMSD", "F1",
                     "nat_correct", "nat_total", "clashes")})
        row["capri_class"] = capri_class(result["DockQ"])
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(score, jobs), start=1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  scored {i}/{len(jobs)}", flush=True)

    out = args.out or args.screen_root / "capri.csv"
    fields = ["cell", "shard", "candidate_id", "seed", "is_wt", "capri_class",
              "DockQ", "fnat", "fnonnat", "iRMSD", "LRMSD", "F1",
              "nat_correct", "nat_total", "clashes", "model", "native"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)

    if incomplete:
        # Repeated at the point of use. A reader who scrolls straight to the
        # counts would otherwise never see the exclusion.
        total = sum(entry["stranded"] for entry in incomplete)
        print(f"\n({len(incomplete)} shard(s) excluded above, {total} "
              "structure(s) stranded -- the counts below are over what remains)")
    print("\nmean/median/best per cell"
          + ("" if args.include_wt else ", excluding the unmodified start"))
    scored = rows if args.include_wt else [r for r in rows if not r["is_wt"]]
    for cell in sorted({r["cell"] for r in scored}):
        summarize([r for r in scored if r["cell"] == cell], cell,
                  args.irmsd_threshold)
    summarize(scored, "ALL", args.irmsd_threshold)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
