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
against. The chain mapping is derived from residue counts rather than
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
    the target. Pairing by length is unambiguous here because the two chains
    differ in length, and an equal-length pair raises rather than guesses.
    """
    model = chain_ca_counts(model_path)
    native = chain_ca_counts(native_path)
    for name, chains in (("model", model), ("native", native)):
        if len(chains) != 2:
            raise ValueError(f"{name} {model_path} has chains {chains}, expected 2")
    if sorted(model.values()) != sorted(native.values()):
        raise ValueError(
            f"chain lengths differ: model {model} vs native {native}; the "
            "native does not match this model"
        )
    if len(set(model.values())) != 2:
        raise ValueError(f"model chains are the same length ({model}); mapping "
                         "cannot be derived by length")
    m_designed, m_target = sorted(model, key=lambda c: model[c])
    n_designed, n_target = sorted(native, key=lambda c: native[c])
    return f"{m_designed}{m_target}:{n_designed}{n_target}", model[m_designed]


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


def wt_candidate_ids(cell_root):
    """Candidate ids the run itself marked as the unmodified start."""
    table = cell_root / "run" / "tables" / "jn1_recovery.csv"
    if not table.exists():
        return set()
    with open(table) as fh:
        return {
            int(row["candidate_id"])
            for row in csv.DictReader(fh)
            if row["is_wt"] == "True"
        }


def discover(screen_root, cells):
    found = []
    for cell_root in sorted(screen_root.iterdir()):
        if not (cell_root / "run" / "heldout").is_dir():
            continue
        if cells and cell_root.name not in cells:
            continue
        wt_ids = wt_candidate_ids(cell_root)
        for shard in sorted((cell_root / "run" / "heldout").glob("shard_*")):
            native = shard / "reference.pdb"
            if not native.exists():
                raise FileNotFoundError(f"{shard} has no reference.pdb")
            for cand in sorted((shard / "structures").glob("candidate_*")):
                cid = int(cand.name.split("_")[-1])
                for model in sorted(cand.glob("seed_*.pdb")):
                    found.append(dict(
                        cell=cell_root.name, shard=shard.name, candidate_id=cid,
                        seed=model.stem.split("_")[-1], model=model,
                        native=native, is_wt=cid in wt_ids,
                    ))
    return found


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
                        help="screen directory holding one subdirectory per cell")
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

    jobs = discover(args.screen_root, set(args.cells) if args.cells else None)
    if not jobs:
        parser.error(f"no held-out structures under {args.screen_root}")
    mapping, designed_len = derive_mapping(jobs[0]["model"], jobs[0]["native"])
    print(f"{len(jobs)} structures, chain mapping {mapping} "
          f"(designed chain {designed_len} residues), {args.workers} workers",
          flush=True)

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
