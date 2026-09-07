"""Multi-GPU dispatcher for
p17_opendde_diffusion_steps_convergence_check.py: scores a candidate
sequence across a grid of (diffusion_steps, seed, recycling_steps, use_wt)
combinations in parallel, one process per GPU, then merges the results
into one summary CSV/table. Same job-level-subprocess pattern as
examples/p17_hallucination_dispatch.py.

Four independent sweep axes, each given as a comma list
(--diffusion-steps-list/--seed-list/--recycling-steps-list/--use-wt-list):
every list with length 1 is broadcast up to the longest list's length;
all lists with length > 1 must share that same length (zipped pairwise,
NOT a full cross-product grid -- keep the list lengths equal to the
number of jobs you actually want).

Three uses so far:
  - Sweep steps at a fixed seed/recycling (the original convergence
    check): --diffusion-steps-list 8,16,...,128 --seed-list 0
  - Sweep seed at a fixed step count (isolates sampling noise from step
    count): --diffusion-steps-list 64 --seed-list 0,1,2,3,4,5,6,7
  - WT vs. designed candidate across recycling_steps (isolates whether a
    low ipTM/high RMSD reading is about the candidate or about
    recycling_steps=4 being too few vs. the validated reference's 10):
    --recycling-steps-list 4,6,8,10,4,6,8,10 --use-wt-list 1,1,1,1,0,0,0,0
    --diffusion-steps-list 64

Usage:
    .venv/bin/python examples/p17_opendde_diffusion_steps_convergence_dispatch.py \\
        --devices 0,1,2,3,4,5,6,7 \\
        --diffusion-steps-list 64 \\
        --recycling-steps-list 4,6,8,10,4,6,8,10 \\
        --use-wt-list 1,1,1,1,0,0,0,0 \\
        --sequence <123-aa binder sequence> \\
        --output-dir results/p17_opendde_wt_vs_design_recycling
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = REPO_ROOT / "examples" / "p17_opendde_diffusion_steps_convergence_check.py"


def broadcast_lists(named_lists: dict[str, list]) -> dict[str, list]:
    """Broadcast every length-1 list up to the longest list's length; every
    list with length > 1 must already equal that length."""
    lengths = {name: len(lst) for name, lst in named_lists.items()}
    target = max(lengths.values())
    bad = [name for name, n in lengths.items() if n not in (1, target)]
    if bad:
        raise SystemExit(
            f"length mismatch: {', '.join(f'{n}={lengths[n]}' for n in bad)} vs. "
            f"target length {target} (every list must be length 1 or {target})"
        )
    return {
        name: (lst * target if len(lst) == 1 else lst)
        for name, lst in named_lists.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, required=True,
                    help="Comma-separated physical GPU ids, e.g. 0,1,2,3,4,5,6,7")
    p.add_argument("--diffusion-steps-list", type=str, required=True)
    p.add_argument("--seed-list", type=str, default="0")
    p.add_argument("--recycling-steps-list", type=str, default="4")
    p.add_argument("--use-wt-list", type=str, default="0",
                    help="Comma list of 0/1 -- 1 scores the real WT sequence "
                         "instead of --sequence for that job")
    p.add_argument("--sequence", type=str, default=None,
                    help="Required unless every --use-wt-list entry is 1")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    device_ids = [d.strip() for d in args.devices.split(",")]
    raw = {
        "steps": [int(s.strip()) for s in args.diffusion_steps_list.split(",")],
        "seed": [int(s.strip()) for s in args.seed_list.split(",")],
        "recycling": [int(s.strip()) for s in args.recycling_steps_list.split(",")],
        "use_wt": [int(s.strip()) for s in args.use_wt_list.split(",")],
    }
    bcast = broadcast_lists(raw)
    jobs = list(zip(bcast["steps"], bcast["seed"], bcast["recycling"], bcast["use_wt"]))

    if not args.sequence and not all(use_wt for _, _, _, use_wt in jobs):
        raise SystemExit("--sequence is required unless every --use-wt-list entry is 1")

    max_parallel = len(device_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dispatch] {len(jobs)} jobs (steps, seed, recycling, use_wt)={jobs}, "
          f"max_parallel={max_parallel}, devices={','.join(device_ids)}", flush=True)

    results = []
    for batch_start in range(0, len(jobs), max_parallel):
        batch = jobs[batch_start:batch_start + max_parallel]
        launched = []
        for i, (steps, seed, recycling, use_wt) in enumerate(batch):
            device = device_ids[i % len(device_ids)]
            tag = f"steps_{steps}_seed_{seed}_recyc_{recycling}_{'wt' if use_wt else 'design'}"
            csv_path = args.output_dir / f"{tag}.csv"
            log_path = args.output_dir / f"{tag}.log"

            cmd = [
                sys.executable, "-u", str(WORKER_SCRIPT),
                "--diffusion-steps", str(steps),
                "--recycling-steps", str(recycling),
                "--seed", str(seed),
                "--output", str(csv_path),
            ]
            if use_wt:
                cmd.append("--use-wt")
            else:
                cmd += ["--sequence", args.sequence]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = device

            print(f"[dispatch] start {tag} device={device} -> {csv_path}", flush=True)
            log_handle = log_path.open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            launched.append((proc, log_handle, tag, csv_path, log_path,
                              steps, seed, recycling, use_wt, time.time()))

        for proc, log_handle, tag, csv_path, log_path, steps, seed, recycling, use_wt, start_time in launched:
            returncode = proc.wait()
            log_handle.close()
            elapsed = time.time() - start_time
            status = "ok" if returncode == 0 and csv_path.exists() else "FAILED"
            print(f"[dispatch] finished {tag}: status={status} returncode={returncode} "
                  f"elapsed={elapsed:.0f}s log={log_path}", flush=True)
            results.append({"tag": tag, "steps": steps, "seed": seed, "recycling": recycling,
                             "use_wt": use_wt, "status": status, "csv": csv_path, "log": log_path})

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[dispatch] done: {n_ok}/{len(results)} jobs succeeded", flush=True)
    for r in results:
        if r["status"] != "ok":
            print(f"[dispatch]   FAILED: {r['tag']} (see {r['log']})", flush=True)

    rows = []
    for r in sorted(results, key=lambda r: (r["use_wt"], r["recycling"], r["steps"], r["seed"])):
        if r["status"] != "ok":
            continue
        with open(r["csv"]) as f:
            rows.append(next(csv.DictReader(f)))

    if rows:
        summary_path = args.output_dir / "convergence_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote summary: {summary_path}", flush=True)

        print(f"\n{'seq':>6}  {'steps':>6}  {'recyc':>5}  {'seed':>5}  {'wall_s':>7}  "
              f"{'iptm':>7}  {'binder_pose_rmsd':>17}", flush=True)
        for row in rows:
            seq_label = "WT" if int(row["use_wt"]) else "design"
            print(f"{seq_label:>6}  {row['diffusion_steps']:>6}  {row['recycling_steps']:>5}  "
                  f"{row['seed']:>5}  {float(row['wall_time_s']):>7.1f}  "
                  f"{float(row['iptm']):>7.4f}  {float(row['binder_pose_rmsd']):>17.2f}", flush=True)

        wt_rows = [r for r in rows if int(r["use_wt"])]
        design_rows = [r for r in rows if not int(r["use_wt"])]
        if wt_rows and design_rows:
            wt_best = max(wt_rows, key=lambda r: float(r["iptm"]))
            print(f"\nWT best ipTM: {float(wt_best['iptm']):.4f} at recycling_steps="
                  f"{wt_best['recycling_steps']} -- if this is still far below the "
                  "0.87-0.93 reference, the gap is about scoring settings (or this "
                  "pipeline vs. the raw-torch reference generally), not about this "
                  "specific candidate. If WT scores well but the design doesn't, "
                  "that's a real finding about what the mutations did.", flush=True)

        print("\nReference point: docs/guidance_alphaseq_testing_notes.md section "
              "13.3's raw-torch OpenDDE run (200 diffusion steps, 10 recycles): "
              "ipTM 0.87-0.93, RMSD ~6A for a reasonable design.", flush=True)

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
