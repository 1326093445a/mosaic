"""Multi-GPU dispatcher for
p17_opendde_diffusion_steps_convergence_check.py: scores the SAME fixed
candidate sequence at several different --diffusion-steps values in
parallel, one process per GPU, then merges the results into one summary
CSV/table. Same job-level-subprocess pattern as
examples/p17_hallucination_dispatch.py.

Usage:
    .venv/bin/python examples/p17_opendde_diffusion_steps_convergence_dispatch.py \\
        --devices 0,1,2,3,4,5,6,7 \\
        --diffusion-steps-list 8,16,24,32,48,64,96,128 \\
        --sequence <123-aa binder sequence> \\
        --output-dir results/p17_opendde_diffusion_convergence
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, required=True,
                    help="Comma-separated physical GPU ids, e.g. 0,1,2,3,4,5,6,7")
    p.add_argument("--diffusion-steps-list", type=str, required=True,
                    help="Comma-separated diffusion step counts to test, e.g. "
                         "8,16,24,32,48,64,96,128 -- one job per value, "
                         "assigned round-robin across --devices")
    p.add_argument("--sequence", type=str, required=True)
    p.add_argument("--recycling-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    device_ids = [d.strip() for d in args.devices.split(",")]
    steps_list = [int(s.strip()) for s in args.diffusion_steps_list.split(",")]
    max_parallel = len(device_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dispatch] {len(steps_list)} jobs (diffusion_steps={steps_list}), "
          f"max_parallel={max_parallel}, devices={','.join(device_ids)}", flush=True)

    results = []
    for batch_start in range(0, len(steps_list), max_parallel):
        batch = steps_list[batch_start:batch_start + max_parallel]
        launched = []
        for i, steps in enumerate(batch):
            device = device_ids[i % len(device_ids)]
            tag = f"steps_{steps}"
            csv_path = args.output_dir / f"{tag}.csv"
            log_path = args.output_dir / f"{tag}.log"

            cmd = [
                sys.executable, "-u", str(WORKER_SCRIPT),
                "--sequence", args.sequence,
                "--diffusion-steps", str(steps),
                "--recycling-steps", str(args.recycling_steps),
                "--seed", str(args.seed),
                "--output", str(csv_path),
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = device

            print(f"[dispatch] start {tag} device={device} -> {csv_path}", flush=True)
            log_handle = log_path.open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            launched.append((proc, log_handle, tag, csv_path, log_path, steps, time.time()))

        for proc, log_handle, tag, csv_path, log_path, steps, start_time in launched:
            returncode = proc.wait()
            log_handle.close()
            elapsed = time.time() - start_time
            status = "ok" if returncode == 0 and csv_path.exists() else "FAILED"
            print(f"[dispatch] finished {tag}: status={status} returncode={returncode} "
                  f"elapsed={elapsed:.0f}s log={log_path}", flush=True)
            results.append({"tag": tag, "steps": steps, "status": status,
                             "csv": csv_path, "log": log_path})

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[dispatch] done: {n_ok}/{len(results)} jobs succeeded", flush=True)
    for r in results:
        if r["status"] != "ok":
            print(f"[dispatch]   FAILED: {r['tag']} (see {r['log']})", flush=True)

    rows = []
    for r in sorted(results, key=lambda r: r["steps"]):
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

        print(f"\n{'steps':>6}  {'wall_s':>7}  {'value':>8}  {'iptm':>7}  "
              f"{'binder_pose_rmsd':>17}  {'bt_pae':>7}  {'tb_pae':>7}", flush=True)
        for row in rows:
            print(f"{row['diffusion_steps']:>6}  {float(row['wall_time_s']):>7.1f}  "
                  f"{float(row['value']):>8.4f}  {float(row['iptm']):>7.4f}  "
                  f"{float(row['binder_pose_rmsd']):>17.2f}  "
                  f"{float(row['bt_pae']):>7.2f}  {float(row['tb_pae']):>7.2f}", flush=True)
        print("\nCompare against docs/guidance_alphaseq_testing_notes.md section "
              "13.3's real reference point (raw-torch OpenDDE, 200 diffusion "
              "steps): ipTM 0.87-0.93, RMSD ~6A for a reasonable design. If "
              "these values are still far off that range even at the largest "
              "--diffusion-steps tested, the gap isn't (only) about step count.",
              flush=True)

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
