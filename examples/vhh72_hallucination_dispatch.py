"""Multi-GPU dispatcher for the VHH72 hallucination search
(examples/vhh72_hallucination_search.py): 2 search policies x 2 AbLang2
gradient-flow settings x N seeds = independent jobs, spread across N GPUs
via CUDA_VISIBLE_DEVICES per subprocess.

Job-level parallelism (independent subprocesses, not JAX pmap/sharding) --
mirrors the existing pattern in
src/mosaic/workflows/boltzgen_vhh_guided.py's run_many_from_cli, which is
"intentionally different from DDP inside one JAX process, because the
[jobs] are independent." With more jobs than GPUs, later batches reuse
GPUs once earlier jobs finish -- e.g. 20 jobs on 4 GPUs means each GPU runs
5 jobs sequentially, 5 batches of 4 running in parallel.

IMPORTANT: multi-seed runs (--seeds with more than one value) are only a
meaningful test of whether a finding (e.g. mcmc,stop_grad=0's Pareto win)
replicates once simplex_APGM's continuous phase is no longer a no-op --
see docs/guidance_alphaseq_testing_notes.md section 13.5 (the nnz=1.00
fixed-point finding). While that's still broken, the continuous phase
collapses to the identical WT vertex regardless of seed (the collapse is
structural, not seed-dependent), so different seeds would only vary the
discrete search's own randomness on top of an identical, frozen starting
point -- not what a seed sweep is meant to test.

Usage:
    .venv/bin/python examples/vhh72_hallucination_dispatch.py \\
        --devices 0,1,2,3 --edit-budget 5 --seeds 0,1,2,3,4 \\
        --output-dir results/hallucination_sweep
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Real exit code, not always 0 -- so a shell wrapper chaining this into
# vhh72_hallucination_aggregate.py under `set -e` stops before aggregating
# an incomplete result set instead of silently continuing.

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_SCRIPT = REPO_ROOT / "examples" / "vhh72_hallucination_search.py"

POLICIES = ["greedy", "mcmc"]
STOP_GRAD_SETTINGS = [1, 0]  # 1 = cheap default, 0 = real backprop through AbLang2 (ablation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, default=None,
                    help="Comma-separated physical GPU ids, e.g. 0,1,2,3. "
                         "If omitted, all jobs share the inherited CUDA_VISIBLE_DEVICES "
                         "and run one at a time.")
    p.add_argument("--edit-budget", type=int, default=5)
    p.add_argument("--seeds", type=str, default="0",
                    help="Comma-separated seeds, e.g. 0,1,2,3,4. Default: single seed 0 "
                         "(matches the original 4-job sweep).")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    device_ids = [d.strip() for d in args.devices.split(",")] if args.devices else []
    max_parallel = len(device_ids) if device_ids else 1
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    jobs = [
        {"policy": policy, "stop_grad": stop_grad, "seed": seed}
        for policy in POLICIES
        for stop_grad in STOP_GRAD_SETTINGS
        for seed in seeds
    ]
    print(f"[dispatch] {len(jobs)} jobs ({len(POLICIES)} policies x "
          f"{len(STOP_GRAD_SETTINGS)} stop_grad settings x {len(seeds)} seeds), "
          f"max_parallel={max_parallel}, "
          f"devices={','.join(device_ids) if device_ids else 'inherited'}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for batch_start in range(0, len(jobs), max_parallel):
        batch = jobs[batch_start:batch_start + max_parallel]
        launched = []
        for i, job in enumerate(batch):
            device = device_ids[i % len(device_ids)] if device_ids else None
            tag = f"{job['policy']}_stopgrad{job['stop_grad']}_seed{job['seed']}"
            csv_path = args.output_dir / f"{tag}.csv"
            log_path = args.output_dir / f"{tag}.log"

            cmd = [
                sys.executable, "-u", str(SEARCH_SCRIPT),
                "--policy", job["policy"],
                "--stop-grad", str(job["stop_grad"]),
                "--edit-budget", str(args.edit_budget),
                "--seed", str(job["seed"]),
                "--output", str(csv_path),
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if device is not None:
                env["CUDA_VISIBLE_DEVICES"] = device

            print(f"[dispatch] start {tag} device={device or 'inherited'} -> {csv_path}", flush=True)
            log_handle = log_path.open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            launched.append((proc, log_handle, tag, csv_path, log_path, time.time()))

        for proc, log_handle, tag, csv_path, log_path, start_time in launched:
            returncode = proc.wait()
            log_handle.close()
            elapsed = time.time() - start_time
            status = "ok" if returncode == 0 and csv_path.exists() else "FAILED"
            print(f"[dispatch] finished {tag}: status={status} returncode={returncode} "
                  f"elapsed={elapsed:.0f}s log={log_path}", flush=True)
            results.append({"tag": tag, "status": status, "returncode": returncode,
                             "csv": str(csv_path), "log": str(log_path)})

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[dispatch] done: {n_ok}/{len(results)} jobs succeeded", flush=True)
    for r in results:
        if r["status"] != "ok":
            print(f"[dispatch]   FAILED: {r['tag']} (see {r['log']})", flush=True)

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
