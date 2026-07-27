"""Multi-GPU dispatcher for the P17 hallucination search
(examples/p17_hallucination_search.py): one fixed, validated config
(mcmc, stop_grad=0, argmax/WT-seeded, distogram in-loop, real hotspot
restriction -- all script defaults, unchanged here) x N seeds, spread
across GPUs via CUDA_VISIBLE_DEVICES per subprocess. Simpler than
examples/vhh72_hallucination_dispatch.py because there's no
policy/stop_grad grid to sweep here -- that question is already answered
(docs/guidance_alphaseq_testing_notes.md sections 13.9-13.12); this script
only varies --seed.

Job-level parallelism (independent subprocesses, not JAX pmap/sharding),
same pattern as the VHH72 dispatcher and
src/mosaic/workflows/boltzgen_vhh_guided.py's run_many_from_cli.

Usage:
    .venv/bin/python examples/p17_hallucination_dispatch.py \\
        --devices 0,1,2,3 --num-seeds 250 --edit-budget 5 \\
        --output-dir results/p17_sweep
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_SCRIPT = REPO_ROOT / "examples" / "p17_hallucination_search.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", type=str, default=None,
                    help="Comma-separated physical GPU ids, e.g. 0,1,2,3. "
                         "If omitted, all jobs share the inherited CUDA_VISIBLE_DEVICES "
                         "and run one at a time.")
    p.add_argument("--num-seeds", type=int, default=250,
                    help="Runs seeds 0..num_seeds-1 (use --seed-offset to shift the range).")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--edit-budget", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    device_ids = [d.strip() for d in args.devices.split(",")] if args.devices else []
    max_parallel = len(device_ids) if device_ids else 1
    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))

    print(f"[dispatch] {len(seeds)} jobs (seeds {seeds[0]}..{seeds[-1]}), "
          f"max_parallel={max_parallel}, "
          f"devices={','.join(device_ids) if device_ids else 'inherited'}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for batch_start in range(0, len(seeds), max_parallel):
        batch = seeds[batch_start:batch_start + max_parallel]
        launched = []
        for i, seed in enumerate(batch):
            device = device_ids[i % len(device_ids)] if device_ids else None
            tag = f"p17_mcmc_stopgrad0_seed{seed}"
            csv_path = args.output_dir / f"{tag}.csv"
            log_path = args.output_dir / f"{tag}.log"

            cmd = [
                sys.executable, "-u", str(SEARCH_SCRIPT),
                "--edit-budget", str(args.edit_budget),
                "--seed", str(seed),
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
