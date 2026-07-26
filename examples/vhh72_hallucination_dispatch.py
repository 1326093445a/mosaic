"""Multi-GPU dispatcher for the VHH72 hallucination search
(examples/vhh72_hallucination_search.py): search policies x AbLang2
gradient-flow settings x seeds = independent jobs, spread across GPUs via
CUDA_VISIBLE_DEVICES per subprocess.

Job-level parallelism (independent subprocesses, not JAX pmap/sharding) --
mirrors the existing pattern in
src/mosaic/workflows/boltzgen_vhh_guided.py's run_many_from_cli, which is
"intentionally different from DDP inside one JAX process, because the
[jobs] are independent." With more jobs than GPUs, later batches reuse
GPUs once earlier jobs finish -- e.g. 20 jobs on 4 GPUs means each GPU runs
5 jobs sequentially, 5 batches of 4 running in parallel.

Interpretation depends on --apgm-seed-mode:
  - argmax keeps the original behavior. APGM's hard argmax has been observed
    to return WT, so a multi-seed argmax run is a discrete-search stability
    test, not evidence that APGM seeding contributes.
  - sample/topk deliberately convert APGM's soft distribution into a non-WT
    hard seed, so those modes are the actual APGM-seeding architecture tests.

Usage:
    .venv/bin/python examples/vhh72_hallucination_dispatch.py \\
        --devices 0,1,2,3 --edit-budget 5 --seeds 0,1,2,3,4 \\
        --output-dir results/hallucination_sweep

    # APGM-seeding architecture test, mcmc/stop_grad=0 only:
    .venv/bin/python examples/vhh72_hallucination_dispatch.py \\
        --devices 0,1,2,3 --edit-budget 5 --seeds 0,1,2 \\
        --policies mcmc --stop-grads 0 --apgm-seed-mode sample \\
        --apgm-init-wt-prob 0.80 --apgm-scale 1.0 \\
        --output-dir results/hallucination_sweep_apgm_sample_mcmc_sg0
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


def _parse_csv_strs(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one comma-separated value")
    return out


def _parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in _parse_csv_strs(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
    p.add_argument("--policies", type=str, default=",".join(POLICIES),
                    help="Comma-separated policies to run. Default: greedy,mcmc.")
    p.add_argument("--stop-grads", type=str, default=",".join(map(str, STOP_GRAD_SETTINGS)),
                    help="Comma-separated AbLang2 stop_grad settings. Default: 1,0.")
    p.add_argument("--apgm-seed-mode", choices=["argmax", "sample", "topk"], default="argmax",
                    help="Passed through to vhh72_hallucination_search.py. Default keeps "
                         "the original argmax behavior.")
    p.add_argument("--apgm-topk-threshold", type=float, default=0.15,
                    help="Passed through to vhh72_hallucination_search.py when "
                         "--apgm-seed-mode topk is used.")
    p.add_argument("--apgm-init-wt-prob", type=float, default=1.0,
                    help="Passed through to vhh72_hallucination_search.py. "
                         "1.0 preserves original exact one-hot WT start; 0.80 "
                         "matches the softened APGM diagnostic.")
    p.add_argument("--apgm-scale", type=float, default=1.2,
                    help="Passed through to vhh72_hallucination_search.py. "
                         "1.2 preserves original sparsity-encouraging default; "
                         "1.0 matches the softened APGM diagnostic.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    device_ids = [d.strip() for d in args.devices.split(",")] if args.devices else []
    max_parallel = len(device_ids) if device_ids else 1
    seeds = _parse_csv_ints(args.seeds)
    policies = _parse_csv_strs(args.policies)
    stop_grad_settings = _parse_csv_ints(args.stop_grads)
    unknown_policies = sorted(set(policies) - set(POLICIES))
    unknown_stop_grads = sorted(set(stop_grad_settings) - set(STOP_GRAD_SETTINGS))
    if unknown_policies:
        raise SystemExit(f"unknown policies: {unknown_policies}; allowed: {POLICIES}")
    if unknown_stop_grads:
        raise SystemExit(f"unknown stop_grad settings: {unknown_stop_grads}; allowed: {STOP_GRAD_SETTINGS}")

    jobs = [
        {"policy": policy, "stop_grad": stop_grad, "seed": seed}
        for policy in policies
        for stop_grad in stop_grad_settings
        for seed in seeds
    ]
    print(f"[dispatch] {len(jobs)} jobs ({len(policies)} policies x "
          f"{len(stop_grad_settings)} stop_grad settings x {len(seeds)} seeds), "
          f"max_parallel={max_parallel}, "
          f"devices={','.join(device_ids) if device_ids else 'inherited'}, "
          f"apgm_seed_mode={args.apgm_seed_mode}, "
          f"apgm_init_wt_prob={args.apgm_init_wt_prob}, "
          f"apgm_scale={args.apgm_scale}", flush=True)

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
                "--apgm-seed-mode", args.apgm_seed_mode,
                "--apgm-topk-threshold", str(args.apgm_topk_threshold),
                "--apgm-init-wt-prob", str(args.apgm_init_wt_prob),
                "--apgm-scale", str(args.apgm_scale),
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
