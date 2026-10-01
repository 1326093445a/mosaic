"""Sequential pose diagnostic, gated population ablation and held-out rescoring.

Dry-run is read-only and imports neither JAX nor the model stack. Workers use
one allocated GPU each. GPU memory is not pooled across devices.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent


def devices_from_csv(value):
    devices = value.split(",")
    if not devices or any(
        not re.fullmatch(r"[0-9]+|(?:GPU-|MIG-)[A-Za-z0-9/-]+", d) for d in devices
    ):
        raise argparse.ArgumentTypeError("use comma-separated GPU indices or UUIDs")
    devices = [str(int(d)) if d.isdecimal() else d for d in devices]
    if len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("duplicate GPU device")
    return devices


def build_plan(args):
    root = args.output_dir
    common = [
        sys.executable,
        str(REPO / "examples/p17_confidence_search.py"),
        "--policy",
        "population",
        "--selection-seeds",
        "0",
        "1",
        "--sampling-steps",
        str(args.sampling_steps),
        "--opendde-dtype",
        args.opendde_dtype,
        "--max-score-calls",
        str(args.max_score_calls),
        "--max-gradient-calls",
        str(args.max_gradient_calls),
        "--max-proposals",
        str(args.max_proposals),
    ]
    diagnostics, searches, heldout = [], [], []
    for seed in (0, 1):
        name = f"model_seed{seed}"
        diagnostics.append(
            dict(
                name=name,
                command=common
                + [
                    "--output-dir",
                    str(root / "diagnostic" / name),
                    "--pose-diagnostic",
                    "--weight-pose",
                    str(args.weight_pose),
                    "--proposal-model-seed",
                    str(seed),
                    "--diagnostic-max-target-rmsd",
                    str(args.max_target_rmsd),
                    "--diagnostic-min-proposal-tv",
                    str(args.min_proposal_tv),
                    "--diagnostic-repeat-factor",
                    str(args.repeat_factor),
                ],
            )
        )
    for arm, weight, retention in (
        ("A_neither", 0.0, False),
        ("B_guidance", args.weight_pose, False),
        ("C_retention", 0.0, True),
        ("D_both", args.weight_pose, True),
    ):
        for seed in args.search_seeds:
            name = f"{arm}_seed{seed}"
            command = common + [
                "--output-dir",
                str(root / "search" / name),
                "--seed",
                str(seed),
                "--weight-pose",
                str(weight),
            ]
            if retention:
                command += ["--retention-pose-margin", str(args.pose_margin)]
            searches.append(dict(name=name, command=command))
    for shard in range(len(args.devices)):
        heldout.append(
            dict(
                name=f"shard_{shard}",
                command=[
                    sys.executable,
                    str(REPO / "examples/p17_rescore_winners.py"),
                    "--input",
                    str(root / "search"),
                    "--output-dir",
                    str(root / "heldout" / f"shard_{shard}"),
                    "--opendde-dtype",
                    args.opendde_dtype,
                    "--seeds",
                    "101",
                    "102",
                    "103",
                    "--num-shards",
                    str(len(args.devices)),
                    "--shard-index",
                    str(shard),
                ],
            )
        )
    return dict(diagnostic=diagnostics, search=searches, heldout=heldout)


def run_stage(stage, jobs, devices, root):
    """Bounded batches; stop launching batches after any nonzero worker exit."""
    root = Path(root)
    for start in range(0, len(jobs), len(devices)):
        workers = []
        try:
            for job, device in zip(jobs[start : start + len(devices)], devices):
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=device,
                    PYTHONUNBUFFERED="1",
                    JAX_PLATFORMS="cuda",
                )
                env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
                handle = (root / "logs" / f"{stage}_{job['name']}.log").open("x")
                try:
                    process = subprocess.Popen(
                        job["command"],
                        cwd=REPO,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    handle.close()
                    raise
                workers.append((job, device, process, handle))
                print(
                    f"Started {stage}/{job['name']} on GPU {device} (PID {process.pid})",
                    flush=True,
                )
            for job, device, process, handle in workers:
                code = process.wait()
                print(f"Finished {stage}/{job['name']}: exit {code}", flush=True)
        finally:
            # Also handles Ctrl-C, SIGTERM (via main), failed spawn, or failed wait.
            for _, _, process, _ in workers:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for _, _, process, handle in workers:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                handle.close()
            with (root / "status.tsv").open("a", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                for job, device, process, _ in workers:
                    writer.writerow(
                        [stage, job["name"], device, process.pid, process.returncode]
                    )
        if any(process.returncode != 0 for _, _, process, _ in workers):
            raise RuntimeError(
                f"{stage} worker failed; dependent stages were not launched"
            )


def check_gate(root):
    required = {
        "completed",
        "same_coordinate_reporting",
        "interpretable_target_fit",
        "proposal_influence",
        "paired_pose_consistent",
    }
    reports = []
    for seed in (0, 1):
        path = Path(root) / "diagnostic" / f"model_seed{seed}" / "diagnostic.json"
        report = json.loads(path.read_text())
        checks = report.get("checks", {})
        if (
            report.get("schema_version") != 1
            or report.get("passed") is not True
            or any(checks.get(name) is not True for name in required)
            or not report.get("reference_audit", {}).get("geometry")
        ):
            raise RuntimeError(f"Diagnostic gate failed: {path}")
        reports.append(str(path.relative_to(root)))
    return reports


def summarize_search(root, pose_margin):
    """Retain raw scores, per-run ceilings and actual costs for each arm/seed."""
    rows, all_candidates = [], []
    for path in sorted((root / "search").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        config = json.loads(path.with_name("config.json").read_text())
        with (path.parent / "tables/candidates.csv").open() as handle:
            candidates = list(csv.DictReader(handle))
        all_candidates.append(candidates)
        wt_pose = next(
            float(c["worst_pose_rmsd_A"]) for c in candidates if c["is_wt"] == "True"
        )
        rows.append(
            dict(
                run=path.parent.name,
                search_seed=config["config"]["seed"],
                pose_weight=config["arguments"]["weight_pose"],
                retention_margin_A=config["arguments"]["retention_pose_margin"],
                calibrated_ceiling_A=summary["retention_pose_ceiling_A"],
                wt_worst_pose_A=wt_pose,
                best_score=summary["best_score"],
                best_violation=summary["best_constraint_violation"],
                full_prediction_calls=summary["full_prediction_calls"],
                full_gradient_calls=summary["full_gradient_calls"],
                **summary["stats"],
            )
        )
    common_ceiling = (
        statistics.median(row["wt_worst_pose_A"] for row in rows) + pose_margin
    )
    for row, candidates in zip(rows, all_candidates):
        row["common_reporting_ceiling_A"] = common_ceiling
        row["evaluated_feasible_fraction_including_WT"] = sum(
            float(c["worst_pose_rmsd_A"]) <= common_ceiling for c in candidates
        ) / len(candidates)
    with (root / "tables/search_runs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_heldout(root, pose_margin):
    """Use one held-out WT-relative definition for every candidate and source run."""
    with (root / "heldout/tables/predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    with (root / "heldout/tables/source_runs.csv").open() as handle:
        sources = list(csv.DictReader(handle))
    grouped = {}
    for row in predictions:
        grouped.setdefault(row["candidate_id"], []).append(row)
    ceiling = max(float(r["binder_pose_rmsd_A"]) for r in grouped["0"]) + pose_margin
    rows = []
    for candidate, samples in sorted(grouped.items(), key=lambda item: int(item[0])):
        row = dict(
            candidate_id=candidate,
            is_wt=candidate == "0",
            source_runs=";".join(
                s["source_run"] for s in sources if s["candidate_id"] == candidate
            ),
            common_reporting_ceiling_A=ceiling,
        )
        for name in (
            "ipsae_min",
            "binder_pose_rmsd_A",
            "target_aligned_rmsd_A",
            "binder_internal_rmsd_A",
        ):
            values = [float(r[name]) for r in samples]
            row["mean_" + name] = statistics.mean(values)
            row["worst_" + name] = min(values) if name == "ipsae_min" else max(values)
        row["pose_feasible_all_seeds"] = row["worst_binder_pose_rmsd_A"] <= ceiling
        row["pose_feasible_seed_fraction"] = sum(
            float(r["binder_pose_rmsd_A"]) <= ceiling for r in samples
        ) / len(samples)
        rows.append(row)
    with (root / "tables/heldout_candidates.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        type=devices_from_csv,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-score-calls", type=int, default=32)
    parser.add_argument("--max-gradient-calls", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=320)
    parser.add_argument("--weight-pose", type=float, default=1.0)
    parser.add_argument("--pose-margin", type=float, default=3.0)
    parser.add_argument("--max-target-rmsd", type=float, default=3.0)
    parser.add_argument("--min-proposal-tv", type=float, default=1e-4)
    parser.add_argument("--repeat-factor", type=float, default=3.0)
    args = parser.parse_args(argv)
    import math

    if any(
        v < 1
        for v in (
            args.sampling_steps,
            args.max_score_calls,
            args.max_gradient_calls,
            args.max_proposals,
        )
    ):
        parser.error("sampling and budgets must be positive")
    if min(args.search_seeds) < 0 or len(set(args.search_seeds)) != len(
        args.search_seeds
    ):
        parser.error("search seeds must be distinct and nonnegative")
    if (
        any(
            not math.isfinite(v) or v <= 0
            for v in (
                args.weight_pose,
                args.max_target_rmsd,
                args.min_proposal_tv,
                args.repeat_factor,
            )
        )
        or args.min_proposal_tv > 1
        or not math.isfinite(args.pose_margin)
        or args.pose_margin < 0
    ):
        parser.error("invalid pose/diagnostic settings")
    args.output_dir = (
        args.output_dir
        or REPO
        / "results"
        / (
            "p17_pose_experiment_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{os.getpid()}"
        )
    ).resolve()
    root = args.output_dir
    if root.exists():
        parser.error(f"output already exists: {root}")
    plan = build_plan(args)
    print(f"Repo: {REPO}\nGPUs: {','.join(args.devices)}; output: {root}")
    print(f"OpenDDE compute: {args.opendde_dtype}; AbLang2: fp32")
    allocator_environment = {
        key: os.environ[key]
        for key in (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "TF_GPU_ALLOCATOR",
        )
        if key in os.environ
    }
    allocator_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    allocator_assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in allocator_environment.items()
    )
    print(f"Allocator environment: {allocator_assignments}")
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd " + shlex.quote(str(REPO)),
        "# Provenance only: execute the launcher to enforce diagnostic gates.",
    ]
    for stage, jobs in plan.items():
        print(
            f"Stage {stage}: {len(jobs)} workers; bounded parallelism {len(args.devices)}"
        )
        for index, job in enumerate(jobs):
            device = args.devices[index % len(args.devices)]
            command = (
                f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda "
                f"{allocator_assignments} "
                + shlex.join(job["command"])
            )
            commands.append(command)
            if args.dry_run:
                print(command)
    if args.dry_run:
        return
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "tables").mkdir()
    (root / "commands.sh").write_text("\n".join(commands) + "\n")
    (root / "status.tsv").write_text("stage\tworker\tgpu\tpid\texit_code\n")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (root / "plan.json").write_text(
        json.dumps(
            dict(settings=settings, allocator_environment=allocator_environment, stages=plan),
            indent=2,
        ) + "\n"
    )
    (root / "README.md").write_text(
        "# Pose experiment\n\n"
        "diagnostic/: paired gradients, proposal CSVs, reference audit, WT/edit structures and baseline repeats.\n"
        "search/: four population arms, each with candidates/predictions CSVs, structures, confidence arrays and logs.\n"
        "heldout/: new predictions of WT and winners on seeds 101/102/103; merged tables retain source-run links.\n"
        "tables/search_runs.csv: scores, ceilings, feasible fractions and actual call counts.\n"
        "tables/heldout_candidates.csv: mean/worst confidence and RMSDs with common WT-relative feasibility.\n"
        "logs/, status.tsv, commands.sh, plan.json, experiment.json: execution provenance and stage status.\n"
        "Diagnostic thresholds are provisional. A passing gate does not validate affinity or pose improvement.\n"
        "Each retention arm calibrates its ceiling from its own WT predictions; compare recorded ceilings for variability.\n"
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    stage = "patches"
    try:
        with (root / "logs/patches.log").open("x") as handle:
            for name in (
                "outer_product_mean",
                "structural_token_expander",
                "bf16_dtype",
            ):
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "patches" / f"patch_jopendde_{name}.py"),
                    ],
                    cwd=REPO,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        for stage in ("diagnostic", "search", "heldout"):
            run_stage(stage, plan[stage], args.devices, root)
            if stage == "diagnostic":
                reports = check_gate(root)
                (root / "gate.json").write_text(
                    json.dumps(dict(passed=True, reports=reports), indent=2) + "\n"
                )
            elif stage == "search":
                summarize_search(root, args.pose_margin)
            else:
                from p17_rescore_winners import merge_shard_tables

                merge_shard_tables(root / "heldout", len(args.devices))
                summarize_heldout(root, args.pose_margin)
        result = dict(completed=True, last_stage=stage)
    except BaseException as exc:
        result = dict(
            completed=False, failed_stage=stage, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        (root / "experiment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed. Results: {root}", flush=True)


if __name__ == "__main__":
    main()
