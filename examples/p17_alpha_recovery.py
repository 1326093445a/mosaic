"""Driver for the gradient recovery experiments (docs/P17_JN1.md 19.3B, 19.5).

Three stages. The first two are the Alpha *control*; the third is the JN.1
question the control exists to make readable.

`calibrate` (Alpha) forward-scores the undamaged reference and every damage
rung, and reports which rungs actually reached the non-binding regime. Damage
severity is not assumed to work: a rung that still scores like a binder is
useless as a starting point, and one that destroys the fold is a different
experiment.

`search` (Alpha) runs recovery searches from the selected rung.

`jn1` runs the same searches against the JN.1 target starting from WT P17.
There is no damage stage: WT P17 against JN.1 is already in the non-binding
regime (ipSAE 0.000-0.163, pose 22-58 A), so the starting point exists
without being constructed. Unlike the Alpha control, no solution is known to
exist here -- which is exactly why the control is run alongside it.

Both search stages rescore archived winners on held-out structural seeds.
Recovery is measured against each run's *own* measured reference and starting
confidence, not a literature number. Nothing here establishes binding.

Policy defaults to **population** only. Section 17's standing decision keeps
population as the provisional policy, and section 19.4 deprioritizes the
population-versus-independent comparison: it varies search memory, which is
second-order to whether the gradient moves the metrics at all. Spending half
the workers on that comparison buys less than spending them on search seeds.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SELECTION_SEEDS = (0, 1)
HELDOUT_SEEDS = (101, 102, 103)

# Provisional, declared before execution, from the historical native control in
# section 19.1: P17+JN.1 spans ipSAE 0.0000-0.1629 and P17+Alpha 0.7949-0.7954.
# A rung counts as damaged when its mean ipSAE falls at or below the top of the
# non-binding range. This is a threshold on one predictor's confidence, not a
# validated boundary between binding and non-binding.
NON_BINDING_IPSAE_CEILING = 0.1629


def run_workers(jobs, devices, log_dir):
    """Run jobs at most one per device, recording exit codes.

    Each worker preallocates most of its GPU, so devices are not shared within
    a batch. Returns the per-job exit codes in order.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    codes = [None] * len(jobs)
    for start in range(0, len(jobs), len(devices)):
        batch = list(enumerate(jobs[start : start + len(devices)], start=start))
        running = []
        for slot, (index, job) in enumerate(batch):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(devices[slot])
            env["JAX_PLATFORMS"] = "cuda"
            handle = (log_dir / f"{job['name']}.log").open("w")
            print(f"  [{job['name']}] device {devices[slot]}", flush=True)
            running.append(
                (
                    index,
                    job,
                    handle,
                    subprocess.Popen(
                        job["command"], env=env, stdout=handle,
                        stderr=subprocess.STDOUT, cwd=REPO,
                    ),
                )
            )
        for index, job, handle, process in running:
            codes[index] = process.wait()
            handle.close()
            status = "ok" if codes[index] == 0 else f"exit {codes[index]}"
            print(f"  [{job['name']}] {status}", flush=True)
    return codes


def score_command(output_dir, start_sequence, steps, dtype):
    """A minimal confidence-search invocation that only scores its start point.

    Reuses the search entry point rather than a second scoring path, so the
    calibration numbers come from exactly the machinery the search will use.
    Budgets are the smallest the harness accepts; the start sequence is scored
    as that run's own WT.
    """
    command = [
        sys.executable, str(EXAMPLES / "p17_confidence_search.py"),
        "--policy", "independent",
        "--output-dir", str(output_dir),
        "--complex", str(REPO / "P17_Alpha.pdb"),
        "--epitope-mode", "contact",
        "--width", "1",
        "--max-score-calls", "1",
        "--max-gradient-calls", "1",
        "--max-proposals", "1",
        "--sampling-steps", str(steps),
        "--opendde-dtype", dtype,
        "--selection-seeds", *[str(s) for s in SELECTION_SEEDS],
    ]
    if start_sequence is not None:
        command += ["--start-sequence", start_sequence]
    return command


def read_start_metrics(run_dir):
    """Mean ipSAE and worst pose RMSD for candidate 0 -- the run's start point."""
    path = Path(run_dir) / "tables/predictions.csv"
    if not path.is_file():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r["candidate_id"] == "0"]
    if not rows:
        return None
    ipsae = [float(r["ipsae_min"]) for r in rows]
    pose = [float(r["binder_pose_rmsd_A"]) for r in rows]
    target = [float(r["target_aligned_rmsd_A"]) for r in rows]
    return dict(
        seeds=len(rows),
        mean_ipsae=sum(ipsae) / len(ipsae),
        worst_ipsae=min(ipsae),
        mean_pose_rmsd_A=sum(pose) / len(pose),
        worst_pose_rmsd_A=max(pose),
        mean_target_fit_A=sum(target) / len(target),
    )


def calibrate(args):
    ladder = json.loads(Path(args.ladder).read_text())
    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)

    jobs = [
        dict(
            name="reference",
            command=score_command(root / "reference", None, args.steps, args.opendde_dtype),
        )
    ]
    for rung in ladder["rungs"]:
        jobs.append(
            dict(
                name=f"damaged_{rung['n_edits']:02d}",
                command=score_command(
                    root / f"damaged_{rung['n_edits']:02d}",
                    rung["sequence"],
                    args.steps,
                    args.opendde_dtype,
                ),
            )
        )
    print(f"Forward-scoring {len(jobs)} sequences on {len(devices)} device(s)...")
    codes = run_workers(jobs, devices, root / "logs")

    results, failures = [], []
    for job, code in zip(jobs, codes):
        metrics = read_start_metrics(root / job["name"])
        if code != 0 or metrics is None:
            failures.append(dict(name=job["name"], exit_code=code))
            continue
        results.append(dict(name=job["name"], exit_code=code, **metrics))
    reference = next((r for r in results if r["name"] == "reference"), None)
    if reference is None:
        raise RuntimeError(
            "the undamaged reference failed to score; without it there is no "
            f"recovery target to measure against. failures={failures}"
        )

    rungs = []
    for rung in ladder["rungs"]:
        name = f"damaged_{rung['n_edits']:02d}"
        scored = next((r for r in results if r["name"] == name), None)
        if scored is None:
            rungs.append(dict(n_edits=rung["n_edits"], scored=False))
            continue
        rungs.append(
            dict(
                n_edits=rung["n_edits"],
                scored=True,
                sequence=rung["sequence"],
                reached_non_binding=scored["mean_ipsae"] <= args.non_binding_ceiling,
                ipsae_drop_from_reference=reference["mean_ipsae"] - scored["mean_ipsae"],
                **{k: v for k, v in scored.items() if k != "name"},
            )
        )

    usable = [r for r in rungs if r.get("reached_non_binding")]
    payload = dict(
        reference=reference,
        rungs=rungs,
        failures=failures,
        non_binding_ipsae_ceiling=args.non_binding_ceiling,
        selection_seeds=list(SELECTION_SEEDS),
        sampling_steps=args.steps,
        opendde_dtype=args.opendde_dtype,
        recommended_rung=min((r["n_edits"] for r in usable), default=None),
        interpretation=(
            "Forward scoring only. 'reached_non_binding' compares mean ipSAE "
            "against a provisional ceiling taken from the historical P17/JN.1 "
            "non-binding range; it is a confidence threshold, not evidence "
            "about binding. The smallest damaged rung is recommended so "
            "recovery is measured from the mildest starting point that is "
            "actually in the non-binding regime."
        ),
    )
    out = Path(args.output_dir) / "calibration.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nReference: mean ipSAE {reference['mean_ipsae']:.4f}, "
          f"worst pose {reference['worst_pose_rmsd_A']:.2f} A, "
          f"target fit {reference['mean_target_fit_A']:.2f} A")
    for rung in rungs:
        if not rung.get("scored"):
            print(f"  {rung['n_edits']:2d} edits: FAILED TO SCORE")
            continue
        flag = "non-binding" if rung["reached_non_binding"] else "still binder-like"
        print(f"  {rung['n_edits']:2d} edits: mean ipSAE {rung['mean_ipsae']:.4f}, "
              f"worst pose {rung['worst_pose_rmsd_A']:.2f} A  [{flag}]")
    if payload["recommended_rung"] is None:
        print(
            "\nNo rung reached the non-binding regime. Recovery cannot be "
            "measured from a starting point that still scores like a binder; "
            "extend --edits before searching.",
            file=sys.stderr,
        )
        return 3
    print(f"\nRecommended starting rung: {payload['recommended_rung']} edits")
    print(f"Wrote {out}")
    return 0


def search_jobs(root, policies, seeds, start_sequence, budget, args, complex_pdb):
    """One worker per (policy, seed). `complex_pdb` None means the JN.1 default.

    Both targets go through the same entry point with the same ceilings, so
    the Alpha control and the JN.1 run stay directly comparable. The only
    deliberate differences are the reference and the starting sequence.
    """
    jobs = []
    for policy in policies:
        for seed in seeds:
            name = f"{policy}_seed{seed}"
            command = [
                sys.executable, str(EXAMPLES / "p17_confidence_search.py"),
                "--policy", policy,
                "--output-dir", str(root / "search" / name),
                "--seed", str(seed),
                "--edit-budget", str(budget),
                "--max-score-calls", str(args.max_score_calls),
                "--max-gradient-calls", str(args.max_gradient_calls),
                "--max-proposals", str(args.max_proposals),
                "--sampling-steps", str(args.steps),
                "--opendde-dtype", args.opendde_dtype,
                "--selection-seeds",
                *[str(s) for s in getattr(args, "selection_seeds", SELECTION_SEEDS)],
            ]
            if complex_pdb is not None:
                command += [
                    "--complex", str(complex_pdb),
                    "--epitope-mode", "contact",
                ]
            if start_sequence is not None:
                command += ["--start-sequence", start_sequence]
            jobs.append(dict(name=name, command=command))
    return jobs


def run_heldout(root, devices, dtype, complex_pdb=None, suffix="", seeds=None):
    """Rescore every archived winner on seeds the search never selected on.

    `complex_pdb` must name the reference the archived searches actually used.
    `p17_rescore_winners.py` defaults to JN.1 and verifies the archived binder
    and target sequences against whatever reference it loads, so pointing it
    at Alpha runs without this refuses the run rather than scoring the wrong
    complex -- which is what it did on 2026-10-04.

    `suffix` names a retry directory, so a re-run does not have to overwrite
    or delete the shards from a previous attempt.
    """
    stage = f"heldout{suffix}"
    print(f"\nHeld-out rescoring on unseen structural seeds -> {stage}/...")
    shards = []
    for shard in range(len(devices)):
        command = [
            sys.executable, str(EXAMPLES / "p17_rescore_winners.py"),
            "--input", str(root / "search"),
            "--output-dir", str(root / stage / f"shard_{shard}"),
            "--opendde-dtype", dtype,
            "--seeds", *[str(s) for s in (seeds or HELDOUT_SEEDS)],
            "--num-shards", str(len(devices)),
            "--shard-index", str(shard),
        ]
        if complex_pdb is not None:
            command += [
                "--complex", str(complex_pdb),
                "--binder-chain", "B",
                "--target-chain", "A",
            ]
        shards.append(dict(name=f"{stage}_shard_{shard}", command=command))
    codes = run_workers(shards, devices, root / "logs")
    if any(code != 0 for code in codes):
        failed = [s["name"] for s, c in zip(shards, codes) if c != 0]
        raise RuntimeError(
            f"held-out shards failed: {failed}; no recovery table written. "
            f"Logs are in {root / 'logs'}. The completed search runs are "
            "untouched -- re-run the `heldout` stage rather than the search."
        )
    return stage


def heldout(args):
    """Re-run only the held-out stage against an existing search directory.

    The searches are the expensive part. When held-out rescoring fails for a
    fixable reason, repeating them would discard hours of correct work, so
    this stage stands alone and writes into a fresh directory.
    """
    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)
    runs = sorted((root / "search").glob("*/summary.json"))
    if not runs:
        raise RuntimeError(f"no completed search runs under {root / 'search'}")
    print(f"Found {len(runs)} completed search run(s) in {root / 'search'}")

    complex_pdb = REPO / "P17_Alpha.pdb" if args.target == "alpha" else None
    stage = run_heldout(root, devices, args.opendde_dtype, complex_pdb, args.suffix)
    if args.target == "alpha":
        calibration = json.loads(Path(args.calibration).read_text())
        selected = args.select_rung or calibration.get("recommended_rung")
        write_recovery_table(
            root, calibration, selected, args.edit_budget or selected, stage
        )
    else:
        write_jn1_table(root, stage)
    print(f"\nCompleted. Results: {root}")
    return 0


def jn1(args):
    """The JN.1 question: search from WT P17, which is already non-binding.

    The WT baseline comes free -- candidate 0 of every run is WT P17+JN.1
    scored under this exact path, which is also what completes the section
    19.1 scale bar on the non-binding side.
    """
    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    print(
        f"JN.1 recovery: {', '.join(args.policies)} x seeds "
        f"{args.search_seeds}, edit budget {args.edit_budget}"
    )
    print(
        "  No damage stage: WT P17 against JN.1 is already non-binding. "
        "Unlike the Alpha control, no solution is known to exist here."
    )
    jobs = search_jobs(
        root,
        args.policies,
        args.search_seeds,
        None,
        args.edit_budget,
        args,
        None,
    )
    codes = run_workers(jobs, devices, root / "logs")
    if any(code != 0 for code in codes):
        failed = [j["name"] for j, c in zip(jobs, codes) if c != 0]
        raise RuntimeError(f"search workers failed: {failed}; held-out stage not run")

    heldout_seeds = getattr(args, "heldout_seeds", None) or HELDOUT_SEEDS
    stage = run_heldout(
        root, devices, args.opendde_dtype, None, "", heldout_seeds
    )
    write_jn1_table(root, stage, heldout_seeds)
    print(f"\nCompleted. Results: {root}")
    return 0


def write_jn1_table(root, stage="heldout", heldout_seeds=HELDOUT_SEEDS):
    """Held-out metrics per candidate, against this run's own WT baseline."""
    from p17_rescore_winners import merge_shard_tables

    merge_shard_tables(root / stage, len(list((root / stage).glob("shard_*"))))
    predictions = list(
        csv.DictReader((root / stage / "tables/predictions.csv").open())
    )
    grouped = {}
    for row in predictions:
        grouped.setdefault(row["candidate_id"], []).append(row)
    if "0" not in grouped:
        raise ValueError("held-out predictions contain no WT candidate (id 0)")

    def summarize(samples):
        ipsae = [float(r["ipsae_min"]) for r in samples]
        pose = [float(r["binder_pose_rmsd_A"]) for r in samples]
        target = [float(r["target_aligned_rmsd_A"]) for r in samples]
        return dict(
            heldout_seeds=len(samples),
            mean_ipsae=sum(ipsae) / len(ipsae),
            worst_ipsae=min(ipsae),
            mean_pose_rmsd_A=sum(pose) / len(pose),
            worst_pose_rmsd_A=max(pose),
            mean_target_fit_A=sum(target) / len(target),
        )

    wt = summarize(grouped["0"])
    rows = []
    for candidate, samples in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        metrics = summarize(samples)
        rows.append(
            dict(
                candidate_id=candidate,
                is_wt=candidate == "0",
                **metrics,
                # Movement relative to this run's own WT, on held-out seeds.
                # There is no known ceiling here, unlike the Alpha control, so
                # these are differences and not recovered fractions.
                ipsae_gain_over_wt=metrics["mean_ipsae"] - wt["mean_ipsae"],
                pose_change_vs_wt_A=metrics["mean_pose_rmsd_A"]
                - wt["mean_pose_rmsd_A"],
            )
        )
    (root / "tables").mkdir(exist_ok=True)
    with (root / "tables/jn1_recovery.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "tables/jn1_context.json").write_text(
        json.dumps(
            dict(
                wt_baseline=wt,
                heldout_seeds=list(heldout_seeds),
                selection_seeds=list(SELECTION_SEEDS),
                alpha_binding_regime_reference=(
                    "Compare against the Alpha control's measured reference: "
                    "that run reports what this predictor assigns a complex it "
                    "is confident about, in this same path."
                ),
                interpretation=(
                    "Gains are relative to this run's own WT P17+JN.1 "
                    "baseline, on held-out seeds. No solution is known to "
                    "exist in the feasible set, so a null result is only "
                    "interpretable alongside a passing Alpha control. "
                    "Confidence gain does not establish binding, and pose is "
                    "in the proposal gradient, so pose improvement is partly "
                    "circular."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {root / 'tables/jn1_recovery.csv'}")
    print(
        f"  WT baseline: mean ipSAE {wt['mean_ipsae']:.4f}, "
        f"worst pose {wt['worst_pose_rmsd_A']:.2f} A, "
        f"target fit {wt['mean_target_fit_A']:.2f} A"
    )


def search(args):
    ladder = json.loads(Path(args.ladder).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)

    selected = args.select_rung or calibration.get("recommended_rung")
    if selected is None:
        raise RuntimeError(
            "no damaged rung reached the non-binding regime; pass --select-rung "
            "to override deliberately, with the reason recorded"
        )
    rung = next((r for r in ladder["rungs"] if r["n_edits"] == selected), None)
    if rung is None:
        raise RuntimeError(f"rung {selected} is not in the ladder")
    scored = next(
        (r for r in calibration["rungs"] if r["n_edits"] == selected), None
    )
    if scored is None or not scored.get("scored"):
        raise RuntimeError(f"rung {selected} was never successfully scored")
    budget = args.edit_budget or selected

    print(f"Recovering from the {selected}-edit rung, edit budget {budget}.")
    print(f"  damaged mean ipSAE {scored['mean_ipsae']:.4f} -> reference "
          f"{calibration['reference']['mean_ipsae']:.4f}")
    if budget < selected:
        print(
            f"  WARNING: budget {budget} < {selected} damaged positions, so the "
            "undamaged sequence is outside the feasible set and full recovery "
            "is impossible by construction.",
            file=sys.stderr,
        )

    print(f"  policies: {', '.join(args.policies)}, seeds {args.search_seeds}")
    jobs = search_jobs(
        root,
        args.policies,
        args.search_seeds,
        rung["sequence"],
        budget,
        args,
        REPO / "P17_Alpha.pdb",
    )
    codes = run_workers(jobs, devices, root / "logs")
    if any(code != 0 for code in codes):
        failed = [j["name"] for j, c in zip(jobs, codes) if c != 0]
        raise RuntimeError(f"search workers failed: {failed}; held-out stage not run")

    stage = run_heldout(
        root, devices, args.opendde_dtype, REPO / "P17_Alpha.pdb"
    )
    write_recovery_table(root, calibration, selected, budget, stage)
    print(f"\nCompleted. Results: {root}")
    return 0


def write_recovery_table(root, calibration, selected, budget, stage="heldout"):
    """One table placing every scored sequence against the recovery target."""
    from p17_rescore_winners import merge_shard_tables

    merge_shard_tables(root / stage, len(list((root / stage).glob("shard_*"))))
    predictions = list(
        csv.DictReader((root / stage / "tables/predictions.csv").open())
    )
    grouped = {}
    for row in predictions:
        grouped.setdefault(row["candidate_id"], []).append(row)

    reference = calibration["reference"]
    damaged = next(r for r in calibration["rungs"] if r["n_edits"] == selected)
    rows = []
    for candidate, samples in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        ipsae = [float(r["ipsae_min"]) for r in samples]
        pose = [float(r["binder_pose_rmsd_A"]) for r in samples]
        mean_ipsae = sum(ipsae) / len(ipsae)
        span = reference["mean_ipsae"] - damaged["mean_ipsae"]
        rows.append(
            dict(
                candidate_id=candidate,
                heldout_seeds=len(samples),
                mean_ipsae=mean_ipsae,
                worst_ipsae=min(ipsae),
                mean_pose_rmsd_A=sum(pose) / len(pose),
                worst_pose_rmsd_A=max(pose),
                # Fraction of the damage closed, on confidence. Negative means
                # worse than the damaged start. 1.0 means the reference's own
                # measured confidence was reached -- on held-out seeds, which
                # the search never selected on.
                ipsae_recovery_fraction=(
                    (mean_ipsae - damaged["mean_ipsae"]) / span if span > 0 else None
                ),
            )
        )
    (root / "tables").mkdir(exist_ok=True)
    with (root / "tables/recovery.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "tables/recovery_context.json").write_text(
        json.dumps(
            dict(
                selected_rung_edits=selected,
                edit_budget=budget,
                reference_metrics=reference,
                damaged_metrics=damaged,
                heldout_seeds=list(HELDOUT_SEEDS),
                selection_seeds=list(SELECTION_SEEDS),
                interpretation=(
                    "ipsae_recovery_fraction is relative to this run's own "
                    "measured reference and damaged confidence, on held-out "
                    "seeds. It measures whether the gradient restored the "
                    "predictor's opinion, not whether anything binds."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {root / 'tables/recovery.csv'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    common = dict(devices="0", steps=64, opendde_dtype="bf16")
    c = sub.add_parser("calibrate")
    c.add_argument("--ladder", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    c.add_argument("--devices", default=common["devices"])
    c.add_argument("--steps", type=int, default=common["steps"])
    c.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    c.add_argument(
        "--non-binding-ceiling", type=float, default=NON_BINDING_IPSAE_CEILING
    )
    c.set_defaults(func=calibrate)

    def add_search_options(sub_parser, *, seed_default):
        sub_parser.add_argument("--output-dir", type=Path, required=True)
        sub_parser.add_argument("--devices", default=common["devices"])
        sub_parser.add_argument(
            "--search-seeds", type=int, nargs="+", default=seed_default
        )
        sub_parser.add_argument(
            "--policies",
            nargs="+",
            choices=["population", "independent"],
            default=["population"],
            help="Default population only: section 17 keeps it as the "
            "provisional policy and section 19.4 deprioritizes the "
            "population-versus-independent comparison, so the workers are "
            "better spent on search seeds than on that contrast.",
        )
        sub_parser.add_argument("--steps", type=int, default=common["steps"])
        sub_parser.add_argument(
            "--opendde-dtype", choices=["fp32", "bf16"], default="bf16"
        )
        sub_parser.add_argument("--max-score-calls", type=int, default=32)
        sub_parser.add_argument("--max-gradient-calls", type=int, default=32)
        sub_parser.add_argument("--max-proposals", type=int, default=320)
        sub_parser.add_argument(
            "--selection-seeds",
            type=int,
            nargs="+",
            default=list(SELECTION_SEEDS),
            help="structural seeds the search selects on (default: "
            "%(default)s). A smoke run uses one to halve its cost.",
        )
        sub_parser.add_argument(
            "--heldout-seeds",
            type=int,
            nargs="+",
            default=list(HELDOUT_SEEDS),
            help="structural seeds reserved for evaluation, never used for "
            "selection (default: %(default)s).",
        )

    s = sub.add_parser("search", help="Alpha recovery from a damaged rung")
    s.add_argument("--ladder", type=Path, required=True)
    s.add_argument("--calibration", type=Path, required=True)
    s.add_argument("--select-rung", type=int, default=None)
    s.add_argument("--edit-budget", type=int, default=None)
    add_search_options(s, seed_default=[0, 1, 2, 3])
    s.set_defaults(func=search)

    j = sub.add_parser("jn1", help="JN.1 search from WT P17")
    j.add_argument(
        "--edit-budget",
        type=int,
        default=5,
        help="WT-relative cap (default 5, the project default; section 7 "
        "argues 7 is also defensible and must be a separate run).",
    )
    add_search_options(j, seed_default=[0, 1, 2, 3])
    j.set_defaults(func=jn1)

    h = sub.add_parser(
        "heldout",
        help="re-run only held-out rescoring against an existing search/ dir",
    )
    h.add_argument("--output-dir", type=Path, required=True)
    h.add_argument("--devices", default=common["devices"])
    h.add_argument("--target", choices=["alpha", "jn1"], required=True)
    h.add_argument(
        "--suffix",
        default="_retry",
        help="appended to the stage directory so a previous failed attempt is "
        "preserved rather than overwritten (default: %(default)s)",
    )
    h.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    h.add_argument(
        "--calibration", type=Path, default=None, help="required for --target alpha"
    )
    h.add_argument("--select-rung", type=int, default=None)
    h.add_argument("--edit-budget", type=int, default=None)
    h.set_defaults(func=heldout)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is heldout and args.target == "alpha":
        if args.calibration is None:
            parser.error("--target alpha requires --calibration")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
