"""Driver for the P17+Alpha gradient recovery control (docs/P17_JN1.md 19.3B).

Two stages, invoked by `run_p17_alpha_recovery.sh`:

`calibrate` forward-scores the undamaged reference and every damage rung, and
reports which rungs actually reached the non-binding regime. Damage severity is
not assumed to work: a rung that still scores like a binder is useless as a
starting point, and one that destroys the fold is a different experiment.

`search` runs recovery searches from the selected rung and rescores the
archived winners on held-out structural seeds.

Recovery is measured against the reference's *own* measured confidence and
pose, not against a literature number. Nothing here establishes binding.
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

    jobs = []
    for policy in ("independent", "population"):
        for seed in args.search_seeds:
            name = f"{policy}_seed{seed}"
            jobs.append(
                dict(
                    name=name,
                    command=[
                        sys.executable, str(EXAMPLES / "p17_confidence_search.py"),
                        "--policy", policy,
                        "--output-dir", str(root / "search" / name),
                        "--complex", str(REPO / "P17_Alpha.pdb"),
                        "--epitope-mode", "contact",
                        "--start-sequence", rung["sequence"],
                        "--seed", str(seed),
                        "--edit-budget", str(budget),
                        "--max-score-calls", str(args.max_score_calls),
                        "--max-gradient-calls", str(args.max_gradient_calls),
                        "--max-proposals", str(args.max_proposals),
                        "--sampling-steps", str(args.steps),
                        "--opendde-dtype", args.opendde_dtype,
                        "--selection-seeds", *[str(s) for s in SELECTION_SEEDS],
                    ],
                )
            )
    codes = run_workers(jobs, devices, root / "logs")
    if any(code != 0 for code in codes):
        failed = [j["name"] for j, c in zip(jobs, codes) if c != 0]
        raise RuntimeError(f"search workers failed: {failed}; held-out stage not run")

    print("\nHeld-out rescoring on unseen structural seeds...")
    heldout = [
        dict(
            name=f"shard_{shard}",
            command=[
                sys.executable, str(EXAMPLES / "p17_rescore_winners.py"),
                "--input", str(root / "search"),
                "--output-dir", str(root / "heldout" / f"shard_{shard}"),
                "--opendde-dtype", args.opendde_dtype,
                "--seeds", *[str(s) for s in HELDOUT_SEEDS],
                "--num-shards", str(len(devices)),
                "--shard-index", str(shard),
            ],
        )
        for shard in range(len(devices))
    ]
    codes = run_workers(heldout, devices, root / "logs")
    if any(code != 0 for code in codes):
        raise RuntimeError("held-out shards failed; no recovery table written")

    write_recovery_table(root, calibration, selected, budget)
    print(f"\nCompleted. Results: {root}")
    return 0


def write_recovery_table(root, calibration, selected, budget):
    """One table placing every scored sequence against the recovery target."""
    from p17_rescore_winners import merge_shard_tables

    merge_shard_tables(root / "heldout", len(list((root / "heldout").glob("shard_*"))))
    predictions = list(csv.DictReader((root / "heldout/tables/predictions.csv").open()))
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

    s = sub.add_parser("search")
    s.add_argument("--ladder", type=Path, required=True)
    s.add_argument("--calibration", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--devices", default=common["devices"])
    s.add_argument("--search-seeds", type=int, nargs="+", default=[0, 1])
    s.add_argument("--select-rung", type=int, default=None)
    s.add_argument("--edit-budget", type=int, default=None)
    s.add_argument("--steps", type=int, default=common["steps"])
    s.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    s.add_argument("--max-score-calls", type=int, default=32)
    s.add_argument("--max-gradient-calls", type=int, default=32)
    s.add_argument("--max-proposals", type=int, default=320)
    s.set_defaults(func=search)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
