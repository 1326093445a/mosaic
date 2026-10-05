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
# The Alpha control's reference. `None` elsewhere means the search script's
# own default reference, which is JN.1.
ALPHA_COMPLEX_PDB = REPO / "P17_Alpha.pdb"
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


def score_command(output_dir, start_sequence, steps, dtype, complex_pdb):
    """A minimal confidence-search invocation that only scores its start point.

    Reuses the search entry point rather than a second scoring path, so the
    calibration numbers come from exactly the machinery the search will use.
    Budgets are the smallest the harness accepts; the start sequence is scored
    as that run's own WT.

    `complex_pdb` is required and has no default: `None` means the search
    script's own reference (JN.1). It used to be hardcoded to Alpha, which
    silently pointed the decoy arm's fold check at the wrong reference -- the
    decoy was built to the 184 aa JN.1 target and then validated against
    Alpha's 195 aa one, so the length guard refused the run (exit 2) on
    2026-10-05. Every caller now states which reference it means.
    """
    command = [
        sys.executable, str(EXAMPLES / "p17_confidence_search.py"),
        "--policy", "independent",
        "--output-dir", str(output_dir),
        "--width", "1",
        "--max-score-calls", "1",
        "--max-gradient-calls", "1",
        "--max-proposals", "1",
        "--sampling-steps", str(steps),
        "--opendde-dtype", dtype,
        "--selection-seeds", *[str(s) for s in SELECTION_SEEDS],
    ]
    if complex_pdb is not None:
        command += ["--complex", str(complex_pdb), "--epitope-mode", "contact"]
    if start_sequence is not None:
        command += ["--start-sequence", start_sequence]
    return command


def chain_plddt(run_dir, rows):
    """Mean per-chain pLDDT for the given prediction rows, or None.

    Read from the saved confidence arrays rather than a table column, so this
    works on runs archived before the columns existed. Returns None when the
    arrays are absent instead of failing: the caller decides whether a missing
    intrinsic measure is fatal.
    """
    import numpy as np

    binder, target = [], []
    for row in rows:
        path = Path(run_dir) / row.get("confidence_file", "")
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=True) as data:
            if "plddt" not in data or "asym_id" not in data:
                return None
            plddt = np.asarray(data["plddt"], dtype=float)
            asym = np.asarray(data["asym_id"]).astype(int)
        is_binder = asym == asym[0]
        if is_binder.all():
            return None
        binder.append(float(plddt[is_binder].mean()))
        target.append(float(plddt[~is_binder].mean()))
    if not binder:
        return None
    return dict(
        mean_binder_plddt=sum(binder) / len(binder),
        mean_target_plddt=sum(target) / len(target),
        worst_target_plddt=min(target),
    )


def read_start_metrics(run_dir):
    """Mean ipSAE and worst pose RMSD for candidate 0 -- the run's start point.

    pLDDT fields are present only when the confidence arrays were saved.
    """
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
        **(chain_plddt(run_dir, rows) or {}),
    )


def calibrate(args):
    ladder = json.loads(Path(args.ladder).read_text())
    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)

    jobs = [
        dict(
            name="reference",
            command=score_command(
                root / "reference", None, args.steps, args.opendde_dtype,
                ALPHA_COMPLEX_PDB,
            ),
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
                    ALPHA_COMPLEX_PDB,
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
            for flag, attr in (
                ("--target-entropy", "target_entropy"),
                ("--acceptance-temperature", "acceptance_temperature"),
                ("--weight-pose", "weight_pose"),
                ("--width", "width"),
            ):
                value = getattr(args, attr, None)
                if value is not None:
                    command += [flag, str(value)]
            decoy = getattr(args, "_decoy_sequence", None)
            if decoy is not None:
                command += ["--target-sequence", decoy]
            if getattr(args, "save_saliency", False):
                command += ["--save-saliency"]
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


def decoy(args):
    """Negative control: the identical search against a target it should fail on.

    Section 20.11 item 1. Every other control asks whether a solution exists;
    none asks whether this pipeline reports success regardless of the target.
    If confidence still climbs to the levels section 20.5 reports, those
    numbers say nothing about JN.1 specifically.

    Two stages, because the decoy has a confound. A shuffled target has no
    native fold, so a low score could mean "no complementarity" or merely "the
    predictor could not fold the target" -- the second makes the control
    vacuous. The first stage scores the decoy's own WT and checks that the
    target still places consistently; only then does the search run.
    """
    from p17_alpha_reference import (
        epitope_scrambled_target,
        real_decoy_target,
        shuffled_target,
    )

    devices = [d for d in args.devices.replace(" ", "").split(",") if d]
    root = Path(args.output_dir)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    real_target = reference_target_sequence(args.complex, args.target_chain)
    if args.decoy_sequence and args.decoy_structure:
        raise ValueError("pass either --decoy-sequence or --decoy-structure")
    if args.decoy_structure:
        decoy_seq, provenance = real_decoy_target(
            args.decoy_structure, args.decoy_structure_chain, len(real_target)
        )
    elif args.decoy_mode == "epitope" and not args.decoy_sequence:
        real_target, epitope_idx = reference_epitope(
            args.complex, args.binder_chain, args.target_chain
        )
        decoy_seq, provenance = epitope_scrambled_target(
            real_target, epitope_idx, args.decoy_seed
        )
    elif args.decoy_sequence:
        decoy_seq = Path(args.decoy_sequence).read_text().strip().upper() \
            if Path(args.decoy_sequence).is_file() else args.decoy_sequence.strip().upper()
        provenance = dict(kind="supplied", length=len(decoy_seq))
        if len(decoy_seq) != len(real_target):
            raise ValueError(
                f"supplied decoy is {len(decoy_seq)} aa, the reference target "
                f"is {len(real_target)}; lengths must match"
            )
    else:
        decoy_seq, provenance = shuffled_target(real_target, args.decoy_seed)
    args._decoy_sequence = decoy_seq
    print(f"Decoy target: {provenance}")

    # Stage 1: does the decoy target fold? Without this the control can pass
    # for the wrong reason.
    probe = root / "fold_check"
    # The fold check must use the same reference the decoy was built from, or
    # the length guard in the search script refuses it.
    command = score_command(
        probe, None, args.steps, args.opendde_dtype, args.complex
    )
    command += ["--target-sequence", decoy_seq]
    codes = run_workers(
        [dict(name="fold_check", command=command)], devices[:1], root / "logs"
    )
    metrics = read_start_metrics(probe) if codes[0] == 0 else None
    if metrics is None:
        raise RuntimeError(
            f"decoy fold check failed (exit {codes[0]}); see {root / 'logs'}"
        )
    # The gate is the decoy target's own pLDDT, not its fit to the reference
    # target's coordinates. Measured on 2026-10-05: the shuffled decoy's
    # target fit is 17.40 A, but so would a *perfectly folded* unrelated
    # protein's be -- `target_aligned_rmsd_A` superimposes the predicted decoy
    # on JN.1's coordinates, so for a decoy it measures shape difference from
    # JN.1 and says nothing about folding. pLDDT is intrinsic to the chain.
    # (The shuffled decoy fails both, at 0.405 target pLDDT against the
    # binder's 0.843 in the same prediction -- the right verdict, but the
    # RMSD gate reached it for the wrong reason.)
    plddt = metrics.get("mean_target_plddt")
    if plddt is None:
        raise RuntimeError(
            "decoy fold check saved no confidence arrays, so the decoy's own "
            f"pLDDT cannot be read; see {root / 'logs'}"
        )
    interpretable = plddt >= args.min_target_plddt
    print(
        f"  decoy WT: mean ipSAE {metrics['mean_ipsae']:.4f}, target pLDDT "
        f"{plddt:.3f} (binder {metrics['mean_binder_plddt']:.3f}), fit to the "
        f"real target {metrics['mean_target_fit_A']:.2f} A "
        f"({'interpretable' if interpretable else 'NOT INTERPRETABLE'})"
    )
    (root / "fold_check.json").write_text(
        json.dumps(
            dict(
                decoy=provenance,
                decoy_sequence=decoy_seq,
                wt_metrics=metrics,
                min_target_plddt=args.min_target_plddt,
                interpretable=interpretable,
                interpretation=(
                    "A decoy target the predictor cannot fold makes this "
                    "negative control vacuous: low confidence would follow "
                    "from the target not folding rather than from absent "
                    "complementarity. The gate is the decoy chain's own "
                    "pLDDT. target_aligned_rmsd_A is recorded but is NOT a "
                    "fold test here: it superimposes the predicted decoy on "
                    "the real target's coordinates, so a correctly folded "
                    "unrelated protein scores badly on it by construction."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    if not interpretable and not args.force:
        raise RuntimeError(
            f"decoy target pLDDT {plddt:.3f} is below {args.min_target_plddt}, "
            "so the predictor cannot fold this decoy and a low decoy score "
            "would be uninterpretable. Supply a real unrelated protein with "
            "--decoy-sequence or --decoy-structure, or pass --force to "
            "proceed and record why."
        )

    jobs = search_jobs(
        root, args.policies, args.search_seeds, None, args.edit_budget, args,
        args.complex,
    )
    codes = run_workers(jobs, devices, root / "logs")
    if any(code != 0 for code in codes):
        failed = [j["name"] for j, c in zip(jobs, codes) if c != 0]
        raise RuntimeError(f"decoy search workers failed: {failed}")

    heldout_seeds = getattr(args, "heldout_seeds", None) or HELDOUT_SEEDS
    stage = run_heldout(
        root, devices, args.opendde_dtype, args.complex, "", heldout_seeds
    )
    write_jn1_table(root, stage, heldout_seeds)
    print(
        "\nCompare mean_ipsae against the real JN.1 arm. If the decoy reaches "
        "comparable confidence, that arm's gains are not about its target."
    )
    print(f"Completed. Results: {root}")
    return 0


def reference_epitope(complex_pdb, binder_chain, target_chain):
    """Target sequence and contact-epitope indices for the reference in use.

    Uses the same 8 A CA-CA definition and the same reference file the search
    itself loads, so the positions an epitope-scrambled decoy destroys are the
    positions this run calls the interface.
    """
    sys.path.insert(0, str(EXAMPLES))
    from p17_alpha_reference import contact_epitope
    from p17_hallucination_search import CONTACT_DISTANCE

    if complex_pdb is None:
        from p17_hallucination_search import (
            load_structure,
            reference_binder_target_ca,
        )

        model, _, target = load_structure()
        binder_ca, target_ca = reference_binder_target_ca(model)
    else:
        from p17_alpha_reference import load_complex

        reference = load_complex(
            complex_pdb, binder_chain or "B", target_chain or "A"
        )
        target = reference["target_seq"]
        binder_ca = reference["binder_ca"]
        target_ca = reference["target_ca"]
    return target, contact_epitope(binder_ca, target_ca, CONTACT_DISTANCE)


def reference_target_sequence(complex_pdb, target_chain):
    """The target sequence a decoy replaces, from whichever reference is used."""
    if complex_pdb is None:
        import sys as _sys

        _sys.path.insert(0, str(EXAMPLES))
        from p17_hallucination_search import load_structure

        _, _, target = load_structure()
        return target
    from p17_alpha_reference import load_complex

    return load_complex(complex_pdb, "B", target_chain or "A")["target_seq"]


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
        # Exploration knobs. Already on the search CLI but never threaded
        # through a launcher, and never swept: section 20.8 found six of eight
        # runs plateauing under the cold default chain.
        sub_parser.add_argument(
            "--save-saliency",
            action="store_true",
            help="record the per-(position, residue) delta matrix at every "
            "gradient step; it cannot be recovered after a run.",
        )
        sub_parser.add_argument("--target-entropy", type=float, default=None)
        sub_parser.add_argument("--acceptance-temperature", type=float, default=None)
        sub_parser.add_argument("--width", type=int, default=None)
        sub_parser.add_argument(
            "--weight-pose",
            type=float,
            default=None,
            help="0 runs the pose-held-out arm: pose is still measured and "
            "reported but no longer guides proposals, which is what separates "
            "a real pose gain from the objective reporting on itself "
            "(section 19.6 item 1).",
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

    d = sub.add_parser(
        "decoy",
        help="negative control: the same search against a target it should fail on",
    )
    d.add_argument(
        "--complex",
        type=Path,
        default=None,
        help="reference to draw the real target from; default is JN.1",
    )
    d.add_argument("--target-chain", default=None)
    d.add_argument("--binder-chain", default=None)
    d.add_argument(
        "--decoy-mode",
        choices=["epitope", "shuffled"],
        default="epitope",
        help="how to build the decoy when no sequence or structure is given. "
        "`epitope` scrambles only the contact epitope, keeping the rest of "
        "the real target so the predictor can still fold it. `shuffled` "
        "scrambles the whole chain and does NOT fold in this predictor "
        "(0.405 mean target pLDDT, measured), so it fails its own gate "
        "(default: %(default)s).",
    )
    d.add_argument(
        "--decoy-sequence",
        default=None,
        help="a real unrelated protein of matching length, inline or as a "
        "file. Avoids the shuffled decoy's fold confound.",
    )
    d.add_argument("--decoy-seed", type=int, default=0)
    d.add_argument(
        "--decoy-structure",
        type=Path,
        default=None,
        help="a structure to draw a real unrelated decoy target from, trimmed "
        "symmetrically to the reference target's length. Preferred over the "
        "shuffled decoy, which does not fold (0.405 pLDDT, measured).",
    )
    d.add_argument("--decoy-structure-chain", default="A")
    d.add_argument(
        "--min-target-plddt",
        type=float,
        default=0.7,
        help="the decoy chain's own mean pLDDT must reach this, or the "
        "predictor cannot fold it and a low decoy score is uninterpretable "
        "(default: %(default)s). The real JN.1 target reaches ~0.9; the "
        "shuffled decoy measured 0.405.",
    )
    d.add_argument(
        "--force",
        action="store_true",
        help="proceed despite a failed fold check, recording that it failed.",
    )
    d.add_argument("--edit-budget", type=int, default=5)
    add_search_options(d, seed_default=[0, 1, 2, 3])
    d.set_defaults(func=decoy)

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
