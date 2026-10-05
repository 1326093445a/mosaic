"""Re-predict WT and saved P17 run winners; save CIF/PDB and pose diagnostics.

These are new predictions of archived sequences, not recovered original poses.
This does not run an optimization or train models. Seed 0 revisits the original
pilot scoring seed; seeds 1 and 2 were not used for selection in that pilot.
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from p17_search_outputs import SearchOutputs, compact_prediction


def check_reference_consistency(
    wt, target, original, complex_pdb, binder_chain, target_chain
):
    """Refuse to rescore archived sequences against the wrong reference.

    Returns the target sequence that must actually be folded, which is not
    always the reference's own.

    The target normally must match exactly: it defines the complex, and a
    mismatch means pose would be measured against the wrong structure. This is
    what caught a run on 2026-10-04 that pointed Alpha searches (195-residue
    target, chain A) at the default JN.1 reference (184 residues, chain T).

    The negative control is the one case where the archived target is
    *supposed* to differ: it replaces the target sequence with a decoy while
    keeping the reference's coordinates, since pose against a decoy is not
    interpretable anyway (§20.11 item 1). For such an archive the real target
    is checked against the reference instead, the archived target is checked
    against the recorded decoy, and the decoy is returned so the rescoring
    folds the complex the search actually scored. Without this the gate
    refused the arm outright -- which it did on 2026-10-05 -- and relaxing the
    gate alone would have silently rescored every decoy winner against the
    real JN.1 target.

    The binder is compared against the *reference's* sequence, which is not the
    archived `binder_sequence` when a run started somewhere else. The recovery
    control starts from a deliberately damaged sequence, so comparing that to
    the reference would refuse every such run -- the second half of the same
    bug. Archives predating `reference_binder_sequence` fall back to
    `start_differs_from_reference`, and length is always checked because a
    different-length chain cannot be compared at all.
    """
    expected_binder = original.get("reference_binder_sequence")
    if expected_binder is None:
        expected_binder = (
            None
            if original.get("start_differs_from_reference")
            else original["binder_sequence"]
        )
    name = Path(complex_pdb).name
    decoy = original.get("decoy_target")
    if decoy is not None:
        if target != decoy["real_target_sequence"]:
            raise ValueError(
                "local reference target differs from the real target this "
                f"decoy run replaced: {name} chain {target_chain} gives "
                f"{len(target)} aa, archive expects "
                f"{len(decoy['real_target_sequence'])} aa. Pass "
                "--complex/--target-chain for a non-JN.1 reference."
            )
        if original["target_sequence"] != decoy["decoy_target_sequence"]:
            raise ValueError(
                "archived target sequence does not match the decoy recorded "
                "in the same config; the archive is internally inconsistent "
                "and must not be rescored"
            )
        if len(decoy["decoy_target_sequence"]) != len(target):
            raise ValueError(
                "decoy and reference target lengths differ, so the reference "
                "coordinates cannot be reused"
            )
        scored_target = decoy["decoy_target_sequence"]
    elif target != original["target_sequence"]:
        raise ValueError(
            "local reference target differs from archived input: "
            f"{name} chain {target_chain} gives {len(target)} aa, archive "
            f"expects {len(original['target_sequence'])} aa. Pass "
            "--complex/--target-chain for a non-JN.1 reference."
        )
    else:
        scored_target = target
    if expected_binder is not None and wt != expected_binder:
        raise ValueError(
            "local reference binder differs from archived input: "
            f"{name} chain {binder_chain} gives {len(wt)} aa, archive expects "
            f"{len(expected_binder)} aa. Pass --complex/--binder-chain for a "
            "non-JN.1 reference."
        )
    if len(wt) != len(original["binder_sequence"]):
        raise ValueError(
            f"reference binder is {len(wt)} aa but archived sequences are "
            f"{len(original['binder_sequence'])} aa; pose comparison would be "
            "between different-length chains"
        )
    return scored_target


def load_candidates(source):
    """Read flat or organized batch results, optionally directly from a tarball."""
    source = Path(source)
    with ExitStack() as stack:
        if source.is_dir():
            configs = {
                str(p.relative_to(source)): p for p in source.rglob("config.json")
            }

            def read(path):
                return json.loads((source / path).read_text())
        else:
            archive = stack.enter_context(tarfile.open(source))
            files = {m.name: m for m in archive.getmembers() if m.isfile()}
            configs = {p: p for p in files if p.endswith("/config.json")}

            def read(path):
                return json.load(archive.extractfile(files[path]))

        if not configs:
            raise ValueError("no run configurations found")
        candidates, links, baseline = [], [], None

        def register(sequence):
            for row in candidates:
                if row["sequence"] == sequence:
                    return row["candidate_id"]
            idx = len(candidates)
            candidates.append(dict(candidate_id=idx, sequence=sequence, is_wt=idx == 0))
            return idx

        for path in sorted(configs):
            config = read(path)
            if baseline is None:
                baseline = config
                register(config["binder_sequence"])
            for field in (
                "binder_sequence",
                "target_sequence",
                "checkpoint",
                "recycling_steps",
            ):
                if config[field] != baseline[field]:
                    raise ValueError(f"inconsistent batch {field}")
            for field in (
                "sampling_steps",
                "pae_cutoff",
                "distance_cutoff",
                "selection_seeds",
            ):
                if config["arguments"][field] != baseline["arguments"][field]:
                    raise ValueError(f"inconsistent batch {field}")
            summary = read(str(Path(path).with_name("summary.json")))
            winner = summary["best_sequence"]
            wt = baseline["binder_sequence"]
            if len(winner) != len(wt) or any(
                a not in "ARNDCQEGHILKMFPSTWYV" for a in winner
            ):
                raise ValueError("invalid winner sequence")
            edits = {i for i, (a, b) in enumerate(zip(wt, winner)) if a != b}
            if (
                not edits <= set(config["designable_positions_0idx"])
                or len(edits) > config["config"]["edit_budget"]
            ):
                raise ValueError("winner violates recorded sequence constraints")
            links.append(
                dict(
                    candidate_id=register(winner),
                    source_run=str(Path(path).parent),
                    source_candidate_id=summary["best_id"],
                    original_score=summary["best_score"],
                )
            )
    return candidates, links, baseline


def select_shard(candidates, num_shards, shard_index):
    """Keep global candidate IDs stable across validation and repeat stages."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num-shards >= 1 and 0 <= shard-index < num-shards")
    return [c for c in candidates if c["candidate_id"] % num_shards == shard_index]


def merge_shard_tables(root, num_shards):
    """Build stage-level tables with file links relative to the stage directory."""
    root = Path(root)
    merged = {
        name: [] for name in ("candidates.csv", "source_runs.csv", "predictions.csv")
    }
    seen_candidates, seen_predictions = set(), set()
    total_calls = 0
    for index in range(num_shards):
        directory = root / f"shard_{index}"
        summary = json.loads((directory / "summary.json").read_text())
        if not summary.get("completed"):
            raise ValueError(f"incomplete shard {index}")
        total_calls += summary["prediction_calls"]
        if summary["sequences"] == 0:
            continue
        for name in merged:
            with (directory / "tables" / name).open() as f:
                rows = list(csv.DictReader(f))
            if name == "predictions.csv" and len(rows) != summary["prediction_calls"]:
                raise ValueError(f"prediction count mismatch in shard {index}")
            for row in rows:
                if name == "candidates.csv":
                    key = row["candidate_id"]
                    if key in seen_candidates:
                        raise ValueError("candidate repeated across shards")
                    seen_candidates.add(key)
                if name == "predictions.csv":
                    key = (row["candidate_id"], row["selection_seed"])
                    if key in seen_predictions:
                        raise ValueError("prediction repeated across shards")
                    seen_predictions.add(key)
                    for field in ("structure_file", "cif_file", "confidence_file"):
                        path = directory / row[field]
                        if not path.is_file():
                            raise ValueError(f"missing prediction artifact: {path}")
                        row[field] = path.relative_to(root).as_posix()
                row["shard"] = index
                merged[name].append(row)
    (root / "tables").mkdir()
    for name, rows in merged.items():
        if not rows:
            continue
        rows.sort(
            key=lambda r: (int(r["candidate_id"]), int(r.get("selection_seed", 0)))
        )
        with (root / "tables" / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                completed=True,
                shards=num_shards,
                sequences=len(seen_candidates),
                prediction_calls=total_calls,
                predictions_table="tables/predictions.csv",
            ),
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Completed batch directory or tar.gz archive",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Fresh output directory"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--sampling-steps", type=int, default=None, help="Default: original run setting"
    )
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    # Which reference the archived sequences were searched against. Omitting
    # this keeps the JN.1 default. The Alpha recovery control searches a
    # different target (195 residues, chain A, against JN.1's 184 on chain T),
    # so rescoring its winners needs the reference told to it: the sequence
    # consistency check below would otherwise correctly refuse the run.
    parser.add_argument("--complex", type=Path, default=None)
    parser.add_argument("--binder-chain", default=None)
    parser.add_argument("--target-chain", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("sampling steps must be positive")
    candidates, links, original = load_candidates(args.input)
    try:
        candidates = select_shard(candidates, args.num_shards, args.shard_index)
    except ValueError as exc:
        parser.error(str(exc))
    selected_ids = {c["candidate_id"] for c in candidates}
    links = [row for row in links if row["candidate_id"] in selected_ids]
    steps = args.sampling_steps or original["arguments"]["sampling_steps"]
    precision = args.opendde_dtype or original["arguments"].get("opendde_dtype", "fp32")
    print(
        f"Shard {args.shard_index}/{args.num_shards}, candidate IDs {sorted(selected_ids)}: "
        f"{len(candidates)} sequences, seeds {args.seeds}: "
        f"{len(candidates) * len(args.seeds)} forward predictions; {steps} sampling steps.",
        flush=True,
    )
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not candidates:
        (args.output_dir / "summary.json").write_text(
            json.dumps(
                dict(
                    completed=True,
                    sequences=0,
                    prediction_calls=0,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                )
            )
            + "\n"
        )
        return

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.structure_prediction import TargetChain
    from mosaic.losses.structure_prediction import predicted_tm_score
    from p17_confidence_search import confidence_metrics, record_memory_call
    from p17_hallucination_search import (
        COMPLEX_PDB,
        BINDER_CHAIN,
        TARGET_CHAIN,
        load_structure,
        reference_binder_target_ca,
    )

    if args.complex is None:
        if args.binder_chain or args.target_chain:
            raise ValueError("--binder-chain/--target-chain require --complex")
        complex_pdb = COMPLEX_PDB
        binder_chain, target_chain = BINDER_CHAIN, TARGET_CHAIN
        reference, wt, reference_target = load_structure()
        binder_ca, target_ca = reference_binder_target_ca(reference)
    else:
        from p17_alpha_reference import load_complex

        complex_pdb = args.complex
        binder_chain = args.binder_chain or "B"
        target_chain = args.target_chain or "A"
        loaded = load_complex(complex_pdb, binder_chain, target_chain)
        wt, reference_target = loaded["binder_seq"], loaded["target_seq"]
        binder_ca, target_ca = loaded["binder_ca"], loaded["target_ca"]

    # The sequence to fold, which is the decoy for a negative-control archive.
    # From here on `scored_target` is the only target sequence in scope: the
    # reference's own is deliberately not reusable, because on 2026-10-05 this
    # function folded the decoy and then exported the prediction labelled with
    # the reference target, which `SearchOutputs.save_prediction` refused --
    # correctly, but only after the whole arm had run.
    scored_target = check_reference_consistency(
        wt, reference_target, original, complex_pdb, binder_chain, target_chain
    )
    del reference_target
    outputs = SearchOutputs(args.output_dir, binder_ca, target_ca)
    shutil.copy2(complex_pdb, args.output_dir / "reference.pdb")
    (args.output_dir / "README.md").write_text(
        "# P17 winner pose review\n\n"
        "New forward predictions of archived WT and winners; no optimization.\n"
        "`reference.pdb` is the target/binder pose reference. Align predicted target "
        f"to reference target chain {target_chain} to inspect binder pose "
        f"(reference binder chain {binder_chain}). Prediction chain mappings are in predictions.csv.\n"
        "`tables/candidates.csv` maps new IDs to sequences; "
        "`tables/source_runs.csv` maps them back to original winners.\n"
        "`tables/predictions.csv` contains per-seed ipSAE, ipTM, pose RMSD, target "
        "fit RMSD and independently aligned binder RMSD (angstroms).\n"
        "`structures/` has PDB + CIF for every sequence and seed; `confidence/` "
        "has matching PAE/pLDDT and coordinates. No RMSD acceptance gate is applied.\n"
        "These poses may differ from the unsaved original predictions. See config.json "
        "for which seeds were used in the original search.\n"
    )
    for filename, records in [
        ("candidates.csv", candidates),
        ("source_runs.csv", links),
    ]:
        with (args.output_dir / "tables" / filename).open("x", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0])
                if records
                else [
                    "candidate_id",
                    "source_run",
                    "source_candidate_id",
                    "original_score",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
    config = dict(
        input=str(args.input.resolve()),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        candidate_ids=sorted(selected_ids),
        seeds=args.seeds,
        sampling_steps=steps,
        original_selection_seeds=original["arguments"]["selection_seeds"],
        recycling_steps=original["recycling_steps"],
        checkpoint=original["checkpoint"],
        opendde_compute_precision=precision,
        reference_binder_chain=binder_chain,
        reference_target_chain=target_chain,
        reference_source=str(complex_pdb),
        reference_sha256=hashlib.sha256(complex_pdb.read_bytes()).hexdigest(),
        pae_cutoff=original["arguments"]["pae_cutoff"],
        distance_cutoff=original["arguments"]["distance_cutoff"],
        # Says plainly which target was folded, so a decoy archive's numbers
        # cannot be read as the real target's.
        scored_target_is_decoy=original.get("decoy_target") is not None,
        decoy_identity_to_real_target=(
            original["decoy_target"].get("identity_to_real_target")
            if original.get("decoy_target") is not None
            else None
        ),
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"OpenDDE compute: {precision}", flush=True)
    model = OpenDDEModelAbag(compute_precision=precision)
    features, _ = model.binder_features(
        len(wt), [TargetChain(scored_target, use_msa=False)]
    )

    @eqx.filter_jit
    def predict(x, key):
        output = model.model_output(
            PSSM=x,
            features=features,
            recycling_steps=config["recycling_steps"],
            sampling_steps=steps,
            key=key,
        )
        iptm = predicted_tm_score(
            output.pae_logits,
            output.pae_bins,
            pair_mask=output.asym_id[:, None] != output.asym_id[None, :],
        ).max()
        return compact_prediction(output), iptm

    with (args.output_dir / "logs/memory.jsonl").open("x") as log:

        def log_memory(kind, before, after, **fields):
            log.write(
                json.dumps(
                    dict(event=kind, before=before, after=after, **fields),
                    allow_nan=False,
                )
                + "\n"
            )
            log.flush()

        for candidate in candidates:
            seq = np.array(
                [TOKENS.index(a) for a in candidate["sequence"]], dtype=np.int32
            )
            for seed in args.seeds:
                with record_memory_call(log_memory, "rescore", seq, seed=seed):
                    output, iptm = predict(
                        jax.nn.one_hot(jnp.asarray(seq), len(TOKENS)),
                        jax.random.key(seed),
                    )
                    output = jax.tree.map(np.asarray, output)
                    metrics = confidence_metrics(
                        output.pae,
                        output.backbone_coordinates[:, 1],
                        len(wt),
                        config["pae_cutoff"],
                        config["distance_cutoff"],
                    )
                    row = outputs.save_prediction(
                        candidate["candidate_id"],
                        seed,
                        candidate["sequence"],
                        scored_target,
                        output,
                        dict(iptm=float(iptm), **metrics),
                    )
                print(
                    f"candidate {candidate['candidate_id']}, seed {seed}: ipSAE={row['ipsae_min']:.4f}, "
                    f"pose RMSD={row['binder_pose_rmsd_A']:.2f} A",
                    flush=True,
                )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            dict(
                completed=True,
                sequences=len(candidates),
                prediction_calls=len(candidates) * len(args.seeds),
                predictions_table="tables/predictions.csv",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
