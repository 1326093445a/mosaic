"""P17 population/independent search with full-confidence retention.

Starts each slot from WT. By default, proposals use full OpenDDE gradients from
contacts, pose RMSD, ipTM, bidirectional interface PAE and pTMEnergy, plus AbLang2
and the edit penalty. --proposal-path distogram retains the earlier cheap path.
Every new proposal separately receives forward-only full OpenDDE scoring.
Selection maximizes mean ipSAE-min across fixed structural seeds, optionally
prioritizing a worst-seed, WT-relative pose constraint.
This is a pilot harness, not a validated binding-optimization pipeline.

Run on a suitably provisioned GPU through run_p17_confidence_search.sh. Use the
same arguments except --policy/--output-dir for the two comparison arms. Actual
call counts and timings, including compilation, are recorded; equal ceilings
alone do not establish matched compute. No model weights are trained.

Per-device memory is sampled around every gradient and confidence call and
written to logs/memory.jsonl (see device_memory_stats()). `peak_bytes_in_use` is a
running maximum since process start, not reset per call -- see that function's
docstring for what this does and does not tell you.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

import numpy as np

from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search
from p17_search_outputs import SearchOutputs, compact_prediction, pose_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def device_memory_stats():
    """Per-device XLA memory counters, or an explicit "unsupported" marker.

    Keyed by str(device) (e.g. "cuda:0") so a two-GPU run stays disambiguated.
    Each device's stats dict comes straight from jax.Device.memory_stats():
    bytes_in_use is live usage at the moment of the call; peak_bytes_in_use is
    a RUNNING MAXIMUM SINCE PROCESS START (JAX/XLA expose no per-call reset),
    so a new shape does not reset it. Earlier work can dominate even the first
    call of a shape. An increase identifies a new high-water mark during that
    interval; an unchanged peak does not identify the call's maximum. Imports jax
    lazily so --help/argument validation stay independent of the model stack.
    """
    import jax

    stats = {}
    for device in jax.devices():
        try:
            device_stats = device.memory_stats()
        except Exception as exc:  # backend may not implement this
            device_stats = None
            note = repr(exc)
        else:
            note = None if device_stats is not None else "memory_stats() returned None"
        if device_stats is None:
            stats[str(device)] = {"supported": False, "note": note}
        else:
            stats[str(device)] = {"supported": True, **device_stats}
    return stats


@contextmanager
def record_memory_call(log_memory, kind, sequence, **fields):
    """Record input allocation and synchronized results; preserve model errors.

    A process killed without Python cleanup cannot be covered by finally.
    """
    before = device_memory_stats()
    error = None
    try:
        yield
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            log_memory(
                kind,
                before,
                device_memory_stats(),
                sequence=np.asarray(sequence).tolist(),
                status="error" if error is not None else "ok",
                error=error,
                **fields,
            )
        except Exception:
            if error is None:
                raise


def build_confidence_terms(weight_iptm, weight_interface_pae, weight_ptm_energy):
    """Confidence losses available from the full OpenDDE path."""
    from mosaic.losses.structure_prediction import (
        BinderTargetPAE,
        IPTMLoss,
        TargetBinderPAE,
        pTMEnergy,
    )

    return (
        weight_iptm * IPTMLoss()
        + weight_interface_pae * BinderTargetPAE()
        + weight_interface_pae * TargetBinderPAE()
        + weight_ptm_energy * pTMEnergy()
    )


def confidence_metrics(pae, ca, binder_length, pae_cutoff, distance_cutoff):
    """Use the same directional, mean-PAE ipSAE as the native control analysis.

    Distance cutoff affects only interface-count diagnostics, not the score.
    Do not substitute mosaic's logit-expectation IPSAE_min: its definition differs.
    """
    from p17_alpha_vs_jn1_native_opendde_analysis import ipsae_d0res_asym_max

    pae, ca = np.asarray(pae), np.asarray(ca)
    if (
        pae.ndim != 2
        or pae.shape[0] != pae.shape[1]
        or ca.shape != (len(pae), 3)
        or not 0 < binder_length < len(pae)
    ):
        raise ValueError("invalid PAE/coordinate shapes or binder length")
    if not np.all(np.isfinite(pae)) or not np.all(np.isfinite(ca)):
        raise ValueError("full prediction contains nonfinite PAE or coordinates")
    asym = np.concatenate([np.zeros(binder_length), np.ones(len(pae) - binder_length)])
    distances = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    bt, nb, nt = ipsae_d0res_asym_max(
        pae, distances, asym, 0, 1, pae_cutoff, distance_cutoff
    )
    tb, _, _ = ipsae_d0res_asym_max(
        pae, distances, asym, 1, 0, pae_cutoff, distance_cutoff
    )
    return dict(
        ipsae_min=min(bt, tb),
        bt_ipsae=bt,
        tb_ipsae=tb,
        interface_binder_count=nb,
        interface_target_count=nt,
    )


def score_prediction_samples(
    predict, sequence, seeds, pae_cutoff, distance_cutoff, *, references=None
):
    """Score every fixed seed, then average per-seed directional minima.

    ``predict(sequence, seed)`` returns (PAE, CA coordinates, ipTM). It runs a
    forward prediction only. Keep per-seed values so selection is auditable.
    """
    samples, metrics = [], {}
    for seed in seeds:
        pae, ca, iptm = predict(sequence, seed)
        values = confidence_metrics(
            np.asarray(pae), np.asarray(ca), len(sequence), pae_cutoff, distance_cutoff
        )
        values["iptm"] = float(iptm)
        if references is not None:
            values.update(pose_metrics(ca, *references))
        samples.append(values)
        metrics.update({f"seed_{seed}_{name}": value for name, value in values.items()})
    metrics.update(
        {
            name: float(np.mean([sample[name] for sample in samples]))
            for name in samples[0]
        }
    )
    if references is not None:
        metrics["worst_pose_rmsd_A"] = max(v["binder_pose_rmsd_A"] for v in samples)
    return ConfidenceScore(metrics["ipsae_min"], metrics)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=["population", "independent"], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Search RNG only.")
    parser.add_argument("--proposal-model-seed", type=int, default=0)
    parser.add_argument(
        "--selection-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Fixed structural seeds shared across candidates and arms.",
    )
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--edit-budget", type=int, default=5)
    parser.add_argument(
        "--max-score-calls",
        type=int,
        default=100,
        help="Unique scored sequences, including WT; each uses all selection seeds.",
    )
    parser.add_argument("--max-gradient-calls", type=int, default=100)
    parser.add_argument("--max-proposals", type=int, default=1000)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.6,
        help="Provisional normalized proposal entropy, shared across arms.",
    )
    parser.add_argument(
        "--acceptance-temperature",
        type=float,
        default=0.02,
        help="Provisional temperature in confidence-score units; 0 is greedy.",
    )
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--proposal-path",
        choices=["full", "distogram"],
        default="full",
        help="full includes confidence and coordinate-RMSD gradients; distogram is the earlier proxy objective.",
    )
    parser.add_argument("--weight-pose", type=float, default=1.0)
    parser.add_argument(
        "--retention-pose-margin",
        type=float,
        default=None,
        help="Optional worst-seed RMSD ceiling: measured WT plus this margin (angstroms).",
    )
    parser.add_argument("--pose-diagnostic", action="store_true")
    parser.add_argument("--diagnostic-max-target-rmsd", type=float, default=3.0)
    parser.add_argument("--diagnostic-min-proposal-tv", type=float, default=1e-4)
    parser.add_argument("--diagnostic-repeat-factor", type=float, default=3.0)
    parser.add_argument("--weight-iptm", type=float, default=0.025)
    parser.add_argument(
        "--weight-interface-pae",
        type=float,
        default=0.05,
        help="Applied to each interface direction in the full proposal loss.",
    )
    parser.add_argument("--weight-ptm-energy", type=float, default=0.025)
    parser.add_argument(
        "--pose-rmsd-tolerance",
        type=float,
        default=0.0,
        help="Full-path proposal RMSD tolerance in angstroms; 0 matches the existing full-gradient example. Not a retention gate.",
    )
    parser.add_argument("--pae-cutoff", type=float, default=12.0)
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=12.0,
        help="Interface-count diagnostic only.",
    )
    # Recovery-control inputs. Omitting all of these reproduces the previous
    # P17/JN.1 behavior exactly, including the literature hotspot epitope.
    parser.add_argument(
        "--complex",
        type=Path,
        default=None,
        help="Reference complex; default is the JN.1 reference with its "
        "hotspot epitope. Point this at P17_Alpha.pdb for the recovery "
        "control of docs/P17_JN1.md section 19.3B.",
    )
    parser.add_argument("--binder-chain", default=None)
    parser.add_argument("--target-chain", default=None)
    parser.add_argument(
        "--epitope-mode",
        choices=["hotspots", "contact"],
        default=None,
        help="'hotspots' uses the five JN.1 literature positions (default "
        "with no --complex). 'contact' derives the epitope from the "
        "reference's own CA contacts, required for a non-JN.1 target whose "
        "numbering those constants do not describe.",
    )
    parser.add_argument(
        "--start-sequence",
        default=None,
        help="Binder sequence the search starts from, instead of the "
        "reference's own. The recovery control passes a damaged sequence "
        "here; edit counts and the WT-relative cap are measured against it.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = SearchConfig(
            **{
                name: getattr(args, name)
                for name in (
                    "policy",
                    "width",
                    "edit_budget",
                    "max_score_calls",
                    "max_gradient_calls",
                    "max_proposals",
                    "target_entropy",
                    "acceptance_temperature",
                    "seed",
                )
            }
        )
        if args.proposal_model_seed < 0 or any(
            seed < 0 for seed in args.selection_seeds
        ):
            raise ValueError("model seeds must be nonnegative")
        if len(set(args.selection_seeds)) != len(args.selection_seeds):
            raise ValueError("selection seeds must be distinct")
        if args.sampling_steps < 1:
            raise ValueError("sampling steps must be >= 1")
        if any(
            not np.isfinite(v) or v < 0
            for v in (
                args.weight_pose,
                args.weight_iptm,
                args.weight_interface_pae,
                args.weight_ptm_energy,
                args.pose_rmsd_tolerance,
            )
        ):
            raise ValueError(
                "proposal weights and RMSD tolerance must be finite and nonnegative"
            )
        if any(
            not np.isfinite(v) or v <= 0
            for v in (args.pae_cutoff, args.distance_cutoff)
        ):
            raise ValueError("cutoffs must be positive and finite")
        if args.retention_pose_margin is not None and (
            not np.isfinite(args.retention_pose_margin)
            or args.retention_pose_margin < 0
        ):
            raise ValueError("retention pose margin must be finite and nonnegative")
        if (
            any(
                not np.isfinite(v) or v <= 0
                for v in (
                    args.diagnostic_max_target_rmsd,
                    args.diagnostic_min_proposal_tv,
                    args.diagnostic_repeat_factor,
                )
            )
            or args.diagnostic_min_proposal_tv > 1
        ):
            raise ValueError("invalid diagnostic thresholds")
        if args.pose_diagnostic and (
            args.proposal_path != "full"
            or args.weight_pose <= 0
            or args.retention_pose_margin is not None
            or args.edit_budget == 0
        ):
            raise ValueError(
                "diagnostic requires full path, positive pose weight/edit budget, no retention"
            )
    except ValueError as exc:
        parser.error(str(exc))
    # Prevent accidentally replacing an earlier experiment's evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # Keep --help and argument validation independent of the heavy model stack.
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from mosaic.common import TOKENS
    from mosaic.losses.ablang2 import load_ablang2
    from mosaic.losses.structure_prediction import predicted_tm_score
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.optimizers import _ranking_leaf
    from mosaic.structure_prediction import TargetChain
    from p17_hallucination_search import (
        CDR_RESIDUE_INDICES_1IDX,
        CONTACT_DISTANCE,
        HOTSPOT_TARGET_RESIDUE_INDICES_1IDX,
        OPENDDE_RECYCLING_STEPS,
        POSE_DRIFT_MARGIN,
        WEIGHT_OPENDDE_CONTACT,
        WEIGHT_ABLANG2,
        WEIGHT_EDIT_BUDGET,
        build_composite_losses,
        load_structure,
        measure_wt_distogram_pose_drift,
        reference_binder_target_ca,
        reference_binder_target_ca_distances,
    )

    setup_start = time.perf_counter()
    reference_info = None
    if args.complex is None:
        if args.binder_chain or args.target_chain:
            parser.error("--binder-chain/--target-chain require --complex")
        if (args.epitope_mode or "hotspots") != "hotspots":
            parser.error("--epitope-mode contact requires --complex")
        model, binder_seq, target_seq = load_structure()
        references = reference_binder_target_ca_distances(model)
        binder_ca, target_ca = reference_binder_target_ca(model)
        epitope_idx = np.array(
            sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX)
        )
        epitope_mode = "hotspots"
    else:
        # A different reference means a different target numbering, so the
        # JN.1 hotspot constants do not describe it. Require the epitope to be
        # derived from the structure rather than silently reusing indices that
        # would land on unrelated residues.
        from p17_alpha_reference import (
            ca_distance_matrix,
            ca_spacing_report,
            contact_epitope,
            load_complex,
        )

        epitope_mode = args.epitope_mode or "contact"
        if epitope_mode != "contact":
            parser.error(
                "--complex requires --epitope-mode contact: the hotspot "
                "constants are JN.1 target positions and do not transfer"
            )
        reference_info = load_complex(
            args.complex,
            args.binder_chain or "B",
            args.target_chain or "A",
        )
        binder_seq = reference_info["binder_seq"]
        target_seq = reference_info["target_seq"]
        binder_ca = reference_info["binder_ca"]
        target_ca = reference_info["target_ca"]
        references = ca_distance_matrix(binder_ca, target_ca)
        epitope_idx = contact_epitope(binder_ca, target_ca, CONTACT_DISTANCE)
        reference_info["spacing"] = [
            ca_spacing_report(binder_ca, "binder"),
            ca_spacing_report(target_ca, "target"),
        ]
        for report in reference_info["spacing"]:
            if not report["passed"]:
                raise ValueError(f"reference backbone spacing failed: {report}")
        reference_info["epitope_idx"] = [int(i) for i in epitope_idx]
        reference_info["binder_ca"] = None
        reference_info["target_ca"] = None
        model = None
        print(
            f"Reference {args.complex.name}: binder {len(binder_seq)} aa, "
            f"target {len(target_seq)} aa, contact epitope "
            f"{epitope_idx.size} residues at {CONTACT_DISTANCE} A",
            flush=True,
        )

    mask = np.array(
        [i + 1 in CDR_RESIDUE_INDICES_1IDX for i in range(len(binder_seq))]
    )
    designable_idx = np.flatnonzero(mask)

    # The search's origin. Edit budget and the Hamming cap are relative to
    # this sequence, so for the recovery control the damaged sequence is the
    # origin and the reference's own sequence is a point the search may or may
    # not reach -- it is never given as a target.
    if args.start_sequence is not None:
        start_seq = args.start_sequence.strip().upper()
        if len(start_seq) != len(binder_seq):
            parser.error(
                f"--start-sequence has {len(start_seq)} residues, reference "
                f"binder has {len(binder_seq)}"
            )
        if set(start_seq) - set(TOKENS[:20]):
            parser.error("--start-sequence has non-standard residues")
        changed = np.flatnonzero(
            np.array(list(start_seq)) != np.array(list(binder_seq))
        )
        if changed.size and not set(changed.tolist()) <= set(designable_idx.tolist()):
            parser.error(
                "--start-sequence differs from the reference outside the "
                "designable mask, so the search could not undo those changes"
            )
        print(
            f"Start sequence differs from the reference at {changed.size} "
            f"designable positions: {sorted(int(i) + 1 for i in changed)}",
            flush=True,
        )
        binder_seq = start_seq
    wt = np.array([TOKENS.index(aa) for aa in binder_seq], dtype=np.int32)
    outputs = SearchOutputs(args.output_dir, binder_ca, target_ca)

    metadata = dict(
        output_layout_version=2,
        reference_source=str(args.complex) if args.complex else "P17_JN1.pdb",
        epitope_mode=epitope_mode,
        reference_details=reference_info,
        start_sequence=binder_seq,
        start_differs_from_reference=args.start_sequence is not None,
        config=asdict(config),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        binder_sequence=binder_seq,
        target_sequence=target_seq,
        designable_positions_0idx=designable_idx.tolist(),
        epitope_positions_0idx=epitope_idx.tolist(),
        confidence_metric="mean across seeds of directional-min ipSAE d0res (mean PAE)",
        checkpoint="opendde_abag.pt",
        opendde_compute_precision=args.opendde_dtype,
        ablang2_compute_precision="fp32",
        recycling_steps=OPENDDE_RECYCLING_STEPS,
        proposal_objective=dict(
            path=args.proposal_path,
            weights=dict(
                contact=WEIGHT_OPENDDE_CONTACT,
                pose=args.weight_pose,
                ablang2=WEIGHT_ABLANG2,
                edit_budget=WEIGHT_EDIT_BUDGET,
                iptm=args.weight_iptm if args.proposal_path == "full" else 0.0,
                interface_pae_each_direction=args.weight_interface_pae
                if args.proposal_path == "full"
                else 0.0,
                ptm_energy=args.weight_ptm_energy
                if args.proposal_path == "full"
                else 0.0,
            ),
            sampling_steps=args.sampling_steps
            if args.proposal_path == "full"
            else None,
            stop_grad_conf_coords=False if args.proposal_path == "full" else None,
        ),
        environment={
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "JOPENDDE_ATTENTION_DTYPE",
                "MOSAIC_OPENDDE_AGGREGATION",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "TF_GPU_ALLOCATOR",
            )
        },
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata["git_commit"] = commit.stdout.strip() or None
    metadata["source_sha256"] = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in (
            "src/mosaic/search.py",
            "examples/p17_confidence_search.py",
            "examples/p17_search_outputs.py",
            "examples/p17_pose_diagnostics.py",
            "examples/p17_structure_audit.py",
            "examples/p17_hallucination_search.py",
            "examples/p17_alpha_vs_jn1_native_opendde_analysis.py",
            "src/mosaic/models/opendde.py",
            "src/mosaic/opendde_precision.py",
            "src/mosaic/losses/opendde.py",
            "src/mosaic/losses/structure_prediction.py",
            "src/mosaic/losses/transformations.py",
            "src/mosaic/losses/ablang2.py",
        )
    }
    metadata["versions"] = {}
    for package in ("jax", "jaxlib", "equinox", "numpy", "jopendde", "jablang"):
        try:
            metadata["versions"][package] = version(package)
        except PackageNotFoundError:
            metadata["versions"][package] = None
    if args.pose_diagnostic:
        from p17_structure_audit import DISTANCE_LIMITS

        metadata["predicted_backbone_distance_limits_A"] = DISTANCE_LIMITS
        metadata["diagnostic_schema_version"] = 2
    metadata_path = args.output_dir / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    reference_audit = None
    if args.pose_diagnostic:
        from p17_pose_diagnostics import audit_reference, geometry_checks, write_report
        from p17_hallucination_search import BINDER_CHAIN, TARGET_CHAIN, COMPLEX_PDB

        # Audit whatever reference this run actually loaded, not the module
        # default: auditing P17_JN1.pdb while scoring against Alpha would
        # report a passing correspondence for the wrong file.
        complex_pdb = args.complex if args.complex is not None else COMPLEX_PDB
        chains = (
            (args.binder_chain or "B", args.target_chain or "A")
            if args.complex is not None
            else (BINDER_CHAIN, TARGET_CHAIN)
        )
        try:
            if model is None:
                # `audit_reference` walks a gemmi model; the recovery control's
                # loader already validated numbering, CA uniqueness, altlocs
                # and spacing, and `reference_info` records all of it.
                reference_audit = dict(
                    source="p17_alpha_reference.load_complex",
                    reference=reference_info,
                    binder_residues=len(binder_seq),
                    target_residues=len(target_seq),
                )
            else:
                reference_audit = audit_reference(
                    model,
                    chains,
                    (binder_seq, target_seq),
                    (binder_ca, target_ca),
                )
            reference_audit["source_sha256"] = hashlib.sha256(
                complex_pdb.read_bytes()
            ).hexdigest()
            reference_audit["geometry"] = geometry_checks(binder_ca, target_ca)
            shutil.copyfile(complex_pdb, args.output_dir / "reference.pdb")
            np.savez_compressed(
                args.output_dir / "confidence/reference_ca.npz",
                binder_ca=binder_ca,
                target_ca=target_ca,
                designable_mask=mask,
            )
            metadata["reference_audit"] = reference_audit
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        except Exception as exc:
            write_report(
                args.output_dir,
                dict(
                    schema_version=2,
                    passed=False,
                    checks={"reference_geometry": False},
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise

    print("Loading frozen OpenDDE and AbLang2...", flush=True)
    print(f"OpenDDE compute: {args.opendde_dtype}; AbLang2: fp32", flush=True)
    opendde = OpenDDEModelAbag(compute_precision=args.opendde_dtype)
    features, _ = opendde.binder_features(
        len(wt), [TargetChain(target_seq, use_msa=False)]
    )
    ablang_model, ablang_tokenizer = load_ablang2()
    model_key = jax.random.key(args.proposal_model_seed)
    wt_pose = None
    if args.proposal_path == "distogram":
        wt_pose = measure_wt_distogram_pose_drift(
            opendde=opendde,
            features=features,
            reference_distances=references,
            binder_seq=binder_seq,
            key=model_key,
        )
        pose_tolerance = wt_pose + POSE_DRIFT_MARGIN
        confidence_loss = None
    else:
        pose_tolerance = args.pose_rmsd_tolerance
        confidence_loss = build_confidence_terms(
            args.weight_iptm,
            args.weight_interface_pae,
            args.weight_ptm_energy,
        )
    proposal_kwargs = dict(
        opendde=opendde,
        features=features,
        ablang2_model=ablang_model,
        ablang2_tokenizer=ablang_tokenizer,
        reference_distances=references,
        reference_binder_ca=binder_ca,
        reference_target_ca=target_ca,
        binder_seq=binder_seq,
        designable_idx=designable_idx,
        epitope_idx=epitope_idx,
        edit_budget=args.edit_budget,
        stop_grad_ablang2=False,
        opendde_path=args.proposal_path,
        pose_tolerance=pose_tolerance,
        opendde_sampling_steps=args.sampling_steps,
        opendde_num_samples=1,
        confidence_loss=confidence_loss,
    )

    def make_gradient(weight):
        proposal_loss, _ = build_composite_losses(**proposal_kwargs, pose_weight=weight)
        return eqx.filter_jit(eqx.filter_value_and_grad(proposal_loss, has_aux=True))

    gradient_eval = make_gradient(args.weight_pose)

    # Bound to a real file handle just before run_gradient_search executes (see
    # below). gradient_fn/predict_sequence close over this name; reassigning it
    # in main()'s own body -- not a nested function -- needs no `nonlocal`, and
    # Python resolves the free variable at call time, so the closures see the
    # real handle once it exists. A memory snapshot is worth having even if a
    # call raises, so failures are logged too, not silently dropped.
    memory_log = None
    memory_call_index = 0

    def log_memory(kind, before, after, **fields):
        nonlocal memory_call_index
        if memory_log is None:
            return
        memory_call_index += 1
        record = dict(
            event=kind,
            call_index=memory_call_index,
            before=before,
            after=after,
            **fields,
        )
        memory_log.write(json.dumps(record, allow_nan=False) + "\n")
        memory_log.flush()

    def gradient_details(sequence, evaluator, weight):
        with record_memory_call(
            log_memory,
            "gradient",
            sequence,
            proposal_path=args.proposal_path,
            seed=args.proposal_model_seed,
            pose_weight=weight,
        ):
            x = jax.nn.one_hot(jnp.asarray(sequence), len(TOKENS))
            (value, aux), gradient = evaluator(x, key=model_key)
            value = float(value)
            gradient = np.asarray(gradient)
            metrics = {}
            for name in (
                "target_contact",
                "binder_pose_rmsd",
                "pose_target_fit_rmsd",
                "binder_target_distogram_drift",
                "iptm",
                "bt_pae",
                "tb_pae",
                "pTMEnergy",
                "ablang2_ppl",
            ):
                leaf = _ranking_leaf(aux, name)
                if leaf is not None:
                    metrics[name] = float(leaf)
            on_event(
                dict(
                    event="proposal_metrics",
                    sequence=sequence.tolist(),
                    proposal_path=args.proposal_path,
                    proposal_loss=value,
                    pose_weight=weight,
                    metrics=metrics,
                )
            )
        return value, gradient, metrics

    def gradient_fn(sequence):
        return gradient_details(sequence, gradient_eval, args.weight_pose)[:2]

    @eqx.filter_jit
    def prediction(model, features, x, key):
        output = model.model_output(
            PSSM=x,
            features=features,
            recycling_steps=OPENDDE_RECYCLING_STEPS,
            sampling_steps=args.sampling_steps,
            key=key,
        )
        iptm = predicted_tm_score(
            output.pae_logits,
            output.pae_bins,
            pair_mask=output.asym_id[:, None] != output.asym_id[None, :],
        ).max()
        # Keep exact scored atoms for export, without transferring large logits.
        return compact_prediction(output), iptm

    def predict_sequence(sequence, seed, *, store=None, include_geometry=False):
        store = outputs if store is None else store
        with record_memory_call(log_memory, "confidence", sequence, seed=seed):
            x = jax.nn.one_hot(jnp.asarray(sequence), len(TOKENS))
            output, iptm = prediction(opendde, features, x, jax.random.key(seed))
            output = jax.tree.map(np.asarray, output)
            pae, ca, iptm = output.pae, output.backbone_coordinates[:, 1], float(iptm)
            metrics = confidence_metrics(
                pae, ca, len(sequence), args.pae_cutoff, args.distance_cutoff
            )
            store.save_prediction(
                store.candidate_id(sequence),
                seed,
                "".join(TOKENS[i] for i in sequence),
                target_seq,
                output,
                dict(iptm=iptm, **metrics),
            )
        if include_geometry:
            from p17_structure_audit import BACKBONE_SLOTS, backbone_geometry

            geometry = backbone_geometry(
                output.atom37_coords[:, BACKBONE_SLOTS],
                output.atom37_mask[:, BACKBONE_SLOTS],
                output.asym_id,
                output.residue_idx,
            )
            return pae, ca, iptm, geometry
        return pae, ca, iptm

    retention_ceiling = None

    def confidence_fn(sequence):
        nonlocal retention_ceiling
        score = score_prediction_samples(
            predict_sequence,
            sequence,
            args.selection_seeds,
            args.pae_cutoff,
            args.distance_cutoff,
            references=(binder_ca, target_ca),
        )
        if args.retention_pose_margin is None:
            return score
        if retention_ceiling is None:
            if not np.array_equal(sequence, wt):
                raise ValueError("WT must be scored first to calibrate retention")
            retention_ceiling = (
                score.metrics["worst_pose_rmsd_A"] + args.retention_pose_margin
            )
            metadata["retention"] = dict(
                kind="worst_seed_WT_relative",
                ceiling_A=retention_ceiling,
                margin_A=args.retention_pose_margin,
                interpretation="baseline preservation, not a correct-pose claim",
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        violation = max(0.0, score.metrics["worst_pose_rmsd_A"] - retention_ceiling)
        return ConfidenceScore(
            score.value,
            dict(
                score.metrics,
                pose_ceiling_A=retention_ceiling,
                pose_feasible=float(violation == 0),
            ),
            violation,
        )

    metadata.update(
        pose_tolerance=pose_tolerance,
        wt_pose_drift=wt_pose,
        devices=[str(device) for device in jax.devices()],
        setup_seconds=time.perf_counter() - setup_start,
        setup_cheap_forward_calls=int(args.proposal_path == "distogram"),
        memory_at_setup=device_memory_stats(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with (
        (args.output_dir / "logs/events.jsonl").open("x") as log,
        (args.output_dir / "logs/memory.jsonl").open("x") as memory_handle,
    ):
        memory_log = memory_handle

        def on_event(event):
            if event["event"] == "gradient":
                event = dict(event)
                event["proposal_loss"] = event.pop("cheap_loss")
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()
            if event["event"] == "evaluation":
                if outputs.candidate_id(event["sequence"]) != event["candidate_id"]:
                    raise RuntimeError("saved structure/candidate ID mismatch")
                print(
                    f"evaluated {event['candidate_id']}: ipSAE={event['score']:.4f} "
                    f"edits={event['edit_count']}",
                    flush=True,
                )

        if args.pose_diagnostic:
            from p17_pose_diagnostics import run_diagnostic, write_report

            try:
                off_eval = make_gradient(0.0)
                repeat_root = args.output_dir / "repeat"
                repeat_root.mkdir()
                repeat_outputs = SearchOutputs(repeat_root, binder_ca, target_ca)
                report = run_diagnostic(
                    root=args.output_dir,
                    wt=wt,
                    mask=mask,
                    config=config,
                    gradient_on=lambda seq: gradient_details(
                        seq, gradient_eval, args.weight_pose
                    ),
                    gradient_off=lambda seq: gradient_details(seq, off_eval, 0.0),
                    predict=lambda seq, seed: predict_sequence(
                        seq, seed, include_geometry=True
                    ),
                    repeat_predict=lambda seq, seed: predict_sequence(
                        seq, seed, store=repeat_outputs, include_geometry=True
                    ),
                    references=(binder_ca, target_ca),
                    seeds=args.selection_seeds,
                    pae_cutoff=args.pae_cutoff,
                    distance_cutoff=args.distance_cutoff,
                    max_target_rmsd=args.diagnostic_max_target_rmsd,
                    min_proposal_tv=args.diagnostic_min_proposal_tv,
                    repeat_factor=args.diagnostic_repeat_factor,
                )
                report["reference_audit"] = reference_audit
                report["memory_at_completion"] = device_memory_stats()
                write_report(args.output_dir, report)
            except Exception as exc:
                write_report(
                    args.output_dir,
                    dict(
                        schema_version=2,
                        passed=False,
                        checks={"completed": False},
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                raise
            if not report["passed"]:
                raise SystemExit("Pose diagnostic failed; see diagnostic.json")
            return

        result = run_gradient_search(
            wt=wt,
            designable_mask=mask,
            gradient_fn=gradient_fn,
            confidence_fn=confidence_fn,
            config=config,
            on_event=on_event,
        )
    active_ids = {candidate.id for candidate in result.active}
    ranking = {
        c.id: i + 1
        for i, c in enumerate(
            sorted(
                result.evaluated, key=lambda c: (c.constraint_violation, -c.score, c.id)
            )
        )
    }
    with (args.output_dir / "tables/candidates.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "rank",
                "is_wt",
                "structure_dir",
                "sequence",
                "edits",
                "score",
                "constraint_violation",
                "worst_pose_rmsd_A",
                "proposal_loss",
                "active",
                "best",
            ],
        )
        writer.writeheader()
        for candidate in result.evaluated:
            writer.writerow(
                dict(
                    id=candidate.id,
                    rank=ranking[candidate.id],
                    is_wt=candidate.id == 0,
                    structure_dir=f"structures/candidate_{candidate.id:05d}",
                    sequence="".join(TOKENS[i] for i in candidate.sequence),
                    edits=int(np.count_nonzero(wt != candidate.sequence)),
                    score=candidate.score,
                    constraint_violation=candidate.constraint_violation,
                    worst_pose_rmsd_A=candidate.metrics["worst_pose_rmsd_A"],
                    proposal_loss=result.cheap_losses.get(candidate.id, ""),
                    active=candidate.id in active_ids,
                    best=candidate.id == result.best.id,
                )
            )
    best_structure_files = outputs.copy_best(result.best.id, args.selection_seeds)
    summary = dict(
        output_layout_version=2,
        best_structure_files=best_structure_files,
        predictions_table="tables/predictions.csv",
        candidates_table="tables/candidates.csv",
        stats=result.stats,
        stop_reason=result.stop_reason,
        best_id=result.best.id,
        best_score=result.best.score,
        best_constraint_violation=result.best.constraint_violation,
        retention_pose_ceiling_A=retention_ceiling,
        best_sequence="".join(TOKENS[i] for i in result.best.sequence),
        active_ids=[candidate.id for candidate in result.active],
        full_prediction_calls=result.stats["score_calls"] * len(args.selection_seeds),
        full_gradient_calls=result.stats["gradient_calls"]
        if args.proposal_path == "full"
        else 0,
        distogram_gradient_calls=result.stats["gradient_calls"]
        if args.proposal_path == "distogram"
        else 0,
        # Same running-peak-since-process-start caveat as memory.jsonl and
        # device_memory_stats(): this is the highest usage seen anywhere in
        # the run, not necessarily attributable to the single most expensive
        # call in isolation.
        memory_at_completion=device_memory_stats(),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Finished: {result.stop_reason}; best ipSAE={result.best.score:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
