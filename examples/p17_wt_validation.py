"""WT-only coordinate investigation; no search, mutation selection or gradients.

Three paths × two sampling budgets × two model seeds, two identical-input repeats
per worker. Native Torch and JAX randomness/precision are not bitwise equivalent.
"""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys
from time import perf_counter

from p17_pose_experiment import REPO, devices_from_csv, run_stage

PATHS = ("mosaic", "direct", "native")
ATOM_FIELDS = (
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "ref_atom_name_chars",
    "ref_mask",
    "ref_pos",
    "asym_id",
    "residue_index",
    "restype",
    "ref_element",
    "ref_charge",
    "frame_atom_index",
    "has_frame",
    "distogram_rep_atom_mask",
    "pae_rep_atom_mask",
    "structural_token_index",
    "parent_residue_idx",
    "subtoken_role_id",
    "atom_to_structural_token_idx",
    "atom_to_structural_tokatom_idx",
    "structural_has_frame",
    "structural_frame_atom_index",
    "structural_distogram_rep_atom_mask",
    "structural_pae_rep_atom_mask",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(args):
    jobs = []
    modes = getattr(args, "aggregation_modes", ["stable"])
    for steps in args.steps:
        for mode in args.paths:
            controls = [None] if mode == "native" else modes
            for aggregation in controls:
                for seed in args.seeds:
                    suffix = f"_{aggregation}" if aggregation and len(modes) > 1 else ""
                    name = f"{mode}{suffix}_steps{steps}_seed{seed}"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--path",
                        mode,
                        "--sampling-steps",
                        str(steps),
                        "--seed",
                        str(seed),
                        "--recycles",
                        str(args.recycles),
                        "--opendde-dtype",
                        args.opendde_dtype,
                        "--reference",
                        str(args.reference),
                        "--output-dir",
                        str(args.output_dir / "workers" / name),
                    ]
                    if aggregation:
                        command += ["--aggregation-mode", aggregation]
                    jobs.append(
                        dict(name=name, command=command, aggregation_mode=aggregation)
                    )
    return jobs


def load_reference(path):
    import gemmi
    import numpy as np

    model = gemmi.read_structure(str(path))[0]
    chains = [model[name] for name in ("B", "T")]
    sequences = [
        gemmi.one_letter_code([r.name for r in chain]).upper() for chain in chains
    ]
    if not all(sequences) or any(
        set(s) - set("ARNDCQEGHILKMFPSTWYV") for s in sequences
    ):
        raise ValueError("reference must have standard protein chains B and T")
    arrays = np.array(
        [
            [[*residue[name][0].pos] for name in ("N", "CA", "C", "O")]
            for chain in chains
            for residue in chain
        ]
    )
    asym = np.repeat([0, 1], [len(c) for c in chains])
    index = np.concatenate([np.arange(len(c)) for c in chains])
    return sequences, [r.name for c in chains for r in c], arrays, asym, index


def seed_preparation():
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def feature_arrays(feat):
    import numpy as np

    def host(value):
        if hasattr(value, "detach"):
            return (
                value.detach().cpu().float().numpy()
                if value.dtype.is_floating_point
                else value.detach().cpu().numpy()
            )
        a = np.asarray(value)
        return a.astype(np.float32) if str(a.dtype) == "bfloat16" else a

    return {
        name: host(feat[name] if isinstance(feat, dict) else getattr(feat, name))
        for name in ATOM_FIELDS
    }


def write_raw_cif(path, coords, arrays, residue_names):
    import gemmi
    import numpy as np
    from p17_structure_audit import decode_atom_names

    names = decode_atom_names(arrays["ref_atom_name_chars"])
    tokens = arrays["atom_to_token_idx"].astype(int)
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    asym = arrays["asym_id"]
    for number, chain_id in enumerate(dict.fromkeys(asym.tolist())):
        chain = gemmi.Chain(("B", "T")[number])
        for token in np.flatnonzero(asym == chain_id):
            residue = gemmi.Residue()
            residue.name = residue_names[token]
            residue.seqid = gemmi.SeqId(int(arrays["residue_index"][token]) + 1, " ")
            for atom_id in np.flatnonzero(
                (tokens == token) & (arrays["ref_mask"] > 0.5)
            ):
                atom = gemmi.Atom()
                atom.name = str(names[atom_id])
                atom.element = gemmi.Element(atom.name[0])
                atom.pos = gemmi.Position(*map(float, coords[atom_id]))
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


def native_prediction_input(data, seed):
    """Native diffusion reads a dedicated rollout seed, not just Torch's RNG."""
    import torch

    batch = copy.deepcopy(data)
    batch["input_feature_dict"]["inference_seed"] = torch.tensor(seed, dtype=torch.long)
    return batch


def prepare_native(args, sequences):
    import torch
    from jopendde.fast_init import fast_init
    from jopendde.inference import Predictor, _set_asset_cache_dir, spec_from_sequences
    from mosaic.cache import cache_dir
    from opendde.config.inference import (
        build_inference_config,
        update_gpu_compatible_configs,
    )
    from opendde.model.opendde import update_input_feature_dict
    from runner.inference import InferenceRunner

    cfg = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
    _set_asset_cache_dir(cfg, cache_dir("opendde"))
    cfg.load_checkpoint_path = str(
        cache_dir("opendde", "checkpoint", "opendde_abag.pt")
    )
    if not Path(cfg.load_checkpoint_path).is_file():
        raise FileNotFoundError(cfg.load_checkpoint_path)
    cfg.use_msa = cfg.use_template = cfg.use_rna_msa = False
    cfg.triangle_multiplicative = cfg.triangle_attention = "torch"
    cfg.enable_diffusion_shared_vars_cache = cfg.enable_efficient_fusion = False
    cfg.model.N_cycle = args.recycles
    cfg.sample_diffusion.N_step = args.sampling_steps
    cfg.sample_diffusion.N_sample = 1
    cfg.dtype = args.opendde_dtype
    cfg.dump_dir = str(args.output_dir / "native_runtime")
    cfg = update_gpu_compatible_configs(cfg)
    with fast_init():
        runner = InferenceRunner(cfg)
    # Use the native loader without constructing a second, converted model.
    predictor = Predictor(None, runner, cfg, None)
    seed_preparation()
    data, atom_array = predictor._load_batch(
        spec_from_sequences(sequences, name="WT", seed=0)
    )
    raw = copy.deepcopy(data["input_feature_dict"])
    with torch.no_grad():
        raw = runner.model.relative_position_encoding.generate_relp(raw, lazy=False)
        raw = update_input_feature_dict(raw)
    arrays = feature_arrays(raw)
    # Independent native atom-array names must agree with the model metadata.
    from p17_structure_audit import decode_atom_names
    import numpy as np

    if not np.array_equal(
        decode_atom_names(arrays["ref_atom_name_chars"]), atom_array.atom_name
    ):
        raise ValueError("native atom-array names disagree with feature names")
    write_json(args.output_dir / "native_config.json", cfg.to_dict())

    def evaluate():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        result = runner.predict(native_prediction_input(data, args.seed))
        return result["coordinate"][0].detach().float().cpu().numpy(), None

    return arrays, evaluate, None


def prepare_jax(args, sequences):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.common import TOKENS
    from mosaic.losses.opendde import set_binder_sequence
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.opendde_precision import cast_float_arrays, compute_dtype
    from mosaic.structure_prediction import TargetChain

    model = OpenDDEModelAbag(compute_precision=args.opendde_dtype)
    seed_preparation()
    # The design path splits this same key internally. Mirror that split once,
    # retain its exact refreshed features, and call the common forward explicitly.
    key, geom_key = jax.random.split(jax.random.key(args.seed))
    if args.path == "mosaic":
        bundle, _ = model.binder_features(
            len(sequences[0]), [TargetChain(sequences[1], use_msa=False)]
        )
        x = jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in sequences[0]]), 20)
        feat = set_binder_sequence(x, bundle, geom_key)
    else:
        feat, _ = model.target_only_features(
            [TargetChain(s, use_msa=False) for s in sequences]
        )
    if args.opendde_dtype == "bf16":
        feat = cast_float_arrays(feat, compute_dtype("bf16"))
    arrays = feature_arrays(feat)

    @eqx.filter_jit
    def forward(model, feat, key):
        output = model.model_output(
            features=feat,
            recycling_steps=args.recycles,
            sampling_steps=args.sampling_steps,
            key=key,
        )
        # Keep raw coordinates, while discarding large confidence logits.
        return eqx.tree_at(
            lambda o: (o.distogram_logits, o.distogram_bins, o.pae_logits, o.pae_bins),
            output,
            replace=(None,) * 4,
        )

    def evaluate():
        output = jax.tree.map(np.asarray, forward(model, feat, key))
        return np.asarray(output.structure_coordinates), output

    return arrays, evaluate, np.asarray(jax.random.key_data(key)).tolist()


def memory_stats(native):
    if native:
        import torch

        return dict(
            allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
    from p17_confidence_search import device_memory_stats

    return device_memory_stats()


def run_worker(args):
    # Must precede any JAX import; native Torch should own its GPU exclusively.
    if args.path == "native":
        os.environ["JAX_PLATFORMS"] = "cpu"
    if args.path != "native":
        os.environ["MOSAIC_OPENDDE_AGGREGATION"] = args.aggregation_mode
    import numpy as np
    from mosaic.cache import cache_dir
    from p17_structure_audit import (
        audit_raw_mapping,
        backbone_geometry,
        decode_atom_names,
    )

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("arrays", "structures", "reports", "logs"):
        (root / folder).mkdir()
    sequences, residue_names, reference_bb, asym, index = load_reference(args.reference)
    reference_geometry = backbone_geometry(
        reference_bb, np.ones(reference_bb.shape[:2]), asym, index
    )
    write_json(root / "reports/reference_geometry.json", reference_geometry)
    if not reference_geometry["passed"]:
        raise ValueError("reference backbone failed sanity checks")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(
        reference_sha256=sha256_file(args.reference),
        checkpoint_sha256=sha256_file(
            cache_dir("opendde", "checkpoint", "opendde_abag.pt")
        ),
        source_sha256={
            name: sha256_file(REPO / "examples" / name)
            for name in (
                "p17_wt_validation.py",
                "p17_structure_audit.py",
                "p17_pose_experiment.py",
                "run_p17_wt_validation.sh",
            )
        },
        preparation_seed=0,
        sample_count=1,
        repeats=2,
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith("XLA_")
            or k.startswith("OPENDDE_FORCE_")
            or k.startswith("MOSAIC_OPENDDE_")
            or k == "JOPENDDE_ATTENTION_DTYPE"
            or k in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "TF_GPU_ALLOCATOR")
        },
        precision_note="Native uses Torch autocast/skip_amp; Mosaic uses its BF16 layer policy. Random streams are not matched across frameworks.",
    )
    write_json(root / "config.json", settings)
    started = perf_counter()
    arrays, evaluate, model_key = (
        prepare_native(args, sequences)
        if args.path == "native"
        else prepare_jax(args, sequences)
    )
    from mosaic.common import TOKENS

    observed = "".join(TOKENS[i] for i in arrays["restype"][:, :20].argmax(-1))
    if observed != "".join(sequences):
        raise ValueError("model token sequence differs from reference")
    import importlib.metadata

    if args.path != "native":
        # The dependency patch imports these at trace time. Import them now so
        # the pre-prediction provenance report includes the exact kernel source.
        for module_name in ("mosaic.opendde_numerics", "mosaic.opendde_padding"):
            importlib.import_module(module_name)

    versions = {}
    for package in (
        "jax",
        "jaxlib",
        "torch",
        "equinox",
        "opendde",
        "jopendde",
        "mosaic",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(
            (
                "jopendde.",
                "mosaic.models.opendde",
                "mosaic.losses.opendde",
                "mosaic.opendde_precision",
                "mosaic.opendde_numerics",
                "mosaic.opendde_padding",
                "opendde.model.",
                "runner.inference",
            )
        ):
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py") and Path(path).is_file():
                sources[name] = dict(path=path, sha256=sha256_file(path))
    write_json(root / "reports/software.json", dict(versions=versions, sources=sources))
    np.savez_compressed(root / "arrays/atom_metadata.npz", **arrays)
    write_json(
        root / "reports/atom_metadata_hashes.json",
        {
            k: dict(
                shape=list(v.shape),
                dtype=str(v.dtype),
                sha256=hashlib.sha256(v.tobytes()).hexdigest(),
            )
            for k, v in arrays.items()
        },
    )
    reports, first = [], None
    names = decode_atom_names(arrays["ref_atom_name_chars"])
    for repeat in range(2):
        before = memory_stats(args.path == "native")
        tick = perf_counter()
        status = "error"
        try:
            coords, output = evaluate()
            np.savez_compressed(
                root / f"arrays/repeat_{repeat}.npz",
                raw_coordinates=coords,
                **(
                    {
                        "atom37_coords": output.atom37_coords,
                        "atom37_mask": output.atom37_mask,
                        "pae": output.pae,
                        "plddt": output.plddt,
                    }
                    if output is not None
                    else {}
                ),
            )
            report = audit_raw_mapping(
                coords,
                arrays["atom_to_token_idx"],
                names,
                arrays["ref_mask"],
                arrays["asym_id"],
                arrays["residue_index"],
                output.atom37_coords if output is not None else None,
                output.atom37_mask if output is not None else None,
            )
            report.update(repeat=repeat, elapsed_seconds=perf_counter() - tick)
            # Invalid finite geometry is evidence to preserve, not a worker crash.
            if np.isfinite(coords).all():
                write_raw_cif(
                    root / f"structures/repeat_{repeat}_raw.cif",
                    coords,
                    arrays,
                    residue_names,
                )
            if output is not None and np.isfinite(output.atom37_coords).all():
                output.to_structure().make_mmcif_document().write_file(
                    str(root / f"structures/repeat_{repeat}_mapped.cif")
                )
            if first is not None:
                valid = arrays["ref_mask"] > 0.5
                delta = np.linalg.norm(coords[valid] - first[valid], axis=-1)
                report["repeat_raw_max_displacement_A"] = (
                    float(delta.max()) if np.isfinite(delta).all() else None
                )
                report["repeat_bitwise_identical"] = bool(np.array_equal(coords, first))
            first = coords.copy()
            write_json(root / f"reports/repeat_{repeat}.json", report)
            reports.append(report)
            status = "completed"
        finally:
            with (root / "logs/memory.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(
                            repeat=repeat,
                            status=status,
                            before=before,
                            after=memory_stats(args.path == "native"),
                        )
                    )
                    + "\n"
                )
    write_json(
        root / "summary.json",
        dict(
            completed=True,
            path=args.path,
            aggregation_mode=args.aggregation_mode if args.path != "native" else None,
            seed=args.seed,
            sampling_steps=args.sampling_steps,
            recycles=args.recycles,
            model_key_data=model_key,
            geometry_passed=all(r["passed"] for r in reports),
            reports=reports,
            elapsed_seconds=perf_counter() - started,
            interpretation="WT-only numerical/geometry diagnostic. No optimization was run. Geometry failure does not abort independent controls.",
        ),
    )


def collect_results(root, jobs):
    rows = []
    for job in jobs:
        path = root / "workers" / job["name"] / "summary.json"
        if not path.exists():
            rows.append(
                dict(
                    worker=job["name"],
                    repeat=None,
                    aggregation_mode=job.get("aggregation_mode"),
                    repeat_bitwise_identical=None,
                    repeat_raw_max_displacement_A=None,
                    completed=False,
                    geometry_passed=None,
                    mapping_agrees=None,
                    raw_CA_median_max_A=None,
                    raw_N_CA_median_max_A=None,
                )
            )
            continue
        summary = json.loads(path.read_text())
        for report in summary["reports"]:
            chains = report["raw_geometry"]["chains"]

            def largest_median(name):
                values = [c["distances"][name]["median_A"] for c in chains]
                return max(values) if all(v is not None for v in values) else None

            rows.append(
                dict(
                    worker=job["name"],
                    repeat=report["repeat"],
                    aggregation_mode=summary.get("aggregation_mode"),
                    repeat_bitwise_identical=report.get("repeat_bitwise_identical"),
                    repeat_raw_max_displacement_A=report.get(
                        "repeat_raw_max_displacement_A"
                    ),
                    completed=True,
                    geometry_passed=report["passed"],
                    mapping_agrees=report.get("mapping_agrees"),
                    raw_CA_median_max_A=largest_median("adjacent_CA"),
                    raw_N_CA_median_max_A=largest_median("N_CA"),
                )
            )
    with (root / "tables/geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        root / "summary.json",
        dict(
            completed=all(row["completed"] for row in rows),
            geometry_passed=all(row["geometry_passed"] is True for row in rows),
            rows=rows,
            interpretation="Inspect each path/budget separately; this report never authorizes optimization.",
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=devices_from_csv, default=list("01234567"))
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--paths", choices=PATHS, nargs="+", default=list(PATHS))
    parser.add_argument("--recycles", type=int, default=4)
    parser.add_argument("--opendde-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--reference", type=Path, default=REPO / "P17_JN1.pdb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--aggregation-modes",
        choices=("original", "stable"),
        nargs="+",
        default=["stable"],
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=("original", "stable"),
        default="stable",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--path", choices=PATHS, help=argparse.SUPPRESS)
    parser.add_argument("--sampling-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.recycles < 1
        or any(s < 1 for s in args.steps)
        or any(s < 0 for s in args.seeds)
    ):
        parser.error("positive budgets and nonnegative seeds required")
    for name in ("steps", "seeds", "paths", "aggregation_modes"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"duplicate {name}")
    args.reference = args.reference.resolve()
    if args.worker:
        if (
            args.path is None
            or args.seed is None
            or args.sampling_steps is None
            or args.output_dir is None
        ):
            parser.error("incomplete worker arguments")
        if args.seed < 0 or args.sampling_steps < 1:
            parser.error("positive sampling steps and a nonnegative seed required")
        return run_worker(args)
    args.output_dir = (
        args.output_dir
        or REPO
        / "results"
        / (
            "p17_wt_validation_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{os.getpid()}"
        )
    ).resolve()
    root = args.output_dir
    if root.exists():
        parser.error(f"output already exists: {root}")
    jobs = build_plan(args)
    print(
        f"Repo: {REPO}\nOutput: {root}\nWT-only: {len(jobs)} workers, {2 * len(jobs)} predictions; GPUs {','.join(args.devices)}",
        flush=True,
    )
    commands = []
    for i, job in enumerate(jobs):
        command = (
            f"CUDA_VISIBLE_DEVICES={shlex.quote(args.devices[i % len(args.devices)])} "
            + shlex.join(job["command"])
        )
        commands.append(command)
        if args.dry_run:
            print(command)
    if args.dry_run:
        return
    root.mkdir(parents=True)
    for name in ("logs", "tables"):
        (root / name).mkdir()
    write_json(
        root / "plan.json",
        dict(
            jobs=jobs,
            devices=args.devices,
            allocator_environment={
                k: v
                for k, v in os.environ.items()
                if k.startswith("XLA_") or k == "TF_GPU_ALLOCATOR"
            },
        ),
    )
    (root / "commands.sh").write_text(
        "# Provenance; run the launcher to apply patches and allocator settings.\n"
        + "\n".join(commands)
        + "\n"
    )
    (root / "status.tsv").write_text("stage\tworker\tgpu\tpid\texit_code\n")
    (root / "README.md").write_text(
        "# WT coordinate validation\n\n"
        "tables/geometry.csv and summary.json summarize independent controls.\n"
        "workers/*/{arrays,structures,reports,logs} preserve atom metadata, raw and mapped coordinates, CIFs, geometry and memory.\n"
        "Each worker reuses features and repeats one model seed twice. Native Torch and JAX random streams and mixed precision differ.\n"
        "Aggregation controls run in fresh processes; both receive the same current padding fixes. The original control only restores the original averaging kernel.\n"
        "Worker exit 0 means completed; check geometry_passed separately. No optimization follows this run.\n"
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with (root / "logs/patches.log").open("x") as handle:
            for name in (
                "outer_product_mean",
                "structural_token_expander",
                "bf16_dtype",
                "aggregation",
                "padding",
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
        if any(path != "native" for path in args.paths):
            # Build a schema-updated template cache once before parallel workers
            # can race to create it with different host RNG states.
            print("Preparing shared atom-template cache on CPU...", flush=True)
            cache_env = dict(os.environ, JAX_PLATFORMS="cpu")
            with (root / "logs/template_cache.log").open("x") as handle:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import random, numpy as np, torch; "
                            "random.seed(0); np.random.seed(0); torch.manual_seed(0); "
                            "from mosaic.models.opendde import _get_atom_templates; "
                            "_get_atom_templates()"
                        ),
                    ],
                    cwd=REPO,
                    env=cache_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        run_stage("wt", jobs, args.devices, root)
    finally:
        collect_results(root, jobs)
        signal.signal(signal.SIGTERM, previous)
    print(f"Controls completed. Review {root / 'tables/geometry.csv'}", flush=True)


if __name__ == "__main__":
    main()
