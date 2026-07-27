"""Real (expensive) OpenDDE structure prediction for a small, hand-picked
set of final hallucination-search candidates -- writes real CIFs to
actually look at, and computes the real coordinate-based BinderPoseRMSD
(not the cheap, coordinate-free BinderPoseDistogramDrift the search loop
used, which can't be inspected visually). See
docs/guidance_alphaseq_testing_notes.md section 13.2.

Why this is a separate script, not folded into the search: real diffusion
coordinate sampling (opendde_forward_from_trunk) is exactly the cost
build_distogram_only_loss exists to avoid paying thousands of times during
search. Here we pay it once per requested candidate, which is what it's
for -- see BinderPoseRMSD's own docstring
(src/mosaic/losses/structure_prediction.py) for this exact division of
labor, planned when it was built, not improvised now.

IMPORTANT (a real mistake caught by actually running this, not just
review): opendde_forward_from_trunk MUST be called under eqx.filter_jit at
real complex sizes -- calling it eagerly reproduces the exact
catastrophic memory blowup already diagnosed in
docs/guidance_alphaseq_testing_notes.md section 9b
(examples/vhh72_opendde_structure_prediction.py's module docstring
explains why: without JIT, XLA can't fuse operations across the
structural-token refiner's transformer layers, so every intermediate over
the full pair representation gets materialized). This script initially
called it eagerly despite citing that exact reference script, and OOM'd
requesting 234GiB on a 24GB GPU -- fixed below.

Usage:
    .venv/bin/python examples/vhh72_predict_hallucination_structures.py \\
        --combined-csv results/hallucination_sweep_20260725_234632/combined.csv \\
        --select mcmc:0:5,mcmc:1:5,greedy:0:5,greedy:1:5 \\
        --output-dir results/hallucination_sweep_20260725_234632/structures

    # Multi-seed sweeps: include seed as policy:stop_grad:seed:edit_count
    # to avoid silently selecting the first matching seed.
    .venv/bin/python examples/vhh72_predict_hallucination_structures.py \\
        --combined-csv results/hallucination_sweep_discrete_3seed/combined.csv \\
        --select mcmc:0:1:4 \\
        --output-dir results/hallucination_sweep_discrete_3seed/structures
"""
import argparse
import csv
import functools
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.opendde import opendde_forward_from_trunk, set_binder_sequence
from mosaic.losses.structure_prediction import BinderPoseRMSD
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_CIF = REPO_ROOT / "vhh72_wt_wt_rbd.cif"
N_DIFFUSION_STEPS = 8  # matches examples/vhh72_opendde_structure_prediction.py's real-prediction default
RECYCLING_STEPS = 4


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def load_structure():
    st = gemmi.read_structure(str(COMPLEX_CIF))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model["A"]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model["A2"]]).upper()
    return model, binder_seq, target_seq


def reference_binder_target_ca(model):
    def ca_coords(chain):
        coords = []
        for res in chain:
            for a in res:
                if a.name == "CA":
                    coords.append([a.pos.x, a.pos.y, a.pos.z])
                    break
        return np.array(coords, dtype=np.float32)
    return ca_coords(model["A"]), ca_coords(model["A2"])


def parse_select(select_str, rows):
    """--select entries:

    - legacy single-seed form: policy:stop_grad:edit_count
    - multi-seed form:         policy:stop_grad:seed:edit_count

    The 3-field form is kept for backward compatibility but is ambiguous for
    multi-seed combined CSVs; if more than one row matches, fail loudly instead
    of silently taking the first seed.
    """
    wanted = []
    for spec in select_str.split(","):
        parts = spec.split(":")
        if len(parts) == 3:
            policy, stop_grad, edit_count = parts
            wanted.append((policy, int(stop_grad), None, int(edit_count), spec))
        elif len(parts) == 4:
            policy, stop_grad, seed, edit_count = parts
            wanted.append((policy, int(stop_grad), int(seed), int(edit_count), spec))
        else:
            raise ValueError(
                f"bad --select entry {spec!r}; expected policy:stop_grad:edit_count "
                "or policy:stop_grad:seed:edit_count"
            )
    selected = []
    for policy, stop_grad, seed, edit_count, spec in wanted:
        match = [r for r in rows if r["policy"] == policy
                 and int(r["stop_grad"]) == stop_grad and int(r["edit_count"]) == edit_count]
        if seed is not None:
            match = [r for r in match if int(r.get("seed", 0)) == seed]
        if not match:
            print(f"WARNING: no row found for {spec}, skipping", flush=True)
            continue
        if seed is None and len(match) > 1:
            seeds = sorted({r.get("seed", "") for r in match})
            raise SystemExit(
                f"--select {spec!r} is ambiguous: matched {len(match)} rows "
                f"with seeds {seeds}. Use policy:stop_grad:seed:edit_count."
            )
        selected.append(match[0])
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=Path, required=True)
    p.add_argument("--select", type=str, required=True,
                    help="comma-separated policy:stop_grad:edit_count for single-seed CSVs, "
                         "or policy:stop_grad:seed:edit_count for multi-seed CSVs, "
                         "e.g. mcmc:0:1:4")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-wt", action=argparse.BooleanOptionalAction, default=True,
                    help="Include the combined.csv edit_count=0 WT row for reference. "
                         "Default: true. Parallel wrappers can disable this for all "
                         "but one job to avoid repeated WT refolds.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.combined_csv)))
    selected = parse_select(args.select, rows)
    wt_row = [r for r in rows if int(r["edit_count"]) == 0]
    if args.include_wt and wt_row and not any(int(r["edit_count"]) == 0 for r in selected):
        selected = [wt_row[0]] + selected  # always include WT for comparison
    print(f"predicting real structures for {len(selected)} candidates "
          f"(including WT reference): {[(r['policy'], r['stop_grad'], r['edit_count']) for r in selected]}", flush=True)

    model, binder_seq, target_seq = load_structure()
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(model)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    rmsd_loss = BinderPoseRMSD(reference_binder_ca, reference_target_ca, rmsd_tolerance=0.0)

    # MUST be jitted at real complex sizes -- see module docstring. eqx.filter_jit
    # because opendde_forward_from_trunk closes over the eqx.Module `opendde.model`
    # and Features pytree, not just plain arrays (same reasoning and pattern as
    # examples/vhh72_opendde_structure_prediction.py).
    _forward_jit = eqx.filter_jit(
        functools.partial(
            opendde_forward_from_trunk,
            n_step=N_DIFFUSION_STEPS,
            dense_atom_to_atom37=opendde.dense_atom_to_atom37,
            pae_bin_params=opendde.pae_bin_params,
            plddt_bin_params=opendde.plddt_bin_params,
        )
    )

    key = jax.random.key(args.seed)
    results = []
    for row in selected:
        seed_part = f"_seed{row['seed']}" if "seed" in row and row["seed"] != "" else ""
        tag = f"{row['policy']}_stopgrad{row['stop_grad']}{seed_part}_edit{row['edit_count']}"
        print(f"\n=== {tag} ===", flush=True)
        cand_one_hot = seq_to_one_hot(row["sequence"])

        key, geom_key, diff_key = jax.random.split(key, 3)
        feat = set_binder_sequence(cand_one_hot, features, geom_key)
        s_inputs, s, z = opendde.model.get_pairformer_output(feat, RECYCLING_STEPS)
        output = _forward_jit(opendde.model, feat, s_inputs, s, z, diff_key)

        rmsd_val, rmsd_aux = rmsd_loss(cand_one_hot, output, key=key)
        plddt_mean = float(np.mean(np.asarray(output.plddt)))
        print(f"  real coordinate-based binder_pose_rmsd: {float(rmsd_aux['binder_pose_rmsd']):.2f}A "
              f"(target-aligned, real Kabsch)", flush=True)
        print(f"  mean pLDDT: {plddt_mean:.3f}", flush=True)

        structure = output.to_structure()
        cif_path = args.output_dir / f"{tag}.cif"
        doc = structure.make_mmcif_document()
        doc.write_file(str(cif_path))
        print(f"  wrote {cif_path}", flush=True)

        results.append({
            "tag": tag, "policy": row["policy"], "stop_grad": row["stop_grad"],
            "edit_count": row["edit_count"], "real_binder_pose_rmsd": float(rmsd_aux["binder_pose_rmsd"]),
            "mean_plddt": plddt_mean, "cif": str(cif_path),
        })

    summary_path = args.output_dir / "real_structure_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "policy", "stop_grad", "edit_count",
                                                "real_binder_pose_rmsd", "mean_plddt", "cif"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote summary: {summary_path}", flush=True)
    print("\nDone. Open the .cif files directly (e.g. PyMOL/ChimeraX) to inspect the pose, "
          "or compare real_binder_pose_rmsd across rows -- this is the real, coordinate-"
          "based, target-aligned Kabsch RMSD, not the cheap distogram-drift proxy used "
          "during search.", flush=True)


if __name__ == "__main__":
    main()
