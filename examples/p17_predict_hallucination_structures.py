"""Real (expensive) OpenDDE structure prediction for a small, hand-picked
set of P17 hallucination-search candidates -- writes real CIFs, and
computes the real coordinate-based BinderPoseRMSD (target-aligned Kabsch
RMSD against the real P17_JN1.pdb reference) plus real PAE-based ipSAE
(BinderTargetIPSAE/TargetBinderIPSAE/IPSAE_min), answering directly: after
CDR mutation, does this candidate's real structural read look like the
binding class (P17-vs-Alpha) or the non-binding class (P17-vs-JN.1) the
in-house OpenDDE check found a large ipSAE/pose gap between -- not "is
this a high-affinity binder" (ipSAE has no published KD/SPR/ITC
correlation, per the ipSAE paper itself and docs/guidance_alphaseq_testing_notes.md
§4; affinity ranking is out of scope here, left to a downstream assay).
Mirrors examples/vhh72_predict_hallucination_structures.py -- same
division of labor (cheap distogram-only path during search, real
diffusion coordinate + confidence-head path here, once per requested
candidate, not thousands of times) and the same eqx.filter_jit requirement
at full complex size (see that script's docstring for the exact OOM this
avoids). ipSAE is deliberately NOT computed in-loop during search --
`interaction_prediction_score`'s hard `pae < pae_cutoff` mask plus
max/min reductions make it non-smooth/unsuitable for gradient guidance
(see docs/guidance_alphaseq_testing_notes.md §4/§5.3); it belongs here,
post-refold, as a selection/success criterion instead.

This check matters more for P17 than it did for VHH72: VHH72 had real
AlphaSeq data to validate designs against; P17 has none (confirmed: no
P17-related entries anywhere in the AlphaSeq dataset). This real structural
RMSD + ipSAE check is the primary validation available here, not a
secondary sanity check.

Usage:
    .venv/bin/python examples/p17_predict_hallucination_structures.py \\
        --combined-csv results/p17_sweep/combined.csv \\
        --select 0:5,1:5,2:4 \\
        --output-dir results/p17_sweep/structures
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
from mosaic.losses.structure_prediction import (
    BinderPoseRMSD,
    BinderTargetIPSAE,
    IPSAE_min,
    TargetBinderIPSAE,
)
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"
N_DIFFUSION_STEPS = 8  # matches vhh72_opendde_structure_prediction.py's real-prediction default
RECYCLING_STEPS = 4
IPSAE_PAE_CUTOFF = 10.0  # Dunbrack ipSAE paper default


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def load_structure():
    st = gemmi.read_structure(str(COMPLEX_PDB))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model[BINDER_CHAIN]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model[TARGET_CHAIN]]).upper()
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
    return ca_coords(model[BINDER_CHAIN]), ca_coords(model[TARGET_CHAIN])


def parse_select(select_str, rows):
    """--select entries: seed:edit_count, e.g. 0:5,1:5,2:4.

    Unlike VHH72's combined.csv, P17's sweep has no policy/stop_grad axis
    (one fixed, validated config -- see p17_hallucination_search.py's
    defaults), so selection only needs seed + edit_count.
    """
    wanted = []
    for spec in select_str.split(","):
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"bad --select entry {spec!r}; expected seed:edit_count")
        seed, edit_count = parts
        wanted.append((int(seed), int(edit_count), spec))
    selected = []
    for seed, edit_count, spec in wanted:
        match = [r for r in rows if int(r["seed"]) == seed and int(r["edit_count"]) == edit_count]
        if not match:
            print(f"WARNING: no row found for {spec}, skipping", flush=True)
            continue
        selected.append(match[0])
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", type=Path, required=True)
    p.add_argument("--select", type=str, required=True,
                    help="comma-separated seed:edit_count, e.g. 0:5,1:5,2:4")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-wt", action=argparse.BooleanOptionalAction, default=True,
                    help="Include the combined.csv edit_count=0 WT/JN.1-baseline row "
                         "for reference. Default: true.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.combined_csv)))
    selected = parse_select(args.select, rows)
    wt_row = [r for r in rows if int(r["edit_count"]) == 0]
    if args.include_wt and wt_row and not any(int(r["edit_count"]) == 0 for r in selected):
        selected = [wt_row[0]] + selected  # always include the JN.1-baseline for comparison
    print(f"predicting real structures for {len(selected)} candidates "
          f"(including WT/JN.1 reference): "
          f"{[(r['seed'], r['edit_count']) for r in selected]}", flush=True)

    model, binder_seq, target_seq = load_structure()
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(model)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    rmsd_loss = BinderPoseRMSD(reference_binder_ca, reference_target_ca, rmsd_tolerance=0.0)
    bt_ipsae_loss = BinderTargetIPSAE(pae_cutoff=IPSAE_PAE_CUTOFF)
    tb_ipsae_loss = TargetBinderIPSAE(pae_cutoff=IPSAE_PAE_CUTOFF)
    ipsae_min_loss = IPSAE_min(pae_cutoff=IPSAE_PAE_CUTOFF)

    # MUST be jitted at real complex sizes -- see module docstring and
    # vhh72_predict_hallucination_structures.py's docstring for the exact
    # 234GiB OOM this avoids at eager call.
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
        tag = f"seed{row['seed']}_edit{row['edit_count']}"
        print(f"\n=== {tag} ===", flush=True)
        cand_one_hot = seq_to_one_hot(row["sequence"])

        key, geom_key, diff_key = jax.random.split(key, 3)
        feat = set_binder_sequence(cand_one_hot, features, geom_key)
        s_inputs, s, z = opendde.model.get_pairformer_output(feat, RECYCLING_STEPS)
        output = _forward_jit(opendde.model, feat, s_inputs, s, z, diff_key)

        rmsd_val, rmsd_aux = rmsd_loss(cand_one_hot, output, key=key)
        _, bt_aux = bt_ipsae_loss(cand_one_hot, output, key=key)
        _, tb_aux = tb_ipsae_loss(cand_one_hot, output, key=key)
        _, ipsae_min_aux = ipsae_min_loss(cand_one_hot, output, key=key)
        plddt_mean = float(np.mean(np.asarray(output.plddt)))
        print(f"  real coordinate-based binder_pose_rmsd: {float(rmsd_aux['binder_pose_rmsd']):.2f}A "
              f"(target-aligned, real Kabsch, vs. real P17_JN1.pdb reference)", flush=True)
        print(f"  mean pLDDT: {plddt_mean:.3f}", flush=True)
        print(f"  ipSAE (pae_cutoff={IPSAE_PAE_CUTOFF}): "
              f"bt_ipsae={float(bt_aux['bt_ipsae']):.3f} "
              f"tb_ipsae={float(tb_aux['tb_ipsae']):.3f} "
              f"ipsae_min={float(ipsae_min_aux['ipsae_min']):.3f}", flush=True)

        structure = output.to_structure()
        cif_path = args.output_dir / f"{tag}.cif"
        doc = structure.make_mmcif_document()
        doc.write_file(str(cif_path))
        print(f"  wrote {cif_path}", flush=True)

        results.append({
            "tag": tag, "seed": row["seed"], "edit_count": row["edit_count"],
            "real_binder_pose_rmsd": float(rmsd_aux["binder_pose_rmsd"]),
            "mean_plddt": plddt_mean,
            "bt_ipsae": float(bt_aux["bt_ipsae"]),
            "tb_ipsae": float(tb_aux["tb_ipsae"]),
            "ipsae_min": float(ipsae_min_aux["ipsae_min"]),
            "cif": str(cif_path),
        })

    summary_path = args.output_dir / "real_structure_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "seed", "edit_count", "real_binder_pose_rmsd", "mean_plddt",
            "bt_ipsae", "tb_ipsae", "ipsae_min", "cif",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote summary: {summary_path}", flush=True)
    print("\nDone. Compare real_binder_pose_rmsd and ipsae_min across rows against the "
          "edit_count=0 JN.1-baseline row's own value -- that baseline is the real "
          "reference point (OpenDDE's own prediction error on the true, unmutated "
          "sequence), not 0A/0. A design whose RMSD is close to the baseline's is not "
          "meaningfully drifted; one substantially higher is a real structural warning "
          "sign. For ipsae_min, the success criterion is NOT 'higher is better affinity' "
          "(ipSAE has no published KD/SPR/ITC correlation) -- it's whether the candidate's "
          "ipsae_min/pose sits closer to your in-house P17-vs-Alpha (binding) reference "
          "than to the P17-vs-JN.1 (non-binding) reference: a binding-class classification "
          "check, not an affinity ranking. Open the .cif files directly (e.g. PyMOL/"
          "ChimeraX) to inspect the pose visually.", flush=True)


if __name__ == "__main__":
    main()
