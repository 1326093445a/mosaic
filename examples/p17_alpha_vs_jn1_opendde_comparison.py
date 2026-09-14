"""Real, forward-only OpenDDE structure prediction, P17-vs-Alpha (binding)
vs. P17-vs-JN.1 (non-binding), under the SAME settings for both: no
template (template_chain left unset, matching this repo's existing
default), no MSA (use_msa=False), recycling_steps=3 -- the specific
apples-to-apples internal reference the user asked for directly: "you can
run the real structure prediction, between P17 and Alpha and P17 and
JN.1, no template, no msa just 3 recycle you can see binding group alpha
can have significant difference between".

Both complexes share the identical 123-aa P17 binder sequence (chain
"B" in both PDBs) -- P17_Alpha.pdb's binder and P17_JN1.pdb's binder are
byte-identical; only the target differs (Alpha spike RBD, chain "A", 195
real residues vs. JN.1 spike RBD, chain "T", 184 real residues). Since
target numbering differs between the two complexes, the known JN.1
hotspot/epitope indices (115,117,146,148,150) are NOT applicable to
Alpha's numbering -- BinderTargetContact is run WITHOUT an epitope
restriction (paratope-to-any-target-residue) for both complexes so the
two are comparable; only the epitope-restricted read is skipped for
Alpha.

Runs each complex across --num-seeds seeds (default 3) to check the gap
is a real effect, not a single lucky/unlucky diffusion trajectory (same
motivation as p17_opendde_diffusion_steps_convergence_dispatch.py's
seed-sweep mode).

Usage:
    .venv/bin/python examples/p17_alpha_vs_jn1_opendde_comparison.py \\
        --diffusion-steps 64 --recycling-steps 3 --num-seeds 3 \\
        --output results/p17_alpha_vs_jn1/comparison.csv
"""
import argparse
import csv
import functools
import time
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.opendde import opendde_forward_from_trunk, set_binder_sequence
from mosaic.losses.structure_prediction import (
    BinderPoseRMSD,
    BinderTargetContact,
    BinderTargetPAE,
    IPTMLoss,
    TargetBinderPAE,
    pTMEnergy,
)
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent

CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
JN1_HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

CONTACT_DISTANCE = 8.0

COMPLEXES = [
    {
        "name": "P17_vs_Alpha",
        "label": "Alpha (binding)",
        "pdb": REPO_ROOT / "P17_Alpha.pdb",
        "binder_chain": "B",
        "target_chain": "A",
        "epitope_idx": None,  # JN.1 hotspot numbering doesn't map onto Alpha's target
    },
    {
        "name": "P17_vs_JN1",
        "label": "JN.1 (non-binding)",
        "pdb": REPO_ROOT / "P17_JN1.pdb",
        "binder_chain": "B",
        "target_chain": "T",
        "epitope_idx": np.array(
            sorted(i - 1 for i in JN1_HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32
        ),
    },
]


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def load_structure(pdb_path: Path, binder_chain: str, target_chain: str):
    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model[binder_chain]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model[target_chain]]).upper()
    return model, binder_seq, target_seq


def reference_binder_target_ca(model, binder_chain: str, target_chain: str):
    def ca_coords(chain):
        coords = []
        for res in chain:
            for a in res:
                if a.name == "CA":
                    coords.append([a.pos.x, a.pos.y, a.pos.z])
                    break
        return np.array(coords, dtype=np.float32)

    return ca_coords(model[binder_chain]), ca_coords(model[target_chain])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion-steps", type=int, default=64)
    p.add_argument("--recycling-steps", type=int, default=3,
                    help="User-specified value for this apples-to-apples check "
                         "(distinct from the search's own default of 4).")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== P17-vs-Alpha (binding) vs. P17-vs-JN.1 (non-binding) OpenDDE "
          f"comparison: no template, no MSA, recycling_steps={args.recycling_steps}, "
          f"diffusion_steps={args.diffusion_steps}, seeds="
          f"{list(range(args.seed_offset, args.seed_offset + args.num_seeds))} ===", flush=True)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()

    _forward_jit = eqx.filter_jit(
        functools.partial(
            opendde_forward_from_trunk,
            n_step=args.diffusion_steps,
            dense_atom_to_atom37=opendde.dense_atom_to_atom37,
            pae_bin_params=opendde.pae_bin_params,
            plddt_bin_params=opendde.plddt_bin_params,
        )
    )

    rows = []
    for cx in COMPLEXES:
        print(f"\n--- {cx['name']} ({cx['label']}) ---", flush=True)
        model, binder_seq, target_seq = load_structure(
            cx["pdb"], cx["binder_chain"], cx["target_chain"]
        )
        reference_binder_ca, reference_target_ca = reference_binder_target_ca(
            model, cx["binder_chain"], cx["target_chain"]
        )
        print(f"  binder ({cx['binder_chain']}): {len(binder_seq)} aa; "
              f"target ({cx['target_chain']}): {len(target_seq)} aa", flush=True)

        designable_mask = np.zeros(len(binder_seq), dtype=bool)
        for i in CDR_RESIDUE_INDICES_1IDX:
            designable_mask[i - 1] = True
        designable_idx = np.nonzero(designable_mask)[0]

        features, _ = opendde.binder_features(
            len(binder_seq), [TargetChain(target_seq, use_msa=False)]
        )  # template_chain left unset -> no template

        contact_loss = BinderTargetContact(
            paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
            epitope_idx=cx["epitope_idx"],
        )
        rmsd_loss = BinderPoseRMSD(reference_binder_ca, reference_target_ca, rmsd_tolerance=0.0)
        iptm_loss = IPTMLoss()
        bt_pae_loss = BinderTargetPAE()
        tb_pae_loss = TargetBinderPAE()
        ptm_energy_loss = pTMEnergy()

        x = seq_to_one_hot(binder_seq)

        for seed in range(args.seed_offset, args.seed_offset + args.num_seeds):
            print(f"  seed={seed}: scoring...", flush=True)
            t0 = time.time()
            key = jax.random.key(seed)
            key, geom_key, diff_key = jax.random.split(key, 3)
            feat = set_binder_sequence(x, features, geom_key)
            s_inputs, s, z = opendde.model.get_pairformer_output(feat, args.recycling_steps)
            output = _forward_jit(opendde.model, feat, s_inputs, s, z, diff_key)

            _, contact_aux = contact_loss(x, output, key=key)
            _, rmsd_aux = rmsd_loss(x, output, key=key)
            _, iptm_aux = iptm_loss(x, output, key=key)
            _, bt_pae_aux = bt_pae_loss(x, output, key=key)
            _, tb_pae_aux = tb_pae_loss(x, output, key=key)
            _, ptm_energy_aux = ptm_energy_loss(x, output, key=key)
            jax.block_until_ready(rmsd_aux)
            wall = time.time() - t0

            row = {
                "complex": cx["name"],
                "label": cx["label"],
                "seed": seed,
                "diffusion_steps": args.diffusion_steps,
                "recycling_steps": args.recycling_steps,
                "wall_time_s": wall,
                "target_contact": float(contact_aux["target_contact"]),
                "binder_pose_rmsd": float(rmsd_aux["binder_pose_rmsd"]),
                "iptm": float(iptm_aux["iptm"]),
                "bt_pae": float(bt_pae_aux["bt_pae"]),
                "tb_pae": float(tb_pae_aux["tb_pae"]),
                "ptm_energy": float(ptm_energy_aux["pTMEnergy"]),
                "mean_plddt": float(np.mean(np.asarray(output.plddt))),
            }
            print(f"    ({wall:.1f}s) iptm={row['iptm']:.4f} "
                  f"binder_pose_rmsd={row['binder_pose_rmsd']:.2f}A "
                  f"bt_pae={row['bt_pae']:.2f} tb_pae={row['tb_pae']:.2f} "
                  f"target_contact={row['target_contact']:.3f} "
                  f"mean_plddt={row['mean_plddt']:.3f}", flush=True)
            rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}", flush=True)

    print(f"\n{'complex':>14}  {'seed':>4}  {'iptm':>7}  {'rmsd':>7}  "
          f"{'bt_pae':>7}  {'tb_pae':>7}  {'contact':>8}  {'plddt':>6}", flush=True)
    for r in rows:
        print(f"{r['complex']:>14}  {r['seed']:>4}  {r['iptm']:>7.4f}  "
              f"{r['binder_pose_rmsd']:>7.2f}  {r['bt_pae']:>7.2f}  "
              f"{r['tb_pae']:>7.2f}  {r['target_contact']:>8.3f}  "
              f"{r['mean_plddt']:>6.3f}", flush=True)

    print("\n=== per-complex mean across seeds ===", flush=True)
    for cx in COMPLEXES:
        sub = [r for r in rows if r["complex"] == cx["name"]]
        if not sub:
            continue
        mean = {k: np.mean([r[k] for r in sub]) for k in
                 ("iptm", "binder_pose_rmsd", "bt_pae", "tb_pae", "target_contact", "mean_plddt")}
        print(f"  {cx['name']} ({cx['label']}): iptm={mean['iptm']:.4f} "
              f"binder_pose_rmsd={mean['binder_pose_rmsd']:.2f}A "
              f"bt_pae={mean['bt_pae']:.2f} tb_pae={mean['tb_pae']:.2f} "
              f"target_contact={mean['target_contact']:.3f} "
              f"mean_plddt={mean['mean_plddt']:.3f}", flush=True)

    print("\nThis is the internal, apples-to-apples reference: whatever gap shows up "
          "here between Alpha (binding) and JN.1 (non-binding) under IDENTICAL settings "
          "(no template, no MSA, recycling_steps=3, same diffusion_steps, same binder "
          "sequence) is what a genuine binding-classification signal looks like under "
          "this exact pipeline -- not the earlier VHH72-derived 0.87-0.93 ipTM reference, "
          "which came from a different complex entirely.", flush=True)


if __name__ == "__main__":
    main()
