"""Single-candidate, single-setting worker: scores ONE fixed binder
sequence through OpenDDE's full confidence-aware path at caller-specified
--diffusion-steps/--recycling-steps, to check whether the rescoring run's
suspiciously-bad numbers (ipTM 0.17-0.21, binder_pose_rmsd 19-51A, vs. the
one validated real check in this repo's history --
docs/guidance_alphaseq_testing_notes.md section 13.3's raw-torch OpenDDE
run at 200 diffusion steps/10 recycles: ipTM 0.87-0.93, RMSD ~6A) are a
real structural finding or a scoring-settings artifact.

FORWARD-ONLY, deliberately -- earlier versions of this script used
eqx.filter_value_and_grad (matching the actual in-search rescoring
mechanism being validated), but this check doesn't need a gradient at all,
and the gradient's backward pass is what made recycling_steps>4 OOM (~112GiB
needed after XLA's own rematerialization pass, vs. this device's 104.88GiB
-- confirmed by direct log: "Can't reduce memory use below 97.24GiB by
rematerialization; only reduced to 112.15GiB, down from 125.18GiB
originally"). Forward-only, mirroring examples/p17_predict_hallucination_structures.py's
already-validated pattern (opendde_forward_from_trunk + set_binder_sequence
directly, no eqx.filter_value_and_grad, each loss class's __call__ invoked
directly on the resulting `output`), avoids that backward-pass memory
inflation entirely and lets recycling_steps go much higher within budget.

Meant to be launched once per (diffusion-steps, recycling-steps, seed,
use-wt) combination, one process per GPU, by
run_p17_opendde_diffusion_steps_convergence_check.sh (which also runs the
dispatcher, examples/p17_opendde_diffusion_steps_convergence_dispatch.py)
-- not normally invoked directly.

Usage:
    .venv/bin/python examples/p17_opendde_diffusion_steps_convergence_check.py \\
        --sequence <123-aa binder sequence> \\
        --diffusion-steps 64 --recycling-steps 10 \\
        --output results/convergence/steps_64_recyc_10.csv
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
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"

CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

CONTACT_DISTANCE = 8.0
OPENDDE_RECYCLING_STEPS = 4


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=str, default=None,
                    help="123-aa binder sequence to score (from an existing "
                         "rescoring CSV's `sequence` column). Ignored if --use-wt.")
    p.add_argument("--use-wt", action="store_true",
                    help="Score the real, unmutated P17 WT binder sequence instead "
                         "of --sequence -- the real baseline to check whether a low "
                         "ipTM/high RMSD reading is about this candidate or about "
                         "the scoring settings (recycling_steps in particular).")
    p.add_argument("--diffusion-steps", type=int, required=True)
    p.add_argument("--recycling-steps", type=int, default=OPENDDE_RECYCLING_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.use_wt and not args.sequence:
        raise SystemExit("--sequence is required unless --use-wt is set")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== convergence check (forward-only): diffusion_steps={args.diffusion_steps} "
          f"recycling_steps={args.recycling_steps} use_wt={args.use_wt} ===", flush=True)

    model, binder_seq, target_seq = load_structure()
    sequence = binder_seq if args.use_wt else args.sequence
    if len(sequence) != len(binder_seq):
        raise ValueError(
            f"--sequence length {len(sequence)} != real P17 binder length "
            f"{len(binder_seq)}; did you pass the right column from the CSV?"
        )
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(model)

    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    epitope_idx = np.array(
        sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32
    )

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    contact_loss = BinderTargetContact(
        paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        epitope_idx=epitope_idx,
    )
    rmsd_loss = BinderPoseRMSD(reference_binder_ca, reference_target_ca, rmsd_tolerance=0.0)
    iptm_loss = IPTMLoss()
    bt_pae_loss = BinderTargetPAE()
    tb_pae_loss = TargetBinderPAE()
    ptm_energy_loss = pTMEnergy()

    # Same pattern as examples/p17_predict_hallucination_structures.py's
    # already-validated forward-only real prediction: MUST be jitted at
    # real complex sizes (see that script's docstring for the eager-call
    # OOM this avoids), no eqx.filter_value_and_grad since there's nothing
    # to differentiate here.
    _forward_jit = eqx.filter_jit(
        functools.partial(
            opendde_forward_from_trunk,
            n_step=args.diffusion_steps,
            dense_atom_to_atom37=opendde.dense_atom_to_atom37,
            pae_bin_params=opendde.pae_bin_params,
            plddt_bin_params=opendde.plddt_bin_params,
        )
    )

    x = seq_to_one_hot(sequence)
    key = jax.random.key(args.seed)

    print(f"scoring (forward-only) at diffusion_steps={args.diffusion_steps} "
          f"recycling_steps={args.recycling_steps}...", flush=True)
    t0 = time.time()
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

    metrics = {
        "target_contact": float(contact_aux["target_contact"]),
        "binder_pose_rmsd": float(rmsd_aux["binder_pose_rmsd"]),
        "iptm": float(iptm_aux["iptm"]),
        "bt_pae": float(bt_pae_aux["bt_pae"]),
        "tb_pae": float(tb_pae_aux["tb_pae"]),
        "ptm_energy": float(ptm_energy_aux["pTMEnergy"]),
        "mean_plddt": float(np.mean(np.asarray(output.plddt))),
    }
    print(f"done ({wall:.1f}s)", flush=True)
    for k, v in metrics.items():
        print(f"  {k}={v:.4f}", flush=True)

    row = {
        "diffusion_steps": args.diffusion_steps,
        "recycling_steps": args.recycling_steps,
        "seed": args.seed,
        "use_wt": int(args.use_wt),
        "wall_time_s": wall,
        **metrics,
    }
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
