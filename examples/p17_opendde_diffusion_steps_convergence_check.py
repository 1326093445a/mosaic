"""Single-candidate, single-diffusion-step-count worker: scores ONE fixed
binder sequence through OpenDDE's full confidence-aware path (same loss
shape as p17_hallucination_mcmc_with_full_opendde_rescoring.py's
build_rescoring_loss) at a caller-specified `--diffusion-steps`, to check
whether the rescoring run's suspiciously-bad numbers (ipTM 0.17-0.21,
binder_pose_rmsd 19-51A, vs. the one validated real check in this repo's
history -- docs/guidance_alphaseq_testing_notes.md section 13.3's raw-torch
OpenDDE run at 200 diffusion steps: ipTM 0.87-0.93, RMSD ~6A) are a real
structural finding or just an artifact of N_DIFFUSION_STEPS=8 being too few
for the diffusion sampler to converge.

Meant to be launched once per diffusion-step-count value, one process per
GPU, by run_p17_opendde_diffusion_steps_convergence_check.sh (which also
runs the dispatcher, examples/p17_opendde_diffusion_steps_convergence_dispatch.py) --
not normally invoked directly.

Usage:
    .venv/bin/python examples/p17_opendde_diffusion_steps_convergence_check.py \\
        --sequence <123-aa binder sequence> \\
        --diffusion-steps 64 \\
        --output results/convergence/steps_64.csv
"""
import argparse
import csv
import time
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import (
    BinderPoseRMSD,
    BinderTargetContact,
    BinderTargetPAE,
    IPTMLoss,
    TargetBinderPAE,
    pTMEnergy,
)
from mosaic.losses.transformations import ClippedGradient
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.optimizers import _ranking_leaf
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"

CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

WEIGHT_OPENDDE_CONTACT = 0.5
WEIGHT_IPTM = 0.025
WEIGHT_INTERFACE_PAE = 0.05
WEIGHT_PTM_ENERGY = 0.025
CONTACT_DISTANCE = 8.0
CLIP_GRADIENT_NORM = 1.0
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
    p.add_argument("--sequence", type=str, required=True,
                    help="123-aa binder sequence to score (from an existing "
                         "rescoring CSV's `sequence` column)")
    p.add_argument("--diffusion-steps", type=int, required=True)
    p.add_argument("--recycling-steps", type=int, default=OPENDDE_RECYCLING_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== convergence check: diffusion_steps={args.diffusion_steps} "
          f"recycling_steps={args.recycling_steps} ===", flush=True)

    model, binder_seq, target_seq = load_structure()
    if len(args.sequence) != len(binder_seq):
        raise ValueError(
            f"--sequence length {len(args.sequence)} != real P17 binder length "
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

    contact_loss = WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
        paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        epitope_idx=epitope_idx,
    )
    pose_loss = ClippedGradient(
        BinderPoseRMSD(
            reference_binder_ca=reference_binder_ca,
            reference_target_ca=reference_target_ca,
            rmsd_tolerance=0.0,
        ),
        CLIP_GRADIENT_NORM,
    )
    confidence_loss = (
        WEIGHT_IPTM * IPTMLoss()
        + WEIGHT_INTERFACE_PAE * BinderTargetPAE()
        + WEIGHT_INTERFACE_PAE * TargetBinderPAE()
        + WEIGHT_PTM_ENERGY * pTMEnergy()
    )
    loss = ClippedGradient(
        opendde.build_loss(
            loss=contact_loss + pose_loss + confidence_loss,
            features=features,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.diffusion_steps,
        ),
        CLIP_GRADIENT_NORM,
    )
    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))

    x = seq_to_one_hot(args.sequence)
    key = jax.random.key(args.seed)

    print(f"scoring at diffusion_steps={args.diffusion_steps}...", flush=True)
    t0 = time.time()
    (value, aux), grad = grad_fn(x, key=key)
    jax.block_until_ready(grad)
    wall = time.time() - t0
    print(f"done ({wall:.1f}s), value={float(value):.4f}", flush=True)

    # AUX_KEY_NAMES maps our snake_case result name -> the loss class's own
    # actual aux dict key (pTMEnergy.__call__ literally returns
    # {"pTMEnergy": E}, camelCase, not "ptm_energy" -- a real mismatch that
    # silently NaN'd this metric in every earlier run).
    AUX_KEY_NAMES = {
        "target_contact": "target_contact", "binder_pose_rmsd": "binder_pose_rmsd",
        "iptm": "iptm", "bt_pae": "bt_pae", "tb_pae": "tb_pae", "ptm_energy": "pTMEnergy",
    }
    metrics = {
        name: _ranking_leaf(aux, aux_key)
        for name, aux_key in AUX_KEY_NAMES.items()
    }
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}={float(v):.4f}", flush=True)

    def _f(v):
        return float(v) if v is not None else float("nan")

    row = {
        "diffusion_steps": args.diffusion_steps,
        "recycling_steps": args.recycling_steps,
        "wall_time_s": wall,
        "value": float(value),
        "target_contact": _f(metrics["target_contact"]),
        "binder_pose_rmsd": _f(metrics["binder_pose_rmsd"]),
        "iptm": _f(metrics["iptm"]),
        "bt_pae": _f(metrics["bt_pae"]),
        "tb_pae": _f(metrics["tb_pae"]),
        "ptm_energy": _f(metrics["ptm_energy"]),
    }
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
