"""Bisects which loss term's gradient through OpenDDE's full path actually
causes the ~199GiB blowup seen in
examples/p17_opendde_full_gradient_smoke_test.py.

That script's failure looked, from HLO shape arithmetic alone, like
OuterProductMean (602^2 * 384^2 * 4 bytes = 199.09GiB, matching the crash's
allocation size exactly) -- patched and verified numerically correct
(patches/patch_jopendde_outer_product_mean.py), but re-running with the
patch applied reproduced the BYTE-IDENTICAL crash. That's a real negative
result: the shape match was evidently a coincidence (or at least not the
whole story), not proof of which op is responsible. Guessing further from
HLO shapes without being able to inspect the GPU directly isn't reliable --
this script isolates the actual culprit empirically instead, by running
each loss component's gradient ALONE through the same real model/features,
one at a time, each in its own subprocess (so one component's OOM doesn't
leave the GPU allocator in a bad state for the next).

Usage:
    # runs all components, reporting pass/fail per component
    .venv/bin/python examples/p17_opendde_full_gradient_bisect.py

    # runs just one component directly (used internally by the driver above,
    # also useful to re-run a single suspect component with more logging)
    .venv/bin/python examples/p17_opendde_full_gradient_bisect.py --component confidence
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.structure_prediction import (
    BinderPoseRMSD,
    BinderTargetContact,
    BinderTargetPAE,
    IPTMLoss,
    TargetBinderPAE,
    pTMEnergy,
)
from mosaic.losses.transformations import ClippedGradient, EditBudget
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"

CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

CONTACT_DISTANCE = 8.0
CLIP_GRADIENT_NORM = 1.0
N_DIFFUSION_STEPS = 8
RECYCLING_STEPS = 4
EDIT_BUDGET = 5

COMPONENTS = ["contact", "pose", "confidence", "ablang2"]


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def load_structure():
    st = gemmi.read_structure(str(COMPLEX_PDB))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model[BINDER_CHAIN]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model[TARGET_CHAIN]]).upper()
    return binder_seq, target_seq


def reference_binder_target_ca(binder_seq, target_seq):
    st = gemmi.read_structure(str(COMPLEX_PDB))
    st.setup_entities()
    model = st[0]

    def ca_coords(chain):
        coords = []
        for res in chain:
            for a in res:
                if a.name == "CA":
                    coords.append([a.pos.x, a.pos.y, a.pos.z])
                    break
        return np.array(coords, dtype=np.float32)

    return ca_coords(model[BINDER_CHAIN]), ca_coords(model[TARGET_CHAIN])


def build_component_loss(component: str, *, binder_seq, opendde, features,
                          reference_binder_ca, reference_target_ca,
                          ablang2_model, ablang2_tokenizer):
    """Each component wrapped so it's the ONLY thing OpenDDE's full path
    needs to differentiate through -- isolates which one is responsible."""
    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    epitope_idx = np.array(
        sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32
    )

    if component == "contact":
        inner = BinderTargetContact(
            paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
            epitope_idx=epitope_idx,
        )
    elif component == "pose":
        inner = BinderPoseRMSD(
            reference_binder_ca=reference_binder_ca,
            reference_target_ca=reference_target_ca,
            rmsd_tolerance=0.0,
        )
    elif component == "confidence":
        inner = (
            0.025 * IPTMLoss()
            + 0.05 * BinderTargetPAE()
            + 0.05 * TargetBinderPAE()
            + 0.025 * pTMEnergy()
        )
    elif component == "ablang2":
        # AbLang2 doesn't touch OpenDDE at all -- included as a negative
        # control (should trivially pass; if THIS OOMs, the problem isn't
        # OpenDDE-related at all).
        return ClippedGradient(
            Ablang2PseudoLikelihood(
                model=ablang2_model, tokenizer=ablang2_tokenizer,
                heavy_len=len(binder_seq),
                designable_positions=jnp.array(designable_idx, dtype=jnp.int32),
                stop_grad=False,
            ),
            CLIP_GRADIENT_NORM,
        )
    else:
        raise ValueError(component)

    return ClippedGradient(
        opendde.build_loss(
            loss=inner,
            features=features,
            recycling_steps=RECYCLING_STEPS,
            sampling_steps=N_DIFFUSION_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )


def run_one(component: str):
    print(f"=== bisect: component={component!r} ===", flush=True)
    binder_seq, target_seq = load_structure()
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(binder_seq, target_seq)

    print("loading OpenDDE (full path)...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    ablang2_model = ablang2_tokenizer = None
    if component == "ablang2":
        print("loading AbLang2...", flush=True)
        ablang2_model, ablang2_tokenizer = load_ablang2()

    loss = build_component_loss(
        component, binder_seq=binder_seq, opendde=opendde, features=features,
        reference_binder_ca=reference_binder_ca, reference_target_ca=reference_target_ca,
        ablang2_model=ablang2_model, ablang2_tokenizer=ablang2_tokenizer,
    )
    x0 = seq_to_one_hot(binder_seq)
    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))

    print(f"[{component}] running forward+backward...", flush=True)
    t0 = time.time()
    (value, aux), grad = grad_fn(x0, key=jax.random.key(0))
    jax.block_until_ready(grad)
    print(f"[{component}] SUCCESS  wall_time={time.time() - t0:.1f}s  loss={float(value):.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component", choices=COMPONENTS, default=None,
                    help="run just this one component directly (no subprocess); "
                         "omit to run all components, each in its own subprocess")
    args = p.parse_args()

    if args.component is not None:
        run_one(args.component)
        return

    results = {}
    for component in COMPONENTS:
        print(f"\n{'=' * 70}\nlaunching subprocess for component={component!r}\n{'=' * 70}", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--component", component],
            capture_output=False,
        )
        results[component] = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"

    print(f"\n{'=' * 70}\nBISECTION SUMMARY\n{'=' * 70}")
    for component, result in results.items():
        print(f"  {component:12s}  {result}")
    print(
        "\nWhichever component(s) FAILED are the actual source(s) of the "
        "blowup -- next step is to look at exactly what that component's "
        "loss class reads off `output` (pae/pae_logits vs coordinates vs "
        "distogram_logits) to find the real op, rather than guessing from "
        "HLO shapes again."
    )


if __name__ == "__main__":
    main()
