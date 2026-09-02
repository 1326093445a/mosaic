"""Single-candidate smoke test: does one OpenDDE FULL-path forward+backward
(real gradient, not just forward inference) fit in GPU memory at P17's real
complex size (123 aa binder + 184 aa target = 307 residues), with a loss that
carries real confidence-metric awareness (IPTMLoss, BinderTargetPAE,
TargetBinderPAE, pTMEnergy) alongside the existing contact/pose/naturalness/
edit-budget terms -- i.e. the loss the discrete edit-budgeted search would
need if it were confidence-aware, not the cheap distogram-only one it
actually uses today.

This deliberately answers a narrower, more useful question than "does full
OpenDDE OOM": the previously-documented 234GiB figure
(docs/guidance_alphaseq_testing_notes.md section 13.3) was an eager-execution
bug, already fixed by JIT-wrapping -- after the fix, a single FORWARD-only
pass needed ~45-50GB (confirmed on VHH72's similarly-sized complex, expected
to fit an H200's 141GB). Nobody has actually measured a single FORWARD+
BACKWARD (gradient) pass, which is what the discrete search's
`_topb_unseen_feasible_mutations`/`batched_eval` machinery actually needs.
That's the real open number this script measures -- no batching, one
candidate, so it isolates the gradient-vs-forward-only cost question from
the separate batch-width question.

Loading is deliberately a separate step from the differentiable call (build
once, call twice: first call pays JIT compile cost, second call gives a
clean execute-only timing/memory read) so the two costs don't get conflated
when reading the output.

Usage (on a real GPU, e.g. an H200 on the cluster -- this will not run
usefully on a small/CPU device):
    .venv/bin/python examples/p17_opendde_full_gradient_smoke_test.py
"""
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

# Same CDR/hotspot definitions as examples/p17_hallucination_search.py --
# real loss shape, not a stripped-down synthetic one.
CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

WEIGHT_OPENDDE_CONTACT = 0.5
WEIGHT_ABLANG2 = 0.10
WEIGHT_EDIT_BUDGET = 5.0
# Real-confidence weights, matching the "default mosaic" template
# (examples/protenij.py) rather than inventing new numbers.
WEIGHT_IPTM = 0.025
WEIGHT_INTERFACE_PAE = 0.05
WEIGHT_PTM_ENERGY = 0.025
CONTACT_DISTANCE = 8.0
CLIP_GRADIENT_NORM = 1.0
N_DIFFUSION_STEPS = 8
RECYCLING_STEPS = 4
EDIT_BUDGET = 5


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


def load_models():
    """Everything expensive and one-time: model weights, features, JIT setup
    inputs. Kept separate from the actual differentiable call below so a
    repeat run (e.g. in a notebook) doesn't have to pay this twice, and so
    the printed timings below aren't muddied by weight-loading I/O."""
    print("[load] reading P17_JN1.pdb...", flush=True)
    binder_seq, target_seq = load_structure()
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(binder_seq, target_seq)
    print(f"[load] binder {len(binder_seq)} aa, target {len(target_seq)} aa, "
          f"complex {len(binder_seq) + len(target_seq)} residues", flush=True)

    print("[load] constructing OpenDDEModelAbag() (full path)...", flush=True)
    t0 = time.time()
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])
    print(f"[load]   done ({time.time() - t0:.1f}s)", flush=True)

    print("[load] loading AbLang2...", flush=True)
    t0 = time.time()
    ablang2_model, ablang2_tokenizer = load_ablang2()
    print(f"[load]   done ({time.time() - t0:.1f}s)", flush=True)

    return {
        "binder_seq": binder_seq,
        "target_seq": target_seq,
        "reference_binder_ca": reference_binder_ca,
        "reference_target_ca": reference_target_ca,
        "opendde": opendde,
        "features": features,
        "ablang2_model": ablang2_model,
        "ablang2_tokenizer": ablang2_tokenizer,
    }


def build_confidence_aware_full_loss(state):
    """Same shape as p17_hallucination_search.py's --opendde-path full
    branch, but with real IPTMLoss/BinderTargetPAE/TargetBinderPAE/
    pTMEnergy terms added at examples/protenij.py's default weights --
    the loss the discrete search would need if it were made
    confidence-aware, not a synthetic test loss."""
    binder_seq = state["binder_seq"]
    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    epitope_idx = np.array(
        sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32
    )

    contact_loss = WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
        paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        epitope_idx=epitope_idx,
    )
    pose_loss = ClippedGradient(
        BinderPoseRMSD(
            reference_binder_ca=state["reference_binder_ca"],
            reference_target_ca=state["reference_target_ca"],
            rmsd_tolerance=0.0,  # unhinged for this smoke test -- real value doesn't matter here
        ),
        CLIP_GRADIENT_NORM,
    )
    confidence_loss = (
        WEIGHT_IPTM * IPTMLoss()
        + WEIGHT_INTERFACE_PAE * BinderTargetPAE()
        + WEIGHT_INTERFACE_PAE * TargetBinderPAE()
        + WEIGHT_PTM_ENERGY * pTMEnergy()
    )

    opendde_loss = ClippedGradient(
        state["opendde"].build_loss(
            loss=contact_loss + pose_loss + confidence_loss,
            features=state["features"],
            recycling_steps=RECYCLING_STEPS,
            sampling_steps=N_DIFFUSION_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )

    ablang2_loss = ClippedGradient(
        Ablang2PseudoLikelihood(
            model=state["ablang2_model"], tokenizer=state["ablang2_tokenizer"],
            heavy_len=len(binder_seq),
            designable_positions=jnp.array(designable_idx, dtype=jnp.int32),
            stop_grad=False,
        ),
        CLIP_GRADIENT_NORM,
    )
    edit_loss = WEIGHT_EDIT_BUDGET * EditBudget.from_residues(
        binder_seq, designable_idx, budget=float(EDIT_BUDGET),
    )

    return opendde_loss + WEIGHT_ABLANG2 * ablang2_loss + edit_loss


def memory_report(label):
    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
    except Exception as e:
        print(f"[{label}] memory_stats() unavailable ({e}); check nvidia-smi manually", flush=True)
        return
    gib = 1024**3
    print(
        f"[{label}] bytes_in_use={stats.get('bytes_in_use', 0) / gib:.2f}GiB  "
        f"peak_bytes_in_use={stats.get('peak_bytes_in_use', 0) / gib:.2f}GiB  "
        f"bytes_limit={stats.get('bytes_limit', 0) / gib:.2f}GiB",
        flush=True,
    )


def main():
    state = load_models()
    loss = build_confidence_aware_full_loss(state)
    x0 = seq_to_one_hot(state["binder_seq"])

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))

    memory_report("before first (compiling) call")
    key = jax.random.key(0)

    print("\n[1/2] first call -- includes JIT compile time, not a clean timing read...", flush=True)
    t0 = time.time()
    (value, aux), grad = grad_fn(x0, key=key)
    jax.block_until_ready(grad)
    print(f"  wall time (compile + execute): {time.time() - t0:.1f}s", flush=True)
    print(f"  loss={float(value):.4f}", flush=True)
    for k in ("target_contact", "binder_pose_rmsd", "iptm", "bt_pae", "tb_pae", "ptm_energy", "ablang2_ppl"):
        if k in aux:
            print(f"  {k}={float(aux[k]):.4f}", flush=True)
    memory_report("after first call")

    print("\n[2/2] second call -- post-compile, clean execute-only timing...", flush=True)
    key2 = jax.random.key(1)
    t0 = time.time()
    (value2, aux2), grad2 = grad_fn(x0, key=key2)
    jax.block_until_ready(grad2)
    print(f"  wall time (execute only): {time.time() - t0:.1f}s", flush=True)
    memory_report("after second call")

    print("\ndone. If this completed without OOM, a single confidence-aware "
          "full-OpenDDE gradient call fits on this device at P17's real "
          "complex size -- next question becomes batch width, not single-"
          "candidate feasibility. If it OOM'd, the peak_bytes_in_use above "
          "(from whichever call got furthest) is the real number to reason "
          "about gradient checkpointing against.", flush=True)


if __name__ == "__main__":
    main()
