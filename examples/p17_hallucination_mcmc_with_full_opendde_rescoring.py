"""Wires real confidence-aware full-OpenDDE rescoring into the ACTUAL P17
discrete MCMC search (edit_budgeted_gradient_mcmc, unmodified), to measure
the real cost of this on one real seed -- not a smoke test, an actual run of
the production search function.

Design, and why it's built this way:
  - `edit_budgeted_gradient_mcmc` (src/mosaic/optimizers.py) is NOT modified.
    It's the validated, production search function every P17 sweep result
    so far has used -- changing its signature/internals to add a rescoring
    hook would be a real risk to already-trusted behavior for a one-seed
    validation run. Instead this script calls it repeatedly in chunks of
    `--rescore-every` steps, using each chunk's returned best_seq as the
    next chunk's starting sequence -- functionally a continued search, just
    interleaved with an external rescoring call between chunks.
  - The search itself still runs on the cheap, budget-aware, already-
    validated distogram-only loss (contact + pose-drift + ablang2 +
    edit-budget), exactly matching examples/p17_hallucination_search.py's
    default config (mcmc, stop_grad=0, --apgm-steps 0 equivalent -- starts
    directly from WT, skipping the continuous-relaxation stage, since this
    is testing the discrete stage's rescoring cost specifically).
  - Between chunks, the CURRENT BEST sequence gets one real, single-
    candidate (unbatched), confidence-aware full-OpenDDE gradient call --
    the exact mechanism verified in
    examples/p17_opendde_full_gradient_smoke_test.py (peak 85.48GiB,
    14.3s/call once JIT-warm on the cluster's real GPU) and confirmed by
    examples/p17_opendde_full_gradient_bisect.py. It's a REscoring step
    only -- its gradient is computed (reusing the exact validated call
    shape) but NOT applied to the sequence; only its value/aux get logged.
    This does not steer the search; it measures what the real signal says
    about candidates the cheap search already found.
  - Known limitation, acceptable for a cost/behavior validation run (not
    claimed to be a drop-in replacement for a single continuous
    `edit_budgeted_gradient_mcmc` call): `seen` (the already-tried-mutation
    set) resets each chunk, so a chunk boundary can waste a few evaluations
    re-proposing an already-rejected mutation. Does not affect correctness,
    only search efficiency near chunk boundaries.

Usage:
    .venv/bin/python examples/p17_hallucination_mcmc_with_full_opendde_rescoring.py \\
        --seed 0 --edit-budget 5 --mcmc-steps 100 --rescore-every 20 \\
        --output results/p17_mcmc_rescoring_seed0.csv
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
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.structure_prediction import (
    BinderPoseDistogramDrift,
    BinderPoseRMSD,
    BinderTargetContact,
    BinderTargetPAE,
    IPTMLoss,
    TargetBinderPAE,
    pTMEnergy,
)
from mosaic.losses.transformations import ClippedGradient, EditBudget
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.optimizers import _ranking_leaf, edit_budgeted_gradient_mcmc
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"

# Same CDR/hotspot definitions as p17_hallucination_search.py.
CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

WEIGHT_OPENDDE_CONTACT = 0.5
WEIGHT_ABLANG2 = 0.10
WEIGHT_EDIT_BUDGET = 5.0
# Real-confidence weights for the rescoring loss, matching the "default
# mosaic" template (examples/protenij.py), same as the smoke test.
WEIGHT_IPTM = 0.025
WEIGHT_INTERFACE_PAE = 0.05
WEIGHT_PTM_ENERGY = 0.025
CONTACT_DISTANCE = 8.0
POSE_DRIFT_MARGIN = 3.0
CLIP_GRADIENT_NORM = 1.0
OPENDDE_RECYCLING_STEPS = 4
N_DIFFUSION_STEPS = 8


def seq_to_one_hot(seq) -> np.ndarray:
    if isinstance(seq, str):
        idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    else:
        idx = np.asarray(seq, dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def one_hot_to_seq(x) -> str:
    idx = np.asarray(x).argmax(-1)
    return "".join(TOKENS[i] for i in idx)


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


def reference_binder_target_ca_distances(model):
    binder_ca, target_ca = reference_binder_target_ca(model)
    diffs = binder_ca[:, None, :] - target_ca[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def measure_wt_distogram_pose_drift(*, opendde, features, reference_distances, binder_seq, key):
    probe = ClippedGradient(
        opendde.build_distogram_only_loss(
            loss=BinderPoseDistogramDrift(reference_distances, tolerance=0.0),
            features=features, recycling_steps=OPENDDE_RECYCLING_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )
    x_wt = seq_to_one_hot(binder_seq)
    _, aux = probe(x_wt, key=key)
    return float(aux["binder_target_distogram_drift"])


def build_cheap_search_loss(*, opendde, features, ablang2_model, ablang2_tokenizer,
                             reference_distances, binder_seq, designable_idx,
                             epitope_idx, edit_budget, pose_tolerance):
    """Exactly p17_hallucination_search.py's default (distogram-path, mcmc,
    stop_grad=0) composite loss -- the search's actual proposal/acceptance
    signal, unchanged."""
    contact_loss = WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
        paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        epitope_idx=epitope_idx,
    )
    pose_loss = ClippedGradient(
        BinderPoseDistogramDrift(reference_distances, tolerance=pose_tolerance),
        CLIP_GRADIENT_NORM,
    )
    opendde_loss = ClippedGradient(
        opendde.build_distogram_only_loss(
            loss=contact_loss + pose_loss,
            features=features,
            recycling_steps=OPENDDE_RECYCLING_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )
    ablang2_loss = ClippedGradient(
        Ablang2PseudoLikelihood(
            model=ablang2_model, tokenizer=ablang2_tokenizer,
            heavy_len=len(binder_seq),
            designable_positions=jnp.array(designable_idx, dtype=jnp.int32),
            stop_grad=False,
        ),
        CLIP_GRADIENT_NORM,
    )
    edit_loss = WEIGHT_EDIT_BUDGET * EditBudget.from_residues(
        binder_seq, designable_idx, budget=float(edit_budget),
    )
    return opendde_loss + WEIGHT_ABLANG2 * ablang2_loss + edit_loss


def build_rescoring_loss(*, opendde, features, reference_binder_ca, reference_target_ca,
                          designable_idx, epitope_idx):
    """Real, full-path, confidence-aware loss -- single candidate, no
    batching, the exact shape verified in
    p17_opendde_full_gradient_smoke_test.py. Used ONLY to score, never to
    steer the search."""
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
    return ClippedGradient(
        opendde.build_loss(
            loss=contact_loss + pose_loss + confidence_loss,
            features=features,
            recycling_steps=OPENDDE_RECYCLING_STEPS,
            sampling_steps=N_DIFFUSION_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edit-budget", type=int, default=5)
    p.add_argument("--mcmc-steps", type=int, default=100)
    p.add_argument("--rescore-every", type=int, default=20,
                    help="chunk size: run this many MCMC steps, then rescore "
                         "the current best with real full-OpenDDE confidence "
                         "terms, repeat until --mcmc-steps is reached")
    p.add_argument("--output", type=Path, required=True,
                    help="CSV path for the rescoring log")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== P17 MCMC + full-OpenDDE rescoring: seed={args.seed} "
          f"edit_budget={args.edit_budget} mcmc_steps={args.mcmc_steps} "
          f"rescore_every={args.rescore_every} ===", flush=True)

    model, binder_seq, target_seq = load_structure()
    print(f"binder ({len(binder_seq)} aa): {binder_seq}", flush=True)
    print(f"target ({len(target_seq)} aa): {target_seq}", flush=True)

    reference_distances = reference_binder_target_ca_distances(model)
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(model)

    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    epitope_idx = np.array(
        sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32
    )
    print(f"designable positions: {len(designable_idx)}", flush=True)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    print("loading AbLang2...", flush=True)
    ablang2_model, ablang2_tokenizer = load_ablang2()

    key = jax.random.key(args.seed)
    key, wt_key = jax.random.split(key)
    wt_pose = measure_wt_distogram_pose_drift(
        opendde=opendde, features=features, reference_distances=reference_distances,
        binder_seq=binder_seq, key=wt_key,
    )
    pose_tolerance = wt_pose + POSE_DRIFT_MARGIN
    print(f"WT baseline distogram drift: {wt_pose:.2f}A -> pose_tolerance={pose_tolerance:.2f}A", flush=True)

    cheap_loss = build_cheap_search_loss(
        opendde=opendde, features=features, ablang2_model=ablang2_model,
        ablang2_tokenizer=ablang2_tokenizer, reference_distances=reference_distances,
        binder_seq=binder_seq, designable_idx=designable_idx, epitope_idx=epitope_idx,
        edit_budget=args.edit_budget, pose_tolerance=pose_tolerance,
    )
    rescore_loss = build_rescoring_loss(
        opendde=opendde, features=features,
        reference_binder_ca=reference_binder_ca, reference_target_ca=reference_target_ca,
        designable_idx=designable_idx, epitope_idx=epitope_idx,
    )
    rescore_grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(rescore_loss, has_aux=True))

    parent = np.array([TOKENS.index(c) for c in binder_seq], dtype=np.int32)
    current_seq = parent.copy()  # start from WT -- equivalent to --apgm-steps 0

    rows = []
    steps_done = 0
    chunk_idx = 0
    while steps_done < args.mcmc_steps:
        chunk_steps = min(args.rescore_every, args.mcmc_steps - steps_done)
        chunk_idx += 1
        key, mcmc_key = jax.random.split(key)

        print(f"\n[chunk {chunk_idx}] running {chunk_steps} MCMC steps "
              f"(steps {steps_done}..{steps_done + chunk_steps})...", flush=True)
        t0 = time.time()
        best_seq, best_val, pareto = edit_budgeted_gradient_mcmc(
            cheap_loss, current_seq, parent=parent, budget=args.edit_budget,
            designable_mask=designable_mask, steps=chunk_steps, key=mcmc_key,
        )
        chunk_wall = time.time() - t0
        current_seq = best_seq
        n_mut = int((current_seq != parent).sum())
        print(f"[chunk {chunk_idx}] cheap search done ({chunk_wall:.1f}s), "
              f"best_val={best_val:.4f}, mutations={n_mut}", flush=True)

        x = seq_to_one_hot(current_seq)
        print(f"[chunk {chunk_idx}] rescoring current best with real "
              f"confidence-aware full-OpenDDE gradient...", flush=True)
        t0 = time.time()
        key, rescore_key = jax.random.split(key)
        (real_value, real_aux), _grad = rescore_grad_fn(x, key=rescore_key)
        jax.block_until_ready(_grad)
        rescore_wall = time.time() - t0
        print(f"[chunk {chunk_idx}] rescore done ({rescore_wall:.1f}s), "
              f"real_value={float(real_value):.4f}", flush=True)

        # real_aux is a nested pytree (ClippedGradient/LinearCombination
        # wrap each sub-loss's own aux dict, same ".0.0.target_contact"-style
        # nesting the cheap MCMC log lines above already show) -- not a flat
        # dict, so look up each named metric via the same nested-aux helper
        # optimizers.py's own biohub_optimizer uses for exactly this.
        real_metrics = {
            name: _ranking_leaf(real_aux, name)
            for name in ("target_contact", "binder_pose_rmsd", "iptm", "bt_pae", "tb_pae", "ptm_energy")
        }
        for k, v in real_metrics.items():
            if v is not None:
                print(f"    {k}={float(v):.4f}", flush=True)

        def _f(v):
            return float(v) if v is not None else float("nan")

        rows.append({
            "chunk": chunk_idx, "steps_done": steps_done + chunk_steps,
            "mutations": n_mut, "sequence": one_hot_to_seq(x),
            "cheap_best_val": float(best_val),
            "cheap_chunk_wall_s": chunk_wall,
            "real_value": float(real_value),
            "real_target_contact": _f(real_metrics["target_contact"]),
            "real_binder_pose_rmsd": _f(real_metrics["binder_pose_rmsd"]),
            "real_iptm": _f(real_metrics["iptm"]),
            "real_bt_pae": _f(real_metrics["bt_pae"]),
            "real_tb_pae": _f(real_metrics["tb_pae"]),
            "real_ptm_energy": _f(real_metrics["ptm_energy"]),
            "rescore_wall_s": rescore_wall,
        })

        steps_done += chunk_steps

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote rescoring log: {args.output}", flush=True)

    total_cheap = sum(r["cheap_chunk_wall_s"] for r in rows)
    total_rescore = sum(r["rescore_wall_s"] for r in rows)
    print(f"\ntotals: cheap_search={total_cheap:.1f}s  rescoring={total_rescore:.1f}s "
          f"({len(rows)} rescoring calls)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
