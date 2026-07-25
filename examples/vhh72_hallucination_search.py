"""Hallucination-based VHH72 CDR redesign: direct gradient optimization
through OpenDDE (binding, cheap distogram-only path) + AbLang2 (naturalness,
masked PLL) + EditBudget (hard mutation-count budget) + BinderPoseDistogramDrift
(pose anchor, distogram-space, cheap) -- no BoltzGen diffusion involved at
all. See docs/guidance_alphaseq_testing_notes.md section 13 for why guided
diffusion via BoltzGen was set aside for this problem, and section 12.0/9a
for why this uses the full, uncropped complex (decided, not a default).

Two-stage optimization, both real, both loss-agnostic mosaic infrastructure
(src/mosaic/optimizers.py), not built for this task specifically:
  1. simplex_APGM: continuous relaxation, sparsity-encouraging (scale>1.0).
  2. edit_budgeted_greedy_descent OR edit_budgeted_gradient_mcmc (--policy):
     hard, exact edit-budget-constrained discrete search from the continuous
     result, real loss evaluation (not gradient-approximated) at every
     accept/reject decision, full Pareto front across every edit count from
     0 to --edit-budget.

Composite loss, every term ClippedGradient-wrapped (matches the existing
BoltzGen-guidance pattern's per-objective clip, cfg.clip_gradient_norm):
  - OpenDDE bind (BinderTargetContact) + BinderPoseDistogramDrift (pose
    anchor), combined INSIDE one DistogramOnlyOpenDDELoss call (one OpenDDE
    forward pass produces the distogram both terms read, not two), the
    whole combination clipped as one unit -- mirrors
    build_opendde_guidance_loss's existing pattern exactly.
  - AbLang2 naturalness (Ablang2PseudoLikelihood, masked PLL) evaluated on
    the full reconstructed binder sequence but averaged only over the
    designable CDR positions. --stop-grad controls whether its gradient
    backprops through AbLang2's own encoder or only reweights by fixed
    per-token scores (see docs/guidance_alphaseq_testing_notes.md's
    Germinal comparison) -- a real ablation axis for this experiment round,
    not yet decided which is better here.
  - EditBudget: soft hinge on total edit distance from WT (already produces
    zero pressure within budget; the HARD budget enforcement is the
    discrete search's job, not this term's).

Usage:
    .venv/bin/python examples/vhh72_hallucination_search.py \\
        --policy greedy --stop-grad 1 --edit-budget 5 \\
        --output results/greedy_stopgrad1.csv
"""
import argparse
import csv
from pathlib import Path

import gemmi
import jax
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.structure_prediction import (
    BinderPoseDistogramDrift,
    BinderTargetContact,
)
from mosaic.losses.transformations import ClippedGradient, EditBudget, SetPositions
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.optimizers import (
    edit_budgeted_gradient_mcmc,
    edit_budgeted_greedy_descent,
    simplex_APGM,
)
from mosaic.structure_prediction import TargetChain

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPLEX_CIF = REPO_ROOT / "vhh72_wt_wt_rbd.cif"

# Same ANARCI-verified, 1-indexed CDR boundaries used throughout this
# project (examples/anarci_vhh72_cdr_boundaries.py, examples/crop_vhh72_wt_rbd.py).
CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 115))

# Defaults, held fixed for this round (docs/guidance_alphaseq_testing_notes.md
# section 12c's "class 2/3 knobs held fixed, sweep class 1" framing) --
# matching the values already used/verified elsewhere in this project.
WEIGHT_OPENDDE_CONTACT = 0.5
WEIGHT_ABLANG2 = 0.10
WEIGHT_EDIT_BUDGET = 5.0
CONTACT_DISTANCE = 8.0
# BinderPoseDistogramDrift's tolerance is NOT a fixed absolute Angstrom
# value -- calibrated at runtime as measure_wt_pose_drift() + this margin.
# A fixed absolute tolerance (originally 2.0A) was wrong: OpenDDE's cheap
# distogram-only path (trunk + distogram head, no diffusion sampling) has
# real, substantial inherent noise even on the true WT structure -- measured
# directly at ~13A, not near zero (see docs/guidance_alphaseq_testing_notes.md
# section 13/9c on this path's confidence). The hinge should fire on
# ADDITIONAL drift a candidate introduces beyond that baseline, not on the
# baseline itself. This margin is a first-pass choice, not derived from
# anything -- tune against real results, same caveat as every other
# first-pass default in this project.
POSE_DRIFT_MARGIN = 3.0
CLIP_GRADIENT_NORM = 1.0
OPENDDE_RECYCLING_STEPS = 4
APGM_STEPS = 200
APGM_STEPSIZE = 0.05
APGM_SCALE = 1.2  # >1.0 encourages sparsity, per simplex_APGM's own docstring
APGM_MOMENTUM = 0.5
GREEDY_STEPS = 200
GREEDY_BATCH_SIZE = 16
MCMC_STEPS = 100


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def one_hot_to_seq(x: np.ndarray) -> str:
    idx = np.asarray(x).argmax(-1)
    return "".join(TOKENS[i] for i in idx)


def load_structure():
    st = gemmi.read_structure(str(COMPLEX_CIF))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model["A"]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model["A2"]]).upper()
    return model, binder_seq, target_seq


def reference_binder_target_ca_distances(model) -> np.ndarray:
    """Real WT-bound Calpha-Calpha binder x target distance matrix -- the
    reference BinderPoseDistogramDrift measures drift against."""
    def ca_coords(chain):
        coords = []
        for res in chain:
            for a in res:
                if a.name == "CA":
                    coords.append([a.pos.x, a.pos.y, a.pos.z])
                    break
        return np.array(coords, dtype=np.float32)

    binder_ca = ca_coords(model["A"])
    target_ca = ca_coords(model["A2"])
    diffs = binder_ca[:, None, :] - target_ca[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def measure_wt_pose_drift(*, opendde, features, reference_distances, binder_seq, key) -> float:
    """Real baseline: OpenDDE's own distogram-only prediction of the TRUE WT
    sequence, scored against the TRUE WT-bound reference distances.
    tolerance=0.0 here (an unhinged raw read, not the search-time loss) --
    this is a measurement, not a penalty."""
    probe = ClippedGradient(
        opendde.build_distogram_only_loss(
            loss=BinderPoseDistogramDrift(reference_distances, tolerance=0.0),
            features=features, recycling_steps=OPENDDE_RECYCLING_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )
    x_wt = seq_to_one_hot(binder_seq)
    _, aux = probe(x_wt, key=key)
    # ClippedGradient/DistogramOnlyOpenDDELoss both pass their wrapped
    # loss's (value, aux) straight through unmodified -- probe wraps only
    # BinderPoseDistogramDrift (a single term, not a LinearCombination), so
    # aux is exactly its own plain dict, verified directly against the
    # smoke test's printed aux structure before writing this.
    return float(aux["binder_target_distogram_drift"])


def build_composite_loss(*, opendde, features, ablang2_model, ablang2_tokenizer,
                          reference_distances, binder_seq, designable_idx,
                          edit_budget: int, stop_grad_ablang2: bool, pose_drift_tolerance: float):
    bind_and_pose = (
        WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
            paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        )
        + ClippedGradient(
            BinderPoseDistogramDrift(reference_distances, tolerance=pose_drift_tolerance),
            CLIP_GRADIENT_NORM,
        )
    )
    opendde_loss = ClippedGradient(
        opendde.build_distogram_only_loss(
            loss=bind_and_pose, features=features, recycling_steps=OPENDDE_RECYCLING_STEPS,
        ),
        CLIP_GRADIENT_NORM,
    )

    ablang2_loss = ClippedGradient(
        Ablang2PseudoLikelihood(
            model=ablang2_model, tokenizer=ablang2_tokenizer,
            heavy_len=len(binder_seq), designable_positions=jax.numpy.array(designable_idx, dtype=jax.numpy.int32),
            stop_grad=stop_grad_ablang2,
        ),
        CLIP_GRADIENT_NORM,
    )

    edit_loss = WEIGHT_EDIT_BUDGET * EditBudget.from_residues(
        binder_seq, designable_idx, budget=float(edit_budget),
    )

    full_loss = opendde_loss + WEIGHT_ABLANG2 * ablang2_loss + edit_loss
    wildtype_tokens = jax.numpy.array([TOKENS.index(aa) for aa in binder_seq], dtype=jax.numpy.int32)
    return SetPositions(
        wildtype_tokens,
        jax.numpy.array(designable_idx, dtype=jax.numpy.int32),
        full_loss,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["greedy", "mcmc"], required=True)
    p.add_argument("--stop-grad", type=int, choices=[0, 1], required=True,
                    help="AbLang2 gradient: 1 = fixed per-token reweighting (cheap, "
                         "current default elsewhere in this project), 0 = real backprop "
                         "through AbLang2's encoder (Germinal-style, more expensive)")
    p.add_argument("--edit-budget", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== VHH72 hallucination search: policy={args.policy} "
          f"stop_grad={args.stop_grad} edit_budget={args.edit_budget} ===", flush=True)

    model, binder_seq, target_seq = load_structure()
    print(f"binder ({len(binder_seq)} aa): {binder_seq}", flush=True)
    print(f"target ({len(target_seq)} aa): {target_seq}", flush=True)

    reference_distances = reference_binder_target_ca_distances(model)

    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    print(f"designable positions: {len(designable_idx)}", flush=True)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    print("loading AbLang2...", flush=True)
    ablang2_model, ablang2_tokenizer = load_ablang2()

    key = jax.random.key(args.seed)

    print("\ncalibrating pose-drift tolerance against the real WT baseline...", flush=True)
    key, wt_key = jax.random.split(key)
    wt_drift = measure_wt_pose_drift(
        opendde=opendde, features=features, reference_distances=reference_distances,
        binder_seq=binder_seq, key=wt_key,
    )
    pose_drift_tolerance = wt_drift + POSE_DRIFT_MARGIN
    print(f"WT baseline distogram drift: {wt_drift:.2f}A -> tolerance = "
          f"{wt_drift:.2f} + {POSE_DRIFT_MARGIN} margin = {pose_drift_tolerance:.2f}A", flush=True)

    loss = build_composite_loss(
        opendde=opendde, features=features,
        ablang2_model=ablang2_model, ablang2_tokenizer=ablang2_tokenizer,
        reference_distances=reference_distances, binder_seq=binder_seq,
        designable_idx=designable_idx, edit_budget=args.edit_budget,
        stop_grad_ablang2=bool(args.stop_grad), pose_drift_tolerance=pose_drift_tolerance,
    )

    print(f"\n[1/2] continuous relaxation (simplex_APGM, {APGM_STEPS} steps)...", flush=True)
    x0 = seq_to_one_hot("".join(binder_seq[i] for i in designable_idx))
    x_final, x_best = simplex_APGM(
        loss_function=loss, x=x0, n_steps=APGM_STEPS, stepsize=APGM_STEPSIZE,
        momentum=APGM_MOMENTUM, scale=APGM_SCALE, key=key,
    )
    full_continuous = np.asarray(loss.sequence(x_best))
    continuous_seq = full_continuous.argmax(-1)
    print(f"continuous result (argmax): {one_hot_to_seq(full_continuous)}", flush=True)

    print(f"\n[2/2] discrete budgeted search (--policy {args.policy})...", flush=True)
    parent = np.array([TOKENS.index(c) for c in binder_seq], dtype=np.int32)
    if args.policy == "greedy":
        best_seq, best_val, pareto = edit_budgeted_greedy_descent(
            loss, continuous_seq, parent=parent, budget=args.edit_budget,
            designable_mask=designable_mask, batch_size=GREEDY_BATCH_SIZE,
            steps=GREEDY_STEPS, key=key,
        )
    else:
        best_seq, best_val, pareto = edit_budgeted_gradient_mcmc(
            loss, continuous_seq, parent=parent, budget=args.edit_budget,
            designable_mask=designable_mask, steps=MCMC_STEPS, key=key,
        )

    print(f"\nbest sequence (val={best_val:.4f}): {one_hot_to_seq(np.eye(len(TOKENS))[best_seq])}", flush=True)

    print(f"\nwriting Pareto front ({len(pareto)} edit counts) to {args.output}", flush=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "stop_grad", "edit_count", "total_loss", "sequence", "num_mutations_from_wt"])
        for edit_count in sorted(pareto.keys()):
            val, seq_arr = pareto[edit_count]
            seq_str = "".join(TOKENS[i] for i in seq_arr)
            n_mut = int((seq_arr != parent).sum())
            writer.writerow([args.policy, args.stop_grad, edit_count, float(val), seq_str, n_mut])

    print("done.", flush=True)


if __name__ == "__main__":
    main()
