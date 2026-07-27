"""Hallucination-based P17 nanobody CDR redesign against the real JN.1 RBD
target -- same architecture validated on VHH72
(examples/vhh72_hallucination_search.py, docs/guidance_alphaseq_testing_notes.md
section 13), reused here with real, independently-verified P17-specific
inputs, not copied from VHH72's numbers.

Verified before writing this script (not assumed):
  - CDR ranges: ANARCI (IMGT scheme) run directly on P17's own binder
    sequence (chain B of P17_JN1.pdb). CDR1 26-33 and CDR2 51-58 happen to
    match VHH72's own ranges exactly; CDR3 does NOT (97-109, 13 residues,
    vs VHH72's 97-114, 18 residues -- real biological CDR3-length
    difference, confirmed by direct sequence slicing on the loaded
    structure, not assumed transferable).
  - CDR ranges cross-checked against the real production config at
    ~/boltzgen/Round2_Design/Nanobody/production/P17_JN1.yaml (prior
    BoltzGen work on this exact target) -- exact match, independent
    confirmation.
  - The 5 real hotspot residues on the JN.1 target (real, evidenced
    mutations at the CDR-contact interface, cross-referenced against solved
    structure 9IU1, from that same production YAML) verified directly
    against P17_JN1.pdb chain T: target residue numbering is a clean
    sequential 1..184 (no gaps, confirmed), and all 5 hotspot positions'
    amino acid identities match the YAML's stated post-mutation residues
    exactly (T115=D, T117=W, T146=K, T148=K, T150=P).

Real difference from VHH72's loss, deliberate: `BinderTargetContact` gets
an explicit `epitope_idx` restricting the contact loss to the 5 real
hotspot target residues, not left unrestricted like VHH72's. VHH72 could
afford an unrestricted contact loss because real AlphaSeq scoring caught
bad designs downstream; P17 has no equivalent real validation data (checked
directly: no P17-related entries anywhere in the AlphaSeq dataset), so the
one piece of real, externally-validated knowledge available (the real
hotspot) is used to constrain generation itself instead.

Real defaults for --policy/--stop-grad/--opendde-path/--apgm-seed-mode are
the validated VHH72 recommendation (mcmc, stop_grad=0, distogram in-loop
path, argmax/WT-seeded discrete search -- see
docs/guidance_alphaseq_testing_notes.md sections 13.9-13.12), not left as
required open questions the way VHH72's original script had them, since
that question is already answered. Full OpenDDE in-loop gradient
(--opendde-path full) remains available but is known infeasible at full
complex size for the discrete search (234GiB OOM, confirmed on VHH72's
complex -- P17's complex is a similar size, expect the same).

Usage:
    .venv/bin/python examples/p17_hallucination_search.py \\
        --edit-budget 5 --output results/p17_mcmc_stopgrad0.csv
"""
import argparse
import csv
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.structure_prediction import (
    BinderPoseRMSD,
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
COMPLEX_PDB = REPO_ROOT / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"

# ANARCI-verified (IMGT scheme, run directly on P17's chain B), cross-checked
# against ~/boltzgen/Round2_Design/Nanobody/production/P17_JN1.yaml -- exact
# match. 1-indexed. Do NOT reuse VHH72's CDR3 range (97-114) -- P17's CDR3
# loop is real and shorter (97-109); see module docstring.
CDR_RESIDUE_INDICES_1IDX = set(range(26, 34)) | set(range(51, 59)) | set(range(97, 110))

# Real, evidenced hotspot residues on the JN.1 target (chain T, 1-indexed --
# chain T's own numbering is a clean sequential 1..184, verified directly
# against the loaded structure, so 1-indexed residue N is 0-indexed array
# position N-1 with no offset/gap correction needed). From
# ~/boltzgen/Round2_Design/Nanobody/production/P17_JN1.yaml's real,
# 9IU1-cross-referenced hotspot list (N450D/L452W/N481K/E484K/F486P at JN.1
# positions 115/117/146/148/150) -- amino acid identities at these positions
# verified directly against P17_JN1.pdb before writing this file.
HOTSPOT_TARGET_RESIDUE_INDICES_1IDX = {115, 117, 146, 148, 150}

# Defaults, held fixed for this round -- matching the values already
# used/verified for VHH72 (docs/guidance_alphaseq_testing_notes.md section
# 12c's "class 2/3 knobs held fixed, sweep class 1" framing).
WEIGHT_OPENDDE_CONTACT = 0.5
WEIGHT_ABLANG2 = 0.10
WEIGHT_EDIT_BUDGET = 5.0
CONTACT_DISTANCE = 8.0
# BinderPoseDistogramDrift's tolerance is NOT a fixed absolute Angstrom
# value -- calibrated at runtime as measure_wt_distogram_pose_drift() + this
# margin, same reasoning as VHH72's script (OpenDDE's cheap distogram-only
# path has real, substantial inherent noise even on the true bound
# structure -- this margin is a first-pass choice, tune against real
# results, same caveat as every other first-pass default in this project).
POSE_DRIFT_MARGIN = 3.0
POSE_RMSD_MARGIN = 3.0
CLIP_GRADIENT_NORM = 1.0
OPENDDE_RECYCLING_STEPS = 4
OPENDDE_FULL_NUM_SAMPLES = 1
APGM_STEPS = 200
APGM_STEPSIZE = 0.05
APGM_SCALE = 1.2  # >1.0 encourages sparsity, per simplex_APGM's own docstring
APGM_INIT_WT_PROB = 1.0  # exact one-hot WT unless explicitly softened
APGM_MOMENTUM = 0.5
GREEDY_STEPS = 200
GREEDY_BATCH_SIZE = 16
MCMC_STEPS = 100


def seq_to_one_hot(seq: str) -> np.ndarray:
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    return np.eye(len(TOKENS), dtype=np.float32)[idx]


def seq_to_soft_wt_init(seq: str, wt_prob: float) -> np.ndarray:
    """WT-centered simplex initialization for APGM.

    wt_prob=1.0 is exact one-hot WT. Values below 1.0 put the remaining
    mass uniformly on the other 19 amino acids -- matches the APGM
    diagnostic that found this necessary to avoid the exact-vertex fixed
    point (docs/guidance_alphaseq_testing_notes.md section 13.7), though
    for P17 the validated default keeps 1.0 (argmax/WT-seeded search is
    the recommendation, not APGM-seeded -- section 13.11-13.12 found
    APGM-seeding negative on VHH72).
    """
    if not (0.0 < wt_prob <= 1.0):
        raise ValueError(f"--apgm-init-wt-prob must be in (0, 1], got {wt_prob}")
    idx = np.array([TOKENS.index(c) for c in seq], dtype=np.int32)
    if wt_prob == 1.0:
        return np.eye(len(TOKENS), dtype=np.float32)[idx]
    other_prob = (1.0 - wt_prob) / (len(TOKENS) - 1)
    x = np.full((len(seq), len(TOKENS)), other_prob, dtype=np.float32)
    x[np.arange(len(seq)), idx] = wt_prob
    return x


def one_hot_to_seq(x: np.ndarray) -> str:
    idx = np.asarray(x).argmax(-1)
    return "".join(TOKENS[i] for i in idx)


def apgm_seed_from_soft(full_continuous: np.ndarray, wt_tokens: np.ndarray, mode: str,
                         key, topk_threshold: float,
                         designable_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert simplex_APGM's soft output into the hard discrete sequence
    handed to the discrete search -- argmax/sample/topk, identical logic to
    examples/vhh72_hallucination_search.py's version (see that module's
    docstring for the full mechanism explanation and
    docs/guidance_alphaseq_testing_notes.md sections 13.7-13.12 for why
    argmax is the validated default here)."""
    wt_tokens = np.asarray(wt_tokens, dtype=np.int32)
    if designable_mask is None:
        designable_mask = np.ones_like(wt_tokens, dtype=bool)
    else:
        designable_mask = np.asarray(designable_mask, dtype=bool)
    assert designable_mask.shape == wt_tokens.shape

    if mode == "argmax":
        seq = np.asarray(full_continuous.argmax(-1), dtype=np.int32)
    elif mode == "sample":
        logits = jax.numpy.log(jax.numpy.clip(full_continuous, 1e-12, 1.0))
        seq = np.asarray(jax.random.categorical(key, logits, axis=-1), dtype=np.int32)
    elif mode == "topk":
        seq = np.array(wt_tokens)
        for i in range(full_continuous.shape[0]):
            if not designable_mask[i]:
                continue
            probs = np.array(full_continuous[i])
            probs[wt_tokens[i]] = -1.0
            best_non_wt = int(np.argmax(probs))
            if probs[best_non_wt] >= topk_threshold:
                seq[i] = best_non_wt
    else:
        raise ValueError(f"unknown --apgm-seed-mode {mode!r}")

    seq = np.array(seq, dtype=np.int32, copy=True)
    seq[~designable_mask] = wt_tokens[~designable_mask]
    return seq


def load_structure():
    st = gemmi.read_structure(str(COMPLEX_PDB))
    st.setup_entities()
    model = st[0]
    binder_seq = gemmi.one_letter_code([r.name for r in model[BINDER_CHAIN]]).upper()
    target_seq = gemmi.one_letter_code([r.name for r in model[TARGET_CHAIN]]).upper()
    return model, binder_seq, target_seq


def reference_binder_target_ca(model) -> tuple[np.ndarray, np.ndarray]:
    """Real JN.1-bound binder/target Calpha coordinates."""
    def ca_coords(chain):
        coords = []
        for res in chain:
            for a in res:
                if a.name == "CA":
                    coords.append([a.pos.x, a.pos.y, a.pos.z])
                    break
        return np.array(coords, dtype=np.float32)

    binder_ca = ca_coords(model[BINDER_CHAIN])
    target_ca = ca_coords(model[TARGET_CHAIN])
    return binder_ca, target_ca


def reference_binder_target_ca_distances(model) -> np.ndarray:
    """Real JN.1-bound Calpha-Calpha binder x target distance matrix -- the
    reference BinderPoseDistogramDrift measures drift against."""
    binder_ca, target_ca = reference_binder_target_ca(model)
    diffs = binder_ca[:, None, :] - target_ca[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def measure_wt_distogram_pose_drift(*, opendde, features, reference_distances,
                                    binder_seq, key) -> float:
    """Real baseline: OpenDDE's own distogram-only prediction of the TRUE
    P17 sequence, scored against the TRUE JN.1-bound reference distances.
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
    return float(aux["binder_target_distogram_drift"])


def measure_wt_pose_rmsd(*, opendde, features, reference_binder_ca,
                         reference_target_ca, binder_seq, key,
                         opendde_sampling_steps: int | None,
                         opendde_num_samples: int) -> float:
    """Real baseline for --opendde-path full: run OpenDDE's full coordinate
    path on WT and measure target-aligned binder RMSD against the JN.1-bound
    input structure. tolerance=0.0 is an unhinged measurement."""
    rmsd = BinderPoseRMSD(
        reference_binder_ca=reference_binder_ca,
        reference_target_ca=reference_target_ca,
        rmsd_tolerance=0.0,
    )
    if opendde_num_samples == 1:
        probe = ClippedGradient(
            opendde.build_loss(
                loss=rmsd,
                features=features,
                recycling_steps=OPENDDE_RECYCLING_STEPS,
                sampling_steps=opendde_sampling_steps,
            ),
            CLIP_GRADIENT_NORM,
        )
    else:
        probe = ClippedGradient(
            opendde.build_multisample_loss(
                loss=rmsd,
                features=features,
                recycling_steps=OPENDDE_RECYCLING_STEPS,
                sampling_steps=opendde_sampling_steps,
                num_samples=opendde_num_samples,
            ),
            CLIP_GRADIENT_NORM,
        )
    x_wt = seq_to_one_hot(binder_seq)

    @eqx.filter_jit
    def _eval(loss, x, key):
        return loss(x, key=key)

    _, aux = _eval(probe, x_wt, key)
    values = [np.asarray(v) for v in jax.tree_util.tree_leaves(aux["binder_pose_rmsd"])]
    return float(np.mean(values))


def build_composite_losses(*, opendde, features, ablang2_model, ablang2_tokenizer,
                           reference_distances, reference_binder_ca,
                           reference_target_ca, binder_seq, designable_idx,
                           epitope_idx, edit_budget: int, stop_grad_ablang2: bool,
                           opendde_path: str, pose_tolerance: float,
                           opendde_sampling_steps: int | None,
                           opendde_num_samples: int):
    # epitope_idx restricts the contact loss to the 5 real hotspot target
    # residues -- deliberate difference from VHH72's script (see module
    # docstring): P17 has no downstream real-data validation to catch a
    # design that binds somewhere biologically wrong, so the one piece of
    # real, externally-validated knowledge available is used to constrain
    # generation itself.
    contact_loss = WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
        paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
        epitope_idx=epitope_idx,
    )
    if opendde_path == "distogram":
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
    elif opendde_path == "full":
        pose_loss = ClippedGradient(
            BinderPoseRMSD(
                reference_binder_ca=reference_binder_ca,
                reference_target_ca=reference_target_ca,
                rmsd_tolerance=pose_tolerance,
            ),
            CLIP_GRADIENT_NORM,
        )
        full_opendde_loss = contact_loss + pose_loss
        if opendde_num_samples == 1:
            opendde_loss = ClippedGradient(
                opendde.build_loss(
                    loss=full_opendde_loss,
                    features=features,
                    recycling_steps=OPENDDE_RECYCLING_STEPS,
                    sampling_steps=opendde_sampling_steps,
                ),
                CLIP_GRADIENT_NORM,
            )
        else:
            opendde_loss = ClippedGradient(
                opendde.build_multisample_loss(
                    loss=full_opendde_loss,
                    features=features,
                    recycling_steps=OPENDDE_RECYCLING_STEPS,
                    sampling_steps=opendde_sampling_steps,
                    num_samples=opendde_num_samples,
                ),
                CLIP_GRADIENT_NORM,
            )
    else:
        raise ValueError(f"unknown opendde_path={opendde_path!r}")

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
    variable_only_loss = SetPositions(
        wildtype_tokens,
        jax.numpy.array(designable_idx, dtype=jax.numpy.int32),
        full_loss,
    )
    return full_loss, variable_only_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["greedy", "mcmc"], default="mcmc",
                    help="Validated default (docs/guidance_alphaseq_testing_notes.md "
                         "section 13.9): mcmc.")
    p.add_argument("--stop-grad", type=int, choices=[0, 1], default=0,
                    help="AbLang2 gradient: 1 = fixed per-token reweighting (cheap), "
                         "0 = real backprop through AbLang2's encoder (Germinal-style, "
                         "more expensive). Validated default: 0 (section 13.9's "
                         "significance test, p=0.001).")
    p.add_argument("--edit-budget", type=int, default=5)
    p.add_argument("--use-hotspot", type=int, choices=[0, 1], default=1,
                    help="1 (default) = restrict BinderTargetContact to the 5 real "
                         "hotspot target residues. 0 = unrestricted contact loss, "
                         "matching VHH72's script (only advisable if you have another "
                         "real validation signal to catch bad designs downstream).")
    p.add_argument("--apgm-seed-mode", choices=["argmax", "sample", "topk"], default="argmax",
                    help="Validated default: argmax (section 13.11-13.12 found "
                         "sample/topk seeding negative on VHH72).")
    p.add_argument("--apgm-topk-threshold", type=float, default=0.15,
                    help="Only used with --apgm-seed-mode topk.")
    p.add_argument("--apgm-init-wt-prob", type=float, default=APGM_INIT_WT_PROB,
                    help="WT amino-acid probability for APGM initialization. "
                         "1.0 preserves exact one-hot WT start.")
    p.add_argument("--apgm-scale", type=float, default=APGM_SCALE,
                    help="simplex_APGM scale parameter.")
    p.add_argument("--apgm-steps", type=int, default=APGM_STEPS,
                    help="Number of simplex_APGM continuous-relaxation steps. "
                         "Use 0 to skip APGM and start discrete search from WT.")
    p.add_argument("--greedy-steps", type=int, default=GREEDY_STEPS,
                    help="Maximum greedy discrete-search steps.")
    p.add_argument("--mcmc-steps", type=int, default=MCMC_STEPS,
                    help="Maximum MCMC discrete-search steps.")
    p.add_argument("--opendde-path", choices=["distogram", "full"], default="distogram",
                    help="distogram (default, validated) = cheap trunk+distogram-head "
                         "path. full = OpenDDE full diffusion/coordinate/confidence-head "
                         "path -- known infeasible for gradient-based discrete search at "
                         "full complex size (234GiB OOM on VHH72's similarly-sized "
                         "complex); only use with --policy greedy --greedy-steps 0 or "
                         "very small step counts as a smoke test.")
    p.add_argument("--opendde-sampling-steps", type=int, default=None,
                    help="OpenDDE diffusion sampling steps for --opendde-path full. "
                         "Default: OpenDDE model default.")
    p.add_argument("--opendde-num-samples", type=int, default=OPENDDE_FULL_NUM_SAMPLES,
                    help="Number of full OpenDDE coordinate samples per loss call in "
                         "--opendde-path full.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.apgm_steps < 0:
        raise SystemExit("--apgm-steps must be >= 0")
    if args.greedy_steps < 0:
        raise SystemExit("--greedy-steps must be >= 0")
    if args.mcmc_steps < 0:
        raise SystemExit("--mcmc-steps must be >= 0")
    if args.opendde_num_samples < 1:
        raise SystemExit("--opendde-num-samples must be >= 1")

    print(f"=== P17 hallucination search: policy={args.policy} "
          f"stop_grad={args.stop_grad} edit_budget={args.edit_budget} "
          f"opendde_path={args.opendde_path} use_hotspot={args.use_hotspot} ===", flush=True)

    model, binder_seq, target_seq = load_structure()
    print(f"binder ({len(binder_seq)} aa): {binder_seq}", flush=True)
    print(f"target ({len(target_seq)} aa): {target_seq}", flush=True)

    reference_distances = reference_binder_target_ca_distances(model)
    reference_binder_ca, reference_target_ca = reference_binder_target_ca(model)

    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]
    print(f"designable positions: {len(designable_idx)}", flush=True)

    epitope_idx = None
    if args.use_hotspot:
        epitope_idx = np.array(sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX), dtype=np.int32)
        hotspot_aas = "".join(target_seq[i] for i in epitope_idx)
        print(f"hotspot target positions (0-idx): {epitope_idx.tolist()} "
              f"(residues: {hotspot_aas})", flush=True)

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    print("loading AbLang2...", flush=True)
    ablang2_model, ablang2_tokenizer = load_ablang2()

    key = jax.random.key(args.seed)

    print(f"\ncalibrating pose tolerance against the real JN.1 baseline "
          f"(--opendde-path {args.opendde_path})...", flush=True)
    key, wt_key = jax.random.split(key)
    if args.opendde_path == "distogram":
        wt_pose = measure_wt_distogram_pose_drift(
            opendde=opendde, features=features, reference_distances=reference_distances,
            binder_seq=binder_seq, key=wt_key,
        )
        pose_tolerance = wt_pose + POSE_DRIFT_MARGIN
        print(f"WT baseline distogram drift: {wt_pose:.2f}A -> tolerance = "
              f"{wt_pose:.2f} + {POSE_DRIFT_MARGIN} margin = {pose_tolerance:.2f}A", flush=True)
    else:
        wt_pose = measure_wt_pose_rmsd(
            opendde=opendde, features=features,
            reference_binder_ca=reference_binder_ca,
            reference_target_ca=reference_target_ca,
            binder_seq=binder_seq, key=wt_key,
            opendde_sampling_steps=args.opendde_sampling_steps,
            opendde_num_samples=args.opendde_num_samples,
        )
        pose_tolerance = wt_pose + POSE_RMSD_MARGIN
        sampling_steps = (
            "model default" if args.opendde_sampling_steps is None
            else str(args.opendde_sampling_steps)
        )
        print(f"WT baseline coordinate RMSD: {wt_pose:.2f}A -> tolerance = "
              f"{wt_pose:.2f} + {POSE_RMSD_MARGIN} margin = {pose_tolerance:.2f}A "
              f"(sampling_steps={sampling_steps}, num_samples={args.opendde_num_samples})",
              flush=True)

    full_loss, variable_only_loss = build_composite_losses(
        opendde=opendde, features=features,
        ablang2_model=ablang2_model, ablang2_tokenizer=ablang2_tokenizer,
        reference_distances=reference_distances,
        reference_binder_ca=reference_binder_ca,
        reference_target_ca=reference_target_ca,
        binder_seq=binder_seq,
        designable_idx=designable_idx, epitope_idx=epitope_idx,
        edit_budget=args.edit_budget,
        stop_grad_ablang2=bool(args.stop_grad), opendde_path=args.opendde_path,
        pose_tolerance=pose_tolerance,
        opendde_sampling_steps=args.opendde_sampling_steps,
        opendde_num_samples=args.opendde_num_samples,
    )

    parent = np.array([TOKENS.index(c) for c in binder_seq], dtype=np.int32)

    print(f"\n[1/2] continuous relaxation (simplex_APGM, {args.apgm_steps} steps)...", flush=True)
    print(f"APGM init WT prob: {args.apgm_init_wt_prob}  scale: {args.apgm_scale}", flush=True)
    x0 = seq_to_soft_wt_init("".join(binder_seq[i] for i in designable_idx),
                             args.apgm_init_wt_prob)
    key, apgm_key, seed_key = jax.random.split(key, 3)
    x_final, x_best = simplex_APGM(
        loss_function=variable_only_loss, x=x0, n_steps=args.apgm_steps, stepsize=APGM_STEPSIZE,
        momentum=APGM_MOMENTUM, scale=args.apgm_scale, key=apgm_key,
    )
    full_continuous = np.asarray(variable_only_loss.sequence(x_best))
    continuous_seq = apgm_seed_from_soft(full_continuous, parent, args.apgm_seed_mode,
                                          seed_key, args.apgm_topk_threshold,
                                          designable_mask=designable_mask)
    print(f"continuous result (argmax): {one_hot_to_seq(full_continuous)}", flush=True)
    print(f"discrete search seed (--apgm-seed-mode {args.apgm_seed_mode}): "
          f"{''.join(TOKENS[i] for i in continuous_seq)}", flush=True)
    print(f"seed differs from WT at {int((continuous_seq != parent).sum())} positions", flush=True)

    print(f"\n[2/2] discrete budgeted search (--policy {args.policy})...", flush=True)
    if args.policy == "greedy":
        best_seq, best_val, pareto = edit_budgeted_greedy_descent(
            full_loss, continuous_seq, parent=parent, budget=args.edit_budget,
            designable_mask=designable_mask, batch_size=GREEDY_BATCH_SIZE,
            steps=args.greedy_steps, key=key,
        )
    else:
        best_seq, best_val, pareto = edit_budgeted_gradient_mcmc(
            full_loss, continuous_seq, parent=parent, budget=args.edit_budget,
            designable_mask=designable_mask, steps=args.mcmc_steps, key=key,
        )

    print(f"\nbest sequence (val={best_val:.4f}): {one_hot_to_seq(np.eye(len(TOKENS))[best_seq])}", flush=True)

    print(f"\nwriting Pareto front ({len(pareto)} edit counts) to {args.output}", flush=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "stop_grad", "seed", "opendde_path", "use_hotspot", "edit_count",
            "total_loss", "sequence", "num_mutations_from_wt",
        ])
        for edit_count in sorted(pareto.keys()):
            val, seq_arr = pareto[edit_count]
            seq_str = "".join(TOKENS[i] for i in seq_arr)
            n_mut = int((seq_arr != parent).sum())
            writer.writerow([
                args.policy, args.stop_grad, args.seed, args.opendde_path, args.use_hotspot,
                edit_count, float(val), seq_str, n_mut,
            ])

    print("done.", flush=True)


if __name__ == "__main__":
    main()
