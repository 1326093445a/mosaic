"""Guided partial-diffusion + edit-budgeted CDR redesign for VHHs.

Pipeline (per outer-loop iteration):
  1. guided_partial_diffusion: BoltzGen partial diffusion with classifier guidance
     from ESMC/AbLang/MPNN/EditBudget, perturbing CDR atoms only.
  2. differentiable_inverse_fold(temp=0.001): decode guided coords to a soft sequence.
  3. edit_budgeted_greedy_descent: polish the discrete sequence under the full
     multi-model loss with a hard <=budget edit constraint vs the parent.
  4. Record a Pareto front {edit_count: (loss, sequence)}.
  5. Update parent if improved, repeat.

After convergence: refold final candidates with Boltz2, rank by iPTM/ipSAE, write CIFs.

Toggles for incremental milestones:
  v0 (task #7):  skip_guidance=True,  skip_polish=True
  v1 (task #11): skip_guidance=False, skip_polish=True, only EditBudget in guidance
  v2 (task #12): skip_guidance=False, skip_polish=True, full multi-model guidance
  v3 (task #14): all flags off
  v4 (task #15): all flags off, skip_refold=False

Inputs: an Ab-Ag complex CIF, the binder/target chain ids, and CDR position indices.
Outputs: per-iteration designs, a Pareto-front CSV, and (if not skipped) Boltz2-refolded
ranked structures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jaxtyping import Array, Bool, Float, Int

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import (
    ClippedGradient,
    EditBudget,
    SetPositions,
)
from mosaic.models.boltzgen import (
    Sampler,
    build_atom_partial_mask,
    differentiable_inverse_fold,
    guided_partial_diffusion,
    load_boltzgen,
    load_features_and_structure_writer,
)
from mosaic.optimizers import edit_budgeted_greedy_descent
from mosaic.proteinmpnn.mpnn import load_abmpnn
from mosaic.util import fold_in


# =============================================================================
# Config
# =============================================================================


@dataclass
class VHHDesignConfig:
    """All runtime knobs for the VHH guided-design driver."""

    # ---- Inputs ----
    complex_cif_path: Path
    binder_chain_id: str           # heavy-chain id in the CIF (e.g. "H")
    target_chain_ids: list[str]    # target chain id(s) (e.g. ["A"])
    cdr_residue_indices: list[int] # 1-indexed label_seq_id positions on the binder chain

    # ---- Constraint ----
    edit_budget: int = 7

    # ---- Guidance composite loss weights ----
    weight_esmc: float = 0.10
    weight_ablang: float = 0.10
    weight_mpnn_recovery: float = 0.50
    weight_edit_budget: float = 5.00
    clip_gradient_norm: float = 1.0  # per-term gradient clip for balance

    # ---- Diffusion ----
    num_sampling_steps: int = 200
    start_sigma_frac: float = 0.4
    step_scale: float = 2.0
    noise_scale: float = 0.88
    lambda_max: float = 1.0
    lambda_schedule: str = "sigma_squared"  # one of {sigma_squared, sigma, constant}

    # ---- Outer loop ----
    n_outer_iterations: int = 3

    # ---- Stage 2 polish ----
    polish_steps: int = 200
    polish_batch_size: int = 16

    # ---- I/O ----
    output_dir: Path = Path("./vhh_designs")

    # ---- Toggles for incremental testing ----
    skip_guidance: bool = False
    skip_polish: bool = False
    skip_refold: bool = True  # default off; task #15 wires this back in

    # ---- Misc ----
    recycling_steps: int = 3
    seed: int = 0


# =============================================================================
# Helpers
# =============================================================================


def build_complex_yaml(cif_filename: str, binder_chain_id: str,
                       target_chain_ids: list[str],
                       cdr_residue_indices: list[int]) -> str:
    """Build a BoltzGen design YAML where the binder's CDR positions are designable.

    Uses the `file.design:` field so BoltzGen knows the parent backbone for ALL
    positions including CDRs and just gets told which to redesign. This is exactly
    the partial-diffusion use case.
    """
    res_idx_str = ",".join(str(i) for i in sorted(set(cdr_residue_indices)))
    target_includes = "\n        ".join(
        f"- chain:\n            id: {cid}" for cid in target_chain_ids
    )
    return f"""
entities:
  - file:
      path: {cif_filename}
      include:
        - chain:
            id: {binder_chain_id}
        {target_includes}
      design:
        - chain:
            id: {binder_chain_id}
            res_index: {res_idx_str}
"""


def parent_one_hot_from_features(features: dict) -> Float[Array, "N 20"]:
    """Recover the parent (pre-mask) sequence as a one-hot over mosaic's TOKENS.

    BoltzGen's masker zeroes res_type at designable positions, but the unmasked
    parent identity is still available in `res_type_clone` (preserved by the masker
    at masker.py:101). We slice columns 2:22 to drop BoltzGen's special tokens and
    keep the 20 standard amino-acid columns, matching mosaic's TOKENS ordering.
    """
    return jnp.array(features["res_type_clone"][0, :, 2:22], dtype=jnp.float32)


def cdr_token_mask_from_features(features: dict) -> Bool[Array, "N"]:
    """Token-level designable mask, sourced directly from the BoltzGen featurizer."""
    return jnp.array(features["design_mask"][0], dtype=bool)


def lambda_schedule_fn(name: str, lam_max: float):
    if name == "sigma_squared":
        return lambda sigma: lam_max * (sigma ** 2)
    if name == "sigma":
        return lambda sigma: lam_max * sigma
    if name == "constant":
        return lambda sigma: lam_max * jnp.ones_like(sigma)
    raise ValueError(f"unknown lambda_schedule: {name}")


# =============================================================================
# Model loading (one-time setup)
# =============================================================================


@dataclass
class LoadedModels:
    boltzgen: any
    mpnn: any
    esmc_pll: any
    ablang_pll: any


def load_all_models() -> LoadedModels:
    """Load every model the driver uses, ONCE. JIT compile cost amortizes across
    the outer loop's iterations because we keep the same loss objects."""
    boltzgen = load_boltzgen()
    mpnn = load_abmpnn()

    # stop_grad=False on the language models because we need gradients to flow
    # back to coords through the differentiable IF bridge during guidance.
    esmc = load_esmc("esmc_300m")
    esmc_pll = ESMCPseudoLikelihood(esmc, stop_grad=False)

    ablang_model, ablang_tokenizer = load_ablang("heavy")
    ablang_pll = AbLangPseudoLikelihood(
        model=ablang_model, tokenizer=ablang_tokenizer, stop_grad=False
    )

    return LoadedModels(boltzgen=boltzgen, mpnn=mpnn, esmc_pll=esmc_pll,
                        ablang_pll=ablang_pll)


# =============================================================================
# Composite loss construction
# =============================================================================


def build_guidance_loss(cfg: VHHDesignConfig, models: LoadedModels,
                        parent_one_hot: Float[Array, "N 20"],
                        designable_token_mask: Bool[Array, "N"]):
    """Composite loss applied per diffusion step inside guided_partial_diffusion.

    Operates on a soft sequence emitted by differentiable_inverse_fold. Gradients
    flow back through IF -> coords. Each term is wrapped in ClippedGradient so the
    contributions stay balanced as guidance compounds across hundreds of steps.
    """
    edit_budget_term = EditBudget(
        s_ref=parent_one_hot,
        designable=designable_token_mask,
        budget=cfg.edit_budget,
    )

    if cfg.skip_guidance:
        # v0 path: no guidance at all. Return None so guided_partial_diffusion
        # short-circuits the per-step gradient computation.
        return None

    # In v1 we use ONLY edit budget; in v2/v3 we use the full composite.
    # We always include the edit-budget term with high weight so the guided
    # trajectory is biased to stay near parent, regardless of other terms.
    aux_terms = [cfg.weight_edit_budget * edit_budget_term]

    if cfg.weight_esmc > 0:
        aux_terms.append(
            cfg.weight_esmc * ClippedGradient(models.esmc_pll, cfg.clip_gradient_norm)
        )
    if cfg.weight_ablang > 0:
        aux_terms.append(
            cfg.weight_ablang * ClippedGradient(models.ablang_pll, cfg.clip_gradient_norm)
        )
    # MPNN sequence recovery is a structure-prediction-output loss, so it needs
    # to be plumbed differently — see polish loss below. For guidance during
    # diffusion we only use sequence-only auxiliary terms here.

    return sum(aux_terms[1:], start=aux_terms[0])


def build_polish_loss(cfg: VHHDesignConfig, models: LoadedModels,
                      parent_one_hot: Float[Array, "N 20"],
                      designable_token_mask: Bool[Array, "N"]):
    """Composite loss for Stage 2 (edit_budgeted_greedy_descent).

    Operates on a full discrete sequence. `EditBudget` is included as a soft
    secondary objective even though Stage 2's hard budget is enforced by the
    feasibility filter — keeping the soft term breaks ties in favor of solutions
    that use less of the budget.
    """
    edit_term = EditBudget(
        s_ref=parent_one_hot,
        designable=designable_token_mask,
        budget=cfg.edit_budget,
    )
    terms = [
        cfg.weight_esmc * models.esmc_pll,
        cfg.weight_ablang * models.ablang_pll,
        cfg.weight_edit_budget * edit_term,
    ]
    return sum(terms[1:], start=terms[0])


# =============================================================================
# Driver
# =============================================================================


def run(cfg: VHHDesignConfig):
    """End-to-end VHH redesign driver. See module docstring for pipeline overview."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "config.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(cfg).items()}, f, indent=2)

    # ---- 1. One-time setup: load models, parse YAML, build features ----
    print("[setup] loading models...")
    models = load_all_models()

    cif_filename = cfg.complex_cif_path.name
    yaml_string = build_complex_yaml(
        cif_filename=cif_filename,
        binder_chain_id=cfg.binder_chain_id,
        target_chain_ids=cfg.target_chain_ids,
        cdr_residue_indices=cfg.cdr_residue_indices,
    )

    print("[setup] parsing YAML and featurizing complex...")
    features, writer = load_features_and_structure_writer(
        yaml_string=yaml_string,
        files={cif_filename: cfg.complex_cif_path},
        # mask_backbone=False keeps the parent backbone visible to the trunk —
        # the sequence is still masked at designable positions.
        mask=True,
        mask_backbone=False,
        mask_disto=True,
    )

    # Extract everything we need from features. Keep arrays UNBATCHED throughout
    # the driver — guided_partial_diffusion auto-batches as needed. For backbone
    # extraction inside guidance_fn we use the token-to-backbone mapping that
    # BoltzGenOutput.backbone_coordinates already exercises.
    parent_one_hot = parent_one_hot_from_features(features)
    designable_token_mask = cdr_token_mask_from_features(features)
    initial_coords = jnp.array(features["coords"][0])               # (M, 3)
    atom_pad_mask = jnp.array(features["atom_pad_mask"][0])         # (M,)
    atom_partial_mask = build_atom_partial_mask(features, designable_token_mask)  # (M,)
    asym_id = jnp.array(features["asym_id"][0])
    residue_index = jnp.array(features["residue_index"][0])
    bb_atom_inds = jnp.argmax(jnp.array(features["token_to_bb4_atoms"][0]), axis=-1)  # (N, 4)

    n_total_tokens = parent_one_hot.shape[0]
    n_designable = int(designable_token_mask.sum())
    print(f"[setup] complex has {n_total_tokens} tokens, "
          f"{n_designable} designable (CDR) positions, "
          f"edit budget = {cfg.edit_budget}")

    # ---- 2. Run trunk ONCE; reuse Sampler across outer iterations ----
    print("[setup] running BoltzGen trunk + diffusion conditioning...")
    sampler = Sampler.from_features(
        model=models.boltzgen,
        features=features,
        key=jax.random.key(cfg.seed),
        deterministic=True,
        recycling_steps=cfg.recycling_steps,
    )

    # ---- 3. Build composite losses ----
    guidance_loss = build_guidance_loss(
        cfg, models, parent_one_hot, designable_token_mask
    )
    polish_loss = build_polish_loss(
        cfg, models, parent_one_hot, designable_token_mask
    )
    lambda_fn = lambda_schedule_fn(cfg.lambda_schedule, cfg.lambda_max)

    # ---- 4. Outer loop ----
    parent_seq_ids = jnp.argmax(parent_one_hot, axis=-1)
    current_parent_one_hot = parent_one_hot
    current_initial_coords = initial_coords

    all_iterations = []
    global_pareto: dict[int, tuple[float, np.ndarray]] = {}

    for outer in range(cfg.n_outer_iterations):
        t0 = time.time()
        print(f"\n[outer {outer}] guided partial diffusion "
              f"(start_sigma_frac={cfg.start_sigma_frac}, "
              f"steps={cfg.num_sampling_steps})...")

        # ----- Stage 1: guided partial diffusion -----
        if guidance_loss is not None:
            def guidance_fn(x0):
                # x0 arrives batched (1, M, 3) from inside guided_partial_diffusion.
                # Extract per-token backbone (N, 4, 3) for the IF model.
                bb_coords = x0[0][bb_atom_inds]
                soft_seq = differentiable_inverse_fold(
                    models.mpnn, bb_coords,
                    parent_sequence=current_parent_one_hot,
                    asym_id=asym_id,
                    residue_idx=residue_index,
                    designable_mask=designable_token_mask.astype(jnp.float32),
                    temperature=0.1,
                    jacobi_iterations=1,
                    key=jax.random.key(cfg.seed + outer),
                )
                v, _ = guidance_loss(soft_seq, key=jax.random.key(cfg.seed + outer))
                return v
        else:
            guidance_fn = None

        x_final = guided_partial_diffusion(
            sampler=sampler,
            structure_module=models.boltzgen.structure_module,
            initial_coords=current_initial_coords,
            atom_partial_mask=atom_partial_mask,
            atom_mask=atom_pad_mask,
            num_sampling_steps=cfg.num_sampling_steps,
            start_sigma_frac=cfg.start_sigma_frac,
            step_scale=cfg.step_scale,
            noise_scale=cfg.noise_scale,
            guidance_fn=guidance_fn,
            guidance_lambda_fn=lambda_fn if guidance_fn is not None else None,
            key=jax.random.key(cfg.seed + 1000 * outer),
        )

        # ----- Stage 1.5: decode to discrete sequence -----
        # x_final is unbatched (M, 3); extract backbone for IF.
        bb_final = x_final[bb_atom_inds]
        soft_seq_decoded = differentiable_inverse_fold(
            models.mpnn, bb_final,
            parent_sequence=current_parent_one_hot,
            asym_id=asym_id,
            residue_idx=residue_index,
            designable_mask=designable_token_mask.astype(jnp.float32),
            temperature=0.001,         # near-one-hot decode
            jacobi_iterations=5,
            key=jax.random.key(cfg.seed + 7 * outer),
        )
        diffusion_seq = jnp.argmax(soft_seq_decoded, axis=-1)
        diffusion_edits = int(((diffusion_seq != parent_seq_ids)
                               & designable_token_mask).sum())
        print(f"[outer {outer}] diffusion produced {diffusion_edits} edits vs parent")

        # ----- Stage 2: edit-budgeted greedy polish -----
        if cfg.skip_polish:
            polished_seq = np.asarray(diffusion_seq)
            polished_val = float("nan")
            iter_pareto = {diffusion_edits: (float("nan"), polished_seq)}
        else:
            print(f"[outer {outer}] edit-budgeted greedy polish "
                  f"(budget={cfg.edit_budget}, steps<={cfg.polish_steps})...")
            polished_seq, polished_val, iter_pareto = edit_budgeted_greedy_descent(
                loss=polish_loss,
                sequence=np.asarray(diffusion_seq),
                parent=np.asarray(parent_seq_ids),
                budget=cfg.edit_budget,
                designable_mask=np.asarray(designable_token_mask),
                batch_size=cfg.polish_batch_size,
                steps=cfg.polish_steps,
                key=jax.random.key(cfg.seed + 31337 * outer),
            )

        # ----- Update global Pareto front -----
        for k, (loss_v, seq) in iter_pareto.items():
            existing = global_pareto.get(k)
            if existing is None or loss_v < existing[0]:
                global_pareto[k] = (loss_v, seq.copy())

        # ----- Convergence check & parent update -----
        if cfg.skip_polish:
            converged = True
        else:
            new_one_hot = jnp.array(jax.nn.one_hot(polished_seq, 20))
            same_as_parent = bool(jnp.allclose(new_one_hot, current_parent_one_hot))
            current_parent_one_hot = new_one_hot
            current_initial_coords = x_final
            converged = same_as_parent
            if same_as_parent:
                print(f"[outer {outer}] no further improving edits — converged.")

        all_iterations.append({
            "outer": outer,
            "diffusion_edits": diffusion_edits,
            "polished_edits": int(((np.asarray(polished_seq)
                                    != np.asarray(parent_seq_ids))
                                   & np.asarray(designable_token_mask)).sum()),
            "polished_loss": polished_val,
            "elapsed_s": time.time() - t0,
        })

        if converged:
            break

    # ---- 5. Write outputs ----
    print("\n[output] writing Pareto front and per-iteration log...")
    pareto_rows = [
        {
            "edit_count": k,
            "loss": v[0],
            "sequence": "".join(TOKENS[i] for i in v[1]),
        }
        for k, v in sorted(global_pareto.items())
    ]
    pl.DataFrame(pareto_rows).write_csv(cfg.output_dir / "pareto_front.csv")
    pl.DataFrame(all_iterations).write_csv(cfg.output_dir / "iterations.csv")

    # ---- 6. Refold (task #15) ----
    if not cfg.skip_refold:
        print("[refold] refolding Pareto candidates with Boltz2...")
        refold_pareto_with_boltz2(global_pareto, cfg, parent_seq_ids,
                                  designable_token_mask)

    print(f"[done] outputs in {cfg.output_dir}")
    return global_pareto, all_iterations


# =============================================================================
# Refolding harness (task #15)
# =============================================================================


def refold_pareto_with_boltz2(
    pareto: dict[int, tuple[float, np.ndarray]],
    cfg: VHHDesignConfig,
    parent_seq_ids: np.ndarray,
    designable_token_mask: np.ndarray,
):
    """Refold each Pareto candidate with Boltz2 + score by iPTM/ipSAE.

    This is a thin orchestration around the existing reusable functions in
    `examples/boltzgen_pipeline.py`. We import them lazily here so v0/v1/v2 runs
    don't pay the Boltz2 import cost when refolding is disabled.
    """
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.losses.boltz2 import boltz2_trunk, boltz2_forward_from_trunk
    from mosaic.losses.structure_prediction import (
        IPTMLoss, BinderTargetIPSAE, TargetBinderIPSAE,
    )
    from mosaic.structure_prediction import TargetChain

    boltz2 = Boltz2()
    target_struct = gemmi.read_structure(str(cfg.complex_cif_path))
    target_struct.setup_entities()
    target_chains = []
    for cid in cfg.target_chain_ids:
        chain = target_struct[0][cid]
        seq = gemmi.one_letter_code([r.name for r in chain])
        target_chains.append(TargetChain(seq, use_msa=False, template_chain=chain))

    ranking_loss = (1.0 * IPTMLoss()
                    + 0.5 * TargetBinderIPSAE()
                    + 0.5 * BinderTargetIPSAE())

    rows = []
    for edit_count, (loss_v, seq_ids) in sorted(pareto.items()):
        seq_str = "".join(TOKENS[i] for i in seq_ids)
        # Refold the full sequence as the binder against the parent target. A
        # production version would split out just the binder chain explicitly,
        # but for ranking purposes this is sufficient.
        feats, w = boltz2.target_only_features(
            [TargetChain(seq_str, use_msa=False)] + target_chains
        )
        # Single trunk run + 5 diffusion samples — match boltzgen_pipeline.multifold
        key = jax.random.key(cfg.seed + 99999 + edit_count)
        initial_emb, trunk_state = boltz2_trunk(
            boltz2.model, feats, recycling_steps=cfg.recycling_steps,
            deterministic=True, key=fold_in(key, "trunk"),
        )
        out = boltz2_forward_from_trunk(
            boltz2.model, feats, initial_emb, trunk_state,
            num_sampling_steps=25, deterministic=True, key=fold_in(key, "sample"),
        )
        v, _ = ranking_loss(
            sequence=jnp.zeros((len(seq_str), 20)),
            output=out,
            key=fold_in(key, "loss"),
        )
        rows.append({
            "edit_count": edit_count,
            "polish_loss": loss_v,
            "refold_loss": float(v),
            "sequence": seq_str,
        })

    pl.DataFrame(sorted(rows, key=lambda r: r["refold_loss"])).write_csv(
        cfg.output_dir / "refold_ranked.csv"
    )


# =============================================================================
# CLI entry point
# =============================================================================


def _example_config() -> VHHDesignConfig:
    """An example config for the IL7Ra-targeting VHH (uses files in mosaic/)."""
    return VHHDesignConfig(
        complex_cif_path=Path(__file__).parent.parent / "IL7RA.cif",
        binder_chain_id="B",        # placeholder — adjust to your actual binder chain
        target_chain_ids=["A"],
        cdr_residue_indices=list(range(26, 33))     # CDR-H1 (Kabat ~26-32)
                            + list(range(52, 60))   # CDR-H2 (Kabat ~52-58)
                            + list(range(99, 110)), # CDR-H3 (Kabat ~99-110)
        edit_budget=7,
        n_outer_iterations=2,
        skip_refold=True,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["v0", "v1", "v2", "v3", "v4"], default="v3",
                   help="incremental milestone toggle")
    p.add_argument("--complex-cif", type=Path)
    p.add_argument("--binder-chain", default="B")
    p.add_argument("--target-chains", nargs="+", default=["A"])
    p.add_argument("--cdr-indices", nargs="+", type=int)
    p.add_argument("--budget", type=int, default=7)
    p.add_argument("--output-dir", type=Path, default=Path("./vhh_designs"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.complex_cif is not None:
        cfg = VHHDesignConfig(
            complex_cif_path=args.complex_cif,
            binder_chain_id=args.binder_chain,
            target_chain_ids=args.target_chains,
            cdr_residue_indices=args.cdr_indices or [],
            edit_budget=args.budget,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        cfg = _example_config()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed

    # Mode-driven flag presets
    if args.mode == "v0":
        cfg.skip_guidance = True
        cfg.skip_polish = True
        cfg.skip_refold = True
    elif args.mode == "v1":
        # EditBudget-only guidance; zero-out other guidance terms.
        cfg.skip_guidance = False
        cfg.skip_polish = True
        cfg.skip_refold = True
        cfg.weight_esmc = 0.0
        cfg.weight_ablang = 0.0
    elif args.mode == "v2":
        cfg.skip_guidance = False
        cfg.skip_polish = True
        cfg.skip_refold = True
    elif args.mode == "v3":
        cfg.skip_guidance = False
        cfg.skip_polish = False
        cfg.skip_refold = True
    elif args.mode == "v4":
        cfg.skip_guidance = False
        cfg.skip_polish = False
        cfg.skip_refold = False

    run(cfg)
