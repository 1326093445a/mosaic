"""Guided partial-diffusion + edit-budgeted CDR redesign for VHHs.

Pipeline (per outer-loop iteration):
  1. guided_partial_diffusion: BoltzGen partial diffusion with classifier guidance
     from AbLang2/EditBudget and an optional target-aware binding term
     (Boltz2 PAE/ipTM-based, or OpenDDE distogram-based -- mutually exclusive,
     see uses_boltz2_guidance/uses_opendde_guidance), perturbing CDR atoms only.
  2. Convert guided coordinates to soft sequences with JAX BoltzGen-IF; all
     sequence-loss gradients flow through this bridge back to coordinates.
  3. Greedy/MCMC search: polish the discrete sequence under the full multi-model
     loss with a hard <=budget edit constraint vs the parent. When native
     BoltzGen-IF is selected, its fixed-backbone logits contribute an NLL prior.
  4. Record a Pareto front {edit_count: (loss, sequence)}.
  5. Update parent if improved, repeat.

After convergence: refold final candidates (Boltz2 or OpenDDE, see
cfg.refold_backend -- defaults to OpenDDE), rank primarily by ipSAE, and write
interface metrics, target-aligned RMSD, and CIFs. OpenDDE's featurizer does
not support templates, unlike Boltz2's optional target/binder templating --
see opendde_refold_ignores_template_config.

Toggles for incremental milestones:
  v0 (task #7):  skip_guidance=True,  skip_polish=True
  v1 (task #11): skip_guidance=False, skip_polish=True, only EditBudget in guidance
  v2 (task #12): skip_guidance=False, skip_polish=True, full multi-model guidance
  v3 (task #14): all flags off
  v4 (task #15): all flags off, skip_refold=False

Inputs: an Ab-Ag complex CIF, the binder/target chain ids, and CDR position indices.
Outputs: per-iteration designs, a Pareto-front CSV, and (if not skipped)
Boltz2-refolded ranked structures with interface/RMSD metrics.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional


_STAGE_T0 = time.time()


def stage_log(message: str) -> None:
    """Print an elapsed-time stage marker for long cluster runs."""
    print(f"[stage {time.time() - _STAGE_T0:8.1f}s] {message}", flush=True)


stage_log("importing heavy dependencies")

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jaxtyping import Array, Bool, Float, Int

from mosaic.common import TOKENS, LossTerm
from mosaic.legacy.diagnostics import (
    format_summary_text,
    per_step_metrics,
    summarize as summarize_guidance_diagnostics,
    write_report as write_guidance_diagnostics_report,
)
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.transformations import (
    ClippedGradient,
    EditBudget,
    FixedLogitsSequenceLoss,
    SetPositions,
)
from mosaic.models.boltzgen import (
    Sampler,
    build_atom_partial_mask,
    guided_partial_diffusion,
    load_boltzgen,
    load_features_and_structure_writer,
)
from mosaic.legacy.boltzgen_if_jax import (
    decode_with_jax_boltzgen_if,
    differentiable_jax_boltzgen_if,
    load_jax_boltzgen_if,
    prepare_jax_boltzgen_if_context,
)
from mosaic.optimizers import (
    edit_budgeted_greedy_descent,
    edit_budgeted_gradient_mcmc,
)
from mosaic.util import fold_in

stage_log("finished importing heavy dependencies")


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
    boltzgen_yaml_path: Optional[Path] = None

    # ---- Constraint ----
    edit_budget: int = 7

    # ---- Guidance composite loss weights ----
    weight_ablang2: float = 0.10
    weight_edit_budget: float = 5.00
    weight_boltz2_ptm_energy: float = 0.0
    weight_boltz2_interface_pae: float = 0.0
    weight_boltz2_iptm: float = 0.0
    weight_boltz2_ipsae: float = 0.0
    boltz2_guidance_recycling_steps: int = 0
    boltz2_guidance_sampling_steps: int = 5
    boltz2_guidance_target_template: bool = True
    # OpenDDE alternative to Boltz2 guidance (mutually exclusive -- see
    # uses_boltz2_guidance/uses_opendde_guidance). PAE/pLDDT are scoring-only
    # for OpenDDE (see build_opendde_guidance_loss), so these read the
    # distogram instead of PAE.
    weight_opendde_iptm: float = 0.0
    weight_opendde_contact: float = 0.0
    opendde_guidance_recycling_steps: int = 4
    # No opendde_guidance_sampling_steps: both OpenDDE guidance losses are
    # distogram-only (build_opendde_guidance_loss uses
    # build_distogram_only_loss), which skips diffusion sampling entirely --
    # a sampling-steps knob here would be dead configuration.
    opendde_contact_distance: float = 8.0  # BinderTargetContact.contact_distance
    clip_gradient_norm: float = 1.0  # per-term gradient clip for balance

    # ---- Diffusion ----
    num_sampling_steps: int = 200
    start_sigma_frac: float = 0.4
    step_scale: float = 2.0
    noise_scale: float = 0.88
    lambda_max: float = 1.0
    lambda_schedule: str = "sigma_squared"  # one of {sigma_squared, sigma, constant}
    # Real iterative NOS-style prior-compatibility mechanism (section 12a of
    # docs/guidance_alphaseq_testing_notes.md) -- off by default (0 inner
    # steps = one-shot merge, unchanged behavior). A one-shot squared-distance
    # translation of NOS's KL term was tried first and found to be
    # mathematically a no-op; genuine inner iteration is required instead,
    # see guided_partial_diffusion's docstring in models/boltzgen.py.
    nos_inner_steps: int = 0
    nos_inner_step_size: float = 0.05
    nos_lambda_kl: float = 1.0
    nos_langevin_noise: float = 0.0  # 0 = deterministic inner descent
    # Real look-ahead mechanism (section 12b): differentiate through the
    # full denoiser wrt atom_coords_noisy instead of the frozen x0_hat.
    # Mutually exclusive with nos_inner_steps > 0 (raises ValueError).
    lookahead: bool = False

    # ---- Outer loop ----
    n_outer_iterations: int = 3

    # ---- Stage 2 polish ----
    search_mode: str = "greedy"  # one of {greedy, mcmc, both}
    polish_steps: int = 200
    polish_batch_size: int = 16
    mcmc_steps: int = 100
    mcmc_temp: float = 0.02
    mcmc_proposal_temp: float = 0.01
    mcmc_max_path_length: int = 2

    # ---- JAX BoltzGen inverse folding ----
    sequence_decoder: str = "boltzgen_if"
    boltzgen_if_checkpoint: Optional[Path] = None
    boltzgen_if_device: str = "auto"
    boltzgen_if_temperature: float = 0.3
    boltzgen_if_guidance_temperature: float = 0.3
    boltzgen_if_avoid: str = "C"
    weight_boltzgen_if_prior: float = 0.10

    # ---- I/O ----
    output_dir: Path = Path("./vhh_designs")

    # ---- Toggles for incremental testing ----
    skip_guidance: bool = False
    skip_polish: bool = False
    skip_refold: bool = True  # default off; task #15 wires this back in

    # ---- Phase 2 guidance diagnostics (docs/legacy/guidance_implementation_todo.md) ----
    log_guidance_diagnostics: bool = False
    guidance_diagnostics_cos_threshold: float = 0.0
    guidance_diagnostics_sigma_bins: int = 4

    # ---- Misc ----
    recycling_steps: int = 3
    refold_sampling_steps: int = 25
    refold_num_samples: int = 1
    refold_batch_size: int = 1
    refold_binder_template: bool = True
    refold_binder_template_mode: str = "full"  # one of {full, framework, none}
    ipsae_pae_cutoff: float = 12.0
    refold_rmsd_threshold: float = 2.5
    # one of {boltz2, opendde}. Defaults to opendde per explicit request --
    # The differentiable guidance path is GPU-smoke-tested; the full refold
    # path has a separate smoke in examples/opendde_refold_smoke_test.py and
    # still needs completion on a GPU with enough free VRAM. Override with
    # --refold-backend boltz2 when that validation requirement matters.
    refold_backend: str = "opendde"
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


def boltzgen_yaml_files(yaml_path: Path, yaml_string: str) -> dict[str, Path]:
    """Resolve file-backed entities referenced by a BoltzGen YAML."""
    parsed = yaml.safe_load(yaml_string)
    files: dict[str, Path] = {}
    for entity in parsed.get("entities", []):
        if not isinstance(entity, dict) or "file" not in entity:
            continue
        file_path = Path(entity["file"]["path"])
        source = file_path if file_path.is_absolute() else yaml_path.parent / file_path
        files[str(file_path)] = source
    return files


def squeeze_feature(array, name: str, ndim: int):
    """Remove leading singleton batch/multiplicity axes from a feature array."""
    arr = jnp.array(array)
    while arr.ndim > ndim:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Feature {name} has shape {arr.shape}; expected leading "
                f"singleton axes before {ndim}D data."
            )
        arr = arr[0]
    if arr.ndim != ndim:
        raise ValueError(
            f"Feature {name} has shape {arr.shape}; expected {ndim}D data."
        )
    return arr


def parent_one_hot_from_features(features: dict) -> Float[Array, "N 20"]:
    """Recover the parent (pre-mask) sequence as a one-hot over mosaic's TOKENS.

    BoltzGen's masker zeroes res_type at designable positions, but the unmasked
    parent identity is still available in `res_type_clone` (preserved by the masker
    at masker.py:101). We slice columns 2:22 to drop BoltzGen's special tokens and
    keep the 20 standard amino-acid columns, matching mosaic's TOKENS ordering.
    """
    res_type_clone = squeeze_feature(features["res_type_clone"], "res_type_clone", 2)
    return jnp.array(res_type_clone[:, 2:22], dtype=jnp.float32)


def cdr_token_mask_from_features(features: dict) -> Bool[Array, "N"]:
    """Token-level designable mask, sourced directly from the BoltzGen featurizer."""
    return jnp.array(squeeze_feature(features["design_mask"], "design_mask", 1), dtype=bool)


def binder_indices_from_design_mask(
    asym_id: Int[Array, "N"],
    designable_token_mask: Bool[Array, "N"],
) -> Int[Array, "M"]:
    """Infer binder tokens as chains containing at least one designable residue."""
    asym_id_np = np.asarray(asym_id)
    design_mask_np = np.asarray(designable_token_mask, dtype=bool)
    binder_asym_ids = np.unique(asym_id_np[design_mask_np])
    if binder_asym_ids.size == 0:
        raise ValueError("No designable residues found; cannot infer binder chain.")
    return jnp.asarray(np.where(np.isin(asym_id_np, binder_asym_ids))[0], dtype=jnp.int32)


def lambda_schedule_fn(name: str, lam_max: float):
    if name == "sigma_squared":
        return lambda sigma: lam_max * (sigma ** 2)
    if name == "sigma":
        return lambda sigma: lam_max * sigma
    if name == "constant":
        return lambda sigma: lam_max * jnp.ones_like(sigma)
    raise ValueError(f"unknown lambda_schedule: {name}")


@dataclass(frozen=True)
class AtomRecord:
    chain_id: str
    residue_key: tuple[str, int, str, str]
    residue_name: str
    atom_name: str
    element: str
    coord: np.ndarray


def _atom_coord(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def _element_name(atom: gemmi.Atom) -> str:
    name = getattr(atom.element, "name", "")
    if name:
        return str(name).upper()
    atom_name = atom.name.strip()
    return atom_name[0].upper() if atom_name else ""


def _heavy_atoms_by_role(
    structure: gemmi.Structure,
    binder_chain_id: str,
    target_chain_ids: list[str],
) -> tuple[list[AtomRecord], list[AtomRecord]]:
    """Collect heavy atoms for binder and target chains from a gemmi structure."""
    binder_atoms: list[AtomRecord] = []
    target_atoms: list[AtomRecord] = []
    target_set = set(target_chain_ids)

    for chain in structure[0]:
        is_binder = chain.name == binder_chain_id
        is_target = chain.name in target_set
        if not is_binder and not is_target:
            continue
        for residue in chain:
            seq_num = int(residue.seqid.num)
            icode = residue.seqid.icode.strip()
            residue_key = (chain.name, seq_num, icode, residue.name)
            for atom in residue:
                element = _element_name(atom)
                if element == "H" or atom.name.strip().upper().startswith("H"):
                    continue
                record = AtomRecord(
                    chain_id=chain.name,
                    residue_key=residue_key,
                    residue_name=residue.name,
                    atom_name=atom.name.strip(),
                    element=element,
                    coord=_atom_coord(atom),
                )
                if is_binder:
                    binder_atoms.append(record)
                else:
                    target_atoms.append(record)
    return binder_atoms, target_atoms


def _close_atom_pairs(
    left: list[AtomRecord],
    right: list[AtomRecord],
    cutoff: float,
    *,
    chunk_size: int = 512,
) -> list[tuple[int, int]]:
    if not left or not right:
        return []

    left_coords = np.stack([a.coord for a in left])
    right_coords = np.stack([a.coord for a in right])
    cutoff_sq = cutoff * cutoff
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(left), chunk_size):
        chunk = left_coords[start:start + chunk_size]
        d2 = np.sum((chunk[:, None, :] - right_coords[None, :, :]) ** 2, axis=-1)
        close = np.argwhere(d2 <= cutoff_sq)
        pairs.extend((start + int(i), int(j)) for i, j in close)
    return pairs


def _is_positive_salt_atom(atom: AtomRecord) -> bool:
    return (
        (atom.residue_name == "LYS" and atom.atom_name == "NZ")
        or (atom.residue_name == "ARG" and atom.atom_name in {"NE", "NH1", "NH2"})
        or (atom.residue_name == "HIS" and atom.atom_name in {"ND1", "NE2"})
    )


def _is_negative_salt_atom(atom: AtomRecord) -> bool:
    return (
        (atom.residue_name == "ASP" and atom.atom_name in {"OD1", "OD2"})
        or (atom.residue_name == "GLU" and atom.atom_name in {"OE1", "OE2"})
    )


def _is_hydrophobic_atom(atom: AtomRecord) -> bool:
    hydrophobic_res = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    return (
        atom.element == "C"
        and atom.residue_name in hydrophobic_res
        and atom.atom_name not in backbone_atoms
    )


def interface_geometry_metrics(
    structure: gemmi.Structure,
    *,
    binder_chain_id: str = "A",
    target_chain_ids: Optional[list[str]] = None,
) -> dict[str, float | int]:
    """Compute lightweight BoltzGen-like interface metrics on a refolded complex.

    These are geometry heuristics, not a full PLIP run. They are useful for
    ranking/diagnostics when we do not run the full BoltzGen analysis task.
    """
    if target_chain_ids is None:
        chain_ids = [chain.name for chain in structure[0]]
        target_chain_ids = [cid for cid in chain_ids if cid != binder_chain_id]

    binder_atoms, target_atoms = _heavy_atoms_by_role(
        structure, binder_chain_id, target_chain_ids
    )

    contact_pairs = _close_atom_pairs(binder_atoms, target_atoms, cutoff=4.5)
    contact_residue_pairs = {
        (binder_atoms[i].residue_key, target_atoms[j].residue_key)
        for i, j in contact_pairs
    }

    polar = {"N", "O", "S"}
    hbond_pairs = [
        (i, j)
        for i, j in _close_atom_pairs(binder_atoms, target_atoms, cutoff=3.5)
        if binder_atoms[i].element in polar and target_atoms[j].element in polar
    ]
    hbond_residue_pairs = {
        (binder_atoms[i].residue_key, target_atoms[j].residue_key)
        for i, j in hbond_pairs
    }

    salt_pairs = [
        (i, j)
        for i, j in _close_atom_pairs(binder_atoms, target_atoms, cutoff=5.5)
        if (
            _is_positive_salt_atom(binder_atoms[i])
            and _is_negative_salt_atom(target_atoms[j])
        )
        or (
            _is_negative_salt_atom(binder_atoms[i])
            and _is_positive_salt_atom(target_atoms[j])
        )
    ]
    salt_residue_pairs = {
        (binder_atoms[i].residue_key, target_atoms[j].residue_key)
        for i, j in salt_pairs
    }

    hydrophobic_pairs = [
        (i, j)
        for i, j in _close_atom_pairs(binder_atoms, target_atoms, cutoff=4.5)
        if _is_hydrophobic_atom(binder_atoms[i])
        and _is_hydrophobic_atom(target_atoms[j])
    ]
    hydrophobic_residue_pairs = {
        (binder_atoms[i].residue_key, target_atoms[j].residue_key)
        for i, j in hydrophobic_pairs
    }

    interaction_score = (
        len(hbond_residue_pairs)
        + len(salt_residue_pairs)
        + len(hydrophobic_residue_pairs)
    )
    return {
        "geom_interface_atom_contacts_refolded": len(contact_pairs),
        "geom_interface_residue_contacts_refolded": len(contact_residue_pairs),
        "geom_hbonds_refolded": len(hbond_residue_pairs),
        "geom_hbond_atom_pairs_refolded": len(hbond_pairs),
        "geom_saltbridges_refolded": len(salt_residue_pairs),
        "geom_saltbridge_atom_pairs_refolded": len(salt_pairs),
        "geom_hydrophobic_contacts_refolded": len(hydrophobic_residue_pairs),
        "geom_hydrophobic_atom_pairs_refolded": len(hydrophobic_pairs),
        "geom_interaction_score_refolded": interaction_score,
    }


def _ca_coords_by_chain(
    structure: gemmi.Structure,
    chain_ids: list[str],
) -> tuple[np.ndarray, list[tuple[str, int, str, str]]]:
    wanted = set(chain_ids)
    coords = []
    keys = []
    for chain in structure[0]:
        if chain.name not in wanted:
            continue
        for residue in chain:
            for atom in residue:
                if atom.name.strip() != "CA":
                    continue
                coords.append(_atom_coord(atom))
                keys.append(
                    (
                        chain.name,
                        int(residue.seqid.num),
                        residue.seqid.icode.strip(),
                        residue.name,
                    )
                )
                break
    if not coords:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.stack(coords), keys


def _fit_transform(mobile: np.ndarray, reference: np.ndarray):
    n = min(len(mobile), len(reference))
    if n == 0:
        return None, None
    mobile = mobile[:n]
    reference = reference[:n]
    mobile_mean = mobile.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    mobile_centered = mobile - mobile_mean
    reference_centered = reference - reference_mean
    u, _, vt = np.linalg.svd(mobile_centered.T @ reference_centered)
    # Row-vector Kabsch: coordinates are transformed as coords @ rotation.
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    translation = reference_mean - mobile_mean @ rotation
    return rotation, translation


def _apply_transform(coords: np.ndarray, rotation, translation) -> np.ndarray:
    if rotation is None or translation is None or len(coords) == 0:
        return coords
    return coords @ rotation + translation


def _rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    n = min(len(coords_a), len(coords_b))
    if n == 0:
        return float("nan")
    diff = coords_a[:n] - coords_b[:n]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def target_aligned_rmsd_metrics(
    original_structure: gemmi.Structure,
    refolded_structure: gemmi.Structure,
    *,
    original_binder_chain_id: str,
    original_target_chain_ids: list[str],
    refolded_binder_chain_id: str = "A",
    refolded_target_chain_ids: Optional[list[str]] = None,
    cdr_residue_indices: Optional[list[int]] = None,
) -> dict[str, float | int]:
    """Align refolded target onto the input target, then score binder movement."""
    if refolded_target_chain_ids is None:
        refolded_chain_ids = [chain.name for chain in refolded_structure[0]]
        refolded_target_chain_ids = [
            cid for cid in refolded_chain_ids if cid != refolded_binder_chain_id
        ]

    orig_target, _ = _ca_coords_by_chain(original_structure, original_target_chain_ids)
    ref_target, _ = _ca_coords_by_chain(refolded_structure, refolded_target_chain_ids)
    orig_binder, orig_binder_keys = _ca_coords_by_chain(
        original_structure, [original_binder_chain_id]
    )
    ref_binder, _ = _ca_coords_by_chain(refolded_structure, [refolded_binder_chain_id])

    orig_complex = np.concatenate([orig_binder, orig_target], axis=0)
    ref_complex = np.concatenate([ref_binder, ref_target], axis=0)

    complex_rotation, complex_translation = _fit_transform(ref_complex, orig_complex)
    ref_complex_aligned = _apply_transform(
        ref_complex, complex_rotation, complex_translation
    )

    binder_rotation, binder_translation = _fit_transform(ref_binder, orig_binder)
    ref_binder_self_aligned = _apply_transform(
        ref_binder, binder_rotation, binder_translation
    )

    target_rotation, target_translation = _fit_transform(ref_target, orig_target)
    ref_target_aligned = _apply_transform(ref_target, target_rotation, target_translation)
    ref_binder_target_aligned = _apply_transform(
        ref_binder, target_rotation, target_translation
    )

    metrics = {
        # BoltzGen-like RMSD filter columns. In BoltzGen Filter defaults,
        # from_inverse_folded=True maps filter_rmsd <- bb_rmsd and
        # filter_rmsd_design <- bb_rmsd_design. Here we use CA-only analogues
        # because refolded gemmi structures do not retain the full BoltzGen
        # atom-mask feature tensors.
        "complex_ca_rmsd": _rmsd(ref_complex_aligned, orig_complex),
        "binder_ca_rmsd_self_aligned": _rmsd(ref_binder_self_aligned, orig_binder),
        "filter_rmsd": _rmsd(ref_complex_aligned, orig_complex),
        "filter_rmsd_design": _rmsd(ref_binder_self_aligned, orig_binder),
        "target_ca_rmsd_target_aligned": _rmsd(ref_target_aligned, orig_target),
        "binder_ca_rmsd_target_aligned": _rmsd(
            ref_binder_target_aligned, orig_binder
        ),
        "bb_target_aligned<2.5": _passes_max_threshold(
            _rmsd(ref_binder_target_aligned, orig_binder), 2.5
        ),
        "complex_ca_rmsd_n": min(len(ref_complex), len(orig_complex)),
        "target_ca_rmsd_n": min(len(ref_target), len(orig_target)),
        "binder_ca_rmsd_n": min(len(ref_binder), len(orig_binder)),
    }

    if cdr_residue_indices:
        # Mosaic/BoltzGen YAML `res_index` values are 1-based chain-order
        # positions, not necessarily PDB author residue numbers. This matters
        # for VHHs with Kabat insertion codes such as H52A or H100A-H.
        cdr_set = {int(i) for i in cdr_residue_indices}
        cdr_positions = [
            i for i in range(min(len(orig_binder_keys), len(ref_binder_target_aligned)))
            if (i + 1) in cdr_set
        ]
        if cdr_positions:
            metrics["cdr_ca_rmsd_target_aligned"] = _rmsd(
                ref_binder_target_aligned[cdr_positions],
                orig_binder[cdr_positions],
            )
            metrics["cdr_ca_rmsd_n"] = len(cdr_positions)
        else:
            metrics["cdr_ca_rmsd_target_aligned"] = float("nan")
            metrics["cdr_ca_rmsd_n"] = 0
    return metrics


def mask_binder_template_design_residues(
    chain: gemmi.Chain,
    design_residue_indices: list[int],
) -> tuple[gemmi.Chain, int]:
    """Keep binder framework templated while removing CDR/design coordinates.

    The chain sequence and residue slots remain present, but residues whose
    1-based chain-order positions are designable are made atomless. Boltz2 then
    gets framework coordinates without a structural template for mutable CDRs.
    """
    masked = chain.clone()
    design_set = {int(i) for i in design_residue_indices}
    removed_atoms = 0
    for position, residue in enumerate(masked, start=1):
        if position not in design_set:
            continue
        while len(residue):
            del residue[0]
            removed_atoms += 1
    return masked, removed_atoms


def write_structure_cif(structure: gemmi.Structure, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = structure.make_mmcif_document()
    doc.write_file(str(output_path))


def _rank_value(value, *, descending: bool):
    if value is None:
        return float("inf")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return -value if descending else value


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _passes_max_threshold(value, threshold: float) -> bool:
    """Return True when value is finite and <= threshold; threshold <= 0 disables."""
    if threshold <= 0:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value <= threshold


def _refold_rank_key(row: dict):
    return (
        not _truthy(row.get("rmsd_pass")),
        _rank_value(row.get("ipsae_min"), descending=True),
        _rank_value(row.get("iptm"), descending=True),
        _rank_value(row.get("ipae_min"), descending=False),
        _rank_value(row.get("geom_interaction_score_refolded"), descending=True),
        _rank_value(row.get("binder_ca_rmsd_target_aligned"), descending=False),
    )


def parse_device_ids(raw: Optional[str]) -> list[str]:
    """Parse a BoltzGen-like devices argument for job-level GPU fan-out."""
    if raw is None or str(raw).strip() == "":
        return []

    raw = str(raw).strip()
    if raw.lower() == "auto":
        count = jax.local_device_count()
        return [str(i) for i in range(count)]

    if raw.isdigit():
        return [str(i) for i in range(int(raw))]

    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_child_cuda_visible_devices(device: Optional[str]) -> Optional[str]:
    """Map a requested child device through the parent's visible-device list.

    If the parent was launched with CUDA_VISIBLE_DEVICES=1,2,3 and the user passes
    --devices 0,1,2, the child jobs should land on physical GPUs 1,2,3 rather
    than accidentally escaping to physical GPU 0.
    """
    if device is None:
        return None

    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not parent_visible or parent_visible in {"NoDevFiles", "-1"}:
        return device

    visible = [part.strip() for part in parent_visible.split(",") if part.strip()]
    if device.isdigit():
        idx = int(device)
        if 0 <= idx < len(visible):
            return visible[idx]
    return device


def write_combined_refold_ranking(root: Path, manifest_rows: list[dict]) -> int:
    """Combine per-seed refold_ranked.csv files into one root-level ranking CSV."""
    combined_rows = []
    for manifest_row in manifest_rows:
        if manifest_row.get("status") != "ok":
            continue
        out_dir = Path(manifest_row["output_dir"])
        ranked_csv = out_dir / "refold_ranked.csv"
        if not ranked_csv.exists():
            continue

        with ranked_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                combined_rows.append(
                    {
                        "seed": manifest_row["seed"],
                        "local_rank": row.get("rank", ""),
                        "edit_count": row.get("edit_count", ""),
                        "sample_idx": row.get("sample_idx", ""),
                        "sequence": row.get("sequence", ""),
                        "ipsae_min": row.get("ipsae_min", ""),
                        "iptm": row.get("iptm", ""),
                        "ipae_min": row.get("ipae_min", ""),
                        "bt_ipsae": row.get("bt_ipsae", ""),
                        "tb_ipsae": row.get("tb_ipsae", ""),
                        "rmsd_pass": row.get("rmsd_pass", ""),
                        "refold_cif": row.get("refold_cif", ""),
                        "output_dir": str(out_dir),
                        "log": manifest_row.get("log", ""),
                    }
                )

    columns = [
        "rank",
        "seed",
        "local_rank",
        "edit_count",
        "sample_idx",
        "sequence",
        "ipsae_min",
        "iptm",
        "ipae_min",
        "bt_ipsae",
        "tb_ipsae",
        "rmsd_pass",
        "refold_cif",
        "output_dir",
        "log",
    ]

    passing_rows = sorted(
        [row for row in combined_rows if _truthy(row.get("rmsd_pass"))],
        key=_refold_rank_key,
    )
    failed_rows = sorted(
        [row for row in combined_rows if not _truthy(row.get("rmsd_pass"))],
        key=_refold_rank_key,
    )
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(passing_rows, start=1)
    ] + [
        {"rank": None, **row}
        for row in failed_rows
    ]

    output_path = root / "combined_refold_ranked.csv"
    if ranked_rows:
        normalized_rows = [
            {
                column: "" if row.get(column) is None else str(row.get(column, ""))
                for column in columns
            }
            for row in ranked_rows
        ]
        pl.DataFrame(
            normalized_rows,
            schema={column: pl.String for column in columns},
        ).write_csv(output_path)
    else:
        output_path.write_text(",".join(columns) + "\n")
    return len(ranked_rows)


def _append_option(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def uses_boltz2_guidance(cfg: VHHDesignConfig) -> bool:
    # weight_boltz2_ipsae is deliberately excluded: ipSAE is post-refold-only
    # (see build_boltz2_guidance_loss), so it never triggers in-loop Boltz2
    # guidance on its own.
    return (
        cfg.weight_boltz2_ptm_energy > 0
        or cfg.weight_boltz2_interface_pae > 0
        or cfg.weight_boltz2_iptm > 0
    )


def uses_opendde_guidance(cfg: VHHDesignConfig) -> bool:
    return cfg.weight_opendde_iptm > 0 or cfg.weight_opendde_contact > 0


def guidance_anchor_is_empty_edit_budget(cfg: VHHDesignConfig) -> bool:
    """True when build_guidance_loss will promote edit_loss into the `bind`
    anchor slot (no Boltz2/OpenDDE binding signal configured -- see its
    fallback branch) *and* that promoted anchor is itself a no-op because
    `weight_edit_budget <= 0`.

    In that case guided_partial_diffusion still runs (grad_bind is not None
    -- edit_loss exists as an object), but its gradient is all-zero, so
    `_compat_project(g_nat, g_bind=0)` returns g_nat completely unprojected
    (a zero anchor has nothing to conflict with) and g_total collapses to
    `alpha_fn(t_hat) * g_nat` alone. Guidance silently becomes
    naturalness-only despite `GuidanceLosses.bind`'s docstring calling it
    "the anchor objective guidance is built around" -- see the warning in
    run() that uses this."""
    return (
        not cfg.skip_guidance
        and not uses_boltz2_guidance(cfg)
        and not uses_opendde_guidance(cfg)
        and cfg.weight_edit_budget <= 0
    )


def opendde_refold_ignores_template_config(cfg: VHHDesignConfig) -> bool:
    """True when refold_binder_template config implies templated refolding
    that --refold-backend opendde silently can't do (its featurizer rejects
    templates outright -- see models/opendde.py). Both refold_binder_template
    fields default to template-on, so this is true for most default-config
    OpenDDE refold runs -- see the warning in run() that uses this."""
    return (
        cfg.refold_backend == "opendde"
        and cfg.refold_binder_template
        and cfg.refold_binder_template_mode != "none"
    )


def build_single_design_command(args, *, seed: int, output_dir: Path) -> list[str]:
    """Reconstruct the CLI for one child design job."""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "mosaic.legacy.boltzgen_vhh_guided",
        "--mode",
        args.mode,
    ]

    _append_option(cmd, "--complex-cif", args.complex_cif)
    _append_option(cmd, "--boltzgen-yaml", args.boltzgen_yaml)
    _append_option(cmd, "--binder-chain", args.binder_chain)

    if args.target_chains:
        cmd.append("--target-chains")
        cmd.extend(str(x) for x in args.target_chains)
    if args.cdr_indices:
        cmd.append("--cdr-indices")
        cmd.extend(str(x) for x in args.cdr_indices)

    _append_option(cmd, "--budget", args.budget)
    _append_option(cmd, "--output-dir", output_dir)
    _append_option(cmd, "--seed", seed)
    _append_option(cmd, "--num-sampling-steps", args.num_sampling_steps)
    _append_option(cmd, "--start-sigma-frac", args.start_sigma_frac)
    _append_option(cmd, "--step-scale", args.step_scale)
    _append_option(cmd, "--noise-scale", args.noise_scale)
    _append_option(cmd, "--lambda-max", args.lambda_max)
    _append_option(cmd, "--lambda-schedule", args.lambda_schedule)
    _append_option(cmd, "--nos-inner-steps", args.nos_inner_steps)
    _append_option(cmd, "--nos-inner-step-size", args.nos_inner_step_size)
    _append_option(cmd, "--nos-lambda-kl", args.nos_lambda_kl)
    _append_option(cmd, "--nos-langevin-noise", args.nos_langevin_noise)
    _append_option(cmd, "--lookahead", args.lookahead)
    _append_option(cmd, "--n-outer-iterations", args.n_outer_iterations)
    _append_option(cmd, "--search-mode", args.search_mode)
    _append_option(cmd, "--polish-steps", args.polish_steps)
    _append_option(cmd, "--polish-batch-size", args.polish_batch_size)
    _append_option(cmd, "--mcmc-steps", args.mcmc_steps)
    _append_option(cmd, "--mcmc-temp", args.mcmc_temp)
    _append_option(cmd, "--mcmc-proposal-temp", args.mcmc_proposal_temp)
    _append_option(cmd, "--mcmc-max-path-length", args.mcmc_max_path_length)
    _append_option(cmd, "--sequence-decoder", args.sequence_decoder)
    _append_option(cmd, "--boltzgen-if-checkpoint", args.boltzgen_if_checkpoint)
    _append_option(cmd, "--boltzgen-if-device", args.boltzgen_if_device)
    _append_option(cmd, "--boltzgen-if-temperature", args.boltzgen_if_temperature)
    _append_option(
        cmd,
        "--boltzgen-if-guidance-temperature",
        args.boltzgen_if_guidance_temperature,
    )
    _append_option(cmd, "--boltzgen-if-avoid", args.boltzgen_if_avoid)
    _append_option(
        cmd,
        "--weight-boltzgen-if-prior",
        args.weight_boltzgen_if_prior,
    )
    _append_option(cmd, "--recycling-steps", args.recycling_steps)
    _append_option(cmd, "--refold-sampling-steps", args.refold_sampling_steps)
    _append_option(cmd, "--refold-num-samples", args.refold_num_samples)
    _append_option(cmd, "--refold-batch-size", args.refold_batch_size)
    _append_option(cmd, "--refold-binder-template", args.refold_binder_template)
    _append_option(
        cmd,
        "--refold-binder-template-mode",
        args.refold_binder_template_mode,
    )
    _append_option(cmd, "--ipsae-pae-cutoff", args.ipsae_pae_cutoff)
    _append_option(cmd, "--refold-rmsd-threshold", args.refold_rmsd_threshold)
    _append_option(cmd, "--weight-ablang2", args.weight_ablang2)
    _append_option(cmd, "--weight-edit-budget", args.weight_edit_budget)
    _append_option(cmd, "--weight-boltz2-ptm-energy", args.weight_boltz2_ptm_energy)
    _append_option(
        cmd,
        "--weight-boltz2-interface-pae",
        args.weight_boltz2_interface_pae,
    )
    _append_option(cmd, "--weight-boltz2-iptm", args.weight_boltz2_iptm)
    _append_option(cmd, "--weight-boltz2-ipsae", args.weight_boltz2_ipsae)
    _append_option(
        cmd,
        "--boltz2-guidance-recycling-steps",
        args.boltz2_guidance_recycling_steps,
    )
    _append_option(
        cmd,
        "--boltz2-guidance-sampling-steps",
        args.boltz2_guidance_sampling_steps,
    )
    _append_option(
        cmd,
        "--boltz2-guidance-target-template",
        args.boltz2_guidance_target_template,
    )
    _append_option(cmd, "--weight-opendde-iptm", args.weight_opendde_iptm)
    _append_option(cmd, "--weight-opendde-contact", args.weight_opendde_contact)
    _append_option(
        cmd,
        "--opendde-guidance-recycling-steps",
        args.opendde_guidance_recycling_steps,
    )
    _append_option(cmd, "--opendde-contact-distance", args.opendde_contact_distance)
    _append_option(cmd, "--refold-backend", args.refold_backend)
    _append_option(cmd, "--clip-gradient-norm", args.clip_gradient_norm)
    _append_option(cmd, "--skip-guidance", args.skip_guidance)
    _append_option(cmd, "--skip-polish", args.skip_polish)
    _append_option(cmd, "--skip-refold", args.skip_refold)
    _append_option(cmd, "--log-guidance-diagnostics", args.log_guidance_diagnostics)
    _append_option(
        cmd,
        "--guidance-diagnostics-cos-threshold",
        args.guidance_diagnostics_cos_threshold,
    )
    _append_option(
        cmd,
        "--guidance-diagnostics-sigma-bins",
        args.guidance_diagnostics_sigma_bins,
    )
    return cmd


def seed_output_complete(out_dir: Path, *, skip_refold: bool) -> bool:
    """Return True when a seed directory has the expected terminal output."""
    expected = out_dir / ("pareto_front.csv" if skip_refold else "refold_ranked.csv")
    return expected.exists() and expected.stat().st_size > 0


def run_many_skip_refold(args) -> bool:
    """Resolve child skip-refold behavior for resume completeness checks."""
    if args.skip_refold is not None:
        return bool(args.skip_refold)
    return args.mode in {"v0", "v1", "v2", "v3"}


def run_many_from_cli(args):
    """Run multiple independent designs from the core Python entry point.

    This is job-level multi-GPU orchestration: each child process sees one GPU via
    CUDA_VISIBLE_DEVICES. It is intentionally different from DDP inside one JAX
    process, because the design seeds are independent.
    """
    num_designs = int(args.num_designs)
    start_seed = args.start_seed if args.start_seed is not None else args.seed
    device_ids = parse_device_ids(args.devices)
    max_parallel = len(device_ids) if device_ids else 1

    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resume = bool(args.resume)
    skip_refold_for_completion = run_many_skip_refold(args)

    print("[multi] launching independent design jobs")
    print(f"[multi] output root: {root}")
    print(f"[multi] num_designs: {num_designs}")
    print(f"[multi] start_seed: {start_seed}")
    print(f"[multi] resume: {resume}")
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent_visible:
        print(f"[multi] parent CUDA_VISIBLE_DEVICES: {parent_visible}")
    print(f"[multi] devices: {','.join(device_ids) if device_ids else 'inherited'}")
    print(f"[multi] max_parallel: {max_parallel}")

    manifest_rows = []
    failures = []

    for batch_start in range(0, num_designs, max_parallel):
        launched = []
        batch_end = min(num_designs, batch_start + max_parallel)
        for design_idx in range(batch_start, batch_end):
            seed = start_seed + design_idx
            device = device_ids[design_idx % len(device_ids)] if device_ids else None
            out_dir = root / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "driver.log"
            child_cuda_visible = resolve_child_cuda_visible_devices(device)
            if resume and seed_output_complete(
                out_dir,
                skip_refold=skip_refold_for_completion,
            ):
                print(
                    f"[multi] resume skip seed={seed} status=ok "
                    f"log={log_path}",
                    flush=True,
                )
                manifest_rows.append(
                    {
                        "seed": seed,
                        "device": device if device is not None else "",
                        "cuda_visible_devices": (
                            child_cuda_visible if child_cuda_visible is not None else ""
                        ),
                        "output_dir": str(out_dir),
                        "log": str(log_path),
                        "returncode": 0,
                        "status": "ok",
                        "resumed": True,
                    }
                )
                continue

            cmd = build_single_design_command(args, seed=seed, output_dir=out_dir)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if child_cuda_visible is not None:
                env["CUDA_VISIBLE_DEVICES"] = child_cuda_visible

            log_handle = log_path.open("w")
            print(
                f"[multi] start seed={seed} device={device or 'inherited'} "
                f"cuda_visible={env.get('CUDA_VISIBLE_DEVICES', 'inherited')} "
                f"-> {out_dir}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            launched.append(
                (proc, log_handle, seed, device, child_cuda_visible, out_dir, log_path)
            )

        for proc, log_handle, seed, device, child_cuda_visible, out_dir, log_path in launched:
            ret = proc.wait()
            log_handle.close()
            status = "ok" if ret == 0 else "failed"
            print(f"[multi] done seed={seed} status={status} log={log_path}", flush=True)
            row = {
                "seed": seed,
                "device": device if device is not None else "",
                "cuda_visible_devices": (
                    child_cuda_visible if child_cuda_visible is not None else ""
                ),
                "output_dir": str(out_dir),
                "log": str(log_path),
                "returncode": ret,
                "status": status,
                "resumed": False,
            }
            manifest_rows.append(row)
            if ret != 0:
                failures.append(row)

    pl.DataFrame(manifest_rows).write_csv(root / "multi_design_manifest.csv")
    combined_count = write_combined_refold_ranking(root, manifest_rows)
    print(
        f"[multi] wrote {combined_count} combined ranked refold row(s) -> "
        f"{root / 'combined_refold_ranked.csv'}"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} design job(s) failed; see "
            f"{root / 'multi_design_manifest.csv'}"
        )


# =============================================================================
# Model loading (one-time setup)
# =============================================================================


@dataclass
class LoadedModels:
    boltzgen: any
    boltzgen_if_jax: any = None
    boltzgen_if_torch: any = None
    ablang2_model: any = None
    ablang2_tokenizer: any = None
    boltz2: any = None
    opendde: any = None


def load_core_models(cfg: VHHDesignConfig) -> LoadedModels:
    """Load only the BoltzGen model needed to build the sampler/trunk state."""
    del cfg
    stage_log("loading BoltzGen model")
    boltzgen = load_boltzgen()
    stage_log("loaded BoltzGen model")
    return LoadedModels(boltzgen=boltzgen)


def load_guidance_models(cfg: VHHDesignConfig, models: LoadedModels) -> LoadedModels:
    """Load IF/language guidance models after BoltzGen trunk conditioning.

    `Sampler.from_features` has a large compile/runtime peak. Deferring AbLang2
    until after that stage avoids keeping those model weights resident during
    the BoltzGen sampler allocation.
    """
    # stop_grad=False on the language model because we need gradients to flow
    # back to coords through the differentiable IF bridge during guidance.
    if cfg.weight_ablang2 > 0 and models.ablang2_model is None:
        stage_log("loading AbLang2 model")
        models.ablang2_model, models.ablang2_tokenizer = load_ablang2()
        stage_log("loaded AbLang2 model")
    elif cfg.weight_ablang2 <= 0:
        stage_log("skipping AbLang2 model because weight_ablang2 <= 0")

    if uses_boltz2_guidance(cfg) and uses_opendde_guidance(cfg):
        raise ValueError(
            "Both Boltz2 (--weight-boltz2-*) and OpenDDE (--weight-opendde-*) "
            "in-loop guidance weights are set; only one can own the bind "
            "objective. Zero out one backend's weights."
        )

    if uses_boltz2_guidance(cfg) and models.boltz2 is None:
        stage_log("loading Boltz2 model for target-aware guidance")
        from mosaic.models.boltz2 import Boltz2

        models.boltz2 = Boltz2()
        stage_log("loaded Boltz2 model for target-aware guidance")
    elif not uses_boltz2_guidance(cfg):
        stage_log("skipping Boltz2 guidance model because its weights are <= 0")

    if uses_opendde_guidance(cfg) and models.opendde is None:
        stage_log("loading OpenDDE (Abag) model for target-aware guidance")
        from mosaic.models.opendde import OpenDDEModelAbag

        models.opendde = OpenDDEModelAbag()
        stage_log("loaded OpenDDE (Abag) model for target-aware guidance")
    elif not uses_opendde_guidance(cfg):
        stage_log("skipping OpenDDE guidance model because its weights are <= 0")
    return models


def ensure_boltzgen_if_loaded(
    cfg: VHHDesignConfig, models: LoadedModels
) -> LoadedModels:
    """Load native IF only when the final decoder first needs it."""
    if models.boltzgen_if_jax is None:
        checkpoint = (
            str(cfg.boltzgen_if_checkpoint)
            if cfg.boltzgen_if_checkpoint is not None
            else "official default"
        )
        stage_log(
            "loading native BoltzGen-IF "
            f"(checkpoint={checkpoint}, device={cfg.boltzgen_if_device})"
        )
        models.boltzgen_if_jax, models.boltzgen_if_torch = load_jax_boltzgen_if(
            cfg.boltzgen_if_checkpoint,
            torch_device=cfg.boltzgen_if_device,
        )
        stage_log("loaded native BoltzGen-IF weights into JAX")
    return models


def load_all_models(cfg: VHHDesignConfig) -> LoadedModels:
    """Load every model the driver uses, kept for non-staged call sites."""
    models = load_core_models(cfg)
    return load_guidance_models(cfg, models)


# =============================================================================
# Composite loss construction
# =============================================================================


class SequenceSubsetLoss(LossTerm):
    """Apply a sequence-only loss to selected token positions."""

    loss: LossTerm
    indices: Int[Array, "M"] = eqx.field(converter=jnp.array)

    def __call__(self, seq: Float[Array, "N 20"], *, key):
        return self.loss(seq[self.indices], key=key)


def make_ablang2_loss(
    models: LoadedModels,
    sequence_loss_indices: Int[Array, "M"],
    *,
    stop_grad: bool,
) -> Ablang2PseudoLikelihood:
    if models.ablang2_model is None or models.ablang2_tokenizer is None:
        raise ValueError("AbLang2 model was not loaded; check weight_ablang2")
    return Ablang2PseudoLikelihood(
        model=models.ablang2_model,
        tokenizer=models.ablang2_tokenizer,
        heavy_len=int(sequence_loss_indices.shape[0]),
        stop_grad=stop_grad,
    )


def build_boltz2_guidance_loss(
    cfg: VHHDesignConfig,
    models: LoadedModels,
    binder_length: int,
) -> LossTerm:
    """Build the target-aware sequence loss used inside diffusion guidance.

    The Boltz2 feature graph is binder-first: a placeholder binder sequence is
    followed by the target chain(s). During guidance, SequenceSubsetLoss passes
    JAX BoltzGen-IF's soft full-binder sequence replaces the placeholder.
    Target templates are allowed; binder templates are intentionally not used
    here so the confidence loss stays sensitive to sequence changes.
    """
    if models.boltz2 is None:
        raise ValueError("Boltz2 model was not loaded; check Boltz2 guidance weights")

    from mosaic.losses.structure_prediction import (
        BinderTargetPAE,
        IPTMLoss,
        TargetBinderPAE,
        pTMEnergy,
    )
    from mosaic.structure_prediction import TargetChain

    if cfg.boltz2_guidance_recycling_steps < 0:
        raise ValueError("--boltz2-guidance-recycling-steps must be >= 0")
    if cfg.boltz2_guidance_sampling_steps < 2:
        raise ValueError("--boltz2-guidance-sampling-steps must be >= 2")

    stage_log(f"Boltz2 guidance: reading target templates from {cfg.complex_cif_path}")
    target_struct = gemmi.read_structure(str(cfg.complex_cif_path))
    target_struct.setup_entities()
    target_chains = []
    for cid in cfg.target_chain_ids:
        chain = target_struct[0][cid]
        seq = gemmi.one_letter_code([r.name for r in chain])
        template_chain = (
            chain.clone() if cfg.boltz2_guidance_target_template else None
        )
        target_chains.append(
            TargetChain(seq, use_msa=False, template_chain=template_chain)
        )

    stage_log(
        "Boltz2 guidance: featurizing binder placeholder + target "
        f"(binder_len={binder_length}, target_template="
        f"{cfg.boltz2_guidance_target_template})"
    )
    features, _ = models.boltz2.binder_features(binder_length, target_chains)

    loss_terms = []
    if cfg.weight_boltz2_ptm_energy > 0:
        # pTMEnergy is negative when confidence is high, so minimizing it
        # maximizes cross-chain TM-style confidence.
        loss_terms.append(cfg.weight_boltz2_ptm_energy * pTMEnergy())
    if cfg.weight_boltz2_interface_pae > 0:
        mean_interface_pae = 0.5 * BinderTargetPAE() + 0.5 * TargetBinderPAE()
        loss_terms.append(cfg.weight_boltz2_interface_pae * mean_interface_pae)
    if cfg.weight_boltz2_iptm > 0:
        loss_terms.append(cfg.weight_boltz2_iptm * IPTMLoss())
    if cfg.weight_boltz2_ipsae > 0:
        # ipSAE's hard PAE cutoff and max/best-row reduction are not a stable
        # differentiable per-step objective. In-loop binding guidance should use
        # ipTM and interface PAE/iPAE; ipSAE stays a post-refold rank/filter.
        raise ValueError(
            "--weight-boltz2-ipsae is not supported for in-loop guidance; "
            "ipSAE is not differentiable enough for per-step guidance. Use "
            "--weight-boltz2-iptm and/or --weight-boltz2-interface-pae for "
            "guidance, then use ipsae_pae_cutoff-based refold ranking "
            "(--skip-refold false) afterward."
        )
    if not loss_terms:
        raise ValueError("Boltz2 guidance requested without positive loss weights")

    boltz2_sequence_loss = sum(loss_terms[1:], start=loss_terms[0])
    stage_log(
        "Boltz2 guidance: built loss "
        f"(pTMEnergy={cfg.weight_boltz2_ptm_energy}, "
        f"interface_PAE={cfg.weight_boltz2_interface_pae}, "
        f"ipTM={cfg.weight_boltz2_iptm}, "
        f"recycle={cfg.boltz2_guidance_recycling_steps}, "
        f"sample_steps={cfg.boltz2_guidance_sampling_steps})"
    )
    return models.boltz2.build_loss(
        loss=boltz2_sequence_loss,
        features=features,
        recycling_steps=cfg.boltz2_guidance_recycling_steps,
        sampling_steps=cfg.boltz2_guidance_sampling_steps,
    )


def build_opendde_guidance_loss(
    cfg: VHHDesignConfig,
    models: LoadedModels,
    designable_token_mask: Bool[Array, "N"],
    sequence_loss_indices: Int[Array, "M"],
) -> LossTerm:
    """Build the target-aware sequence loss used inside diffusion guidance,
    OpenDDE backend (alternative to `build_boltz2_guidance_loss`).

    OpenDDE's PAE/pLDDT are scoring-only, not a gradient channel (see
    src/mosaic/models/opendde.py's module docstring: "the differentiable
    design signal rides the distogram... pae/pLDDT are consumed as scored
    values, not as a gradient channel"). So unlike the Boltz2 loss above
    (built from PAE/ipTM), this reads `distogram_logits` via
    `DistogramIPTMProxy`/`BinderTargetContact` -- both already generic over
    `StructureModelOutput`, no OpenDDE-specific loss classes needed.

    OpenDDE's featurizer does not support templates (`TargetChain(...,
    template_chain=...)` raises `NotImplementedError` there), unlike Boltz2's
    optional target templating -- target chains are always template-free here.

    Both currently-supported OpenDDE guidance losses (`DistogramIPTMProxy`,
    `BinderTargetContact`) read only `distogram_logits`/`distogram_bins`, so
    this uses `build_distogram_only_loss` rather than `build_loss` --
    `build_loss` always runs OpenDDE's diffusion coordinate sampling and
    confidence head regardless of whether the wrapped loss touches them,
    which would otherwise be paid on every BoltzGen diffusion step for
    nothing. This would matter again only if a PAE/coordinate-based OpenDDE
    guidance loss is added later, which `build_distogram_only_loss`'s
    `_DistogramOnlyOutput` stand-in deliberately can't support -- see
    `mosaic.losses.opendde.DistogramOnlyOpenDDELoss`.
    """
    if models.opendde is None:
        raise ValueError("OpenDDE model was not loaded; check OpenDDE guidance weights")

    from mosaic.losses.structure_prediction import BinderTargetContact, DistogramIPTMProxy
    from mosaic.structure_prediction import TargetChain

    if cfg.opendde_guidance_recycling_steps < 0:
        raise ValueError("--opendde-guidance-recycling-steps must be >= 0")

    binder_length = int(sequence_loss_indices.shape[0])

    stage_log(f"OpenDDE guidance: reading target chains from {cfg.complex_cif_path}")
    target_struct = gemmi.read_structure(str(cfg.complex_cif_path))
    target_struct.setup_entities()
    target_chains = []
    for cid in cfg.target_chain_ids:
        chain = target_struct[0][cid]
        seq = gemmi.one_letter_code([r.name for r in chain])
        target_chains.append(TargetChain(seq, use_msa=False))

    stage_log(
        "OpenDDE guidance: featurizing binder placeholder + target "
        f"(binder_len={binder_length})"
    )
    features, _ = models.opendde.binder_features(binder_length, target_chains)

    # Binder-local (0-indexed into the binder-only sequence array) positions
    # of the designable/CDR residues, for BinderTargetContact's paratope_idx --
    # mirrors examples/protenij_vhh.py's pattern of restricting contact scoring
    # to the redesigned positions rather than the whole binder chain.
    paratope_idx = np.nonzero(
        np.asarray(designable_token_mask)[np.asarray(sequence_loss_indices)]
    )[0]

    loss_terms = []
    if cfg.weight_opendde_iptm > 0:
        loss_terms.append(cfg.weight_opendde_iptm * DistogramIPTMProxy())
    if cfg.weight_opendde_contact > 0:
        loss_terms.append(
            cfg.weight_opendde_contact * BinderTargetContact(
                paratope_idx=paratope_idx,
                contact_distance=cfg.opendde_contact_distance,
            )
        )
    if not loss_terms:
        raise ValueError("OpenDDE guidance requested without positive loss weights")

    opendde_sequence_loss = sum(loss_terms[1:], start=loss_terms[0])
    stage_log(
        "OpenDDE guidance: built distogram-only loss "
        f"(distogram_iptm={cfg.weight_opendde_iptm}, "
        f"contact={cfg.weight_opendde_contact}, "
        f"recycle={cfg.opendde_guidance_recycling_steps}; "
        "diffusion sampling skipped, see build_distogram_only_loss)"
    )
    return models.opendde.build_distogram_only_loss(
        loss=opendde_sequence_loss,
        features=features,
        recycling_steps=cfg.opendde_guidance_recycling_steps,
    )


@dataclass
class GuidanceLosses:
    """The three separate objectives Phase 1's guidance controller
    (`guided_partial_diffusion`'s `guidance_fn_bind`/`_nat`/`_edit`) expects,
    as opposed to the single pre-summed loss the old single-`guidance_fn`
    interface took. See docs/legacy/guidance_design_notes.md section 5 / 10.

    `bind` is the anchor objective guidance is built around; if it is
    `None`, `guided_partial_diffusion` treats guidance as fully disabled
    regardless of whether `nat`/`edit` are present (there is no anchor to
    merge them against) — see `build_guidance_loss`'s fallback below for how
    that's avoided when Boltz2 guidance isn't configured.
    """

    bind: LossTerm | None
    nat: LossTerm | None
    edit: LossTerm | None


def build_guidance_loss(cfg: VHHDesignConfig, models: LoadedModels,
                        parent_one_hot: Float[Array, "N 20"],
                        designable_token_mask: Bool[Array, "N"],
                        sequence_loss_indices: Int[Array, "M"]) -> GuidanceLosses | None:
    """Build the three separate per-step objectives for guided_partial_diffusion.

    Operates on a soft sequence emitted by JAX BoltzGen-IF. Gradients flow
    back through IF -> coords. Each term is wrapped in ClippedGradient so the
    per-objective sequence-space gradient stays bounded before it is pulled
    back through the IF Jacobian; guided_partial_diffusion's own controller
    (mask/de-mean/RMS-normalize/PCGrad-merge/trust-radius-clip) then handles
    balancing the three *coordinate-space* gradients against each other,
    which sequence-space clipping alone cannot guarantee (see
    docs/legacy/guidance_design_notes.md section 3 for why).
    """
    edit_budget_term = EditBudget(
        s_ref=parent_one_hot,
        designable=designable_token_mask,
        budget=cfg.edit_budget,
    )

    if cfg.skip_guidance:
        # v0 path: no guidance at all. Return None so the driver skips
        # building any guidance_fn_* closures and guided_partial_diffusion
        # short-circuits the per-step gradient computation.
        return None

    # Edit-budget is the locality/edit-restraint objective (L_edit). It is
    # always available when guidance is active at all (unlike naturalness or
    # Boltz2 binding, which are individually optional via their weights).
    edit_loss = cfg.weight_edit_budget * edit_budget_term

    # Naturalness (L_nat): AbLang2, if enabled.
    nat_terms = []
    if cfg.weight_ablang2 > 0:
        ablang2_pll = make_ablang2_loss(
            models, sequence_loss_indices, stop_grad=False
        )
        nat_terms.append(
            cfg.weight_ablang2
            * SequenceSubsetLoss(
                ClippedGradient(ablang2_pll, cfg.clip_gradient_norm),
                sequence_loss_indices,
            )
        )
    nat_loss = sum(nat_terms[1:], start=nat_terms[0]) if nat_terms else None

    # Binding confidence (L_bind): the Boltz2- or OpenDDE-based
    # interface-confidence surrogate, if either is configured. Mutually
    # exclusive -- load_guidance_models raises if both are set.
    bind_loss = None
    if uses_boltz2_guidance(cfg):
        boltz2_guidance_loss = build_boltz2_guidance_loss(
            cfg,
            models,
            binder_length=int(sequence_loss_indices.shape[0]),
        )
        bind_loss = SequenceSubsetLoss(
            ClippedGradient(boltz2_guidance_loss, cfg.clip_gradient_norm),
            sequence_loss_indices,
        )
    elif uses_opendde_guidance(cfg):
        opendde_guidance_loss = build_opendde_guidance_loss(
            cfg,
            models,
            designable_token_mask,
            sequence_loss_indices,
        )
        bind_loss = SequenceSubsetLoss(
            ClippedGradient(opendde_guidance_loss, cfg.clip_gradient_norm),
            sequence_loss_indices,
        )

    if bind_loss is not None:
        return GuidanceLosses(bind=bind_loss, nat=nat_loss, edit=edit_loss)

    # No Boltz2/OpenDDE binding signal configured — this is v1's
    # "EditBudget-only guidance" mode. guided_partial_diffusion requires a
    # bind objective for guidance to be active at all (it is the anchor
    # everything else is projected against), so use edit_loss as the anchor
    # in this case instead of as a separate regularizer. This preserves v1's
    # original behavior exactly: with no naturalness terms either, guidance
    # reduces to edit-budget alone driving the trajectory, matching what the
    # single merged guidance_fn used to do when only weight_edit_budget was
    # nonzero.
    return GuidanceLosses(bind=edit_loss, nat=nat_loss, edit=None)


def build_polish_loss(cfg: VHHDesignConfig, models: LoadedModels,
                      parent_one_hot: Float[Array, "N 20"],
                      designable_token_mask: Bool[Array, "N"],
                      sequence_loss_indices: Int[Array, "M"]):
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
    terms = [cfg.weight_edit_budget * edit_term]
    if cfg.weight_ablang2 > 0:
        ablang2_pll = make_ablang2_loss(
            models, sequence_loss_indices, stop_grad=False
        )
        terms.append(
            cfg.weight_ablang2
            * SequenceSubsetLoss(ablang2_pll, sequence_loss_indices)
        )
    return sum(terms[1:], start=terms[0])


# =============================================================================
# Driver
# =============================================================================


def run(cfg: VHHDesignConfig):
    """End-to-end VHH redesign driver. See module docstring for pipeline overview."""
    stage_log(f"starting run seed={cfg.seed} output_dir={cfg.output_dir}")
    if cfg.num_sampling_steps < 2:
        raise ValueError("--num-sampling-steps must be >= 2")
    if cfg.sequence_decoder != "boltzgen_if":
        raise ValueError(
            f"Unknown sequence decoder {cfg.sequence_decoder!r}; "
            "this workflow requires boltzgen_if"
        )
    if cfg.boltzgen_if_guidance_temperature <= 0:
        raise ValueError("--boltzgen-if-guidance-temperature must be > 0")
    if cfg.weight_boltzgen_if_prior < 0:
        raise ValueError("--weight-boltzgen-if-prior must be >= 0")
    if cfg.weight_boltz2_ipsae > 0:
        # Validated unconditionally here, not inside build_boltz2_guidance_loss:
        # uses_boltz2_guidance() deliberately excludes weight_boltz2_ipsae (see
        # its docstring), so if ipSAE were the *only* positive Boltz2 weight,
        # build_boltz2_guidance_loss would never be called and its ValueError
        # guard would never fire -- ipSAE would be silently ignored instead of
        # rejected. Checking here catches that case too.
        raise ValueError(
            "--weight-boltz2-ipsae is not supported for in-loop guidance; "
            "ipSAE is not differentiable enough for per-step guidance. Use "
            "--weight-boltz2-iptm and/or --weight-boltz2-interface-pae for "
            "guidance, then use ipsae_pae_cutoff-based refold ranking "
            "(--skip-refold false) afterward."
        )
    if uses_boltz2_guidance(cfg) and uses_opendde_guidance(cfg):
        # Same reasoning as the ipSAE check above: validated here (before any
        # model loading) rather than only inside load_guidance_models, so it
        # fails fast and is cheaply testable without a real BoltzGen load.
        raise ValueError(
            "Both Boltz2 (--weight-boltz2-*) and OpenDDE (--weight-opendde-*) "
            "in-loop guidance weights are set; only one can own the bind "
            "objective. Zero out one backend's weights."
        )
    if cfg.refold_backend not in {"boltz2", "opendde"}:
        raise ValueError("--refold-backend must be one of boltz2, opendde")
    if guidance_anchor_is_empty_edit_budget(cfg):
        # Warned here (before any model loading), same reasoning as the
        # other config-only checks in this block.
        print(
            "[guidance] WARNING: no Boltz2/OpenDDE binding signal is "
            "configured and --weight-edit-budget <= 0, so the promoted "
            "`bind` anchor (see build_guidance_loss) has an all-zero "
            "gradient. Guidance will collapse to naturalness-only "
            "(AbLang2) if --weight-ablang2 > 0, or be a no-op entirely if "
            "it is not, despite the anchor/regularizer framing implying "
            "edit-budget is driving the trajectory. Set "
            "--weight-edit-budget > 0 if that framing should hold, or "
            "ignore this if naturalness-only guidance is intentional.",
            flush=True,
        )
    if not cfg.skip_refold and opendde_refold_ignores_template_config(cfg):
        # Warned here (before any model loading) rather than only right
        # before refolding runs, so it's visible before paying for the
        # entire design/search loop, not immediately before the refold step
        # at the very end of a run that could take a long time.
        print(
            "[refold] WARNING: --refold-binder-template is set "
            f"(mode={cfg.refold_binder_template_mode!r}) but "
            "--refold-backend opendde does not support templates -- this "
            "run's refolding will be template-free regardless. Set "
            "--refold-binder-template-mode none to make that explicit, or "
            "use --refold-backend boltz2 if templated refolding matters.",
            flush=True,
        )
    if not cfg.skip_refold:
        # Both refold_pareto_with_boltz2 and refold_pareto_with_opendde score
        # samples via eqx.filter_jit(jax.vmap(score_one)) -- fusing a whole
        # trunk-to-confidence-heads forward (with its unrolled per-step
        # diffusion sampling loop) into one compiled program. That produces a
        # compiled kernel binary over 1GB; loading it competes with JAX's
        # preallocated device-memory arena for a *separate* memory pool, and
        # XLA's autotuner additionally grabs large transient scratch while
        # benchmarking candidate kernels during compilation. Both cause
        # spurious CUDA_ERROR_OUT_OF_MEMORY on a single 24GB GPU even for
        # tiny inputs -- confirmed via jax's compiled memory_analysis(),
        # whose reported peak *data* usage was ~1.3GB, nowhere near the
        # failure. Disabling autotuning avoids both; XLA falls back to
        # default (not empirically-searched) kernel choices, a minor
        # runtime-speed tradeoff for a program that otherwise won't load at
        # all on this hardware.
        #
        # Set here, before load_core_models (a few lines below) -- the
        # earliest JAX device touch in this process -- since XLA reads
        # XLA_FLAGS once, lazily, at backend initialization. Every
        # --devices multi-GPU child re-executes run() from scratch as its
        # own process (see run_many_from_cli/build_single_design_command),
        # so this self-applies per child with no separate multi-GPU-specific
        # handling needed.
        existing_xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_autotune_level" not in existing_xla_flags:
            os.environ["XLA_FLAGS"] = (
                existing_xla_flags + " --xla_gpu_autotune_level=0"
            ).strip()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "config.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(cfg).items()}, f, indent=2)
    stage_log("wrote config.json")

    # ---- 1. One-time setup: load BoltzGen, parse YAML, build features ----
    print("[setup] loading BoltzGen model...", flush=True)
    models = load_core_models(cfg)

    if cfg.boltzgen_yaml_path is not None:
        stage_log(f"reading BoltzGen YAML {cfg.boltzgen_yaml_path}")
        yaml_string = cfg.boltzgen_yaml_path.read_text()
        input_files = boltzgen_yaml_files(cfg.boltzgen_yaml_path, yaml_string)
    else:
        stage_log("building BoltzGen YAML from CLI inputs")
        cif_filename = cfg.complex_cif_path.name
        yaml_string = build_complex_yaml(
            cif_filename=cif_filename,
            binder_chain_id=cfg.binder_chain_id,
            target_chain_ids=cfg.target_chain_ids,
            cdr_residue_indices=cfg.cdr_residue_indices,
        )
        input_files = {cif_filename: cfg.complex_cif_path}

    print("[setup] parsing YAML and featurizing complex...", flush=True)
    stage_log("loading and featurizing complex")
    features, writer = load_features_and_structure_writer(
        yaml_string=yaml_string,
        files=input_files,
        # mask_backbone=False keeps the parent backbone visible to the trunk —
        # the sequence is still masked at designable positions.
        mask=True,
        mask_backbone=False,
        mask_disto=True,
    )
    stage_log("finished loading and featurizing complex")

    # Extract everything we need from features. Keep arrays UNBATCHED throughout
    # the driver — guided_partial_diffusion auto-batches as needed. For backbone
    # extraction inside guidance_fn we use the token-to-backbone mapping that
    # BoltzGenOutput.backbone_coordinates already exercises.
    stage_log("building parent sequence and design masks")
    parent_one_hot = parent_one_hot_from_features(features)
    designable_token_mask = cdr_token_mask_from_features(features)
    initial_coords = squeeze_feature(features["coords"], "coords", 2)               # (M, 3)
    atom_pad_mask = squeeze_feature(features["atom_pad_mask"], "atom_pad_mask", 1)  # (M,)
    atom_partial_mask = build_atom_partial_mask(features, designable_token_mask)  # (M,)
    asym_id = squeeze_feature(features["asym_id"], "asym_id", 1)
    binder_token_indices = binder_indices_from_design_mask(
        asym_id, designable_token_mask
    )
    n_total_tokens = parent_one_hot.shape[0]
    n_binder_tokens = int(binder_token_indices.shape[0])
    n_designable = int(designable_token_mask.sum())
    print(f"[setup] complex has {n_total_tokens} tokens, "
          f"{n_binder_tokens} binder tokens, "
          f"{n_designable} designable (CDR) positions, "
          f"edit budget = {cfg.edit_budget}", flush=True)
    stage_log("finished parent sequence and design masks")

    # ---- 2. Run trunk ONCE; reuse Sampler across outer iterations ----
    print("[setup] running BoltzGen trunk + diffusion conditioning...", flush=True)
    stage_log("starting BoltzGen sampler/trunk conditioning")
    sampler = Sampler.from_features(
        model=models.boltzgen,
        features=features,
        key=jax.random.key(cfg.seed),
        deterministic=True,
        recycling_steps=cfg.recycling_steps,
    )
    stage_log("finished BoltzGen sampler/trunk conditioning")

    # ---- 3. Load guidance models and build composite losses ----
    print("[setup] loading guidance models...", flush=True)
    stage_log("starting guidance model loading")
    models = load_guidance_models(cfg, models)
    models = ensure_boltzgen_if_loaded(cfg, models)
    stage_log("finished guidance model loading")

    stage_log("building guidance and polish losses")
    guidance_losses = build_guidance_loss(
        cfg, models, parent_one_hot, designable_token_mask, binder_token_indices
    )
    polish_loss = build_polish_loss(
        cfg, models, parent_one_hot, designable_token_mask, binder_token_indices
    )
    lambda_fn = lambda_schedule_fn(cfg.lambda_schedule, cfg.lambda_max)
    stage_log("finished building guidance and polish losses")

    # ---- 4. Outer loop ----
    parent_seq_ids = jnp.argmax(parent_one_hot, axis=-1)
    current_parent_one_hot = parent_one_hot
    current_initial_coords = initial_coords

    all_iterations = []
    global_pareto: dict[int, tuple[float, np.ndarray]] = {}
    guidance_diagnostics_records = []

    for outer in range(cfg.n_outer_iterations):
        t0 = time.time()
        print(f"\n[outer {outer}] guided partial diffusion "
              f"(start_sigma_frac={cfg.start_sigma_frac}, "
              f"steps={cfg.num_sampling_steps})...", flush=True)

        # ----- Stage 1: guided partial diffusion -----
        if_context = prepare_jax_boltzgen_if_context(
            models.boltzgen_if_torch,
            writer.torch_features,
            np.asarray(current_initial_coords),
            parent_sequence=np.asarray(current_parent_one_hot),
            designable_mask=np.asarray(designable_token_mask),
        )
        if_order = jax.random.permutation(
            jax.random.key(cfg.seed + 7919 * outer),
            if_context.designable_positions,
        )
        def _make_guidance_fn(loss_term):
            """Wrap one GuidanceLosses field into a guidance_fn_* closure of
            the shape guided_partial_diffusion expects: (x0) -> scalar,
            gradient flowing x0 -> soft_seq (via JAX BoltzGen-IF) -> loss_term."""
            def guidance_fn(x0):
                soft_seq = differentiable_jax_boltzgen_if(
                    models.boltzgen_if_jax,
                    if_context,
                    x0[0],
                    key=jax.random.key(cfg.seed + outer),
                    temperature=cfg.boltzgen_if_guidance_temperature,
                    avoid=cfg.boltzgen_if_avoid,
                    order=if_order,
                )
                v, _ = loss_term(soft_seq, key=jax.random.key(cfg.seed + outer))
                return v
            return guidance_fn

        if guidance_losses is not None:
            guidance_fn_bind = (
                _make_guidance_fn(guidance_losses.bind)
                if guidance_losses.bind is not None else None
            )
            guidance_fn_nat = (
                _make_guidance_fn(guidance_losses.nat)
                if guidance_losses.nat is not None else None
            )
            guidance_fn_edit = (
                _make_guidance_fn(guidance_losses.edit)
                if guidance_losses.edit is not None else None
            )
        else:
            guidance_fn_bind = guidance_fn_nat = guidance_fn_edit = None

        # Diagnostics are only meaningful when guidance is actually active --
        # with guidance_fn_bind=None the controller is a no-op and
        # cos(guided,unguided) would trivially be ~1 everywhere.
        want_diagnostics = cfg.log_guidance_diagnostics and guidance_fn_bind is not None

        stage_log(f"outer {outer}: starting guided partial diffusion")
        diffusion_result = guided_partial_diffusion(
            sampler=sampler,
            structure_module=models.boltzgen.structure_module,
            initial_coords=current_initial_coords,
            atom_partial_mask=atom_partial_mask,
            atom_mask=atom_pad_mask,
            num_sampling_steps=cfg.num_sampling_steps,
            start_sigma_frac=cfg.start_sigma_frac,
            step_scale=cfg.step_scale,
            noise_scale=cfg.noise_scale,
            guidance_fn_bind=guidance_fn_bind,
            guidance_fn_nat=guidance_fn_nat,
            guidance_fn_edit=guidance_fn_edit,
            guidance_lambda_fn=lambda_fn if guidance_fn_bind is not None else None,
            guidance_nos_inner_steps=cfg.nos_inner_steps if guidance_fn_bind is not None else 0,
            guidance_nos_inner_step_fn=lambda t: cfg.nos_inner_step_size,
            guidance_nos_lambda_kl_fn=lambda t: cfg.nos_lambda_kl,
            guidance_nos_noise_fn=lambda t: cfg.nos_langevin_noise,
            guidance_lookahead=cfg.lookahead if guidance_fn_bind is not None else False,
            key=jax.random.key(cfg.seed + 1000 * outer),
            return_diagnostics=want_diagnostics,
        )
        if want_diagnostics:
            x_final, step_diagnostics = diffusion_result
            label = f"outer_{outer}"
            summary = summarize_guidance_diagnostics(
                per_step_metrics(step_diagnostics, atom_partial_mask=atom_partial_mask),
                disagreement_cos_threshold=cfg.guidance_diagnostics_cos_threshold,
                n_sigma_bins=cfg.guidance_diagnostics_sigma_bins,
            )
            print(format_summary_text(label, summary), flush=True)
            guidance_diagnostics_records.append(
                {"label": label, "summary": summary, "outcome": None}
            )
            # Write incrementally after every outer iteration, not just at the
            # end of run() -- cluster jobs can be preempted or crash mid-run,
            # and partial diagnostics are still useful.
            write_guidance_diagnostics_report(
                guidance_diagnostics_records,
                cfg.output_dir / "guidance_diagnostics.json",
            )
        else:
            x_final = diffusion_result
        stage_log(f"outer {outer}: finished guided partial diffusion")

        # ----- Stage 1.5: decode to discrete sequence -----
        stage_log(f"outer {outer}: decoding final structure with JAX BoltzGen-IF")
        if_result = decode_with_jax_boltzgen_if(
            models.boltzgen_if_jax,
            if_context,
            x_final,
            seed=cfg.seed + 7 * outer,
            temperature=cfg.boltzgen_if_temperature,
            avoid=cfg.boltzgen_if_avoid,
            order=if_order,
        )
        diffusion_seq = jnp.asarray(if_result.sequence_ids)
        np.savez_compressed(
            cfg.output_dir / f"boltzgen_if_outer_{outer}.npz",
            logits=if_result.logits,
            sequence_ids=if_result.sequence_ids,
            parent_ids=np.asarray(jnp.argmax(current_parent_one_hot, axis=-1)),
            designable_mask=np.asarray(designable_token_mask),
        )
        stage_log(f"outer {outer}: saved JAX BoltzGen-IF logits")

        search_loss = polish_loss
        if cfg.weight_boltzgen_if_prior > 0:
            if_prior = FixedLogitsSequenceLoss.from_logits(
                if_result.logits,
                designable_token_mask,
                name="boltzgen_if_nll",
            )
            search_loss = search_loss + cfg.weight_boltzgen_if_prior * if_prior
            stage_log(
                f"outer {outer}: added fixed-backbone BoltzGen-IF search prior "
                f"(weight={cfg.weight_boltzgen_if_prior})"
            )
        diffusion_edits = int(((diffusion_seq != parent_seq_ids)
                               & designable_token_mask).sum())
        print(
            f"[outer {outer}] diffusion produced {diffusion_edits} edits vs parent",
            flush=True,
        )
        stage_log(
            f"outer {outer}: finished {cfg.sequence_decoder} discrete decode"
        )

        # ----- Stage 2: edit-budgeted sequence search -----
        if cfg.skip_polish:
            polished_seq = np.asarray(diffusion_seq)
            polished_val = float("nan")
            iter_pareto = {diffusion_edits: (float("nan"), polished_seq)}
            polished_label = "none"
        else:
            search_mode = cfg.search_mode.lower()
            if search_mode not in {"greedy", "mcmc", "both"}:
                raise ValueError(
                    f"Unknown search_mode {cfg.search_mode!r}; "
                    "expected greedy, mcmc, or both."
                )
            parent_np = np.asarray(parent_seq_ids)
            design_mask_np = np.asarray(designable_token_mask)
            diffusion_np = np.asarray(diffusion_seq)
            search_results: list[tuple[str, np.ndarray, float, dict]] = []

            if search_mode in {"greedy", "both"}:
                print(
                    f"[outer {outer}] edit-budgeted greedy polish "
                    f"(budget={cfg.edit_budget}, steps<={cfg.polish_steps})...",
                    flush=True,
                )
                stage_log(f"outer {outer}: starting edit-budgeted greedy polish")
                greedy_seq, greedy_val, greedy_pareto = edit_budgeted_greedy_descent(
                    loss=search_loss,
                    sequence=diffusion_np,
                    parent=parent_np,
                    budget=cfg.edit_budget,
                    designable_mask=design_mask_np,
                    batch_size=cfg.polish_batch_size,
                    steps=cfg.polish_steps,
                    key=jax.random.key(cfg.seed + 31337 * outer),
                )
                search_results.append(("greedy", greedy_seq, greedy_val, greedy_pareto))
                stage_log(f"outer {outer}: finished edit-budgeted greedy polish")

            if search_mode in {"mcmc", "both"}:
                print(
                    f"[outer {outer}] edit-budgeted gradient MCMC "
                    f"(budget={cfg.edit_budget}, steps={cfg.mcmc_steps}, "
                    f"temp={cfg.mcmc_temp}, proposal_temp={cfg.mcmc_proposal_temp}, "
                    f"path<={cfg.mcmc_max_path_length})...",
                    flush=True,
                )
                stage_log(f"outer {outer}: starting edit-budgeted gradient MCMC")
                mcmc_seq, mcmc_val, mcmc_pareto = edit_budgeted_gradient_mcmc(
                    loss=search_loss,
                    sequence=diffusion_np,
                    parent=parent_np,
                    budget=cfg.edit_budget,
                    designable_mask=design_mask_np,
                    steps=cfg.mcmc_steps,
                    batch_size=cfg.polish_batch_size,
                    temp=cfg.mcmc_temp,
                    proposal_temp=cfg.mcmc_proposal_temp,
                    max_path_length=cfg.mcmc_max_path_length,
                    key=jax.random.key(cfg.seed + 7331 * outer),
                )
                search_results.append(("mcmc", mcmc_seq, mcmc_val, mcmc_pareto))
                stage_log(f"outer {outer}: finished edit-budgeted gradient MCMC")

            polished_label, polished_seq, polished_val, iter_pareto = min(
                search_results,
                key=lambda item: item[2] if np.isfinite(item[2]) else float("inf"),
            )
            if len(search_results) > 1:
                iter_pareto = {}
                for _, _, _, pareto in search_results:
                    for k, (loss_v, seq) in pareto.items():
                        existing = iter_pareto.get(k)
                        if existing is None or loss_v < existing[0]:
                            iter_pareto[k] = (loss_v, seq.copy())
            print(
                f"[outer {outer}] selected {polished_label} search result "
                f"with loss={polished_val:.4f}",
                flush=True,
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
                print(
                    f"[outer {outer}] no further improving edits — converged.",
                    flush=True,
                )

        all_iterations.append({
            "outer": outer,
            "search_mode": cfg.search_mode,
            "selected_search": polished_label,
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
    print("\n[output] writing Pareto front and per-iteration log...", flush=True)
    stage_log("writing Pareto front and iterations CSV")
    binder_token_indices_np = np.asarray(binder_token_indices)
    pareto_rows = [
        {
            "edit_count": k,
            "loss": v[0],
            "sequence": "".join(TOKENS[i] for i in v[1][binder_token_indices_np]),
            "full_sequence": "".join(TOKENS[i] for i in v[1]),
        }
        for k, v in sorted(global_pareto.items())
    ]
    pl.DataFrame(pareto_rows).write_csv(cfg.output_dir / "pareto_front.csv")
    pl.DataFrame(all_iterations).write_csv(cfg.output_dir / "iterations.csv")

    # ---- 6. Refold (task #15) ----
    if not cfg.skip_refold:
        print(f"[refold] refolding Pareto candidates with {cfg.refold_backend}...", flush=True)
        stage_log(f"starting {cfg.refold_backend} refold")
        if cfg.refold_backend == "opendde":
            refold_pareto_with_opendde(global_pareto, cfg, binder_token_indices)
        else:
            refold_pareto_with_boltz2(global_pareto, cfg, binder_token_indices)
        stage_log(f"finished {cfg.refold_backend} refold")

    print(f"[done] outputs in {cfg.output_dir}", flush=True)
    stage_log("run complete")
    return global_pareto, all_iterations


# =============================================================================
# Refolding harness (task #15)
# =============================================================================


def refold_pareto_with_boltz2(
    pareto: dict[int, tuple[float, np.ndarray]],
    cfg: VHHDesignConfig,
    binder_token_indices: Int[Array, "M"],
):
    """Refold each Pareto candidate with Boltz2 + rank primarily by ipSAE.

    This is a thin orchestration around the existing reusable functions in
    `examples/boltzgen_pipeline.py`. We import them lazily here so v0/v1/v2 runs
    don't pay the Boltz2 import cost when refolding is disabled.
    """
    stage_log("refold: importing Boltz2 helpers")
    from mosaic.models.boltz2 import Boltz2
    from mosaic.losses.boltz2 import boltz2_trunk, boltz2_forward_from_trunk
    from mosaic.losses.structure_prediction import (
        IPTMLoss, BinderTargetIPSAE, TargetBinderIPSAE, IPSAE_min,
    )
    from mosaic.structure_prediction import TargetChain

    stage_log("refold: loading Boltz2 model")
    boltz2 = Boltz2()
    stage_log("refold: loaded Boltz2 model")
    stage_log(f"refold: reading complex {cfg.complex_cif_path}")
    target_struct = gemmi.read_structure(str(cfg.complex_cif_path))
    target_struct.setup_entities()
    if cfg.refold_binder_template_mode not in {"full", "framework", "none"}:
        raise ValueError(
            "--refold-binder-template-mode must be one of full, framework, none"
        )
    binder_template_chain = None
    binder_template_removed_atoms = 0
    if cfg.refold_binder_template and cfg.refold_binder_template_mode != "none":
        parent_binder_chain = target_struct[0][cfg.binder_chain_id]
        if cfg.refold_binder_template_mode == "framework":
            binder_template_chain, binder_template_removed_atoms = (
                mask_binder_template_design_residues(
                    parent_binder_chain,
                    cfg.cdr_residue_indices,
                )
            )
        else:
            binder_template_chain = parent_binder_chain.clone()
    target_templates = []
    for cid in cfg.target_chain_ids:
        chain = target_struct[0][cid]
        seq = gemmi.one_letter_code([r.name for r in chain])
        # Boltz2 template YAML construction renames template chains in-place to
        # A/B/...; pass a clone so the original reference keeps author chain IDs
        # for target-aligned RMSD diagnostics.
        target_templates.append((seq, chain.clone()))
    stage_log(
        "refold: built target chain templates/features "
        f"(binder_template={cfg.refold_binder_template}, "
        f"binder_template_mode={cfg.refold_binder_template_mode}, "
        f"masked_binder_template_atoms={binder_template_removed_atoms})"
    )

    iptm_loss = IPTMLoss()
    bt_ipsae_loss = BinderTargetIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    tb_ipsae_loss = TargetBinderIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    ipsae_min_loss = IPSAE_min(pae_cutoff=cfg.ipsae_pae_cutoff)

    rows = []
    all_sample_rows = []
    binder_token_indices_np = np.asarray(binder_token_indices)
    refold_dir = cfg.output_dir / "refolded_cifs"
    refolded_binder_chain_id = "A"
    refolded_target_chain_ids = [
        chr(ord("B") + i) for i in range(len(cfg.target_chain_ids))
    ]
    if cfg.refold_num_samples < 1:
        raise ValueError("--refold-num-samples must be >= 1")
    if cfg.refold_batch_size < 1:
        raise ValueError("--refold-batch-size must be >= 1")
    refold_batch_size = min(cfg.refold_batch_size, cfg.refold_num_samples)
    print(
        f"[refold] using sample batch size {refold_batch_size} "
        f"for {cfg.refold_num_samples} sample(s) per candidate",
        flush=True,
    )

    def score_refold_batch(
        model,
        features,
        initial_emb,
        trunk_state,
        binder_sequence_placeholder,
        sample_keys,
    ):
        def score_one(sample_key):
            out = boltz2_forward_from_trunk(
                model, features, initial_emb, trunk_state,
                num_sampling_steps=cfg.refold_sampling_steps,
                deterministic=True, key=sample_key,
            )
            _, iptm_aux = iptm_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "iptm"),
            )
            _, bt_aux = bt_ipsae_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "bt_ipsae"),
            )
            _, tb_aux = tb_ipsae_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "tb_ipsae"),
            )
            _, ipsae_min_aux = ipsae_min_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "ipsae_min"),
            )
            binder_len = binder_sequence_placeholder.shape[0]
            bt_pae = out.pae[:binder_len, binder_len:]
            tb_pae = out.pae[binder_len:, :binder_len]
            ipae_min = jnp.minimum(jnp.min(bt_pae), jnp.min(tb_pae))
            return {
                "structure_coordinates": out.structure_coordinates,
                "iptm": iptm_aux["iptm"],
                "bt_ipsae": bt_aux["bt_ipsae"],
                "tb_ipsae": tb_aux["tb_ipsae"],
                "ipsae_min": ipsae_min_aux["ipsae_min"],
                "ipae_min": ipae_min,
                "bt_pae_mean": jnp.mean(bt_pae),
                "tb_pae_mean": jnp.mean(tb_pae),
            }

        return jax.vmap(score_one)(sample_keys)

    score_refold_batch = eqx.filter_jit(score_refold_batch)

    for edit_count, (loss_v, seq_ids) in sorted(pareto.items()):
        stage_log(f"refold: preparing edit_count={edit_count}")
        seq_str = "".join(TOKENS[i] for i in seq_ids[binder_token_indices_np])
        # Query sequence is the designed binder sequence, while the optional
        # binder template supplies the parent pose/backbone to keep refolding
        # close to the input complex instead of freely redocking the binder.
        binder_template = (
            binder_template_chain.clone()
            if binder_template_chain is not None
            else None
        )
        refold_chains = [
            TargetChain(seq_str, use_msa=False, template_chain=binder_template)
        ] + [
            TargetChain(seq, use_msa=False, template_chain=template.clone())
            for seq, template in target_templates
        ]
        feats, w = boltz2.target_only_features(refold_chains)
        key = jax.random.key(cfg.seed + 99999 + edit_count)
        stage_log(f"refold: running trunk edit_count={edit_count}")
        initial_emb, trunk_state = boltz2_trunk(
            boltz2.model, feats, recycling_steps=cfg.recycling_steps,
            deterministic=True, key=fold_in(key, "trunk"),
        )
        stage_log(f"refold: finished trunk edit_count={edit_count}")

        binder_sequence_placeholder = jnp.zeros((len(seq_str), 20))

        best_row = None
        best_structure = None

        for chunk_start in range(0, cfg.refold_num_samples, refold_batch_size):
            chunk_size = min(refold_batch_size, cfg.refold_num_samples - chunk_start)
            stage_log(
                f"refold: sampling edit_count={edit_count} "
                f"samples {chunk_start}-{chunk_start + chunk_size - 1}"
            )
            sample_keys = jax.random.split(
                fold_in(key, f"sample_batch_{chunk_start}"),
                chunk_size,
            )
            batch_scores = score_refold_batch(
                boltz2.model,
                feats,
                initial_emb,
                trunk_state,
                binder_sequence_placeholder,
                sample_keys,
            )
            stage_log(
                f"refold: scored edit_count={edit_count} "
                f"samples {chunk_start}-{chunk_start + chunk_size - 1}"
            )

            for chunk_offset in range(chunk_size):
                sample_idx = chunk_start + chunk_offset

                structure = w(batch_scores["structure_coordinates"][chunk_offset])
                interface_metrics = interface_geometry_metrics(
                    structure,
                    binder_chain_id=refolded_binder_chain_id,
                    target_chain_ids=refolded_target_chain_ids,
                )
                rmsd_metrics = target_aligned_rmsd_metrics(
                    target_struct,
                    structure,
                    original_binder_chain_id=cfg.binder_chain_id,
                    original_target_chain_ids=cfg.target_chain_ids,
                    refolded_binder_chain_id=refolded_binder_chain_id,
                    refolded_target_chain_ids=refolded_target_chain_ids,
                    cdr_residue_indices=cfg.cdr_residue_indices,
                )

                ipsae_min = float(batch_scores["ipsae_min"][chunk_offset])
                row = {
                    "edit_count": edit_count,
                    "sample_idx": sample_idx,
                    "polish_loss": loss_v,
                    "refold_loss": -ipsae_min,
                    "refold_batch_size": refold_batch_size,
                    "ipsae_pae_cutoff": cfg.ipsae_pae_cutoff,
                    "iptm": float(batch_scores["iptm"][chunk_offset]),
                    "bt_ipsae": float(batch_scores["bt_ipsae"][chunk_offset]),
                    "tb_ipsae": float(batch_scores["tb_ipsae"][chunk_offset]),
                    "ipsae_min": ipsae_min,
                    "ipae_min": float(batch_scores["ipae_min"][chunk_offset]),
                    "bt_pae_mean": float(batch_scores["bt_pae_mean"][chunk_offset]),
                    "tb_pae_mean": float(batch_scores["tb_pae_mean"][chunk_offset]),
                    "sequence": seq_str,
                }
                row.update(interface_metrics)
                row.update(rmsd_metrics)
                row["rmsd_filter_threshold"] = cfg.refold_rmsd_threshold
                row["rmsd_pass"] = (
                    _passes_max_threshold(
                        row["filter_rmsd"],
                        cfg.refold_rmsd_threshold,
                    )
                    and _passes_max_threshold(
                        row["filter_rmsd_design"],
                        cfg.refold_rmsd_threshold,
                    )
                )
                all_sample_rows.append(dict(row))

                if (
                    best_row is None
                    or _refold_rank_key(row) < _refold_rank_key(best_row)
                ):
                    best_row = row
                    best_structure = structure

        assert best_row is not None and best_structure is not None
        cif_path = refold_dir / f"edit_{edit_count}_sample_{best_row['sample_idx']}.cif"
        write_structure_cif(best_structure, cif_path)
        best_row["refold_cif"] = str(cif_path)
        rows.append(best_row)

    passing_rows = sorted(
        [row for row in rows if row["rmsd_pass"]],
        key=_refold_rank_key,
    )
    failed_rows = sorted(
        [row for row in rows if not row["rmsd_pass"]],
        key=_refold_rank_key,
    )
    ranked_rows = [
        {
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(passing_rows, start=1)
    ] + [
        {
            "rank": None,
            **row,
        }
        for row in failed_rows
    ]

    pl.DataFrame(ranked_rows).write_csv(cfg.output_dir / "refold_ranked.csv")
    pl.DataFrame(rows).write_csv(cfg.output_dir / "refold_best_by_edit_count.csv")
    pl.DataFrame(all_sample_rows).write_csv(cfg.output_dir / "refold_all_samples.csv")


def refold_pareto_with_opendde(
    pareto: dict[int, tuple[float, np.ndarray]],
    cfg: VHHDesignConfig,
    binder_token_indices: Int[Array, "M"],
):
    """Refold each Pareto candidate with OpenDDE (Abag) + rank primarily by
    ipSAE. Alternative to `refold_pareto_with_boltz2` (see `cfg.refold_backend`).

    At refold time the binder sequence is a fixed, concrete string (not a
    soft PSSM under optimization), so this uses `target_only_features` --
    the same simple path Boltz2's refold uses -- rather than the
    poly-Trp-placeholder + `set_binder_sequence` machinery
    `build_opendde_guidance_loss` needs for in-loop gradient guidance.

    OpenDDE's featurizer does not support templates (`TargetChain(...,
    template_chain=...)` raises `NotImplementedError`), so unlike the Boltz2
    refold path there is no binder/target templating here -- every chain is
    template-free.
    """
    stage_log("refold: importing OpenDDE helpers")
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.losses.opendde import opendde_forward_from_trunk
    from mosaic.losses.structure_prediction import (
        IPTMLoss, BinderTargetIPSAE, TargetBinderIPSAE, IPSAE_min,
    )
    from mosaic.structure_prediction import TargetChain

    stage_log("refold: loading OpenDDE (Abag) model")
    opendde = OpenDDEModelAbag()
    stage_log("refold: loaded OpenDDE (Abag) model")
    stage_log(f"refold: reading complex {cfg.complex_cif_path}")
    target_struct = gemmi.read_structure(str(cfg.complex_cif_path))
    target_struct.setup_entities()
    target_seqs = []
    for cid in cfg.target_chain_ids:
        chain = target_struct[0][cid]
        target_seqs.append(gemmi.one_letter_code([r.name for r in chain]))
    stage_log(f"refold: built {len(target_seqs)} target chain sequence(s)")

    iptm_loss = IPTMLoss()
    bt_ipsae_loss = BinderTargetIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    tb_ipsae_loss = TargetBinderIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    ipsae_min_loss = IPSAE_min(pae_cutoff=cfg.ipsae_pae_cutoff)

    rows = []
    all_sample_rows = []
    binder_token_indices_np = np.asarray(binder_token_indices)
    refold_dir = cfg.output_dir / "refolded_cifs"
    refolded_binder_chain_id = "A"
    refolded_target_chain_ids = [
        chr(ord("B") + i) for i in range(len(cfg.target_chain_ids))
    ]
    if cfg.refold_num_samples < 1:
        raise ValueError("--refold-num-samples must be >= 1")
    if cfg.refold_batch_size < 1:
        raise ValueError("--refold-batch-size must be >= 1")
    refold_batch_size = min(cfg.refold_batch_size, cfg.refold_num_samples)
    print(
        f"[refold] using sample batch size {refold_batch_size} "
        f"for {cfg.refold_num_samples} sample(s) per candidate",
        flush=True,
    )

    def score_refold_batch(feat, s_inputs, s, z, binder_sequence_placeholder, sample_keys):
        def score_one(sample_key):
            out = opendde_forward_from_trunk(
                opendde.model, feat, s_inputs, s, z, sample_key,
                n_step=cfg.refold_sampling_steps,
                dense_atom_to_atom37=opendde.dense_atom_to_atom37,
                pae_bin_params=opendde.pae_bin_params,
                plddt_bin_params=opendde.plddt_bin_params,
            )
            _, iptm_aux = iptm_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "iptm"),
            )
            _, bt_aux = bt_ipsae_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "bt_ipsae"),
            )
            _, tb_aux = tb_ipsae_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "tb_ipsae"),
            )
            _, ipsae_min_aux = ipsae_min_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "ipsae_min"),
            )
            binder_len = binder_sequence_placeholder.shape[0]
            bt_pae = out.pae[:binder_len, binder_len:]
            tb_pae = out.pae[binder_len:, :binder_len]
            ipae_min = jnp.minimum(jnp.min(bt_pae), jnp.min(tb_pae))
            aux = {
                "iptm": iptm_aux["iptm"],
                "bt_ipsae": bt_aux["bt_ipsae"],
                "tb_ipsae": tb_aux["tb_ipsae"],
                "ipsae_min": ipsae_min_aux["ipsae_min"],
                "ipae_min": ipae_min,
                "bt_pae_mean": jnp.mean(bt_pae),
                "tb_pae_mean": jnp.mean(tb_pae),
            }
            # Return the full StructureModelOutput (an eqx.Module / pytree) so
            # to_structure() has every field it needs after vmap+indexing --
            # simpler and less error-prone than hand-picking a subset of
            # fields and reconstructing a partial StructureModelOutput later.
            return out, aux

        return jax.vmap(score_one)(sample_keys)

    score_refold_batch = eqx.filter_jit(score_refold_batch)

    for edit_count, (loss_v, seq_ids) in sorted(pareto.items()):
        stage_log(f"refold: preparing edit_count={edit_count}")
        seq_str = "".join(TOKENS[i] for i in seq_ids[binder_token_indices_np])
        refold_chains = [TargetChain(seq_str, use_msa=False)] + [
            TargetChain(seq, use_msa=False) for seq in target_seqs
        ]
        feat, _ = opendde.target_only_features(refold_chains)
        key = jax.random.key(cfg.seed + 99999 + edit_count)
        stage_log(f"refold: running trunk edit_count={edit_count}")
        s_inputs, s, z = opendde.model.get_pairformer_output(feat, cfg.recycling_steps)
        stage_log(f"refold: finished trunk edit_count={edit_count}")

        binder_sequence_placeholder = jnp.zeros((len(seq_str), 20))

        best_row = None
        best_structure = None

        for chunk_start in range(0, cfg.refold_num_samples, refold_batch_size):
            chunk_size = min(refold_batch_size, cfg.refold_num_samples - chunk_start)
            stage_log(
                f"refold: sampling edit_count={edit_count} "
                f"samples {chunk_start}-{chunk_start + chunk_size - 1}"
            )
            sample_keys = jax.random.split(
                fold_in(key, f"sample_batch_{chunk_start}"),
                chunk_size,
            )
            stacked_out, batch_scores = score_refold_batch(
                feat, s_inputs, s, z, binder_sequence_placeholder, sample_keys,
            )
            stage_log(
                f"refold: scored edit_count={edit_count} "
                f"samples {chunk_start}-{chunk_start + chunk_size - 1}"
            )

            for chunk_offset in range(chunk_size):
                sample_idx = chunk_start + chunk_offset

                sample_output = jax.tree.map(lambda x: x[chunk_offset], stacked_out)
                structure = sample_output.to_structure()
                interface_metrics = interface_geometry_metrics(
                    structure,
                    binder_chain_id=refolded_binder_chain_id,
                    target_chain_ids=refolded_target_chain_ids,
                )
                rmsd_metrics = target_aligned_rmsd_metrics(
                    target_struct,
                    structure,
                    original_binder_chain_id=cfg.binder_chain_id,
                    original_target_chain_ids=cfg.target_chain_ids,
                    refolded_binder_chain_id=refolded_binder_chain_id,
                    refolded_target_chain_ids=refolded_target_chain_ids,
                    cdr_residue_indices=cfg.cdr_residue_indices,
                )

                ipsae_min = float(batch_scores["ipsae_min"][chunk_offset])
                row = {
                    "edit_count": edit_count,
                    "sample_idx": sample_idx,
                    "polish_loss": loss_v,
                    "refold_loss": -ipsae_min,
                    "refold_batch_size": refold_batch_size,
                    "ipsae_pae_cutoff": cfg.ipsae_pae_cutoff,
                    "iptm": float(batch_scores["iptm"][chunk_offset]),
                    "bt_ipsae": float(batch_scores["bt_ipsae"][chunk_offset]),
                    "tb_ipsae": float(batch_scores["tb_ipsae"][chunk_offset]),
                    "ipsae_min": ipsae_min,
                    "ipae_min": float(batch_scores["ipae_min"][chunk_offset]),
                    "bt_pae_mean": float(batch_scores["bt_pae_mean"][chunk_offset]),
                    "tb_pae_mean": float(batch_scores["tb_pae_mean"][chunk_offset]),
                    "sequence": seq_str,
                }
                row.update(interface_metrics)
                row.update(rmsd_metrics)
                row["rmsd_filter_threshold"] = cfg.refold_rmsd_threshold
                row["rmsd_pass"] = (
                    _passes_max_threshold(
                        row["filter_rmsd"],
                        cfg.refold_rmsd_threshold,
                    )
                    and _passes_max_threshold(
                        row["filter_rmsd_design"],
                        cfg.refold_rmsd_threshold,
                    )
                )
                all_sample_rows.append(dict(row))

                if (
                    best_row is None
                    or _refold_rank_key(row) < _refold_rank_key(best_row)
                ):
                    best_row = row
                    best_structure = structure

        assert best_row is not None and best_structure is not None
        cif_path = refold_dir / f"edit_{edit_count}_sample_{best_row['sample_idx']}.cif"
        write_structure_cif(best_structure, cif_path)
        best_row["refold_cif"] = str(cif_path)
        rows.append(best_row)

    passing_rows = sorted(
        [row for row in rows if row["rmsd_pass"]],
        key=_refold_rank_key,
    )
    failed_rows = sorted(
        [row for row in rows if not row["rmsd_pass"]],
        key=_refold_rank_key,
    )
    ranked_rows = [
        {
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(passing_rows, start=1)
    ] + [
        {
            "rank": None,
            **row,
        }
        for row in failed_rows
    ]

    pl.DataFrame(ranked_rows).write_csv(cfg.output_dir / "refold_ranked.csv")
    pl.DataFrame(rows).write_csv(cfg.output_dir / "refold_best_by_edit_count.csv")
    pl.DataFrame(all_sample_rows).write_csv(cfg.output_dir / "refold_all_samples.csv")


# =============================================================================
# CLI entry point
# =============================================================================


def validate_and_normalize_cli_args(parser, args):
    """Fail before model loading and make child-process paths cwd-independent."""
    if args.complex_cif is None and args.boltzgen_yaml is None:
        parser.error("one of --complex-cif or --boltzgen-yaml is required")
    if args.num_designs < 1:
        parser.error("--num-designs must be >= 1")

    for attribute in ("complex_cif", "boltzgen_yaml", "boltzgen_if_checkpoint"):
        value = getattr(args, attribute)
        if value is None:
            continue
        path = value.expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{attribute.replace('_', '-')} not found: {path}")
        setattr(args, attribute, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["v0", "v1", "v2", "v3", "v4"], default="v3",
                   help="incremental milestone toggle")
    p.add_argument("--complex-cif", type=Path)
    p.add_argument("--boltzgen-yaml", type=Path)
    p.add_argument("--binder-chain", default="B")
    p.add_argument("--target-chains", nargs="+", default=["A"])
    p.add_argument("--cdr-indices", nargs="+", type=int)
    p.add_argument("--budget", type=int, default=7)
    p.add_argument("--output-dir", type=Path, default=Path("./vhh_designs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-seed", type=int,
                   help="First seed for --num-designs; defaults to --seed")
    p.add_argument("--num-designs", type=int, default=1,
                   help="Run N independent design jobs from this Python driver")
    p.add_argument("--devices", type=str,
                   help="GPU ids/count for independent jobs, e.g. 4 or 0,1,2,3")
    p.add_argument("--resume", type=int, choices=[0, 1], default=0,
                   help="For --num-designs, skip seed directories with complete outputs")
    p.add_argument("--num-sampling-steps", type=int)
    p.add_argument("--start-sigma-frac", type=float)
    p.add_argument("--step-scale", type=float)
    p.add_argument("--noise-scale", type=float)
    p.add_argument("--lambda-max", type=float)
    p.add_argument("--lambda-schedule",
                   choices=["sigma_squared", "sigma", "constant"])
    p.add_argument("--nos-inner-steps", type=int,
                   help="Real iterative NOS-style prior-compatibility inner loop "
                        "(section 12a); 0 (default) = off, one-shot merge unchanged")
    p.add_argument("--nos-inner-step-size", type=float)
    p.add_argument("--nos-lambda-kl", type=float)
    p.add_argument("--nos-langevin-noise", type=float)
    p.add_argument("--lookahead", type=int, choices=[0, 1],
                   help="Real look-ahead mechanism (section 12b): differentiate through "
                        "the full denoiser instead of the frozen x0_hat. Mutually "
                        "exclusive with --nos-inner-steps > 0")
    p.add_argument("--n-outer-iterations", type=int)
    p.add_argument("--search-mode", choices=["greedy", "mcmc", "both"])
    p.add_argument("--polish-steps", type=int)
    p.add_argument("--polish-batch-size", type=int)
    p.add_argument("--mcmc-steps", type=int)
    p.add_argument("--mcmc-temp", type=float)
    p.add_argument("--mcmc-proposal-temp", type=float)
    p.add_argument("--mcmc-max-path-length", type=int)
    p.add_argument(
        "--sequence-decoder",
        choices=["boltzgen_if"],
        help="JAX BoltzGen-IF decoder (the only decoder supported by this workflow)",
    )
    p.add_argument("--boltzgen-if-checkpoint", type=Path,
                   help="Native BoltzGen-IF checkpoint; downloads official default when omitted")
    p.add_argument("--boltzgen-if-device", type=str,
                   help="Torch device for native IF, e.g. auto, cpu, cuda, or cuda:0")
    p.add_argument("--boltzgen-if-temperature", type=float,
                   help="Autoregressive native IF sampling temperature")
    p.add_argument("--boltzgen-if-guidance-temperature", type=float,
                   help="Softmax temperature for differentiable JAX IF guidance")
    p.add_argument("--boltzgen-if-avoid", type=str,
                   help="One-letter amino acids excluded by native IF (default C)")
    p.add_argument(
        "--weight-boltzgen-if-prior",
        type=float,
        help="Fixed-backbone native IF NLL weight during greedy/MCMC search",
    )
    p.add_argument("--recycling-steps", type=int)
    p.add_argument("--refold-sampling-steps", type=int)
    p.add_argument("--refold-num-samples", type=int)
    p.add_argument("--refold-batch-size", type=int,
                   help="Refold samples to evaluate per batched model call "
                        "(used by both --refold-backend boltz2 and opendde)")
    p.add_argument("--refold-binder-template", type=int, choices=[0, 1],
                   help="Use parent binder chain as a refold template -- "
                        "--refold-backend boltz2 only; opendde's featurizer "
                        "does not support templates and ignores this")
    p.add_argument("--refold-binder-template-mode",
                   choices=["full", "framework", "none"],
                   help="Binder template mode for refold: full, framework-only, "
                        "or none -- --refold-backend boltz2 only; opendde "
                        "ignores this (see --refold-binder-template)")
    p.add_argument("--ipsae-pae-cutoff", type=float)
    p.add_argument("--refold-rmsd-threshold", type=float,
                   help="BoltzGen-like CA RMSD filter using filter_rmsd and filter_rmsd_design; <=0 disables")
    p.add_argument("--refold-backend", choices=["boltz2", "opendde"],
                   help="Structure model for post-refold ranking (--skip-refold 0)")
    p.add_argument("--weight-ablang2", "--weight-ablang", dest="weight_ablang2",
                   type=float, help="AbLang2 PLL weight; --weight-ablang is a deprecated alias")
    p.add_argument("--weight-edit-budget", type=float)
    p.add_argument("--weight-boltz2-ptm-energy", type=float,
                   help="Boltz2 guidance weight for smooth cross-chain pTM energy")
    p.add_argument("--weight-boltz2-interface-pae", "--weight-boltz2-ipae",
                   dest="weight_boltz2_interface_pae", type=float,
                   help="Boltz2 guidance weight for mean interface PAE")
    p.add_argument("--weight-boltz2-iptm", type=float,
                   help="Boltz2 guidance weight for maximizing ipTM")
    p.add_argument("--weight-boltz2-ipsae", type=float,
                   help="Disabled for in-loop guidance (ipSAE is post-refold-"
                        "only); setting > 0 raises an error")
    p.add_argument("--boltz2-guidance-recycling-steps", type=int,
                   help="Boltz2 recycling steps inside per-step guidance")
    p.add_argument("--boltz2-guidance-sampling-steps", type=int,
                   help="Boltz2 structure sampling steps inside per-step guidance")
    p.add_argument("--boltz2-guidance-target-template", type=int, choices=[0, 1],
                   help="Use target chain template for Boltz2 guidance features")
    p.add_argument("--weight-opendde-iptm", type=float,
                   help="OpenDDE guidance weight for the distogram-based ipTM proxy "
                        "(DistogramIPTMProxy); mutually exclusive with --weight-boltz2-*")
    p.add_argument("--weight-opendde-contact", type=float,
                   help="OpenDDE guidance weight for CDR-target distogram contact "
                        "(BinderTargetContact); mutually exclusive with --weight-boltz2-*")
    p.add_argument("--opendde-guidance-recycling-steps", type=int,
                   help="OpenDDE recycling steps inside per-step guidance")
    p.add_argument("--opendde-contact-distance", type=float,
                   help="BinderTargetContact.contact_distance for OpenDDE "
                        "guidance (--weight-opendde-contact); default 8.0 is a "
                        "physical-contact-like threshold, tighter than "
                        "BinderTargetContact's own 20.0 default (broad "
                        "proximity, not a contact criterion)")
    p.add_argument("--clip-gradient-norm", type=float)
    p.add_argument("--skip-guidance", type=int, choices=[0, 1],
                   help="Override the mode preset; 1 disables diffusion guidance")
    p.add_argument("--skip-polish", type=int, choices=[0, 1],
                   help="Override the mode preset; 1 disables sequence polish/search")
    p.add_argument("--skip-refold", type=int, choices=[0, 1],
                   help="Override the mode preset; 1 disables Boltz2 refold/ranking")
    p.add_argument("--log-guidance-diagnostics", type=int, choices=[0, 1],
                   help="Log Phase 2 guidance diagnostics (cosine similarity, "
                        "norm ratio, inter-objective conflict) per outer "
                        "iteration to <output-dir>/guidance_diagnostics.json")
    p.add_argument("--guidance-diagnostics-cos-threshold", type=float,
                   help="cos(guided,unguided) below this counts as strong "
                        "directional disagreement in the diagnostics report")
    p.add_argument("--guidance-diagnostics-sigma-bins", type=int,
                   help="Number of noise-level bins for diagnostics stratification")
    args = validate_and_normalize_cli_args(p, p.parse_args())

    if args.num_designs > 1 or args.devices is not None:
        run_many_from_cli(args)
        raise SystemExit(0)

    complex_cif = args.complex_cif
    if complex_cif is None and args.boltzgen_yaml is not None:
        yaml_string = args.boltzgen_yaml.read_text()
        yaml_files = boltzgen_yaml_files(args.boltzgen_yaml, yaml_string)
        if len(yaml_files) != 1:
            raise ValueError(
                "--complex-cif is required when the YAML references multiple files"
            )
        complex_cif = next(iter(yaml_files.values()))

    cfg = VHHDesignConfig(
        complex_cif_path=complex_cif,
        binder_chain_id=args.binder_chain,
        target_chain_ids=args.target_chains,
        cdr_residue_indices=args.cdr_indices or [],
        boltzgen_yaml_path=args.boltzgen_yaml,
        edit_budget=args.budget,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    overrides = {
        "num_sampling_steps": args.num_sampling_steps,
        "start_sigma_frac": args.start_sigma_frac,
        "step_scale": args.step_scale,
        "noise_scale": args.noise_scale,
        "lambda_max": args.lambda_max,
        "lambda_schedule": args.lambda_schedule,
        "nos_inner_steps": args.nos_inner_steps,
        "nos_inner_step_size": args.nos_inner_step_size,
        "nos_lambda_kl": args.nos_lambda_kl,
        "nos_langevin_noise": args.nos_langevin_noise,
        "lookahead": bool(args.lookahead) if args.lookahead is not None else None,
        "n_outer_iterations": args.n_outer_iterations,
        "search_mode": args.search_mode,
        "polish_steps": args.polish_steps,
        "polish_batch_size": args.polish_batch_size,
        "mcmc_steps": args.mcmc_steps,
        "mcmc_temp": args.mcmc_temp,
        "mcmc_proposal_temp": args.mcmc_proposal_temp,
        "mcmc_max_path_length": args.mcmc_max_path_length,
        "sequence_decoder": args.sequence_decoder,
        "boltzgen_if_checkpoint": args.boltzgen_if_checkpoint,
        "boltzgen_if_device": args.boltzgen_if_device,
        "boltzgen_if_temperature": args.boltzgen_if_temperature,
        "boltzgen_if_guidance_temperature": args.boltzgen_if_guidance_temperature,
        "boltzgen_if_avoid": args.boltzgen_if_avoid,
        "weight_boltzgen_if_prior": args.weight_boltzgen_if_prior,
        "recycling_steps": args.recycling_steps,
        "refold_sampling_steps": args.refold_sampling_steps,
        "refold_num_samples": args.refold_num_samples,
        "refold_batch_size": args.refold_batch_size,
        "refold_binder_template": (
            bool(args.refold_binder_template)
            if args.refold_binder_template is not None else None
        ),
        "refold_binder_template_mode": args.refold_binder_template_mode,
        "ipsae_pae_cutoff": args.ipsae_pae_cutoff,
        "refold_rmsd_threshold": args.refold_rmsd_threshold,
        "refold_backend": args.refold_backend,
        "weight_ablang2": args.weight_ablang2,
        "weight_edit_budget": args.weight_edit_budget,
        "weight_boltz2_ptm_energy": args.weight_boltz2_ptm_energy,
        "weight_boltz2_interface_pae": args.weight_boltz2_interface_pae,
        "weight_boltz2_iptm": args.weight_boltz2_iptm,
        "weight_boltz2_ipsae": args.weight_boltz2_ipsae,
        "boltz2_guidance_recycling_steps": args.boltz2_guidance_recycling_steps,
        "boltz2_guidance_sampling_steps": args.boltz2_guidance_sampling_steps,
        "boltz2_guidance_target_template": (
            bool(args.boltz2_guidance_target_template)
            if args.boltz2_guidance_target_template is not None else None
        ),
        "weight_opendde_iptm": args.weight_opendde_iptm,
        "weight_opendde_contact": args.weight_opendde_contact,
        "opendde_guidance_recycling_steps": args.opendde_guidance_recycling_steps,
        "opendde_contact_distance": args.opendde_contact_distance,
        "clip_gradient_norm": args.clip_gradient_norm,
        "log_guidance_diagnostics": (
            bool(args.log_guidance_diagnostics)
            if args.log_guidance_diagnostics is not None else None
        ),
        "guidance_diagnostics_cos_threshold": args.guidance_diagnostics_cos_threshold,
        "guidance_diagnostics_sigma_bins": args.guidance_diagnostics_sigma_bins,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)

    # Mode-driven flag presets
    if args.mode == "v0":
        cfg.skip_guidance = True
        cfg.skip_polish = True
        cfg.skip_refold = True
        cfg.weight_ablang2 = 0.0
        cfg.weight_boltz2_ptm_energy = 0.0
        cfg.weight_boltz2_interface_pae = 0.0
        cfg.weight_boltz2_iptm = 0.0
        cfg.weight_boltz2_ipsae = 0.0
        cfg.weight_opendde_iptm = 0.0
        cfg.weight_opendde_contact = 0.0
    elif args.mode == "v1":
        # EditBudget-only guidance; zero-out other guidance terms.
        cfg.skip_guidance = False
        cfg.skip_polish = True
        cfg.skip_refold = True
        cfg.weight_ablang2 = 0.0
        cfg.weight_boltz2_ptm_energy = 0.0
        cfg.weight_boltz2_interface_pae = 0.0
        cfg.weight_boltz2_iptm = 0.0
        cfg.weight_boltz2_ipsae = 0.0
        cfg.weight_opendde_iptm = 0.0
        cfg.weight_opendde_contact = 0.0
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

    if args.skip_guidance is not None:
        cfg.skip_guidance = bool(args.skip_guidance)
    if args.skip_polish is not None:
        cfg.skip_polish = bool(args.skip_polish)
    if args.skip_refold is not None:
        cfg.skip_refold = bool(args.skip_refold)

    run(cfg)
