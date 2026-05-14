"""Guided partial-diffusion + edit-budgeted CDR redesign for VHHs.

Pipeline (per outer-loop iteration):
  1. guided_partial_diffusion: BoltzGen partial diffusion with classifier guidance
     from ESM2/AbLang2/EditBudget and optional Boltz2 target-aware terms,
     perturbing CDR atoms only.
  2. differentiable_inverse_fold(temp=0.001): decode guided coords to a soft sequence.
  3. edit_budgeted_greedy_descent: polish the discrete sequence under the full
     multi-model loss with a hard <=budget edit constraint vs the parent.
  4. Record a Pareto front {edit_count: (loss, sequence)}.
  5. Update parent if improved, repeat.

After convergence: refold final candidates with Boltz2, rank primarily by ipSAE,
and write interface metrics, target-aligned RMSD, and CIFs.

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
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.esm import ESM2PseudoLikelihood, load_esm2
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
from mosaic.optimizers import (
    edit_budgeted_greedy_descent,
    edit_budgeted_gradient_mcmc,
)
from mosaic.proteinmpnn.mpnn import load_abmpnn
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
    weight_esm2: float = 0.10
    weight_ablang2: float = 0.10
    weight_mpnn_recovery: float = 0.50
    weight_edit_budget: float = 5.00
    weight_boltz2_ptm_energy: float = 0.0
    weight_boltz2_interface_pae: float = 0.0
    boltz2_guidance_recycling_steps: int = 0
    boltz2_guidance_sampling_steps: int = 5
    boltz2_guidance_target_template: bool = True
    clip_gradient_norm: float = 1.0  # per-term gradient clip for balance
    esm2_model_name: str = "esm2_t33_650M_UR50D"

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
    search_mode: str = "greedy"  # one of {greedy, mcmc, both}
    polish_steps: int = 200
    polish_batch_size: int = 16
    mcmc_steps: int = 100
    mcmc_temp: float = 0.02
    mcmc_proposal_temp: float = 0.01
    mcmc_max_path_length: int = 2

    # ---- I/O ----
    output_dir: Path = Path("./vhh_designs")

    # ---- Toggles for incremental testing ----
    skip_guidance: bool = False
    skip_polish: bool = False
    skip_refold: bool = True  # default off; task #15 wires this back in

    # ---- Misc ----
    recycling_steps: int = 3
    refold_sampling_steps: int = 25
    refold_num_samples: int = 1
    refold_batch_size: int = 1
    refold_binder_template: bool = True
    ipsae_pae_cutoff: float = 12.0
    refold_rmsd_threshold: float = 2.5
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

    rotation, translation = _fit_transform(ref_target, orig_target)
    ref_target_aligned = _apply_transform(ref_target, rotation, translation)
    ref_binder_aligned = _apply_transform(ref_binder, rotation, translation)

    metrics = {
        "target_ca_rmsd_target_aligned": _rmsd(ref_target_aligned, orig_target),
        "binder_ca_rmsd_target_aligned": _rmsd(ref_binder_aligned, orig_binder),
        "target_ca_rmsd_n": min(len(ref_target), len(orig_target)),
        "binder_ca_rmsd_n": min(len(ref_binder), len(orig_binder)),
    }

    if cdr_residue_indices:
        # Mosaic/BoltzGen YAML `res_index` values are 1-based chain-order
        # positions, not necessarily PDB author residue numbers. This matters
        # for VHHs with Kabat insertion codes such as H52A or H100A-H.
        cdr_set = {int(i) for i in cdr_residue_indices}
        cdr_positions = [
            i for i in range(min(len(orig_binder_keys), len(ref_binder_aligned)))
            if (i + 1) in cdr_set
        ]
        if cdr_positions:
            metrics["cdr_ca_rmsd_target_aligned"] = _rmsd(
                ref_binder_aligned[cdr_positions],
                orig_binder[cdr_positions],
            )
            metrics["cdr_ca_rmsd_n"] = len(cdr_positions)
        else:
            metrics["cdr_ca_rmsd_target_aligned"] = float("nan")
            metrics["cdr_ca_rmsd_n"] = 0
    return metrics


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
        {"rank": "", **row}
        for row in failed_rows
    ]

    output_path = root / "combined_refold_ranked.csv"
    if ranked_rows:
        pl.DataFrame(ranked_rows).select(columns).write_csv(output_path)
    else:
        output_path.write_text(",".join(columns) + "\n")
    return len(ranked_rows)


def _append_option(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def uses_boltz2_guidance(cfg: VHHDesignConfig) -> bool:
    return (
        cfg.weight_boltz2_ptm_energy > 0
        or cfg.weight_boltz2_interface_pae > 0
    )


def build_single_design_command(args, *, seed: int, output_dir: Path) -> list[str]:
    """Reconstruct the CLI for one child design job."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", args.mode]

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
    _append_option(cmd, "--n-outer-iterations", args.n_outer_iterations)
    _append_option(cmd, "--search-mode", args.search_mode)
    _append_option(cmd, "--polish-steps", args.polish_steps)
    _append_option(cmd, "--polish-batch-size", args.polish_batch_size)
    _append_option(cmd, "--mcmc-steps", args.mcmc_steps)
    _append_option(cmd, "--mcmc-temp", args.mcmc_temp)
    _append_option(cmd, "--mcmc-proposal-temp", args.mcmc_proposal_temp)
    _append_option(cmd, "--mcmc-max-path-length", args.mcmc_max_path_length)
    _append_option(cmd, "--recycling-steps", args.recycling_steps)
    _append_option(cmd, "--refold-sampling-steps", args.refold_sampling_steps)
    _append_option(cmd, "--refold-num-samples", args.refold_num_samples)
    _append_option(cmd, "--refold-batch-size", args.refold_batch_size)
    _append_option(cmd, "--refold-binder-template", args.refold_binder_template)
    _append_option(cmd, "--ipsae-pae-cutoff", args.ipsae_pae_cutoff)
    _append_option(cmd, "--refold-rmsd-threshold", args.refold_rmsd_threshold)
    _append_option(cmd, "--esm2-model", args.esm2_model)
    _append_option(cmd, "--weight-esm2", args.weight_esm2)
    _append_option(cmd, "--weight-ablang2", args.weight_ablang2)
    _append_option(cmd, "--weight-edit-budget", args.weight_edit_budget)
    _append_option(cmd, "--weight-boltz2-ptm-energy", args.weight_boltz2_ptm_energy)
    _append_option(
        cmd,
        "--weight-boltz2-interface-pae",
        args.weight_boltz2_interface_pae,
    )
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
    _append_option(cmd, "--clip-gradient-norm", args.clip_gradient_norm)
    return cmd


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

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    print("[multi] launching independent design jobs")
    print(f"[multi] output root: {root}")
    print(f"[multi] num_designs: {num_designs}")
    print(f"[multi] start_seed: {start_seed}")
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
            cmd = build_single_design_command(args, seed=seed, output_dir=out_dir)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            child_cuda_visible = resolve_child_cuda_visible_devices(device)
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
                cwd=Path(__file__).resolve().parents[1],
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
    mpnn: any = None
    esm2_pll: any = None
    ablang2_model: any = None
    ablang2_tokenizer: any = None
    boltz2: any = None


def load_core_models(cfg: VHHDesignConfig) -> LoadedModels:
    """Load only the BoltzGen model needed to build the sampler/trunk state."""
    del cfg
    stage_log("loading BoltzGen model")
    boltzgen = load_boltzgen()
    stage_log("loaded BoltzGen model")
    return LoadedModels(boltzgen=boltzgen)


def load_guidance_models(cfg: VHHDesignConfig, models: LoadedModels) -> LoadedModels:
    """Load IF/language guidance models after BoltzGen trunk conditioning.

    `Sampler.from_features` has a large compile/runtime peak. Deferring ESM2 and
    AbLang2 until after that stage avoids keeping those model weights resident
    during the BoltzGen sampler allocation.
    """
    if models.mpnn is None:
        stage_log("loading ABMPNN inverse-folding model")
        models.mpnn = load_abmpnn()
        stage_log("loaded ABMPNN inverse-folding model")

    # stop_grad=False on the language models because we need gradients to flow
    # back to coords through the differentiable IF bridge during guidance.
    if cfg.weight_esm2 > 0 and models.esm2_pll is None:
        stage_log(f"loading ESM2 model {cfg.esm2_model_name}")
        esm2 = load_esm2(cfg.esm2_model_name)
        models.esm2_pll = ESM2PseudoLikelihood(esm2, stop_grad=False)
        stage_log(f"loaded ESM2 model {cfg.esm2_model_name}")
    elif cfg.weight_esm2 <= 0:
        stage_log("skipping ESM2 model because weight_esm2 <= 0")

    if cfg.weight_ablang2 > 0 and models.ablang2_model is None:
        stage_log("loading AbLang2 model")
        models.ablang2_model, models.ablang2_tokenizer = load_ablang2()
        stage_log("loaded AbLang2 model")
    elif cfg.weight_ablang2 <= 0:
        stage_log("skipping AbLang2 model because weight_ablang2 <= 0")

    if uses_boltz2_guidance(cfg) and models.boltz2 is None:
        stage_log("loading Boltz2 model for target-aware guidance")
        from mosaic.models.boltz2 import Boltz2

        models.boltz2 = Boltz2()
        stage_log("loaded Boltz2 model for target-aware guidance")
    elif not uses_boltz2_guidance(cfg):
        stage_log("skipping Boltz2 guidance model because its weights are <= 0")
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
    ABMPNN's soft full-binder sequence into this loss, replacing the placeholder.
    Target templates are allowed; binder templates are intentionally not used
    here so the confidence loss stays sensitive to sequence changes.
    """
    if models.boltz2 is None:
        raise ValueError("Boltz2 model was not loaded; check Boltz2 guidance weights")

    from mosaic.losses.structure_prediction import (
        BinderTargetPAE,
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
    if not loss_terms:
        raise ValueError("Boltz2 guidance requested without positive loss weights")

    boltz2_sequence_loss = sum(loss_terms[1:], start=loss_terms[0])
    stage_log(
        "Boltz2 guidance: built loss "
        f"(pTMEnergy={cfg.weight_boltz2_ptm_energy}, "
        f"interface_PAE={cfg.weight_boltz2_interface_pae}, "
        f"recycle={cfg.boltz2_guidance_recycling_steps}, "
        f"sample_steps={cfg.boltz2_guidance_sampling_steps})"
    )
    return models.boltz2.build_loss(
        loss=boltz2_sequence_loss,
        features=features,
        recycling_steps=cfg.boltz2_guidance_recycling_steps,
        sampling_steps=cfg.boltz2_guidance_sampling_steps,
    )


def build_guidance_loss(cfg: VHHDesignConfig, models: LoadedModels,
                        parent_one_hot: Float[Array, "N 20"],
                        designable_token_mask: Bool[Array, "N"],
                        sequence_loss_indices: Int[Array, "M"]):
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

    if cfg.weight_esm2 > 0:
        aux_terms.append(
            cfg.weight_esm2
            * SequenceSubsetLoss(
                ClippedGradient(models.esm2_pll, cfg.clip_gradient_norm),
                sequence_loss_indices,
            )
        )
    if cfg.weight_ablang2 > 0:
        ablang2_pll = make_ablang2_loss(
            models, sequence_loss_indices, stop_grad=False
        )
        aux_terms.append(
            cfg.weight_ablang2
            * SequenceSubsetLoss(
                ClippedGradient(ablang2_pll, cfg.clip_gradient_norm),
                sequence_loss_indices,
            )
        )
    if uses_boltz2_guidance(cfg):
        boltz2_guidance_loss = build_boltz2_guidance_loss(
            cfg,
            models,
            binder_length=int(sequence_loss_indices.shape[0]),
        )
        aux_terms.append(
            SequenceSubsetLoss(
                ClippedGradient(boltz2_guidance_loss, cfg.clip_gradient_norm),
                sequence_loss_indices,
            )
        )
    # MPNN sequence recovery is a structure-prediction-output loss, so it needs
    # to be plumbed differently — see polish loss below. For guidance during
    # diffusion we use sequence-only auxiliary terms plus the optional Boltz2
    # target-aware sequence loss through the ABMPNN soft-sequence bridge.

    return sum(aux_terms[1:], start=aux_terms[0])


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
    if cfg.weight_esm2 > 0:
        terms.append(
            cfg.weight_esm2 * SequenceSubsetLoss(models.esm2_pll, sequence_loss_indices)
        )
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
    residue_index = squeeze_feature(features["residue_index"], "residue_index", 1)
    binder_token_indices = binder_indices_from_design_mask(
        asym_id, designable_token_mask
    )
    token_to_bb4_atoms = squeeze_feature(
        features["token_to_bb4_atoms"], "token_to_bb4_atoms", 3
    )
    bb_atom_inds = jnp.argmax(token_to_bb4_atoms, axis=-1)  # (N, 4)

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
    stage_log("finished guidance model loading")

    stage_log("building guidance and polish losses")
    guidance_loss = build_guidance_loss(
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

    for outer in range(cfg.n_outer_iterations):
        t0 = time.time()
        print(f"\n[outer {outer}] guided partial diffusion "
              f"(start_sigma_frac={cfg.start_sigma_frac}, "
              f"steps={cfg.num_sampling_steps})...", flush=True)

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

        stage_log(f"outer {outer}: starting guided partial diffusion")
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
        stage_log(f"outer {outer}: finished guided partial diffusion")

        # ----- Stage 1.5: decode to discrete sequence -----
        # x_final is unbatched (M, 3); extract backbone for IF.
        stage_log(f"outer {outer}: decoding final structure with ABMPNN")
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
        print(
            f"[outer {outer}] diffusion produced {diffusion_edits} edits vs parent",
            flush=True,
        )
        stage_log(f"outer {outer}: finished ABMPNN decode")

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
                    loss=polish_loss,
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
                    loss=polish_loss,
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
        print("[refold] refolding Pareto candidates with Boltz2...", flush=True)
        stage_log("starting Boltz2 refold")
        refold_pareto_with_boltz2(global_pareto, cfg, binder_token_indices)
        stage_log("finished Boltz2 refold")

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
    binder_template_chain = None
    if cfg.refold_binder_template:
        binder_template_chain = target_struct[0][cfg.binder_chain_id].clone()
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
        f"(binder_template={cfg.refold_binder_template})"
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
                row["rmsd_pass"] = _passes_max_threshold(
                    row["binder_ca_rmsd_target_aligned"],
                    cfg.refold_rmsd_threshold,
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
    p.add_argument("--num-sampling-steps", type=int)
    p.add_argument("--start-sigma-frac", type=float)
    p.add_argument("--step-scale", type=float)
    p.add_argument("--noise-scale", type=float)
    p.add_argument("--lambda-max", type=float)
    p.add_argument("--lambda-schedule",
                   choices=["sigma_squared", "sigma", "constant"])
    p.add_argument("--n-outer-iterations", type=int)
    p.add_argument("--search-mode", choices=["greedy", "mcmc", "both"])
    p.add_argument("--polish-steps", type=int)
    p.add_argument("--polish-batch-size", type=int)
    p.add_argument("--mcmc-steps", type=int)
    p.add_argument("--mcmc-temp", type=float)
    p.add_argument("--mcmc-proposal-temp", type=float)
    p.add_argument("--mcmc-max-path-length", type=int)
    p.add_argument("--recycling-steps", type=int)
    p.add_argument("--refold-sampling-steps", type=int)
    p.add_argument("--refold-num-samples", type=int)
    p.add_argument("--refold-batch-size", type=int,
                   help="Boltz2 refold samples to evaluate per batched model call")
    p.add_argument("--refold-binder-template", type=int, choices=[0, 1],
                   help="Use parent binder chain as a Boltz2 refold template")
    p.add_argument("--ipsae-pae-cutoff", type=float)
    p.add_argument("--refold-rmsd-threshold", type=float,
                   help="Binder CA RMSD filter after target alignment; <=0 disables")
    p.add_argument("--esm2-model", default=None,
                   help="ESM2 checkpoint name in esm.pretrained, e.g. esm2_t33_650M_UR50D")
    p.add_argument("--weight-esm2", "--weight-esmc", dest="weight_esm2",
                   type=float, help="ESM2 PLL weight; --weight-esmc is a deprecated alias")
    p.add_argument("--weight-ablang2", "--weight-ablang", dest="weight_ablang2",
                   type=float, help="AbLang2 PLL weight; --weight-ablang is a deprecated alias")
    p.add_argument("--weight-edit-budget", type=float)
    p.add_argument("--weight-boltz2-ptm-energy", type=float,
                   help="Boltz2 guidance weight for smooth cross-chain pTM energy")
    p.add_argument("--weight-boltz2-interface-pae", "--weight-boltz2-ipae",
                   dest="weight_boltz2_interface_pae", type=float,
                   help="Boltz2 guidance weight for mean interface PAE")
    p.add_argument("--boltz2-guidance-recycling-steps", type=int,
                   help="Boltz2 recycling steps inside per-step guidance")
    p.add_argument("--boltz2-guidance-sampling-steps", type=int,
                   help="Boltz2 structure sampling steps inside per-step guidance")
    p.add_argument("--boltz2-guidance-target-template", type=int, choices=[0, 1],
                   help="Use target chain template for Boltz2 guidance features")
    p.add_argument("--clip-gradient-norm", type=float)
    args = p.parse_args()

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

    if complex_cif is not None:
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
    else:
        cfg = _example_config()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed

    overrides = {
        "num_sampling_steps": args.num_sampling_steps,
        "start_sigma_frac": args.start_sigma_frac,
        "step_scale": args.step_scale,
        "noise_scale": args.noise_scale,
        "lambda_max": args.lambda_max,
        "lambda_schedule": args.lambda_schedule,
        "n_outer_iterations": args.n_outer_iterations,
        "search_mode": args.search_mode,
        "polish_steps": args.polish_steps,
        "polish_batch_size": args.polish_batch_size,
        "mcmc_steps": args.mcmc_steps,
        "mcmc_temp": args.mcmc_temp,
        "mcmc_proposal_temp": args.mcmc_proposal_temp,
        "mcmc_max_path_length": args.mcmc_max_path_length,
        "recycling_steps": args.recycling_steps,
        "refold_sampling_steps": args.refold_sampling_steps,
        "refold_num_samples": args.refold_num_samples,
        "refold_batch_size": args.refold_batch_size,
        "refold_binder_template": (
            bool(args.refold_binder_template)
            if args.refold_binder_template is not None else None
        ),
        "ipsae_pae_cutoff": args.ipsae_pae_cutoff,
        "refold_rmsd_threshold": args.refold_rmsd_threshold,
        "esm2_model_name": args.esm2_model,
        "weight_esm2": args.weight_esm2,
        "weight_ablang2": args.weight_ablang2,
        "weight_edit_budget": args.weight_edit_budget,
        "weight_boltz2_ptm_energy": args.weight_boltz2_ptm_energy,
        "weight_boltz2_interface_pae": args.weight_boltz2_interface_pae,
        "boltz2_guidance_recycling_steps": args.boltz2_guidance_recycling_steps,
        "boltz2_guidance_sampling_steps": args.boltz2_guidance_sampling_steps,
        "boltz2_guidance_target_template": (
            bool(args.boltz2_guidance_target_template)
            if args.boltz2_guidance_target_template is not None else None
        ),
        "clip_gradient_norm": args.clip_gradient_norm,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)

    # Mode-driven flag presets
    if args.mode == "v0":
        cfg.skip_guidance = True
        cfg.skip_polish = True
        cfg.skip_refold = True
        cfg.weight_esm2 = 0.0
        cfg.weight_ablang2 = 0.0
        cfg.weight_boltz2_ptm_energy = 0.0
        cfg.weight_boltz2_interface_pae = 0.0
    elif args.mode == "v1":
        # EditBudget-only guidance; zero-out other guidance terms.
        cfg.skip_guidance = False
        cfg.skip_polish = True
        cfg.skip_refold = True
        cfg.weight_esm2 = 0.0
        cfg.weight_ablang2 = 0.0
        cfg.weight_boltz2_ptm_energy = 0.0
        cfg.weight_boltz2_interface_pae = 0.0
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
