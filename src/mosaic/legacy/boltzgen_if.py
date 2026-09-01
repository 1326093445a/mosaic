"""Native BoltzGen inverse folding for completed Mosaic backbones.

This module is intentionally a discrete, post-diffusion adapter. It does not
participate in JAX autodiff; the upstream BoltzGen IF ``sample`` path is a
PyTorch autoregressive decoder running under ``torch.no_grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from boltzgen.data import const
from boltzgen.model.models.boltz import Boltz

from mosaic.common import TOKENS


DEFAULT_IF_REPO = "boltzgen/boltzgen-1"
DEFAULT_IF_FILENAME = "boltzgen1_ifold.ckpt"


@dataclass(frozen=True)
class BoltzGenIFResult:
    """One native IF decode in Mosaic's canonical 20-amino-acid alphabet."""

    sequence_ids: np.ndarray
    logits: np.ndarray

    @property
    def sequence(self) -> str:
        return "".join(TOKENS[int(i)] for i in self.sequence_ids)


def resolve_boltzgen_if_checkpoint(
    checkpoint: Path | str | None = None,
    *,
    cache_dir: Path | str = Path("~/.boltz").expanduser(),
) -> Path:
    """Return a local official IF checkpoint, downloading it when absent."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BoltzGen-IF checkpoint not found: {path}")
        return path

    cache_dir = Path(cache_dir).expanduser().resolve()
    expected = cache_dir / DEFAULT_IF_FILENAME
    if expected.is_file():
        return expected

    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=DEFAULT_IF_REPO,
        filename=DEFAULT_IF_FILENAME,
        local_dir=cache_dir,
    )
    return Path(downloaded).resolve()


def resolve_torch_device(device: str = "auto") -> torch.device:
    """Resolve the IF device without assuming the Torch build has CUDA."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"BoltzGen-IF device {device!r} requested, but Torch CUDA is unavailable"
        )
    return resolved


def load_boltzgen_if(
    checkpoint: Path | str | None = None,
    *,
    device: str = "auto",
    temperature: float = 0.3,
    avoid: str = "C",
) -> Boltz:
    """Load the official upstream BoltzGen inverse-folding checkpoint."""
    if temperature <= 0:
        raise ValueError("BoltzGen-IF temperature must be > 0")

    invalid = sorted(set(avoid) - set(const.prot_letter_to_token))
    if invalid:
        raise ValueError(f"Unknown residues in BoltzGen-IF avoid set: {invalid}")

    checkpoint_path = resolve_boltzgen_if_checkpoint(checkpoint)
    torch_device = resolve_torch_device(device)
    model = Boltz.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        weights_only=False,
        # Official checkpoints retain training validators, some of which carry
        # CUDA metric state and defeat map_location="cpu" during Lightning load.
        validators=None,
    ).eval()
    model.structure_module.sampling_temperature = float(temperature)
    model.structure_module.inverse_fold_restriction = [
        const.prot_letter_to_token[one_letter] for one_letter in avoid
    ]
    return model.to(torch_device)


def _move_feature(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {key: _move_feature(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_feature(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_feature(item, device) for item in value)
    return value


def prepare_boltzgen_if_features(
    torch_features: dict[str, Any],
    coords: np.ndarray,
    *,
    parent_sequence: np.ndarray,
    designable_mask: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Build an IF feature dictionary for a final Mosaic atom-coordinate sample."""
    features = {
        key: _move_feature(value, device) for key, value in torch_features.items()
    }

    coords_tensor = torch.tensor(
        np.array(coords, dtype=np.float32, copy=True), device=device
    )
    if coords_tensor.ndim != 2 or coords_tensor.shape[-1] != 3:
        raise ValueError(f"Expected final coords with shape (M, 3), got {coords_tensor.shape}")

    atom_count = int(features["atom_pad_mask"].shape[-1])
    if coords_tensor.shape[0] != atom_count:
        raise ValueError(
            f"Coordinate count {coords_tensor.shape[0]} does not match features {atom_count}"
        )

    parent = torch.tensor(
        np.array(parent_sequence, dtype=np.float32, copy=True), device=device
    )
    mask = torch.tensor(
        np.array(designable_mask, dtype=bool, copy=True), device=device
    )
    token_count = int(features["token_pad_mask"].shape[-1])
    if parent.shape != (token_count, 20):
        raise ValueError(
            f"Expected parent sequence shape {(token_count, 20)}, got {tuple(parent.shape)}"
        )
    if mask.shape != (token_count,):
        raise ValueError(
            f"Expected designable mask shape {(token_count,)}, got {tuple(mask.shape)}"
        )

    # IF's geometry encoder expects (batch, atoms, xyz), whereas the regular
    # BoltzGen featurizer stores one conformation as (batch, 1, atoms, xyz).
    features["coords"] = coords_tensor.unsqueeze(0)

    # Protein token center is CA, represented by bb4 slot 1 ([N, CA, C, O]).
    # Recompute it from the diffused coordinates so KNN edges match the sample.
    ca_to_atom = features["token_to_bb4_atoms"][:, :, 1].float()
    updated_centers = torch.bmm(ca_to_atom, coords_tensor.unsqueeze(0))
    has_ca = ca_to_atom.sum(dim=-1, keepdim=True) > 0
    features["center_coords"] = torch.where(
        has_ca, updated_centers, features["center_coords"].float()
    )

    parent_boltz = torch.zeros(
        (1, token_count, const.num_tokens), dtype=torch.float32, device=device
    )
    parent_boltz[:, :, const.canonicals_offset : const.canonicals_offset + 20] = parent
    original = features["res_type_clone"].float()
    protein_mask = features["mol_type"] == const.chain_type_ids["PROTEIN"]
    features["res_type_clone"] = torch.where(
        protein_mask.unsqueeze(-1), parent_boltz, original
    )
    features["res_type"] = features["res_type_clone"].clone()
    features["inverse_fold_design_mask"] = mask.unsqueeze(0)
    features["design_mask"] = mask.unsqueeze(0).to(features["design_mask"].dtype)
    return features


def decode_with_boltzgen_if(
    model: Boltz,
    torch_features: dict[str, Any],
    coords: np.ndarray,
    *,
    parent_sequence: np.ndarray,
    designable_mask: np.ndarray,
    seed: int,
) -> BoltzGenIFResult:
    """Autoregressively decode one final backbone with native BoltzGen-IF."""
    device = next(model.parameters()).device
    features = prepare_boltzgen_if_features(
        torch_features,
        coords,
        parent_sequence=parent_sequence,
        designable_mask=designable_mask,
        device=device,
    )

    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        output = model(features, recycling_steps=0)

    canonical_slice = slice(
        const.canonicals_offset, const.canonicals_offset + len(const.canonical_tokens)
    )
    logits = output["logits"][0, :, canonical_slice].float().cpu().numpy()
    sampled = output["res_type"][0, :, canonical_slice]
    sequence_ids = sampled.argmax(dim=-1).cpu().numpy().astype(np.int32)

    # Native IF already preserves these positions, but enforcing the contract
    # here protects Mosaic if upstream decoder behavior changes.
    parent_ids = np.asarray(parent_sequence).argmax(axis=-1).astype(np.int32)
    designable = np.asarray(designable_mask, dtype=bool)
    sequence_ids = np.where(designable, sequence_ids, parent_ids)
    return BoltzGenIFResult(sequence_ids=sequence_ids, logits=logits)
