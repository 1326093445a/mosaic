#####
#
# Note: this is pretty rushed, will come back and clean up later
# data loading and structure writing is **terrible**
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import joltzgen
import numpy as np
import torch
from boltzgen.data import const
from boltzgen.data.data import (
    Structure,
    convert_ccd,
)
from boltzgen.data.feature.featurizer import (
    Featurizer,
    res_all_gly,
    res_from_atom14,
    res_from_atom37,
)
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.task.predict.data_from_yaml import DataConfig, FromYamlDataModule
from boltzgen.task.predict.writer import DesignWriter
from jaxtyping import Array, Bool, Float, Int, PyTree

from ..util import pairwise_distance
from .guidance_lookahead import build_lookahead_grad_fn



def load_boltzgen(checkpoint_dir=Path("~/.boltz/").expanduser(), model_diverse=True):
    checkpoints = ["boltzgen1_adherence.ckpt", "boltzgen1_diverse.ckpt"]
    print(f"[load_boltzgen] checking checkpoints in {checkpoint_dir}", flush=True)
    if not all((checkpoint_dir / ckpt).exists() for ckpt in checkpoints):
        print(f"[load_boltzgen] downloading Boltz checkpoints to {checkpoint_dir}", flush=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for ckpt in checkpoints:
            print(f"[load_boltzgen] downloading {ckpt}", flush=True)
            subprocess.run(
                [
                    "wget",
                    "-O",
                    str(checkpoint_dir / ckpt),
                    f"https://huggingface.co/boltzgen/boltzgen-1/resolve/main/{ckpt}?download=true",
                ],
            )
            # ugh, torch is trash
            print(f"[load_boltzgen] cleaning checkpoint metadata for {ckpt}", flush=True)
            cpkt = torch.load(
                checkpoint_dir / ckpt, map_location="cpu", weights_only=False
            )
            del cpkt["hyper_parameters"]["validators"]  # these contain GPU tensors
            del cpkt["validators"]
            torch.save(cpkt, checkpoint_dir / ckpt)

        print("[load_boltzgen] downloading mols.zip", flush=True)
        subprocess.run(
            [
                "wget",
                "-O",
                str(checkpoint_dir / "mols.zip"),
                "https://huggingface.co/datasets/boltzgen/inference-data/resolve/main/mols.zip?download=true",
            ]
        )

    checkpoint = checkpoint_dir / (checkpoints[0] if not model_diverse else checkpoints[1])
    print(f"[load_boltzgen] loading torch checkpoint {checkpoint}", flush=True)
    torch_model = Boltz.load_from_checkpoint(
        checkpoint,
        strict=True,
        map_location="cpu",
    ).eval()
    torch_model.structure_module.time_dilation = 2.667

    print("[load_boltzgen] converting torch model to JAX", flush=True)
    model = joltzgen.from_torch(torch_model)
    print("[load_boltzgen] moving model arrays to JAX device", flush=True)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    loaded = eqx.combine(jax.device_put(_model_params), _model_static)
    print("[load_boltzgen] ready", flush=True)
    return loaded


def _generate_mmcif(
    self,
    prediction: any = None,
    batch: any = None,
    sample_id: str = None,
) -> None:
    if prediction["exception"]:
        self.failed += 1
        return
    n_samples, _, _ = prediction["coords"].shape

    # TODO: remove this which is only here for temporary backward compatibility
    masker = BoltzMasker(mask=True, mask_backbone=False, mask_disto=True)
    feat_masked = masker(batch)
    prediction["ref_element"] = feat_masked["ref_element"]
    prediction["ref_atom_name_chars"] = feat_masked["ref_atom_name_chars"]
    """Write the predictions to disk."""
    # Check for extra molecules
    if batch["extra_mols"] is not None:
        extra_mols = batch["extra_mols"][0]
        for k, v in extra_mols.items():
            with open(self.mol_dir / f"{k}.pkl", "wb") as f:
                pickle.dump(v, f)

    # write samples to disk
    for n in range(n_samples):
        # get structure for all generated coords
        sample, native = {}, {}

        for k in set(prediction.keys()) & set(batch.keys()):
            if k == "coords":
                native[k] = batch[k][0][0].unsqueeze(0)
                sample[k] = prediction[k][n]

            if k in const.token_features:
                sample[k] = prediction[k][0]
                native[k] = batch[k][0]
            elif k in const.atom_features:
                if k == "coords":
                    native[k] = batch[k][0][0].unsqueeze(0)
                    sample[k] = prediction[k][n]
                else:
                    native[k] = batch[k][0]
                    sample[k] = prediction[k][0]
            elif k == "exception":
                sample[k] = prediction[k]
                native[k] = batch[k]
            else:
                # print(k)
                # print(batch[k].shape)
                try:
                    if batch[k] is not None:
                        native[k] = batch[k][0]
                        sample[k] = prediction[k][0]
                        native[k] = batch[k][0]
                except Exception as e:
                    print(e)

        if self.atom14:
            sample = res_from_atom14(sample)
        elif self.atom37:
            sample = res_from_atom37(sample)
        elif self.backbone_only:
            sample = res_all_gly(sample)

        design_mask = batch["design_mask"][0].bool()
        assert design_mask.sum() == sample["design_mask"].sum()

        if self.inverse_fold:
            token_ids = torch.argmax(sample["res_type"], dim=-1)
            tokens = [const.tokens[i] for i in token_ids]
            ccds = [convert_ccd(token) for token in tokens]

            ccds = torch.tensor(ccds).to(sample["res_type"])
            sample["ccd"][design_mask] = ccds[design_mask]

        try:
            structure, _, _ = Structure.from_feat(sample)
            str_native, _, _ = Structure.from_feat(native)

            # write structure to cif

            # design mask bfactor
            design_mask = batch["design_mask"][0].float()
            atom_design_mask = (
                sample["atom_to_token"].float() @ design_mask.unsqueeze(-1).float()
            )
            design_mask = native["design_mask"].float()

            atom_design_mask = atom_design_mask.squeeze().bool()
            bfactor = atom_design_mask * 100

            # binding type bfactor
            binding_type = batch["binding_type"][0].float()
            atom_binding_type = (
                sample["atom_to_token"].float() @ binding_type.unsqueeze(-1).float()
            )

            atom_binding_type = atom_binding_type.squeeze().bool()
            binding_type = native["binding_type"].float()
            bfactor[atom_binding_type == const.binding_type_ids["BINDING"]] = 60

            bfactor = atom_design_mask[sample["atom_pad_mask"].bool()].float()
            str_native.atoms["bfactor"] = bfactor.cpu().numpy()
            structure.atoms["bfactor"] = bfactor.cpu().numpy()

            # Add dummy (0-coord) design side chains if inverse fold
            if self.inverse_fold:
                atom_design_mask_no_pad = atom_design_mask[
                    native["atom_pad_mask"].bool()
                ]
                res_design_mask = np.array(
                    [
                        all(
                            atom_design_mask_no_pad[
                                res["atom_idx"] : res["atom_idx"] + res["atom_num"]
                            ]
                        )
                        for res in structure.residues
                    ]
                )
                structure = Structure.add_side_chains(
                    structure, residue_mask=res_design_mask
                )

            pred_binding_mask = prediction["binding_type"][0].cpu().bool().numpy()
            if self.design:
                chain_design_mask = (
                    prediction["chain_design_mask"][0].cpu().bool().numpy()
                )
            pred_design_mask = prediction["design_mask"][0].cpu().bool().numpy()
            design_color_features = np.ones_like(pred_binding_mask) * 0.8
            design_color_features[pred_binding_mask] = 1.0
            if self.design:
                design_color_features[chain_design_mask] = 0.0
            design_color_features[pred_design_mask] = 0.6

            # Create a mask to identify unique token-to-res mappings.
            # This is for small molecules where multiple tokens can be mapped to the same residue.
            token_to_res = prediction["token_to_res"][0].cpu().numpy()
            unique_mask = np.ones_like(token_to_res, dtype=bool)
            unique_mask[1:] = token_to_res[1:] != token_to_res[:-1]
            design_color_features = design_color_features[unique_mask]
            return gemmi.make_structure_from_block(
                gemmi.cif.read_string(
                    to_mmcif(
                        structure,
                        design_coloring=True,
                        color_features=design_color_features,
                    )
                )[0]
            )

        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()  # noqa: T201
            msg = f"predict/writer.py: Validation structure writing failed on {batch['id'][0]} with error {e}. Skipping."
            print(msg)


@dataclass
class BoltzGenWriter:
    writer: any
    torch_features: dict

    def __call__(self, coords: Float[Array, "... 3"]):
        return _generate_mmcif(
            self.writer,
            prediction=self.torch_features
            | {
                "coords": torch.tensor(np.array(coords)),
                "exception": False,
                "masks": self.torch_features["atom_pad_mask"].unsqueeze(0),
                "extra_mols": None,
                "structure_bonds": [torch.zeros(0)],  # hope you don't need bonds!
            },
            batch=self.torch_features
            | {
                "extra_mols": None,
                "target_msa_mask": torch.zeros(1, 1, 1),
                "structure_bonds": [torch.zeros(0)],  # lol
            },
        )


def load_features_and_structure_writer(
    yaml_string: str,
    moldir: Path = Path("~/.boltz/").expanduser() / "mols.zip",
    files: dict[str, Path] = {},
    mask: bool = True,
    mask_backbone: bool = False,
    mask_disto: bool = True,
):
    """Load BoltzGen features from a design YAML.

    Args:
        yaml_string: BoltzGen design specification YAML (see boltzgen/example/).
        moldir: path to mols.zip for small-molecule support.
        files: optional {filename: path} map to copy into the YAML's working directory
            (use this to resolve `path: <name>.cif` references inside the YAML).
        mask: whether to apply BoltzMasker to the features at all. Set False for
            inverse-fold-style runs where the structure is fully specified.
        mask_backbone: if True, also mask the backbone atoms of designed residues.
            Default False matches partial-design / sequence-redesign use cases that
            keep the parent backbone visible to the trunk.
        mask_disto: whether to mask the distogram loss over designed residues.
    """
    with TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/yaml.yaml", "w") as yaml_file:
            yaml_file.write(yaml_string)
            yaml_file.flush()

        for filename, p in files.items():
            dest_file = Path(f"{temp_dir}/{filename}")
            with open(p, "rb") as src_file, open(dest_file, "wb") as dest_file:
                dest_file.write(src_file.read())

        dataset_config = DataConfig(
            yaml_path=yaml_file.name,
            multiplicity=1,  # Multiplicity isn't used in get_sample, 1 is safe
            tokenizer=Tokenizer(),
            featurizer=Featurizer(),
            moldir=moldir,
            atom14=True,
            backbone_only=False,
            atom37=False,
            disulfide_prob=1.0,
            disulfide_on=True,
        )
        datamodule = FromYamlDataModule(
            dataset_config, batch_size=1, num_workers=0, pin_memory=False
        )
        dl = datamodule.predict_dataloader()

        features = next(iter(dl))

    features["structure_bonds"] = []
    torch_masker = BoltzMasker(
        mask=mask, mask_backbone=mask_backbone, mask_disto=mask_disto
    )
    features = torch_masker(features)

    # convert features to jax
    j_features = jax.tree.map(
        lambda v: jnp.array(v) if isinstance(v, torch.Tensor) else v, features
    ) | {"cyclic_period": jnp.zeros((1, 1))}

    j_features["msa"] = jax.nn.one_hot(j_features["msa"], num_classes=const.num_tokens)

    output_dir = TemporaryDirectory(delete=False).name

    return (
        j_features,
        BoltzGenWriter(
            DesignWriter(
                output_dir=output_dir,
                res_atoms_only=False,
                atom14=True,
                atom37=False,
                write_native=False,
            ),
            features,
        ),
    )


class Sampler(eqx.Module):
    """Hold conditioner information for repeated sampling from stucture module. Can be vmapped, jitted, etc."""

    trunk_s: Float[Array, "N S"]
    s_inputs: Float[Array, "N S"]
    feats: dict[str, any]
    q: Float[Array, "..."]
    c: Float[Array, "..."]
    to_keys: any
    atom_enc_bias: Float[Array, "..."]
    atom_dec_bias: Float[Array, "..."]
    token_trans_bias: Float[Array, "..."]

    @eqx.filter_jit
    @staticmethod
    def from_features(
        *,
        model: joltzgen.JoltzGen,
        features: dict[str, any],
        recycling_steps: int,
        key,
        deterministic: bool = True,
    ):
        initial_embedding = model.embed_inputs(features)

        trunk_state, key = model.recycle(
            initial_embedding=initial_embedding,
            recycling_steps=recycling_steps,
            feats=features,
            key=key,
            deterministic=deterministic,
        )

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            model.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                initial_embedding.relative_position_encoding,
                features,
            )
        )

        return Sampler(
            trunk_s=trunk_state.s,
            s_inputs=initial_embedding.s_inputs,
            feats=features,
            q=q,
            c=c,
            to_keys=to_keys,
            atom_enc_bias=atom_enc_bias,
            atom_dec_bias=atom_dec_bias,
            token_trans_bias=token_trans_bias,
        )

    def __call__(
        self,
        *,
        structure_module: joltzgen.AtomDiffusion,
        num_sampling_steps: int,
        step_scale: float,
        noise_scale: float,
        key,
        sample_schedule="dilated",
    ):
        return structure_module.sample(
            s_trunk=self.trunk_s,
            s_inputs=self.s_inputs,
            feats=self.feats,
            num_sampling_steps=num_sampling_steps,
            atom_mask=self.feats["atom_pad_mask"],
            multiplicity=1,
            diffusion_conditioning={
                "q": self.q,
                "c": self.c,
                "to_keys": self.to_keys,
                "atom_enc_bias": self.atom_enc_bias,
                "atom_dec_bias": self.atom_dec_bias,
                "token_trans_bias": self.token_trans_bias,
            },
            key=jax.random.fold_in(key, 2),
            step_scale=step_scale,
            noise_scale=noise_scale,
            sample_schedule=sample_schedule,
        )

def _coords_to_restype(coords, *, des_idx, threshold: float = 0.5):
    design_coords = coords[des_idx]
    design_coords = design_coords.reshape(len(design_coords) // 14, 14, 3)

    # For each sidechain atom, compute closest backbone atom and count them
    # while excluding those side chain atoms whose distance is above a threshold
    distances = pairwise_distance(
        design_coords[:, :4], design_coords[:, 4:]
    )  # torch.cdist(design_coords[:, :4], design_coords[:, 4:])
    value, argmin = jnp.min(distances, axis=1), jnp.argmin(distances, axis=1)
    argmin = jnp.where(value > threshold, -1, argmin)
    arange = jnp.arange(len(const.ref_atoms["GLY"]))
    counts = (argmin[:, :, None] == arange[None, None, :]).sum(1)
    # counts is num_res x 4
    with jax.ensure_compile_time_eval():
        count_matrix = np.zeros((20, 4))
        for k, v in const.placement_count_to_token.items():
            if const.token_ids[v] != 22:
                count_matrix[const.token_ids[v] - 2] = k

    dists = ((counts[:, None, :] - count_matrix[None, :, :]) ** 2).sum(-1)

    return dists.argmin(-1)


class CoordsToToken(eqx.Module):
    """Convert sampled coordinates to mosaic token indices. Class to make precomputing some things outside of JIT easier..."""

    des_idx: np.ndarray

    def __init__(self, features: dict[str, any]):
        design_mask = np.array(features["design_mask"]).astype(bool)
        mol_type = features["mol_type"]
        atom_to_token = np.array(features["atom_to_token"])
        token_index = np.array(features["token_index"])
        atom_pad_mask = np.array(features["atom_pad_mask"])
        design_mask = np.logical_and(
            design_mask, mol_type == const.chain_type_ids["PROTEIN"]
        )
        # Get designed atom coordinates in shape N//14 x 14 x 3
        atom_to_token = np.argmax(atom_to_token, axis=-1)
        token_indices = token_index[design_mask.astype(bool)]
        atom_design_mask = np.isin(atom_to_token, token_indices)
        atom_design_mask = np.logical_and(atom_design_mask, atom_pad_mask)
        self.des_idx = np.nonzero(atom_design_mask[0])

    @eqx.filter_jit
    def __call__(self, coords: Float[Array, "... 3"]):
        return _coords_to_restype(coords, des_idx=self.des_idx)

class BoltzGenOutput(eqx.Module):
    sample: jax.Array
    features: PyTree
    coords2token: CoordsToToken

    @property
    def full_sequence(self):
        binder_sequence = self.coords2token(self.sample[0])
        binder_sequence = jax.nn.one_hot(binder_sequence, 20, dtype=jnp.int32)
        binder_len = binder_sequence.shape[0]
        return self.features["res_type"][0, :, 2:22].at[:binder_len].set(binder_sequence)

    @property
    def asym_id(self):
        return self.features["asym_id"][0]

    @property
    def residue_idx(self):
        return self.features["residue_index"][0]

    @property
    def backbone_coordinates(self):
        # could precompute the index in load_features to avoid slow operation alarm
        bb_atom_inds = jnp.argmax(self.features["token_to_bb4_atoms"][0], axis=-1)
        return self.sample[0][bb_atom_inds]

    @property
    def structure_coordinates(self):
        return self.sample


def differentiable_inverse_fold(
    mpnn,
    coords: Float[Array, "N 4 3"],
    *,
    parent_sequence: Float[Array, "N 20"],
    asym_id: Int[Array, "N"],
    residue_idx: Int[Array, "N"],
    designable_mask: Float[Array, "N"],
    temperature: float = 0.1,
    jacobi_iterations: int = 1,
    key,
) -> Float[Array, "N 20"]:
    """Soft inverse-fold: given backbone coords, return a per-position softmax over AAs.

    Gradient flows from the output soft sequence back to the input coords via
    `mpnn.encode` (structural features) and through any Jacobi refinement steps
    (which use a softmax instead of argmax). This is the differentiable bridge used
    inside `guided_partial_diffusion` to pull sequence-space loss gradients into
    coord-space.

    The soft sequence is built only at `designable_mask` positions; non-designable
    positions stay at `parent_sequence`. The MPNN decoder always sees the current
    soft sequence as autoregressive context.

    Args:
        mpnn: a `ProteinMPNN` instance (e.g. ABMPNN for VHH design).
        coords: backbone coords, shape (N, 4, 3) for [N, CA, C, O] atoms per residue.
        parent_sequence: one-hot or soft sequence at non-designable positions; also
            the initialization at designable positions before Jacobi refinement.
            Shape (N, 20) in mosaic's TOKENS alphabet.
        asym_id: chain id per token, shape (N,).
        residue_idx: residue index within chain, shape (N,).
        designable_mask: float (N,), 1.0 where positions are designable, 0.0 otherwise.
        temperature: softmax temperature. Lower → sharper. Use ~0.1 for guidance,
            ~0.001 for final decoding to a near-one-hot sequence.
        jacobi_iterations: number of Jacobi refinement steps. 1 is enough for
            guidance (gradient signal); higher for a higher-quality final decode.
        key: jax random key (controls decoding order).

    Returns:
        Soft sequence of shape (N, 20) in mosaic's TOKENS alphabet, with
        non-designable positions equal to `parent_sequence`.
    """
    from ..losses.protein_mpnn import boltz_to_mpnn_matrix

    total_length = parent_sequence.shape[0]
    mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)

    # Adjust residue idx so chains don't overlap; add 100-residue gap between chains
    # (matches the convention in mosaic.losses.protein_mpnn.inverse_fold)
    chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
    res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
    adjusted_residue_idx = (
        residue_idx
        + (asym_id[:, None] == np.arange(16)[None]) @ res_idx_adjustment
        + 100 * asym_id
    )

    # Encode structure (gradient flows from coords here)
    h_V, h_E, E_idx = mpnn.encode(
        X=coords,
        mask=mpnn_mask,
        residue_idx=adjusted_residue_idx,
        chain_encoding_all=asym_id,
        key=key,
    )

    # Decoding order: designable positions get +2.0 so they sort to the end
    # (decoded last, with full structural and non-designable context)
    decoding_order = (
        jax.random.uniform(key, shape=(total_length,))
        + 2.0 * designable_mask.astype(jnp.float32)
    )

    T = jnp.array(boltz_to_mpnn_matrix())  # (20, 21) boltz -> MPNN

    def step(soft_seq, _):
        sequence_mpnn = soft_seq @ T  # (N, 21)
        log_probs_mpnn = mpnn.decode(
            S=sequence_mpnn,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mpnn_mask,
            decoding_order=decoding_order,
        )[0]
        # Convert back to boltz alphabet by selecting the 20 boltz columns from
        # the 21-token MPNN logits.
        logits_boltz = log_probs_mpnn @ T.T  # (N, 20)
        new_soft = jax.nn.softmax(logits_boltz / temperature, axis=-1)
        # Only update designable positions; framework / target stays at parent
        new_soft = jnp.where(
            designable_mask[:, None].astype(bool), new_soft, parent_sequence
        )
        return new_soft, None

    soft_seq, _ = jax.lax.scan(step, parent_sequence, length=jacobi_iterations)
    return soft_seq


def _center(coords, atom_mask):
    """Thin alias for joltzgen.center; matches batched [b m 3] / [b m] convention.

    Imported lazily so this module loads even without joltzgen present (e.g.
    during pure-Python syntax checks).
    """
    from joltzgen import center as _joltzgen_center
    return _joltzgen_center(coords, atom_mask.astype(coords.dtype))


# ---------------------------------------------------------------------------
# Guidance controller primitives (Phase 1 of docs/guidance_implementation_todo.md)
#
# These implement the per-objective mask -> de-mean -> normalize -> PCGrad-style
# conflict projection -> trust-region clip pipeline from
# docs/guidance_design_notes.md section 5 / 10. All operate on batched
# [B, M, 3] coordinate-shaped gradients and [B, M] atom masks.
# ---------------------------------------------------------------------------


def _mask_center_normalize(
    g: Float[Array, "B M 3"],
    atom_partial_mask: Float[Array, "B M"],
    eps: float = 1e-8,
) -> Float[Array, "B M 3"]:
    """Mask to designable atoms, remove rigid-translation drift, RMS-normalize.

    Three steps, each addressing a distinct failure mode identified in
    docs/guidance_design_notes.md section 3:
      1. mask: zero the gradient on frozen atoms, so guidance never touches
         framework/target atoms even transiently.
      2. de-mean: subtract the per-batch mean gradient vector over designable
         atoms, removing the "just translate the whole loop" degenerate
         direction (a rigid shift is nearly free for many losses to exploit
         and carries little real structural information).
      3. RMS-normalize: rescale to unit RMS magnitude over designable atoms,
         so different objectives (which can differ in raw coordinate-space
         gradient magnitude by orders of magnitude once pulled back through
         the inverse-fold Jacobian) become comparable before merging.
    """
    mask = (atom_partial_mask > 0)[..., None]  # (B, M, 1)
    g = jnp.where(mask, g, 0.0)

    n_design = jnp.maximum(jnp.sum(atom_partial_mask, axis=-1, keepdims=True), 1.0)  # (B, 1)
    mean = jnp.sum(g, axis=1) / n_design  # (B, 3)
    g = jnp.where(mask, g - mean[:, None, :], 0.0)

    sum_sq = jnp.sum(g * g, axis=(1, 2), keepdims=True)  # (B, 1, 1)
    rms = jnp.sqrt(sum_sq / (n_design[..., None] * 3.0) + eps)
    return g / rms


def _compat_project(
    g_aux: Float[Array, "B M 3"],
    g_anchor: Float[Array, "B M 3"],
    eps: float = 1e-8,
) -> Float[Array, "B M 3"]:
    """PCGrad-style asymmetric conflict projection.

    If `g_aux` agrees with `g_anchor` (positive dot product), it passes
    through unchanged. If it conflicts (negative dot product), the
    conflicting component is removed. `g_anchor` itself is never modified —
    this is the asymmetric variant from docs/guidance_design_notes.md
    section 4.5: binding leads, auxiliary objectives only regularize.
    """
    dot = jnp.sum(g_aux * g_anchor, axis=(1, 2), keepdims=True)
    anchor_sq = jnp.sum(g_anchor * g_anchor, axis=(1, 2), keepdims=True) + eps
    conflict = jnp.minimum(0.0, dot) / anchor_sq
    return g_aux - conflict * g_anchor


def _clip_rms(
    delta: Float[Array, "B M 3"],
    tau: Float[Array, "..."],
    atom_partial_mask: Float[Array, "B M"],
    eps: float = 1e-8,
) -> Float[Array, "B M 3"]:
    """Trust-region clip: cap delta's RMS magnitude (over designable atoms) at tau.

    This is the guardrail docs/guidance_design_notes.md section 4.6 calls "the
    key missing guardrail" — bounds the worst-case per-step coordinate
    displacement independent of how well-behaved the upstream normalization
    and merge turned out to be.
    """
    mask = (atom_partial_mask > 0)[..., None]
    n_design = jnp.maximum(jnp.sum(atom_partial_mask, axis=-1, keepdims=True), 1.0)
    sum_sq = jnp.sum(jnp.where(mask, delta, 0.0) ** 2, axis=(1, 2), keepdims=True)
    rms = jnp.sqrt(sum_sq / (n_design[..., None] * 3.0) + eps)
    scale = jnp.minimum(1.0, tau / rms)
    return delta * scale


def _euler_step_delta(direction, step_scale: float, sigma_t, t_hat):
    """Scale a raw `(atom_coords_noisy - x0) / t_hat` direction by the same
    `step_scale * (sigma_t - t_hat)` factor the EDM Euler update applies, so
    the result is the actual physical per-step coordinate displacement (not
    just a scale-invariant direction). Used for both the unguided and guided
    diagnostics in `guided_partial_diffusion` — see docs/guidance_implementation_todo.md
    Phase 1/Phase 2 on why the raw direction alone is insufficient for a
    norm-ratio diagnostic.
    """
    return step_scale * (sigma_t - t_hat) * direction


def default_lambda_schedule(t_hat, lam_max: float = 1.0):
    """Default overall guidance-strength schedule: linear decay to 0 as t_hat -> 0.

    First-pass default (docs/guidance_implementation_todo.md Phase 1: "pin
    down actual functional forms"). `lambda ~ t_hat` is one of the "standard
    EDM choices" already noted in this function's docstring history and
    matches the qualitative requirement in guidance_design_notes.md section 9
    ("late diffusion: sharply reduce coordinate guidance magnitude"). Tune
    `lam_max` per target; treat the linear-in-sigma shape itself as a
    starting point pending Phase 2 diagnostics, not a final answer.
    """
    return lam_max * t_hat


def default_tau_schedule(t_hat, tau_max: float = 2.0, tau_min: float = 0.05):
    """Default trust-radius schedule: proportional to the current noise level,
    floored so the cap never collapses to exactly zero.

    Moving no further than a fraction of the noise currently being injected
    is a physically motivated default: at high sigma the model itself is
    still exploring broadly, so a larger guided step is in proportion to what
    the denoiser is already doing; at low sigma the structure should mostly
    be settling, so the cap shrinks with it. `tau_min` prevents a hard stop
    at the final step. First-pass default, same caveat as
    `default_lambda_schedule` — tune against Phase 2 diagnostics.
    """
    return jnp.maximum(tau_max * t_hat, tau_min)


def default_alpha_schedule(t_hat, alpha_max: float = 1.0):
    """Default naturalness-objective weight: ramps up as t_hat -> 0.

    Per guidance_design_notes.md section 9: naturalness should be weak early
    (high sigma, still exploring structurally) and matter more as the
    trajectory progresses. `alpha_max / (1 + t_hat)` increases monotonically
    as t_hat decreases without needing the schedule's start/end sigma bounds
    threaded in as extra arguments. First-pass default, tune against Phase 2
    diagnostics.
    """
    return alpha_max / (1.0 + t_hat)


def default_beta_schedule(t_hat, beta_max: float = 1.0):
    """Default edit-locality-objective weight: same shape as
    `default_alpha_schedule` (ramps up as t_hat -> 0), matching
    guidance_design_notes.md section 9's "edit/locality penalty turns on"
    guidance for mid-to-late diffusion. First-pass default, tune against
    Phase 2 diagnostics.
    """
    return beta_max / (1.0 + t_hat)


def _merge_aux_gradients(x0, grad_bind, grad_nat, grad_edit, atom_partial_mask, alpha_fn, beta_fn, t_hat):
    """One evaluation of the asymmetric mask/normalize/PCGrad merge at a
    given point `x0`. Factored out of the one-shot guidance path so the NOS
    iterative loop (`_nos_iterative_merge`, docs/guidance_alphaseq_testing_notes.md
    section 12a) can call it fresh at each inner step -- re-evaluating
    grad_bind/grad_nat/grad_edit AT THE CURRENT point, not just once at
    x0_hat, is exactly what makes NOS's real mechanism structurally
    different from a one-shot squared-distance penalty (verified directly
    to be a pure rescale of the auxiliary gradient before this was built).
    """
    zeros_like_g = jnp.zeros_like(x0)
    g_bind = _mask_center_normalize(grad_bind(x0), atom_partial_mask)
    g_total = g_bind
    if grad_nat is not None:
        g_nat = _mask_center_normalize(grad_nat(x0), atom_partial_mask)
        g_total = g_total + alpha_fn(t_hat) * _compat_project(g_nat, g_bind)
    else:
        g_nat = zeros_like_g
    if grad_edit is not None:
        g_edit = _mask_center_normalize(grad_edit(x0), atom_partial_mask)
        g_total = g_total + beta_fn(t_hat) * _compat_project(g_edit, g_bind)
    else:
        g_edit = zeros_like_g
    return g_total, g_bind, g_nat, g_edit


def default_nos_inner_step_schedule(t_hat, step_max: float = 0.05):
    """Default per-inner-step size for the NOS iterative loop (docs/
    guidance_alphaseq_testing_notes.md section 12a): a small constant, not a
    function of t_hat. NOS's own Langevin step size is fixed per training
    round (their setting is discrete-diffusion at one noise level per round,
    not a continuous EDM sigma schedule), so there is no literature-grounded
    noise-dependent shape to copy here. First-pass default -- tune against
    results. The outer `_clip_rms(..., tau_fn(t_hat))` safety cap still
    applies to the accumulated result regardless of this value, so a
    too-large step size is bounded, not unsafe.
    """
    return step_max


def default_nos_lambda_kl_schedule(t_hat, lam_kl_max: float = 1.0):
    """Default weight on the NOS consistency term inside the inner loop
    (section 12a). Constant -- same caveat as
    `default_nos_inner_step_schedule`: no noise-dependent shape for this
    term is grounded in the source paper (arXiv:2305.20009), which does not
    operate on a continuous sigma schedule.
    """
    return lam_kl_max


def default_nos_langevin_noise_schedule(t_hat, noise_max: float = 0.0):
    """Default injected-noise std for the NOS inner loop (section 12a).
    Zero by default -- deterministic inner descent, not full Langevin
    sampling. Set > 0 to match NOS's actual stochastic Langevin updates more
    closely; left off by default since injected noise is a real, separate
    behavioral change (genuine sampling stochasticity, not just a different
    guidance direction) that should not turn on silently.
    """
    return noise_max


def _nos_iterative_merge(
    x0_hat,
    grad_bind,
    grad_nat,
    grad_edit,
    atom_partial_mask,
    alpha_fn,
    beta_fn,
    tau_fn,
    t_hat,
    n_inner_steps: int,
    inner_step_fn,
    lambda_kl_fn,
    noise_fn,
    key,
):
    """Real, iterative NOS-style prior-compatibility mechanism
    (docs/guidance_alphaseq_testing_notes.md section 12a) -- not the
    one-shot squared-distance penalty tried first and found to be
    mathematically a no-op (identically zero gradient) or, in a two-stage
    variant, provably just a rescale of the auxiliary gradient with no new
    directional information. Both failures share one cause: a single
    evaluation forces the "current point" to be an affine function of the
    same gradient set being regularized. Genuine iteration breaks that: `K`
    inner steps, each re-evaluating grad_bind/grad_nat/grad_edit at the
    CURRENT (already-moved) point -- not just once at x0_hat -- while a
    consistency gradient keeps pulling back toward the fixed original
    x0_hat.

    Stability, found and fixed during testing (not from the source paper --
    a real property of this translation): the consistency term alone gives
    the linear recursion `(x0_i+1 - anchor) = (1 - 2*step*lambda_kl) *
    (x0_i - anchor)`, which diverges geometrically whenever
    `2*step*lambda_kl > 1` (confirmed directly: `step=0.1, lambda_kl=50`
    grew a small perturbation to a distance of ~2000 over 5 steps instead of
    shrinking it). To keep a hyperparameter sweep (section 12c) from
    silently producing exploded garbage for an unstable `(inner_step_fn,
    lambda_kl_fn)` combination, each inner step's raw displacement is capped
    via the same `_clip_rms` guardrail already used for the outer step,
    scaled down by `n_inner_steps` so `K` capped inner steps can't
    dramatically exceed the outer trust region even before the final outer
    clip in `step_body` runs.

    Requires `n_inner_steps >= 1` (the caller only invokes this when the
    static `guidance_nos_inner_steps > 0`, mirroring the `grad_bind is not
    None` trace-time gating already used elsewhere in this function).

    Returns `(x0_candidate, last_g_bind, last_g_nat, last_g_edit)` -- the
    caller forms `delta = lambda_fn(t_hat) * (x0_hat - x0_candidate)` (the
    same outer strength schedule the one-shot path applies to `g_total` --
    dropping this multiplication was a real bug, found and fixed, see
    docs/guidance_alphaseq_testing_notes.md section 12a) and applies the
    existing outer trust-region clip, exactly like the one-shot path.
    """
    anchor = jax.lax.stop_gradient(x0_hat)
    mask = (atom_partial_mask > 0)[..., None]
    inner_tau = tau_fn(t_hat) / n_inner_steps

    def inner_step(carry, _):
        x0_i, step_key = carry
        g_total_i, g_bind_i, g_nat_i, g_edit_i = _merge_aux_gradients(
            x0_i, grad_bind, grad_nat, grad_edit, atom_partial_mask, alpha_fn, beta_fn, t_hat
        )
        # Consistency gradient: d/dx0_i[ ||x0_i - anchor||^2 ] = 2*(x0_i - anchor).
        # Nonzero whenever x0_i has moved from the fixed anchor -- true from
        # inner step 1 onward whenever g_total_i != 0 at step 0.
        g_consistency_i = jnp.where(mask, 2.0 * (x0_i - anchor), 0.0)
        g_total_i = g_total_i + lambda_kl_fn(t_hat) * g_consistency_i

        step_key, sub = jax.random.split(step_key)
        noise = noise_fn(t_hat) * jax.random.normal(sub, x0_i.shape)
        noise = jnp.where(mask, noise, 0.0)

        raw_step = inner_step_fn(t_hat) * g_total_i
        clipped_step = _clip_rms(raw_step, inner_tau, atom_partial_mask)
        x0_next = x0_i - clipped_step + noise
        return (x0_next, step_key), (g_bind_i, g_nat_i, g_edit_i)

    (x0_final, _), (g_bind_hist, g_nat_hist, g_edit_hist) = jax.lax.scan(
        inner_step, (x0_hat, key), jnp.arange(n_inner_steps)
    )
    return x0_final, g_bind_hist[-1], g_nat_hist[-1], g_edit_hist[-1]


def guided_partial_diffusion(
    *,
    sampler: "Sampler",
    structure_module,
    initial_coords: Float[Array, "M 3"],
    atom_partial_mask: Float[Array, "M"],
    atom_mask: Float[Array, "M"],
    num_sampling_steps: int,
    start_sigma_frac: float,
    step_scale: float,
    noise_scale: float,
    guidance_fn_bind=None,
    guidance_fn_nat=None,
    guidance_fn_edit=None,
    guidance_alpha_fn=None,
    guidance_beta_fn=None,
    guidance_lambda_fn=None,
    guidance_tau_fn=None,
    guidance_nos_inner_steps: int = 0,
    guidance_nos_inner_step_fn=None,
    guidance_nos_lambda_kl_fn=None,
    guidance_nos_noise_fn=None,
    guidance_lookahead: bool = False,
    sidechain_mask: Float[Array, "M"] | None = None,
    sidechain_noise_multiplier: float = 1.0,
    key=None,
    return_diagnostics: bool = False,
):
    """Partial diffusion of BoltzGen, optionally with multi-objective classifier guidance.

    With all `guidance_fn_*` left as `None` this runs vanilla partial
    diffusion: schedule truncation + parent-anchored sampling, no auxiliary
    signal. With `guidance_fn_bind` provided (the only required one of the
    three — naturalness and edit-locality are optional regularizers), each
    step pulls auxiliary-model gradients into the denoised coords via the
    differentiable inverse-fold bridge (classifier-guided diffusion).

    This is the Phase 1 controller rewrite from
    docs/guidance_implementation_todo.md: rather than one merged scalar loss
    differentiated once (the earlier, "too brutal" design analyzed in
    docs/guidance_design_notes.md sections 2-3), three objectives are
    evaluated and differentiated *separately* — `L_bind` (primary, e.g.
    interface confidence), `L_nat` (naturalness prior), `L_edit`
    (locality/edit-restraint prior) — each masked to the designable region,
    de-meaned, and RMS-normalized before being merged. Binding is the anchor
    direction: naturalness and edit gradients are only kept where they agree
    with it (PCGrad-style conflict projection), never the reverse. The merged
    result is trust-region clipped before being applied.

    Why this lives in mosaic and not in joltzgen: partial diffusion is purely a
    sampling-time orchestration on top of the unchanged denoiser network. The
    mechanics here mirror the user's torch-side boltzgen fork at `diffusion.py`,
    but importantly, joltzgen itself needs no custom modification — it only needs
    to expose `preconditioned_network_forward`, which is the standard upstream API.

    Per-step recipe (mirrors `boltzgen/src/.../diffusion.py:580-728`):

      1. Truncate the schedule to start at `start_sigma_frac` (so we begin from a
         partially-noised initial structure rather than pure noise).
      2. Initialize `atom_coords = where(partial_mask, parent + init_sigma*eps, parent)`.
      3. For each step k:
         a. Re-center both `atom_coords` and `initial_coords_rep` on the protein COM
            (keeps them in the same frame for the re-anchor step).
         b. EDM stochastic-churn step: `t_hat = sigma_{k-1} * (1+gamma)`, inject
            additional noise scaled by `(t_hat^2 - sigma_{k-1}^2)`.
         c. Call the BoltzGen denoiser under `jax.lax.stop_gradient` to get
            `x0_hat = D(noisy, t_hat)`. We deliberately do NOT backprop through D —
            the guidance gradient flows only through the small IF + aux-loss subgraph.
            (This is the boundary docs/guidance_design_notes.md section 6 flags as
            the still-open prior-compatibility question: nothing here yet ties the
            guidance direction back to what D itself would predict.)
         d. Compute one gradient per objective: `g_k = ∇_{x0} L_k(x0_hat)` for
            k in {bind, nat, edit}. Mask, de-mean, RMS-normalize each
            independently (`_mask_center_normalize`).
         e. Merge asymmetrically, binding as anchor:
            `g_total = g_bind + alpha(t_hat)*compat(g_nat, g_bind) + beta(t_hat)*compat(g_edit, g_bind)`.
         f. Trust-region clip: `delta = clip_rms(lambda(t_hat) * g_total, tau(t_hat))`.
         g. Apply: `x0_guided = x0_hat - delta`.
         h. Euler step in EDM parameterization:
            `x_next = noisy + step_scale * (sigma_k - t_hat) * (noisy - x0_guided)/t_hat`.
         i. Re-anchor frozen atoms: `where(partial_mask, x_next, initial_coords_rep)`.

    INCLUDED:
      - `weighted_rigid_align` between noisy and denoised, gated on
        `structure_module.alignment_reverse_diff`. The BoltzGen-1 release
        checkpoints set this to True.

    SKIPPED (low impact for guidance — port back if needed):
      - Coordinate augmentation per step (`diffusion.py:661-674`). Both inputs are
        rotated identically and the IF model is SE(3) invariant, so this is a
        Monte Carlo de-biasing trick that does not affect the search dynamics.
      - Heun second-derivative correction. The torch reference uses Euler only,
        so we do too.

    Args:
        sampler: a `Sampler` built via `Sampler.from_features(...)`. Provides the
            cached trunk embeddings and diffusion conditioning so we don't re-run
            the trunk per sample.
        structure_module: `boltzgen.structure_module` (the JAX `AtomDiffusion`).
            Must expose `preconditioned_network_forward(x, sigma, ...)` and
            `sample_schedule_dilated(num_sampling_steps)` separately. If joltzgen
            does not expose `preconditioned_network_forward` directly, port the
            method from `boltzgen/src/boltzgen/model/modules/diffusion.py`.
        initial_coords: parent atom coords, shape (M, 3). Frozen atoms stay here.
        atom_partial_mask: float (M,), 1.0 at designable atoms, 0.0 elsewhere.
            Construct from a token-level mask via `atom_to_token @ token_mask`.
        atom_mask: float (M,), 1.0 at real atoms (not pad).
        num_sampling_steps: number of full-schedule steps. After truncation by
            `start_sigma_frac`, the actual loop length is shorter.
        start_sigma_frac: in (0, 1]. 1.0 = full diffusion from pure noise; smaller
            = start later in the schedule (less noise, more local refinement).
            Typical: 0.3-0.5 for CDR-scale local edits.
        step_scale, noise_scale: EDM step / noise scale (scalars). Match the
            BoltzGen training defaults: ~2.0 and ~0.88 respectively.
        guidance_fn_bind: callable `(x0: Float[B,M,3]) -> scalar`, the primary
            (anchor) objective — e.g. interface-confidence surrogate. If
            `None`, guidance is a no-op regardless of the other two
            `guidance_fn_*` args (there is no anchor to merge against).
        guidance_fn_nat: optional callable, same signature, naturalness prior.
            Only contributes if `guidance_fn_bind` is also provided.
        guidance_fn_edit: optional callable, same signature, locality/edit
            restraint prior. Only contributes if `guidance_fn_bind` is also
            provided.
        guidance_alpha_fn: callable `(t_hat) -> scalar`, naturalness weight in
            the merge. Defaults to `default_alpha_schedule` if not given and
            `guidance_fn_nat` is provided.
        guidance_beta_fn: callable `(t_hat) -> scalar`, edit-restraint weight
            in the merge. Defaults to `default_beta_schedule` if not given and
            `guidance_fn_edit` is provided.
        guidance_lambda_fn: callable `(t_hat) -> scalar`, overall guidance
            strength. Defaults to `default_lambda_schedule` if not given.
        guidance_tau_fn: callable `(t_hat) -> scalar`, trust-radius cap on the
            RMS coordinate displacement per step. Defaults to
            `default_tau_schedule` if not given.
        guidance_nos_inner_steps: int, static (trace-time, not a JAX value).
            `0` (default) = one-shot merge, identical to prior behavior.
            `> 0` = real iterative NOS-style prior-compatibility mechanism
            (docs/guidance_alphaseq_testing_notes.md section 12a):
            `guidance_fn_bind`/`_nat`/`_edit` are re-evaluated at each of
            `guidance_nos_inner_steps` inner points, with a consistency
            gradient pulling back toward the outer step's original x0_hat.
            Real added cost: `guidance_nos_inner_steps` extra forward+backward
            passes through the guidance losses per outer diffusion step.
        guidance_nos_inner_step_fn: callable `(t_hat) -> scalar`, per-inner-
            step size for the NOS loop. Defaults to
            `default_nos_inner_step_schedule`. Only used when
            `guidance_nos_inner_steps > 0`.
        guidance_nos_lambda_kl_fn: callable `(t_hat) -> scalar`, weight on
            the NOS consistency term. Defaults to
            `default_nos_lambda_kl_schedule`.
        guidance_nos_noise_fn: callable `(t_hat) -> scalar`, injected
            Langevin noise std for the NOS loop. Defaults to
            `default_nos_langevin_noise_schedule` (0.0 = deterministic).
        guidance_lookahead: bool, static (trace-time). `False` (default) =
            unchanged behavior. `True` = real look-ahead mechanism (docs/
            guidance_alphaseq_testing_notes.md section 12b, arXiv:2404.14743):
            `guidance_fn_bind`/`_nat`/`_edit` are differentiated through the
            *entire* denoiser forward pass with respect to `atom_coords_noisy`
            (removing the `stop_gradient` boundary the other two paths keep),
            instead of with respect to the frozen `x0_hat`. Real added cost: a
            full backward pass through the denoiser trunk per guidance term
            per outer diffusion step, not just through the small guidance-loss
            subgraph. Mutually exclusive with `guidance_nos_inner_steps > 0`
            (raises `ValueError` if both are set -- combining them would mean
            K full denoiser backward passes per step, not scoped or tested).
        sidechain_mask: optional float (M,), 1.0 at sidechain atoms whose noise
            should be amplified by `sidechain_noise_multiplier`. Useful when you
            trust the parent backbone but want to fully scramble CDR sidechains.
        sidechain_noise_multiplier: amplifier on sidechain init noise (1.0 = same).
        key: jax random key.
        return_diagnostics: if True, `step_body`'s per-step scan output
            switches from `None` to a dict of per-step arrays (stacked over
            the trajectory by `jax.lax.scan`) intended for
            docs/guidance_implementation_todo.md Phase 2 logging:
              - "unguided_direction": the actual `(atom_coords_noisy - x0_hat)
                / t_hat` reverse-update direction the sampler would have used
                with no guidance at all (computed from the same
                `atom_coords_noisy` — post-churn, post optional rigid
                realignment — that the real step uses, not a hand-derived
                shorthand).
              - "guided_direction": the same quantity but with `x0_guided`.
              - "unguided_direction"/"guided_direction" are raw derivatives,
                scale-invariant and valid for cosine-similarity diagnostics,
                but NOT comparable to actual per-step coordinate motion.
              - "unguided_step_delta", "guided_step_delta": the above two
                directions scaled by the same `step_scale * (sigma_t - t_hat)`
                factor the real Euler update applies, i.e. the actual physical
                displacement each would produce. Use these (not the raw
                directions) for the norm-ratio diagnostic Phase 2 calls for.
              - "g_bind", "g_nat", "g_edit": the masked/de-meaned/normalized
                per-objective gradients (zeros where the corresponding
                `guidance_fn_*` was not provided), for the pairwise
                objective-vs-objective conflict logging Phase 2 also calls for.
              - "t_hat", "sigma_t": the noise-axis position of each step
                (scalars), for stratifying any of the above by sigma.
            `False` by default: this branch is resolved at trace time (a plain
            Python bool, not a JAX value), so when unused it costs nothing —
            none of the diagnostic computation is even added to the traced
            graph.

    Returns:
        Final atom coords, shape (M, 3), if `return_diagnostics=False`.
        `(final_coords, diagnostics)` if `return_diagnostics=True`, where
        `diagnostics` is a dict of arrays each with a leading `n_steps` axis.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 1_000_000))

    # ---- Shape normalization (must come BEFORE init code) ------------------
    # joltzgen expects batched shapes [b, m, 3] / [b, m]. The driver typically
    # passes feature arrays already with a leading batch dim of 1; if they're
    # passed unbatched we add one and remember to squeeze back at the end.
    unbatched_input = (initial_coords.ndim == 2)
    if unbatched_input:
        initial_coords = initial_coords[None]                # (1, M, 3)
        atom_partial_mask = atom_partial_mask[None]          # (1, M)
        atom_mask = atom_mask[None]                          # (1, M)
        if sidechain_mask is not None:
            sidechain_mask = sidechain_mask[None]
    # -------------------------------------------------------------------------

    # 1. Build truncated schedule (mirrors diffusion.py:586-595)
    full_sigmas = structure_module.sample_schedule_dilated(num_sampling_steps)
    start_idx = int((1.0 - start_sigma_frac) * (len(full_sigmas) - 1))
    sigmas = full_sigmas[start_idx:]

    # gamma schedule: gamma_0 if sigma > gamma_min else 0 (diffusion.py:596)
    gamma_0 = structure_module.gamma_0
    gamma_min = structure_module.gamma_min
    gammas = jnp.where(sigmas > gamma_min, gamma_0, 0.0)

    # 2. Initialize coords: parent + init_sigma noise on designable atoms only.
    # Shapes after batching: initial_coords (1, M, 3), partial mask (1, M).
    # `[..., None]` adds a trailing axis (broadcasts with the trailing 3 of coords).
    init_sigma = sigmas[0]
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, initial_coords.shape)
    if sidechain_mask is not None and sidechain_noise_multiplier != 1.0:
        sc_sigma = sidechain_noise_multiplier * init_sigma
        atom_coords = jnp.where(
            ((sidechain_mask > 0) & (atom_partial_mask > 0))[..., None],
            initial_coords + sc_sigma * noise,
            jnp.where(
                (atom_partial_mask > 0)[..., None],
                initial_coords + init_sigma * noise,
                initial_coords,
            ),
        )
    else:
        atom_coords = jnp.where(
            (atom_partial_mask > 0)[..., None],
            initial_coords + init_sigma * noise,
            initial_coords,
        )

    # Build the guidance subgraph only if we have a primary (binding) guidance
    # function. Resolved at trace time (plain Python None-checks, not JAX
    # values), so JIT sees a single static path per call — no guidance
    # machinery is even traced when guidance_fn_bind is None.
    grad_bind = jax.grad(guidance_fn_bind) if guidance_fn_bind is not None else None
    grad_nat = jax.grad(guidance_fn_nat) if guidance_fn_nat is not None else None
    grad_edit = jax.grad(guidance_fn_edit) if guidance_fn_edit is not None else None

    lambda_fn = guidance_lambda_fn if guidance_lambda_fn is not None else default_lambda_schedule
    tau_fn = guidance_tau_fn if guidance_tau_fn is not None else default_tau_schedule
    alpha_fn = guidance_alpha_fn if guidance_alpha_fn is not None else default_alpha_schedule
    beta_fn = guidance_beta_fn if guidance_beta_fn is not None else default_beta_schedule
    nos_inner_steps = guidance_nos_inner_steps
    nos_inner_step_fn = (
        guidance_nos_inner_step_fn if guidance_nos_inner_step_fn is not None else default_nos_inner_step_schedule
    )
    nos_lambda_kl_fn = (
        guidance_nos_lambda_kl_fn if guidance_nos_lambda_kl_fn is not None else default_nos_lambda_kl_schedule
    )
    nos_noise_fn = (
        guidance_nos_noise_fn if guidance_nos_noise_fn is not None else default_nos_langevin_noise_schedule
    )
    if guidance_lookahead and nos_inner_steps > 0:
        raise ValueError(
            "guidance_lookahead=True and guidance_nos_inner_steps > 0 are mutually "
            "exclusive -- combining them would mean K full denoiser backward passes "
            "per diffusion step, not scoped or tested. Pick one mechanism."
        )

    # network_condition_kwargs MUST match what joltzgen.AtomDiffusion.sample
    # passes through to preconditioned_network_forward — i.e. everything except
    # `atom_mask` and the explicit sample-loop kwargs. Verified against
    # joltzgen 0.1.0 sample() signature.
    diffusion_conditioning = {
        "q": sampler.q,
        "c": sampler.c,
        "to_keys": sampler.to_keys,
        "atom_enc_bias": sampler.atom_enc_bias,
        "atom_dec_bias": sampler.atom_dec_bias,
        "token_trans_bias": sampler.token_trans_bias,
    }
    network_condition_kwargs = dict(
        s_trunk=sampler.trunk_s,
        s_inputs=sampler.s_inputs,
        feats=sampler.feats,
        diffusion_conditioning=diffusion_conditioning,
        multiplicity=1,
    )

    def step_body(carry, idx):
        atom_coords, init_coords_rep, key = carry
        sigma_tm = sigmas[idx]
        sigma_t = sigmas[idx + 1]
        gamma = gammas[idx + 1]

        t_hat = sigma_tm * (1.0 + gamma)
        noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)

        # Re-center both (diffusion.py:656-659)
        atom_coords = _center(atom_coords, atom_mask)
        init_coords_rep = _center(init_coords_rep, atom_mask)

        # EDM churn noise (diffusion.py:676-677). Matches joltzgen's
        # `noise_scale * sqrt(noise_var)` which expands to
        # `noise_scale^2 * sqrt(t_hat^2 - sigma^2)`.
        key, sub = jax.random.split(key)
        eps = (
            noise_scale
            * jnp.sqrt(jnp.maximum(noise_var, 0.0))
            * jax.random.normal(sub, atom_coords.shape)
        )
        atom_coords_noisy = atom_coords + eps

        # Frozen denoiser call. stop_gradient ensures we do NOT backprop through
        # the BoltzGen network — guidance gradients flow only through the much
        # smaller IF + aux-loss subgraph below.
        # joltzgen 0.1.0 signature:
        #   preconditioned_network_forward(noised_coords, sigma,
        #                                  network_condition_kwargs: dict,
        #                                  *, key) -> denoised_coords  [b m 3]
        key, sub = jax.random.split(key)
        x0_hat = structure_module.preconditioned_network_forward(
            atom_coords_noisy,
            t_hat,
            network_condition_kwargs=network_condition_kwargs,
            key=sub,
        )
        x0_hat = jax.lax.stop_gradient(x0_hat)

        # Optional rigid alignment of noisy onto denoised (matches joltzgen.sample
        # when alignment_reverse_diff=True; the BoltzGen-1 release ckpts have
        # this set to True). The denoiser is SE(3) equivariant but doesn't pin a
        # global frame, so without this the per-step (noisy - x0_hat)/t_hat term
        # picks up a rigid-body component that compounds across steps. Aligns
        # `atom_coords_noisy` to `x0_hat` (NOT to x0_guided) so the Euler step
        # operates in a frame defined purely by the model's prediction.
        if structure_module.alignment_reverse_diff:
            from joltzgen import weighted_rigid_align
            atom_coords_noisy = weighted_rigid_align(
                atom_coords_noisy, x0_hat, atom_mask, atom_mask,
            )

        # The reverse-update direction the sampler would use with NO guidance
        # at all, computed from the same atom_coords_noisy the real step uses
        # (post-churn, post optional rigid realignment above) — not a
        # hand-derived shorthand. This is the quantity
        # docs/guidance_implementation_todo.md Phase 1 calls out to expose
        # for Phase 2's prior-compatibility diagnostics.
        unguided_direction = (atom_coords_noisy - x0_hat) / t_hat

        # === GUIDANCE INJECTION (Phase 1 controller; no-op when guidance_fn_bind is None) ===
        zeros_like_g = jnp.zeros_like(x0_hat)
        if grad_bind is not None:
            if nos_inner_steps > 0:
                # Real iterative NOS-style path (section 12a) -- static
                # branch, resolved at trace time like `grad_bind is not
                # None` above, so the one-shot path below costs nothing when
                # this is off (the default).
                key, nos_key = jax.random.split(key)
                x0_candidate, g_bind, g_nat, g_edit = _nos_iterative_merge(
                    x0_hat, grad_bind, grad_nat, grad_edit, atom_partial_mask,
                    alpha_fn, beta_fn, tau_fn, t_hat, nos_inner_steps,
                    nos_inner_step_fn, nos_lambda_kl_fn, nos_noise_fn, nos_key,
                )
                # lambda_fn(t_hat) must still scale the merged displacement here,
                # exactly as it does in the one-shot path below -- otherwise
                # --lambda-max/--lambda-schedule silently stop controlling
                # guidance strength for NOS runs (real bug, found by review,
                # verified directly against this code before fixing; see
                # docs/guidance_alphaseq_testing_notes.md section 12a).
                # nos_inner_step_size/nos_lambda_kl remain the inner solver's
                # own step size and prior-compatibility strength -- mechanism-
                # internal parameters, not a substitute for the outer schedule.
                delta = lambda_fn(t_hat) * (x0_hat - x0_candidate)
            elif guidance_lookahead:
                # Real look-ahead path (section 12b): differentiate through
                # the FULL denoiser with respect to atom_coords_noisy,
                # instead of with respect to the frozen x0_hat. A fresh
                # denoiser closure is built per step (t_hat and
                # network_condition_kwargs vary step to step, unlike
                # grad_bind/grad_nat/grad_edit above, which are built once
                # outside the scan since they only close over the guidance
                # loss). _merge_aux_gradients is reused unchanged -- the
                # merge machinery doesn't care which point space a closure
                # differentiates in, only that it returns a same-shaped
                # gradient.
                key, la_key = jax.random.split(key)

                def _denoiser_fn(coords, _t_hat=t_hat, _key=la_key):
                    return structure_module.preconditioned_network_forward(
                        coords, _t_hat, network_condition_kwargs=network_condition_kwargs, key=_key,
                    )

                grad_bind_la = build_lookahead_grad_fn(_denoiser_fn, guidance_fn_bind)
                grad_nat_la = (
                    build_lookahead_grad_fn(_denoiser_fn, guidance_fn_nat) if guidance_fn_nat is not None else None
                )
                grad_edit_la = (
                    build_lookahead_grad_fn(_denoiser_fn, guidance_fn_edit) if guidance_fn_edit is not None else None
                )
                g_total, g_bind, g_nat, g_edit = _merge_aux_gradients(
                    atom_coords_noisy, grad_bind_la, grad_nat_la, grad_edit_la,
                    atom_partial_mask, alpha_fn, beta_fn, t_hat,
                )
                # Sign/scale: an earlier version of this line derived a
                # direction-space equivalence by hand (adjust
                # denoised_over_sigma, solve for an x0-equivalent delta) and
                # got it wrong -- shipped as `delta = -t_hat * lambda_fn(t_hat)
                # * g_total`, which empirically made the toy guidance loss
                # WORSE, not better (caught by review, reproduced directly:
                # guidance_loss(x0_hat)=9.12, guidance_loss(x0_hat + t_hat*
                # lam*g_total)=12.67 -- confirms the bug -- vs
                # guidance_loss(x0_hat - lam*g_total)=5.17, a real decrease).
                # That hand derivation was internally self-consistent but
                # never checked against an actual loss decrease, which is
                # the only thing that actually matters here -- self-
                # consistency of an unvalidated premise is not correctness.
                # g_total here (d(loss)/d(atom_coords_noisy) via the chain
                # rule through the denoiser) is empirically well-aligned in
                # direction with d(loss)/d(x0) evaluated at x0_hat (cosine
                # ~0.93 in the toy case checked) -- descending along it in
                # x0-space works the same way descending g_bind does in the
                # one-shot path. No special sign or extra t_hat scaling:
                # treat it exactly like the one-shot/NOS merged gradient.
                delta = lambda_fn(t_hat) * g_total
            else:
                g_total, g_bind, g_nat, g_edit = _merge_aux_gradients(
                    x0_hat, grad_bind, grad_nat, grad_edit, atom_partial_mask, alpha_fn, beta_fn, t_hat
                )
                delta = lambda_fn(t_hat) * g_total

            delta = _clip_rms(delta, tau_fn(t_hat), atom_partial_mask)
            x0_guided = x0_hat - delta
        else:
            g_bind = zeros_like_g
            g_nat = zeros_like_g
            g_edit = zeros_like_g
            x0_guided = x0_hat
        # ===========================

        # Euler step in EDM parameterization (diffusion.py:702-705)
        denoised_over_sigma = (atom_coords_noisy - x0_guided) / t_hat
        guided_step_delta = _euler_step_delta(denoised_over_sigma, step_scale, sigma_t, t_hat)
        atom_coords_next = atom_coords_noisy + guided_step_delta

        # Re-anchor frozen atoms (diffusion.py:707-713)
        atom_coords_next = jnp.where(
            (atom_partial_mask > 0)[..., None], atom_coords_next, init_coords_rep
        )

        if return_diagnostics:
            # unguided_direction/guided_direction are the raw (atom_coords_noisy
            # - x0)/t_hat derivatives -- valid for cosine comparisons (scale
            # invariant) but NOT for the physical displacement norm ratio that
            # docs/guidance_implementation_todo.md's Phase 2 diagnostics call
            # for. *_step_delta below apply the same step_scale * (sigma_t -
            # t_hat) factor the real Euler update uses, so their norms are
            # directly comparable to the actual per-step coordinate motion.
            diagnostics = {
                "unguided_direction": unguided_direction,
                "guided_direction": denoised_over_sigma,
                "unguided_step_delta": _euler_step_delta(
                    unguided_direction, step_scale, sigma_t, t_hat
                ),
                "guided_step_delta": guided_step_delta,
                "g_bind": g_bind,
                "g_nat": g_nat,
                "g_edit": g_edit,
                # noise-axis position of this step, for stratifying Phase 2
                # diagnostics by sigma (docs/guidance_implementation_todo.md
                # Phase 2). t_hat is the churned noise level the guidance
                # gradients above were actually evaluated at; sigma_t is the
                # step's target noise level.
                "t_hat": t_hat,
                "sigma_t": sigma_t,
            }
        else:
            diagnostics = None
        return (atom_coords_next, init_coords_rep, key), diagnostics

    n_steps = len(sigmas) - 1
    (atom_coords_final, _, _), diagnostics = jax.lax.scan(
        step_body, (atom_coords, initial_coords, key), jnp.arange(n_steps)
    )
    if unbatched_input:
        atom_coords_final = atom_coords_final[0]
    if return_diagnostics:
        return atom_coords_final, diagnostics
    return atom_coords_final


def build_atom_partial_mask(
    features: dict,
    designable_token_mask: Bool[Array, "N"],
) -> Float[Array, "M"]:
    """Convert a token-level designable mask to atom-level via the feature's atom_to_token.

    Mirrors the torch sampler's mask conversion at `diffusion.py:565-574`.
    Returns a float array (1.0 at designable atoms, 0.0 elsewhere) for use as
    `atom_partial_mask` in `guided_partial_diffusion`.
    """
    atom_to_token = features["atom_to_token"]  # (B, M, N) or (M, N)
    if atom_to_token.ndim == 3:
        atom_to_token = atom_to_token[0]
    return (atom_to_token.astype(jnp.float32) @ designable_token_mask.astype(jnp.float32))

