#!/usr/bin/env python3
"""CAAT-style Boltz2 sensitivity scan for VHH/RBD optimization.

This is not the original AlphaFold CAAT attention extractor.  It applies the
same practical question to Boltz2: which editable binder positions cause a
confidence/interface response when mutated, and how many edits are needed
before the response rises above same-sequence sampling noise?

The default refold setting mirrors the design-time diagnostic:

  target template + binder framework template, with editable CDR atoms removed.

Outputs:
  - baseline_samples.csv
  - single_mutation_scores.csv
  - position_sensitivity.csv
  - edit_count_curve.csv
  - config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boltz.data.const as const  # noqa: E402
from examples.boltzgen_vhh_guided import interface_geometry_metrics  # noqa: E402
from mosaic.common import TOKENS, tokenize  # noqa: E402
from mosaic.losses.boltz2 import boltz2_forward_from_trunk, boltz2_trunk  # noqa: E402
from mosaic.losses.structure_prediction import (  # noqa: E402
    BinderTargetIPSAE,
    IPSAE_min,
    IPTMLoss,
    TargetBinderIPSAE,
)
from mosaic.models.boltz2 import Boltz2  # noqa: E402
from mosaic.structure_prediction import TargetChain  # noqa: E402
from mosaic.util import fold_in  # noqa: E402


AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

PRESETS = {
    "vhh72": {
        "structure": ROOT / "optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb",
        "binder_chain": "A",
        "target_chain": "E",
        "design_indices": "30..31,52..65,100..111",
        "description": "VHH72 WT / SARS-CoV-2 WT RBD",
    },
    "p17-alpha": {
        "structure": Path(
            "/home/yfeng17/SBSAb/dataset/Affinity_train/structures/"
            "phase2c_abag_ANDD_ANDD_8GZ5_B_X_A_WT.cif"
        ),
        "binder_chain": "B",
        "target_chain": "A",
        "design_indices": "26..33,51..58,97..109",
        "description": "P17 / SARS-CoV-2 Alpha RBD positive reference",
    },
    "p17-jn1": {
        "structure": Path("/home/yfeng17/boltzgen/Round2_Design/Nanobody/pdb/P17_JN1.pdb"),
        "binder_chain": "B",
        "target_chain": "T",
        "design_indices": "26..33,51..58,97..109",
        "description": "P17 / SARS-CoV-2 JN.1 escape reference",
    },
}


@dataclass(frozen=True)
class ScanConfig:
    preset: str
    structure: Path
    binder_chain: str
    target_chain: str
    design_indices: list[int]
    aa_panel: str
    output_dir: Path
    baseline_samples: int
    num_samples: int
    sampling_steps: int
    recycling_steps: int
    ipsae_pae_cutoff: float
    seed: int
    min_delta_ipsae: float
    min_abs_z: float
    max_combo_edits: int
    top_positions: int


def parse_ranges(raw: str) -> list[int]:
    out: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ".." in piece:
            start, end = piece.split("..", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(piece))
    return sorted(dict.fromkeys(out))


def parse_aa_panel(raw: str) -> str:
    if raw.lower() == "all":
        return TOKENS
    seen = []
    for aa in raw.upper():
        if aa not in TOKENS:
            raise ValueError(f"Unknown amino-acid code {aa!r}; allowed: {TOKENS}")
        if aa not in seen:
            seen.append(aa)
    if not seen:
        raise ValueError("AA panel is empty")
    return "".join(seen)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def protein_chain(structure: gemmi.Structure, chain_id: str) -> gemmi.Chain:
    chain = structure[0][chain_id]
    new_chain = gemmi.Chain(chain.name)
    for residue in chain:
        if residue.name in AA3:
            new_chain.add_residue(residue.clone())
    return new_chain


def read_chain_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    seq = gemmi.one_letter_code([res.name for res in protein_chain(structure, chain_id)])
    seq = seq.replace("X", "")
    if not seq:
        raise ValueError(f"No protein residues found for chain {chain_id!r}")
    return seq


def residue_labels(chain: gemmi.Chain) -> dict[int, str]:
    labels = {}
    idx = 0
    for residue in chain:
        if residue.name not in AA3:
            continue
        idx += 1
        icode = residue.seqid.icode.strip()
        author = f"{residue.seqid.num}{icode}" if icode else str(residue.seqid.num)
        labels[idx] = f"{chain.name}:{author}:{residue.name}"
    return labels


def masked_framework_template(chain: gemmi.Chain, design_indices: Iterable[int]) -> gemmi.Chain:
    design_set = set(int(i) for i in design_indices)
    new_chain = gemmi.Chain(chain.name)
    idx = 0
    for residue in chain:
        if residue.name not in AA3:
            continue
        idx += 1
        if idx in design_set:
            masked = gemmi.Residue()
            masked.name = residue.name
            masked.seqid = residue.seqid
            masked.subchain = residue.subchain
            masked.label_seq = residue.label_seq
            masked.entity_id = residue.entity_id
            new_chain.add_residue(masked)
        else:
            new_chain.add_residue(residue.clone())
    return new_chain


def set_binder_sequence_quiet(new_sequence, features):
    """Same as mosaic.losses.boltz2.set_binder_sequence, without stdout spam."""
    features = features | {}
    features["res_type"] = features["res_type"].astype(jnp.float32)
    features["msa"] = features["msa"].astype(jnp.float32)
    features["profile"] = features["profile"].astype(jnp.float32)
    binder_len = new_sequence.shape[0]
    zero_padded_sequence = jnp.pad(new_sequence, ((0, 0), (2, const.num_tokens - 22)))
    n_msa = features["msa"].shape[1]
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)
    return features | {
        "res_type": features["res_type"].at[0, :binder_len, :].set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


def summarize_values(values: list[dict], prefix: str = "") -> dict:
    out = {}
    for key in ["iptm", "bt_ipsae", "tb_ipsae", "ipsae_min", "ipae_min", "bt_pae_mean", "tb_pae_mean"]:
        arr = np.asarray([float(v[key]) for v in values], dtype=np.float64)
        out[f"{prefix}{key}_mean"] = float(np.mean(arr))
        out[f"{prefix}{key}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        out[f"{prefix}{key}_min"] = float(np.min(arr))
        out[f"{prefix}{key}_max"] = float(np.max(arr))
    return out


def finite_z(delta: float, sigma: float) -> float:
    if not math.isfinite(sigma) or sigma <= 1e-8:
        return math.inf if abs(delta) > 0 else 0.0
    return delta / sigma


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="vhh72")
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--binder-chain")
    parser.add_argument("--target-chain")
    parser.add_argument("--design-indices")
    parser.add_argument("--aa-panel", default="FWYRMSTV", help="'all' or one-letter AA list")
    parser.add_argument("--output-dir", type=Path, default=Path("vhh/boltz2_caat_sensitivity"))
    parser.add_argument("--baseline-samples", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-delta-ipsae", type=float, default=0.10)
    parser.add_argument("--min-abs-z", type=float, default=2.0)
    parser.add_argument("--max-combo-edits", type=int, default=7)
    parser.add_argument("--top-positions", type=int, default=12)
    return parser


def resolve_config(args: argparse.Namespace) -> ScanConfig:
    preset = PRESETS[args.preset]
    structure = args.structure or preset["structure"]
    binder_chain = args.binder_chain or preset["binder_chain"]
    target_chain = args.target_chain or preset["target_chain"]
    raw_indices = args.design_indices or preset["design_indices"]
    return ScanConfig(
        preset=args.preset,
        structure=Path(structure),
        binder_chain=binder_chain,
        target_chain=target_chain,
        design_indices=parse_ranges(raw_indices),
        aa_panel=parse_aa_panel(args.aa_panel),
        output_dir=args.output_dir,
        baseline_samples=args.baseline_samples,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        recycling_steps=args.recycling_steps,
        ipsae_pae_cutoff=args.ipsae_pae_cutoff,
        seed=args.seed,
        min_delta_ipsae=args.min_delta_ipsae,
        min_abs_z=args.min_abs_z,
        max_combo_edits=args.max_combo_edits,
        top_positions=args.top_positions,
    )


def main() -> int:
    args = build_parser().parse_args()
    cfg = resolve_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.baseline_samples < 2:
        raise ValueError("--baseline-samples should be >= 2 to estimate sampling noise")
    if cfg.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if not cfg.structure.exists():
        raise FileNotFoundError(cfg.structure)

    print(f"[caat] preset={cfg.preset} structure={cfg.structure}", flush=True)
    structure = gemmi.read_structure(str(cfg.structure))
    structure.setup_entities()
    binder_sequence = read_chain_sequence(structure, cfg.binder_chain)
    target_sequence = read_chain_sequence(structure, cfg.target_chain)
    binder_chain = protein_chain(structure, cfg.binder_chain)
    target_chain = protein_chain(structure, cfg.target_chain)
    labels = residue_labels(binder_chain)
    bad_indices = [i for i in cfg.design_indices if i < 1 or i > len(binder_sequence)]
    if bad_indices:
        raise ValueError(f"Design indices outside binder sequence: {bad_indices}")

    print(
        f"[caat] binder_len={len(binder_sequence)} target_len={len(target_sequence)} "
        f"positions={len(cfg.design_indices)} aa_panel={cfg.aa_panel}",
        flush=True,
    )

    binder_template = masked_framework_template(binder_chain, cfg.design_indices)
    refold_chains = [
        TargetChain(binder_sequence, use_msa=False, template_chain=binder_template),
        TargetChain(target_sequence, use_msa=False, template_chain=target_chain),
    ]

    print("[caat] loading Boltz2/features", flush=True)
    boltz2 = Boltz2()
    features, writer = boltz2.target_only_features(refold_chains)
    del writer
    print("[caat] loaded Boltz2/features", flush=True)

    iptm_loss = IPTMLoss()
    bt_ipsae_loss = BinderTargetIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    tb_ipsae_loss = TargetBinderIPSAE(pae_cutoff=cfg.ipsae_pae_cutoff)
    ipsae_min_loss = IPSAE_min(pae_cutoff=cfg.ipsae_pae_cutoff)

    def score_sequence(seq_ids, sample_keys):
        seq_one_hot = jax.nn.one_hot(seq_ids, 20)
        mutated_features = set_binder_sequence_quiet(seq_one_hot, features)
        initial_emb, trunk_state = boltz2_trunk(
            boltz2.model,
            mutated_features,
            recycling_steps=cfg.recycling_steps,
            deterministic=True,
            key=fold_in(sample_keys[0], "trunk"),
        )

        def score_one(sample_key):
            out = boltz2_forward_from_trunk(
                boltz2.model,
                mutated_features,
                initial_emb,
                trunk_state,
                num_sampling_steps=cfg.sampling_steps,
                deterministic=True,
                key=sample_key,
            )
            _, iptm_aux = iptm_loss(sequence=seq_one_hot, output=out, key=fold_in(sample_key, "iptm"))
            _, bt_aux = bt_ipsae_loss(sequence=seq_one_hot, output=out, key=fold_in(sample_key, "bt_ipsae"))
            _, tb_aux = tb_ipsae_loss(sequence=seq_one_hot, output=out, key=fold_in(sample_key, "tb_ipsae"))
            _, ipsae_aux = ipsae_min_loss(sequence=seq_one_hot, output=out, key=fold_in(sample_key, "ipsae_min"))
            binder_len = seq_one_hot.shape[0]
            bt_pae = out.pae[:binder_len, binder_len:]
            tb_pae = out.pae[binder_len:, :binder_len]
            return {
                "iptm": iptm_aux["iptm"],
                "bt_ipsae": bt_aux["bt_ipsae"],
                "tb_ipsae": tb_aux["tb_ipsae"],
                "ipsae_min": ipsae_aux["ipsae_min"],
                "ipae_min": jnp.minimum(jnp.min(bt_pae), jnp.min(tb_pae)),
                "bt_pae_mean": jnp.mean(bt_pae),
                "tb_pae_mean": jnp.mean(tb_pae),
            }

        return jax.vmap(score_one)(sample_keys)

    score_sequence = eqx.filter_jit(score_sequence)

    def run_scores(label: str, sequence: str, n_samples: int, seed_offset: int) -> list[dict]:
        seq_ids = jnp.asarray(tokenize(sequence))
        keys = jax.random.split(jax.random.key(cfg.seed + seed_offset), n_samples)
        scores = score_sequence(seq_ids, keys)
        jax.tree.map(lambda x: x.block_until_ready(), scores)
        rows = []
        for sample_idx in range(n_samples):
            row = {"label": label, "sample_idx": sample_idx}
            for key, value in scores.items():
                row[key] = float(value[sample_idx])
            rows.append(row)
        return rows

    baseline_rows = run_scores("WT", binder_sequence, cfg.baseline_samples, 1000)
    baseline_summary = summarize_values(baseline_rows)
    write_csv(cfg.output_dir / "baseline_samples.csv", baseline_rows)

    baseline_ipsae = baseline_summary["ipsae_min_mean"]
    baseline_ipae = baseline_summary["ipae_min_mean"]
    sigma_ipsae = baseline_summary["ipsae_min_std"]
    sigma_ipae = baseline_summary["ipae_min_std"]
    print(
        "[caat] baseline "
        f"ipSAE={baseline_ipsae:.4f}+/-{sigma_ipsae:.4f} "
        f"iPAE={baseline_ipae:.3f}+/-{sigma_ipae:.3f}",
        flush=True,
    )

    single_rows = []
    raw_sample_rows = []
    variant_count = 0
    for pos in cfg.design_indices:
        wt = binder_sequence[pos - 1]
        for aa in cfg.aa_panel:
            if aa == wt:
                continue
            variant_count += 1
            mutation = f"{wt}{pos}{aa}"
            seq = binder_sequence[: pos - 1] + aa + binder_sequence[pos:]
            rows = run_scores(mutation, seq, cfg.num_samples, 100000 + variant_count)
            for row in rows:
                raw_sample_rows.append({
                    "mutation": mutation,
                    "position": pos,
                    "wt_aa": wt,
                    "mut_aa": aa,
                    "author_residue": labels.get(pos, ""),
                    **row,
                })
            summary = summarize_values(rows)
            delta_ipsae = summary["ipsae_min_mean"] - baseline_ipsae
            delta_ipae = summary["ipae_min_mean"] - baseline_ipae
            z_ipsae = finite_z(delta_ipsae, sigma_ipsae)
            z_ipae = finite_z(delta_ipae, sigma_ipae)
            acknowledged = (
                abs(delta_ipsae) >= cfg.min_delta_ipsae
                and abs(z_ipsae) >= cfg.min_abs_z
            )
            single_rows.append({
                "mutation": mutation,
                "position": pos,
                "author_residue": labels.get(pos, ""),
                "wt_aa": wt,
                "mut_aa": aa,
                **summary,
                "baseline_ipsae_min_mean": baseline_ipsae,
                "baseline_ipae_min_mean": baseline_ipae,
                "delta_ipsae_min_mean": delta_ipsae,
                "abs_delta_ipsae_min_mean": abs(delta_ipsae),
                "z_delta_ipsae_min_mean": z_ipsae,
                "delta_ipae_min_mean": delta_ipae,
                "z_delta_ipae_min_mean": z_ipae,
                "acknowledged": acknowledged,
            })
            print(
                f"[caat] {mutation:<8} d_ipSAE={delta_ipsae:+.4f} "
                f"z={z_ipsae:+.2f} d_iPAE={delta_ipae:+.3f}",
                flush=True,
            )
            # Keep partial CSVs useful if an expensive scan is interrupted.
            write_csv(cfg.output_dir / "single_mutation_sample_scores.csv", raw_sample_rows)
            write_csv(cfg.output_dir / "single_mutation_scores.csv", single_rows)

    write_csv(cfg.output_dir / "single_mutation_sample_scores.csv", raw_sample_rows)
    write_csv(cfg.output_dir / "single_mutation_scores.csv", single_rows)

    position_rows = []
    by_pos = {pos: [r for r in single_rows if r["position"] == pos] for pos in cfg.design_indices}
    for pos, rows in by_pos.items():
        if not rows:
            continue
        most_sensitive = max(rows, key=lambda r: r["abs_delta_ipsae_min_mean"])
        best_increase = max(rows, key=lambda r: r["delta_ipsae_min_mean"])
        worst_decrease = min(rows, key=lambda r: r["delta_ipsae_min_mean"])
        position_rows.append({
            "position": pos,
            "author_residue": labels.get(pos, ""),
            "wt_aa": binder_sequence[pos - 1],
            "n_mutations_tested": len(rows),
            "n_acknowledged": sum(bool(r["acknowledged"]) for r in rows),
            "mean_abs_delta_ipsae_min": float(np.mean([r["abs_delta_ipsae_min_mean"] for r in rows])),
            "max_abs_delta_ipsae_min": most_sensitive["abs_delta_ipsae_min_mean"],
            "most_sensitive_mutation": most_sensitive["mutation"],
            "most_sensitive_delta_ipsae": most_sensitive["delta_ipsae_min_mean"],
            "best_increase_mutation": best_increase["mutation"],
            "best_delta_ipsae": best_increase["delta_ipsae_min_mean"],
            "worst_decrease_mutation": worst_decrease["mutation"],
            "worst_delta_ipsae": worst_decrease["delta_ipsae_min_mean"],
        })
    position_rows.sort(key=lambda r: r["max_abs_delta_ipsae_min"], reverse=True)
    write_csv(cfg.output_dir / "position_sensitivity.csv", position_rows)

    aa_type_rows = []
    by_mut_aa = {aa: [r for r in single_rows if r["mut_aa"] == aa] for aa in cfg.aa_panel}
    for aa, rows in by_mut_aa.items():
        if not rows:
            continue
        most_sensitive = max(rows, key=lambda r: r["abs_delta_ipsae_min_mean"])
        best_increase = max(rows, key=lambda r: r["delta_ipsae_min_mean"])
        worst_decrease = min(rows, key=lambda r: r["delta_ipsae_min_mean"])
        aa_type_rows.append({
            "mut_aa": aa,
            "n_positions_tested": len(rows),
            "n_acknowledged": sum(bool(r["acknowledged"]) for r in rows),
            "acknowledged_fraction": (
                sum(bool(r["acknowledged"]) for r in rows) / len(rows)
            ),
            "mean_delta_ipsae_min": float(np.mean([r["delta_ipsae_min_mean"] for r in rows])),
            "mean_abs_delta_ipsae_min": float(np.mean([r["abs_delta_ipsae_min_mean"] for r in rows])),
            "max_abs_delta_ipsae_min": most_sensitive["abs_delta_ipsae_min_mean"],
            "most_sensitive_mutation": most_sensitive["mutation"],
            "most_sensitive_position": most_sensitive["position"],
            "most_sensitive_author_residue": most_sensitive["author_residue"],
            "most_sensitive_delta_ipsae": most_sensitive["delta_ipsae_min_mean"],
            "best_increase_mutation": best_increase["mutation"],
            "best_increase_position": best_increase["position"],
            "best_delta_ipsae": best_increase["delta_ipsae_min_mean"],
            "worst_decrease_mutation": worst_decrease["mutation"],
            "worst_decrease_position": worst_decrease["position"],
            "worst_delta_ipsae": worst_decrease["delta_ipsae_min_mean"],
        })
    aa_type_rows.sort(key=lambda r: r["max_abs_delta_ipsae_min"], reverse=True)
    write_csv(cfg.output_dir / "aa_type_sensitivity.csv", aa_type_rows)

    combo_rows = []
    for mode in ["absolute", "increase"]:
        if mode == "absolute":
            selected = sorted(
                [max(rows, key=lambda r: r["abs_delta_ipsae_min_mean"]) for rows in by_pos.values() if rows],
                key=lambda r: r["abs_delta_ipsae_min_mean"],
                reverse=True,
            )
        else:
            selected = sorted(
                [max(rows, key=lambda r: r["delta_ipsae_min_mean"]) for rows in by_pos.values() if rows],
                key=lambda r: r["delta_ipsae_min_mean"],
                reverse=True,
            )
        selected = selected[: cfg.top_positions]
        seq_list = list(binder_sequence)
        used = []
        for edit_count, row in enumerate(selected[: cfg.max_combo_edits], start=1):
            pos = int(row["position"])
            seq_list[pos - 1] = str(row["mut_aa"])
            used.append(row["mutation"])
            combo_label = f"{mode}_top{edit_count}"
            combo_seq = "".join(seq_list)
            rows = run_scores(combo_label, combo_seq, cfg.num_samples, 500000 + 1000 * (mode == "increase") + edit_count)
            summary = summarize_values(rows)
            delta_ipsae = summary["ipsae_min_mean"] - baseline_ipsae
            delta_ipae = summary["ipae_min_mean"] - baseline_ipae
            z_ipsae = finite_z(delta_ipsae, sigma_ipsae)
            combo_rows.append({
                "mode": mode,
                "edit_count": edit_count,
                "mutations": ";".join(used),
                **summary,
                "baseline_ipsae_min_mean": baseline_ipsae,
                "baseline_ipae_min_mean": baseline_ipae,
                "delta_ipsae_min_mean": delta_ipsae,
                "abs_delta_ipsae_min_mean": abs(delta_ipsae),
                "z_delta_ipsae_min_mean": z_ipsae,
                "delta_ipae_min_mean": delta_ipae,
                "acknowledged": (
                    abs(delta_ipsae) >= cfg.min_delta_ipsae
                    and abs(z_ipsae) >= cfg.min_abs_z
                ),
            })
    write_csv(cfg.output_dir / "edit_count_curve.csv", combo_rows)

    acknowledgement_rows = []
    for mode in ["absolute", "increase"]:
        rows = [r for r in combo_rows if r["mode"] == mode]
        first = next((r for r in rows if r["acknowledged"]), None)
        five = next((r for r in rows if int(r["edit_count"]) == 5), None)
        nearest_to_five = max(
            [r for r in rows if int(r["edit_count"]) <= 5],
            key=lambda r: int(r["edit_count"]),
            default=None,
        )
        chosen_five = five or nearest_to_five
        acknowledgement_rows.append({
            "mode": mode,
            "first_acknowledged_edit_count": (
                first["edit_count"] if first is not None else ""
            ),
            "first_acknowledged_delta_ipsae": (
                first["delta_ipsae_min_mean"] if first is not None else ""
            ),
            "first_acknowledged_mutations": (
                first["mutations"] if first is not None else ""
            ),
            "five_edit_available": five is not None,
            "five_or_nearest_edit_count": (
                chosen_five["edit_count"] if chosen_five is not None else ""
            ),
            "five_or_nearest_acknowledged": (
                chosen_five["acknowledged"] if chosen_five is not None else ""
            ),
            "five_or_nearest_delta_ipsae": (
                chosen_five["delta_ipsae_min_mean"] if chosen_five is not None else ""
            ),
            "five_or_nearest_z_delta_ipsae": (
                chosen_five["z_delta_ipsae_min_mean"] if chosen_five is not None else ""
            ),
            "five_or_nearest_mutations": (
                chosen_five["mutations"] if chosen_five is not None else ""
            ),
        })
    write_csv(cfg.output_dir / "acknowledgement_summary.csv", acknowledgement_rows)

    try:
        ref_geom = interface_geometry_metrics(
            structure,
            binder_chain_id=cfg.binder_chain,
            target_chain_ids=[cfg.target_chain],
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        ref_geom = {"interface_geometry_error": str(exc)}

    (cfg.output_dir / "config.json").write_text(
        json.dumps(
            {
                "preset": cfg.preset,
                "structure": str(cfg.structure),
                "binder_chain": cfg.binder_chain,
                "target_chain": cfg.target_chain,
                "design_indices": cfg.design_indices,
                "aa_panel": cfg.aa_panel,
                "baseline_samples": cfg.baseline_samples,
                "num_samples": cfg.num_samples,
                "sampling_steps": cfg.sampling_steps,
                "recycling_steps": cfg.recycling_steps,
                "ipsae_pae_cutoff": cfg.ipsae_pae_cutoff,
                "min_delta_ipsae": cfg.min_delta_ipsae,
                "min_abs_z": cfg.min_abs_z,
                "reference_interface_geometry": ref_geom,
                "top_positions_by_max_abs_delta_ipsae": position_rows[:10],
                "top_aa_types_by_max_abs_delta_ipsae": aa_type_rows[:10],
                "acknowledgement_summary": acknowledgement_rows,
            },
            indent=2,
        )
        + "\n"
    )

    top = position_rows[: min(10, len(position_rows))]
    print("\n[caat] top sensitive positions")
    for row in top:
        print(
            f"  pos {row['position']:>3} {row['author_residue']:<16} "
            f"max_abs_d_ipSAE={row['max_abs_delta_ipsae_min']:.4f} "
            f"mutation={row['most_sensitive_mutation']} "
            f"best={row['best_increase_mutation']}({row['best_delta_ipsae']:+.4f})"
        )

    print("\n[caat] top sensitive amino-acid types")
    for row in aa_type_rows[: min(10, len(aa_type_rows))]:
        print(
            f"  {row['mut_aa']}: max_abs_d_ipSAE={row['max_abs_delta_ipsae_min']:.4f} "
            f"at {row['most_sensitive_mutation']} "
            f"ack={row['n_acknowledged']}/{row['n_positions_tested']} "
            f"best={row['best_increase_mutation']}({row['best_delta_ipsae']:+.4f})"
        )

    print("\n[caat] edit-count acknowledgement")
    for mode in ["absolute", "increase"]:
        rows = [r for r in combo_rows if r["mode"] == mode]
        first = next((r for r in rows if r["acknowledged"]), None)
        if first is None:
            print(f"  {mode}: not acknowledged up to {len(rows)} edits")
        else:
            print(
                f"  {mode}: first acknowledged at {first['edit_count']} edit(s), "
                f"d_ipSAE={first['delta_ipsae_min_mean']:+.4f}, "
                f"mutations={first['mutations']}"
            )
        five = next((r for r in rows if int(r["edit_count"]) == 5), None)
        if five is not None:
            print(
                f"  {mode}: 5 edits acknowledged={five['acknowledged']} "
                f"d_ipSAE={five['delta_ipsae_min_mean']:+.4f} "
                f"z={five['z_delta_ipsae_min_mean']:+.2f} "
                f"mutations={five['mutations']}"
            )

    print(f"\n[caat] wrote {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
