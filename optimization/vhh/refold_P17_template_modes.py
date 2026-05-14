#!/usr/bin/env python3
"""Boltz2 template-mode benchmark for P17 against Alpha/WT/JN.1 RBD.

This diagnostic asks whether Boltz2 confidence metrics separate a known binder
pose from the JN.1 non-binding/escape setting.  It refolds the fixed P17
sequence against the same target sequence under four template regimes:

  1. no_template
  2. target_template
  3. target_plus_framework_binder_template
     Full-length binder template sequence, but CDR residue atoms removed.
  4. full_template

For each prediction it reports ipSAE/ipTM/iPAE, target-aligned binder RMSD to
the input/reference complex, CDR RMSD, and lightweight interface geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from examples.boltzgen_vhh_guided import (  # noqa: E402
    _passes_max_threshold,
    _refold_rank_key,
    interface_geometry_metrics,
    target_aligned_rmsd_metrics,
    write_structure_cif,
)
from mosaic.losses.boltz2 import boltz2_forward_from_trunk, boltz2_trunk  # noqa: E402
from mosaic.losses.structure_prediction import (  # noqa: E402
    BinderTargetIPSAE,
    IPTMLoss,
    IPSAE_min,
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

DEFAULT_OUTPUT = Path("vhh/P17_boltz2_template_mode_benchmark")
DEFAULT_TARGETS = {
    "alpha": {
        "path": Path(
            "/home/yfeng17/SBSAb/dataset/Affinity_train/structures/"
            "phase2c_abag_ANDD_ANDD_8GZ5_B_X_A_WT.cif"
        ),
        "binder_chain": "B",
        "target_chain": "A",
        "expected_binding": "binder",
        "description": "8GZ5 Alpha RBD-P17 crystal/cropped positive reference",
    },
    "wt": {
        "path": Path("/home/yfeng17/boltzgen/Round2_Design/Nanobody/pdb/P17_WT.pdb"),
        "binder_chain": "B",
        "target_chain": "D",
        "expected_binding": "binder",
        "description": "P17 with WT-like RBD local reference",
    },
    "jn1": {
        "path": Path("/home/yfeng17/boltzgen/Round2_Design/Nanobody/pdb/P17_JN1.pdb"),
        "binder_chain": "B",
        "target_chain": "T",
        "expected_binding": "negative_or_escape",
        "description": "P17 with JN.1 RBD local negative/escape reference",
    },
}
DEFAULT_TEMPLATE_MODES = [
    "no_template",
    "target_template",
    "target_plus_framework_binder_template",
    "full_template",
]
P17_CDR_INDICES = [
    *range(26, 34),
    *range(51, 59),
    *range(97, 110),
]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    path: Path
    binder_chain: str
    target_chain: str
    expected_binding: str
    description: str


def read_chain_sequence(structure: gemmi.Structure, chain_id: str) -> str:
    seq = gemmi.one_letter_code([res.name for res in structure[0][chain_id]])
    seq = seq.replace("X", "")
    if not seq:
        raise ValueError(f"No protein residues found for chain {chain_id!r}")
    return seq


def protein_chain(structure: gemmi.Structure, chain_id: str) -> gemmi.Chain:
    chain = structure[0][chain_id]
    new_chain = gemmi.Chain(chain.name)
    for residue in chain:
        if residue.name in AA3:
            new_chain.add_residue(residue.clone())
    return new_chain


def masked_framework_template(chain: gemmi.Chain, cdr_indices: Iterable[int]) -> gemmi.Chain:
    cdr_set = set(int(i) for i in cdr_indices)
    new_chain = gemmi.Chain(chain.name)
    for idx, residue in enumerate(chain, start=1):
        if residue.name not in AA3:
            continue
        if idx in cdr_set:
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


def chain_with_name(chain: gemmi.Chain, name: str) -> gemmi.Chain:
    clone = chain.clone()
    clone.name = name
    return clone


def parse_targets(raw: str) -> list[str]:
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [name for name in names if name not in DEFAULT_TARGETS]
    if bad:
        raise ValueError(f"Unknown target(s): {bad}; choose from {sorted(DEFAULT_TARGETS)}")
    return names


def parse_modes(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [name for name in names if name not in DEFAULT_TEMPLATE_MODES]
    if bad:
        raise ValueError(
            f"Unknown template mode(s): {bad}; choose from {DEFAULT_TEMPLATE_MODES}"
        )
    return names


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


def build_refold_chains(
    binder_sequence: str,
    target_sequence: str,
    binder_template_chain: gemmi.Chain | None,
    target_template_chain: gemmi.Chain | None,
) -> list[TargetChain]:
    return [
        TargetChain(
            binder_sequence,
            use_msa=False,
            template_chain=binder_template_chain.clone()
            if binder_template_chain is not None
            else None,
        ),
        TargetChain(
            target_sequence,
            use_msa=False,
            template_chain=target_template_chain.clone()
            if target_template_chain is not None
            else None,
        ),
    ]


def summarize_reference(spec: TargetSpec, structure: gemmi.Structure) -> dict:
    metrics = interface_geometry_metrics(
        structure,
        binder_chain_id=spec.binder_chain,
        target_chain_ids=[spec.target_chain],
    )
    return {
        "target": spec.name,
        "input_path": str(spec.path),
        "binder_chain": spec.binder_chain,
        "target_chain": spec.target_chain,
        "expected_binding": spec.expected_binding,
        "description": spec.description,
        "binder_sequence": read_chain_sequence(structure, spec.binder_chain),
        "target_sequence": read_chain_sequence(structure, spec.target_chain),
        **{f"input_{k}": v for k, v in metrics.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="alpha,wt,jn1")
    parser.add_argument("--template-modes", default=",".join(DEFAULT_TEMPLATE_MODES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=12.0)
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=2.5,
        help=(
            "BoltzGen-style target-aligned binder CA RMSD filter in A; "
            "<=0 disables the filter"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-structures", type=int, choices=[0, 1], default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_names = parse_targets(args.targets)
    template_modes = parse_modes(args.template_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    print("[P17] loading Boltz2", flush=True)
    boltz2 = Boltz2()
    print("[P17] loaded Boltz2", flush=True)

    iptm_loss = IPTMLoss()
    bt_ipsae_loss = BinderTargetIPSAE(pae_cutoff=args.ipsae_pae_cutoff)
    tb_ipsae_loss = TargetBinderIPSAE(pae_cutoff=args.ipsae_pae_cutoff)
    ipsae_min_loss = IPSAE_min(pae_cutoff=args.ipsae_pae_cutoff)

    def score_batch(features, initial_emb, trunk_state, sample_keys, binder_len: int):
        binder_sequence_placeholder = jnp.zeros((binder_len, 20))

        def score_one(sample_key):
            out = boltz2_forward_from_trunk(
                boltz2.model,
                features,
                initial_emb,
                trunk_state,
                num_sampling_steps=args.sampling_steps,
                deterministic=True,
                key=sample_key,
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
            _, ipsae_aux = ipsae_min_loss(
                sequence=binder_sequence_placeholder,
                output=out,
                key=fold_in(sample_key, "ipsae_min"),
            )
            bt_pae = out.pae[:binder_len, binder_len:]
            tb_pae = out.pae[binder_len:, :binder_len]
            return {
                "structure_coordinates": out.structure_coordinates,
                "iptm": iptm_aux["iptm"],
                "bt_ipsae": bt_aux["bt_ipsae"],
                "tb_ipsae": tb_aux["tb_ipsae"],
                "ipsae_min": ipsae_aux["ipsae_min"],
                "ipae_min": jnp.minimum(jnp.min(bt_pae), jnp.min(tb_pae)),
                "bt_pae_mean": jnp.mean(bt_pae),
                "tb_pae_mean": jnp.mean(tb_pae),
            }

        return jax.vmap(score_one)(sample_keys)

    score_batch = eqx.filter_jit(score_batch)

    reference_rows = []
    sample_rows = []
    summary_rows = []

    for target_name in target_names:
        raw = DEFAULT_TARGETS[target_name]
        spec = TargetSpec(
            name=target_name,
            path=raw["path"],
            binder_chain=raw["binder_chain"],
            target_chain=raw["target_chain"],
            expected_binding=raw["expected_binding"],
            description=raw["description"],
        )
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)

        reference_structure = gemmi.read_structure(str(spec.path))
        reference_structure.setup_entities()
        binder_sequence = read_chain_sequence(reference_structure, spec.binder_chain)
        target_sequence = read_chain_sequence(reference_structure, spec.target_chain)
        binder_template_full = protein_chain(reference_structure, spec.binder_chain)
        target_template_full = protein_chain(reference_structure, spec.target_chain)
        binder_template_framework = masked_framework_template(
            binder_template_full, P17_CDR_INDICES
        )
        reference_rows.append(summarize_reference(spec, reference_structure))

        for mode_idx, mode in enumerate(template_modes):
            target_template = None
            binder_template = None
            if mode in {
                "target_template",
                "target_plus_framework_binder_template",
                "full_template",
            }:
                target_template = target_template_full
            if mode == "target_plus_framework_binder_template":
                binder_template = binder_template_framework
            elif mode == "full_template":
                binder_template = binder_template_full

            print(
                f"[P17] target={target_name} mode={mode} "
                f"samples={args.num_samples} steps={args.sampling_steps}",
                flush=True,
            )
            refold_chains = build_refold_chains(
                binder_sequence,
                target_sequence,
                binder_template,
                target_template,
            )
            features, writer = boltz2.target_only_features(refold_chains)
            target_offset = target_names.index(target_name) * 100000
            key = jax.random.key(args.seed + target_offset + 10000 * mode_idx)

            print(f"[P17] target={target_name} mode={mode} trunk", flush=True)
            initial_emb, trunk_state = boltz2_trunk(
                boltz2.model,
                features,
                recycling_steps=args.recycling_steps,
                deterministic=True,
                key=fold_in(key, "trunk"),
            )
            print(f"[P17] target={target_name} mode={mode} sampling", flush=True)

            mode_rows = []
            for chunk_start in range(0, args.num_samples, args.batch_size):
                chunk_size = min(args.batch_size, args.num_samples - chunk_start)
                sample_keys = jax.random.split(
                    fold_in(key, f"sample_batch_{chunk_start}"),
                    chunk_size,
                )
                batch_scores = score_batch(
                    features,
                    initial_emb,
                    trunk_state,
                    sample_keys,
                    len(binder_sequence),
                )
                jax.tree.map(lambda x: x.block_until_ready(), batch_scores)

                for offset in range(chunk_size):
                    sample_idx = chunk_start + offset
                    structure = writer(batch_scores["structure_coordinates"][offset])
                    structure.setup_entities()
                    if args.write_structures:
                        structure_path = (
                            args.output_dir
                            / "structures"
                            / target_name
                            / mode
                            / f"sample_{sample_idx}.cif"
                        )
                        write_structure_cif(structure, structure_path)
                    else:
                        structure_path = Path("")

                    rmsd = target_aligned_rmsd_metrics(
                        reference_structure,
                        structure,
                        original_binder_chain_id=spec.binder_chain,
                        original_target_chain_ids=[spec.target_chain],
                        refolded_binder_chain_id="A",
                        refolded_target_chain_ids=["B"],
                        cdr_residue_indices=P17_CDR_INDICES,
                    )
                    geom = interface_geometry_metrics(
                        structure,
                        binder_chain_id="A",
                        target_chain_ids=["B"],
                    )
                    row = {
                        "target": target_name,
                        "expected_binding": spec.expected_binding,
                        "template_mode": mode,
                        "sample_idx": sample_idx,
                        "sampling_steps": args.sampling_steps,
                        "recycling_steps": args.recycling_steps,
                        "ipsae_pae_cutoff": args.ipsae_pae_cutoff,
                        "iptm": float(batch_scores["iptm"][offset]),
                        "bt_ipsae": float(batch_scores["bt_ipsae"][offset]),
                        "tb_ipsae": float(batch_scores["tb_ipsae"][offset]),
                        "ipsae_min": float(batch_scores["ipsae_min"][offset]),
                        "ipae_min": float(batch_scores["ipae_min"][offset]),
                        "bt_pae_mean": float(batch_scores["bt_pae_mean"][offset]),
                        "tb_pae_mean": float(batch_scores["tb_pae_mean"][offset]),
                        "structure_path": str(structure_path),
                        **rmsd,
                        **geom,
                    }
                    row["rmsd_filter_threshold"] = args.rmsd_threshold
                    row["rmsd_pass"] = _passes_max_threshold(
                        row["binder_ca_rmsd_target_aligned"],
                        args.rmsd_threshold,
                    )
                    sample_rows.append(row)
                    mode_rows.append(row)

            best_by_rank = min(mode_rows, key=_refold_rank_key)
            best_by_rmsd = min(
                mode_rows,
                key=lambda r: (
                    r["binder_ca_rmsd_target_aligned"],
                    -r["ipsae_min"],
                    r["sample_idx"],
                ),
            )
            best_by_ipsae = max(
                mode_rows,
                key=lambda r: (
                    r["ipsae_min"],
                    -r["binder_ca_rmsd_target_aligned"],
                    -r["ipae_min"],
                ),
            )
            summary_rows.append(
                {
                    "target": target_name,
                    "expected_binding": spec.expected_binding,
                    "template_mode": mode,
                    "n_samples": len(mode_rows),
                    "rmsd_filter_threshold": args.rmsd_threshold,
                    "n_rmsd_pass": sum(bool(r["rmsd_pass"]) for r in mode_rows),
                    "best_ranked_sample_idx": best_by_rank["sample_idx"],
                    "best_ranked_rmsd_pass": best_by_rank["rmsd_pass"],
                    "best_ranked_ipsae": best_by_rank["ipsae_min"],
                    "best_ranked_iptm": best_by_rank["iptm"],
                    "best_ranked_ipae_min": best_by_rank["ipae_min"],
                    "best_ranked_binder_ca_rmsd": best_by_rank[
                        "binder_ca_rmsd_target_aligned"
                    ],
                    "best_ranked_cdr_ca_rmsd": best_by_rank.get(
                        "cdr_ca_rmsd_target_aligned", float("nan")
                    ),
                    "best_ranked_structure_path": best_by_rank["structure_path"],
                    "best_rmsd_sample_idx": best_by_rmsd["sample_idx"],
                    "best_rmsd_binder_ca_rmsd": best_by_rmsd[
                        "binder_ca_rmsd_target_aligned"
                    ],
                    "best_rmsd_cdr_ca_rmsd": best_by_rmsd.get(
                        "cdr_ca_rmsd_target_aligned", float("nan")
                    ),
                    "best_rmsd_ipsae_min": best_by_rmsd["ipsae_min"],
                    "best_rmsd_iptm": best_by_rmsd["iptm"],
                    "best_rmsd_ipae_min": best_by_rmsd["ipae_min"],
                    "best_ipsae_sample_idx": best_by_ipsae["sample_idx"],
                    "best_ipsae": best_by_ipsae["ipsae_min"],
                    "best_ipsae_binder_ca_rmsd": best_by_ipsae[
                        "binder_ca_rmsd_target_aligned"
                    ],
                    "best_ipsae_cdr_ca_rmsd": best_by_ipsae.get(
                        "cdr_ca_rmsd_target_aligned", float("nan")
                    ),
                    "best_ipsae_iptm": best_by_ipsae["iptm"],
                    "best_ipsae_ipae_min": best_by_ipsae["ipae_min"],
                    "best_ipsae_structure_path": best_by_ipsae["structure_path"],
                    "best_rmsd_structure_path": best_by_rmsd["structure_path"],
                }
            )

    write_csv(args.output_dir / "reference_geometry.csv", reference_rows)
    write_csv(args.output_dir / "all_samples.csv", sample_rows)
    write_csv(args.output_dir / "summary_by_target_mode.csv", summary_rows)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "targets": target_names,
                "template_modes": template_modes,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "sampling_steps": args.sampling_steps,
                "recycling_steps": args.recycling_steps,
                "ipsae_pae_cutoff": args.ipsae_pae_cutoff,
                "rmsd_threshold": args.rmsd_threshold,
                "p17_cdr_indices": P17_CDR_INDICES,
            },
            indent=2,
        )
        + "\n"
    )

    print("\n[P17] summary")
    for row in summary_rows:
        print(
            f"{row['target']:>5} {row['template_mode']:<38} "
            f"ranked_ipSAE={row['best_ranked_ipsae']:.4f} "
            f"ranked_RMSD={row['best_ranked_binder_ca_rmsd']:.2f}A "
            f"pass={row['best_ranked_rmsd_pass']} "
            f"n_pass={row['n_rmsd_pass']}/{row['n_samples']} "
            f"best_ipSAE={row['best_ipsae']:.4f} "
            f"RMSD@best_ipSAE={row['best_ipsae_binder_ca_rmsd']:.2f}A "
            f"iPAE@best_ipSAE={row['best_ipsae_ipae_min']:.2f} "
            f"best_RMSD={row['best_rmsd_binder_ca_rmsd']:.2f}A "
            f"ipSAE@bestRMSD={row['best_rmsd_ipsae_min']:.4f}"
        )
    print(f"\n[P17] wrote {args.output_dir / 'summary_by_target_mode.csv'}")
    print(f"[P17] wrote {args.output_dir / 'all_samples.csv'}")
    print(f"[P17] wrote {args.output_dir / 'reference_geometry.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
