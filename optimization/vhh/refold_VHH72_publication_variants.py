#!/usr/bin/env python3
"""Refold fixed VHH72 publication variants with Boltz2 template modes.

This diagnostic bypasses diffusion and sequence search. It refolds WT and the
published triple mutant S56M+L97W+T99V using:

  1. target template only
  2. target template + parent binder template

The scoring path reuses the main Mosaic VHH refold code, so ipSAE/iPTM,
interface metrics, and target-aligned RMSD are directly comparable to pipeline
outputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.boltzgen_vhh_guided import VHHDesignConfig, refold_pareto_with_boltz2
from mosaic.common import TOKENS


DEFAULT_PDB = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb")
DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_OUTPUT = Path("vhh/VHH72_publication_refold_templates")


def read_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    structure = gemmi.read_structure(str(pdb_path))
    seq = gemmi.one_letter_code([res.name for res in structure[0][chain_id]])
    if not seq:
        raise ValueError(f"No residues found for chain {chain_id!r} in {pdb_path}")
    bad = sorted(set(seq) - set(TOKENS))
    if bad:
        raise ValueError(f"Unsupported residue code(s) in binder sequence: {bad}")
    return seq


def read_cdr_indices(cdr_map: Path) -> list[int]:
    indices = []
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["group"] in {"CDR1", "CDR2", "CDR3"}:
                indices.append(int(row["seq_index"]))
    if not indices:
        raise ValueError(f"No CDR1/CDR2/CDR3 rows found in {cdr_map}")
    return indices


def read_known_variants(cdr_map: Path, requested: list[str]) -> list[dict]:
    by_variant = {}
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["group"] == "known_variant" and row["variant"]:
                by_variant[row["variant"].upper()] = row

    variants = []
    for variant in requested:
        key = variant.upper()
        if key not in by_variant:
            raise ValueError(f"Variant {variant!r} not found in {cdr_map}")
        row = by_variant[key]
        variants.append({
            "variant": key,
            "wt": key[0],
            "target": key[-1],
            "seq_index": int(row["seq_index"]),
            "anarci_label": row["anarci_label"],
            "pdb_auth_label": row["pdb_auth_label"],
        })
    return variants


def apply_variants(wt_sequence: str, variants: list[dict]) -> str:
    seq = list(wt_sequence)
    for variant in variants:
        idx0 = variant["seq_index"] - 1
        observed = seq[idx0]
        if observed != variant["wt"]:
            raise ValueError(
                f"{variant['variant']} expects WT {variant['wt']} at "
                f"{variant['anarci_label']} seq_index {variant['seq_index']}, "
                f"but sequence has {observed}"
            )
        seq[idx0] = variant["target"]
    return "".join(seq)


def tokenize(sequence: str) -> np.ndarray:
    return np.asarray([TOKENS.index(aa) for aa in sequence], dtype=np.int32)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rank_float(row: dict, key: str, default: float, *, reverse: bool = False) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not np.isfinite(value):
        value = default
    return -value if reverse else value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--target-chain", default="E")
    parser.add_argument("--variants", default="S56M,L97W,T99V")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--ipsae-pae-cutoff", type=float, default=12.0)
    parser.add_argument("--rmsd-threshold", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    wt_sequence = read_chain_sequence(args.pdb, args.binder_chain)
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = read_known_variants(args.cdr_map, requested)
    triple_label = "+".join(v["variant"] for v in variants)
    triple_sequence = apply_variants(wt_sequence, variants)
    cdr_indices = read_cdr_indices(args.cdr_map)

    candidates = {
        0: ("WT", wt_sequence),
        len(variants): (triple_label, triple_sequence),
    }
    binder_indices = np.arange(len(wt_sequence), dtype=np.int32)

    combined_rows = []
    combined_all_sample_rows = []
    for mode_name, binder_template in [
        ("target_template_only", False),
        ("target_plus_binder_template", True),
    ]:
        mode_dir = args.output_dir / mode_name
        pareto = {
            edit_count: (0.0, tokenize(seq))
            for edit_count, (_label, seq) in candidates.items()
        }
        cfg = VHHDesignConfig(
            complex_cif_path=args.pdb,
            binder_chain_id=args.binder_chain,
            target_chain_ids=[args.target_chain],
            cdr_residue_indices=cdr_indices,
            output_dir=mode_dir,
            edit_budget=len(variants),
            refold_binder_template=binder_template,
            refold_num_samples=args.num_samples,
            refold_batch_size=args.batch_size,
            refold_sampling_steps=args.sampling_steps,
            recycling_steps=args.recycling_steps,
            ipsae_pae_cutoff=args.ipsae_pae_cutoff,
            refold_rmsd_threshold=args.rmsd_threshold,
            seed=args.seed,
        )
        print(
            f"\n=== {mode_name} "
            f"(samples={args.num_samples}, steps={args.sampling_steps}) ===",
            flush=True,
        )
        refold_pareto_with_boltz2(pareto, cfg, binder_indices)
        for row in read_rows(mode_dir / "refold_ranked.csv"):
            edit_count = int(row["edit_count"])
            label, _seq = candidates[edit_count]
            combined_rows.append({
                "template_mode": mode_name,
                "binder_template": binder_template,
                "candidate_label": label,
                **row,
            })
        for row in read_rows(mode_dir / "refold_all_samples.csv"):
            edit_count = int(row["edit_count"])
            label, _seq = candidates[edit_count]
            combined_all_sample_rows.append({
                "template_mode": mode_name,
                "binder_template": binder_template,
                "candidate_label": label,
                **row,
            })

    summary_csv = args.output_dir / "publication_variant_refold_summary.csv"
    all_samples_csv = args.output_dir / "publication_variant_refold_all_samples.csv"
    write_csv(summary_csv, combined_rows)
    write_csv(all_samples_csv, combined_all_sample_rows)

    print("\nSummary:")
    for row in combined_rows:
        print(
            f"  {row['template_mode']} {row['candidate_label']}: "
            f"ipSAE={float(row['ipsae_min']):.4f} "
            f"ipTM={float(row['iptm']):.4f} "
            f"binder_RMSD={float(row['binder_ca_rmsd_target_aligned']):.3f} "
            f"CDR_RMSD={float(row['cdr_ca_rmsd_target_aligned']):.3f} "
            f"contacts={row.get('geom_interface_residue_contacts_refolded', '')} "
            f"hbonds={row.get('geom_hbonds_refolded', '')} "
            f"rmsd_pass={row['rmsd_pass']}"
        )
    if combined_all_sample_rows:
        print("\nBest per candidate across all samples:")
        groups = {}
        for row in combined_all_sample_rows:
            groups.setdefault((row["template_mode"], row["candidate_label"]), []).append(row)
        for (mode_name, label), sample_rows in groups.items():
            best_rmsd = min(
                sample_rows,
                key=lambda row: (
                    rank_float(row, "binder_ca_rmsd_target_aligned", float("inf")),
                    rank_float(row, "ipsae_min", -float("inf"), reverse=True),
                ),
            )
            best_ipsae = min(
                sample_rows,
                key=lambda row: (
                    rank_float(row, "ipsae_min", -float("inf"), reverse=True),
                    rank_float(row, "binder_ca_rmsd_target_aligned", float("inf")),
                ),
            )
            print(
                f"  {mode_name} {label}: "
                f"best_RMSD={float(best_rmsd['binder_ca_rmsd_target_aligned']):.3f} "
                f"(ipSAE={float(best_rmsd['ipsae_min']):.4f}, "
                f"sample={best_rmsd['sample_idx']}); "
                f"best_ipSAE={float(best_ipsae['ipsae_min']):.4f} "
                f"(RMSD={float(best_ipsae['binder_ca_rmsd_target_aligned']):.3f}, "
                f"sample={best_ipsae['sample_idx']})"
            )
    print(f"\nWrote summary CSV: {summary_csv}")
    print(f"Wrote all-samples CSV: {all_samples_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
