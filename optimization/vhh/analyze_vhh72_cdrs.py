#!/usr/bin/env python
"""Map VHH72 Kabat CDRs to BoltzGen/Mosaic residue indices.

The VHH72 PDB uses Kabat-style author numbering with insertion codes
(for example H52A and H100A-H). BoltzGen YAML `res_index` fields are parsed as
1-based positions in chain order, so this helper runs ANARCI, verifies the
numbering, and emits the chain-order design ranges needed by Mosaic.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser


DEFAULT_ANARCI_PYTHON = Path("/home/yfeng17/anaconda3/envs/anarci/bin/python")
DEFAULT_ANARCI_SCRIPT = Path("/home/yfeng17/anaconda3/envs/anarci/bin/ANARCI")
DEFAULT_HMMER_DIR = Path("/home/yfeng17/anaconda3/envs/anarci/bin")

PAPER_CDRS = {
    "CDR1": ((30, ""), (31, "")),
    "CDR2": ((52, ""), (64, "")),
    "CDR3": ((96, ""), (100, "G")),
}

KABAT_FULL_CDRS = {
    "CDR1": ((31, ""), (35, "")),
    "CDR2": ((50, ""), (65, "")),
    "CDR3": ((95, ""), (102, "")),
}

KNOWN_VARIANTS = {
    "S56M": (56, "", "S", "M"),
    "L97W": (97, "", "L", "W"),
    "T99V": (99, "", "T", "V"),
}

DEFAULT_RBD_HOTSPOTS = [374, 377, 378, 379, 383, 384]


@dataclass(frozen=True)
class ResidueRecord:
    seq_index: int
    auth_num: int
    insertion: str
    resname: str
    aa: str

    @property
    def auth_label(self) -> str:
        return f"{self.auth_num}{self.insertion}"


@dataclass(frozen=True)
class NumberedResidue:
    chain_type: str
    number: int
    insertion: str
    aa: str
    record: ResidueRecord

    @property
    def anarci_label(self) -> str:
        return f"H{self.number}{self.insertion}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb",
        type=Path,
        default=Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb"),
    )
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--target-chain", default="E")
    parser.add_argument("--scheme", default="kabat")
    parser.add_argument("--anarci-python", type=Path, default=DEFAULT_ANARCI_PYTHON)
    parser.add_argument("--anarci-script", type=Path, default=DEFAULT_ANARCI_SCRIPT)
    parser.add_argument("--hmmer-dir", type=Path, default=DEFAULT_HMMER_DIR)
    parser.add_argument(
        "--target-hotspots",
        default="auto",
        help=(
            "'auto' to use relaxed CDR-target heavy-atom contacts, or a comma "
            "list of author residue numbers on the target chain."
        ),
    )
    parser.add_argument(
        "--hotspot-cutoff",
        type=float,
        default=4.5,
        help="Heavy-atom contact cutoff in Angstroms for --target-hotspots auto.",
    )
    parser.add_argument(
        "--out-yaml",
        type=Path,
        default=Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD.yaml"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv"),
    )
    parser.add_argument(
        "--out-hotspots-csv",
        type=Path,
        default=Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_hotspots.csv"),
    )
    return parser.parse_args()


def chain_residues(pdb_path: Path, chain_id: str) -> list[ResidueRecord]:
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(iter(structure))
    if chain_id not in [chain.id for chain in model]:
        raise ValueError(f"Chain {chain_id!r} not found in {pdb_path}")

    records = []
    seen = set()
    for residue in model[chain_id]:
        hetflag, auth_num, insertion = residue.id
        if hetflag != " " or not residue.has_id("CA"):
            continue
        key = (auth_num, insertion)
        if key in seen:
            continue
        seen.add(key)
        aa = protein_letters_3to1.get(residue.resname, "X")
        records.append(
            ResidueRecord(
                seq_index=len(records) + 1,
                auth_num=int(auth_num),
                insertion=insertion.strip(),
                resname=residue.resname,
                aa=aa,
            )
        )
    return records


def run_anarci(
    sequence: str,
    *,
    scheme: str,
    anarci_python: Path,
    anarci_script: Path,
    hmmer_dir: Path,
) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{hmmer_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        [
            str(anarci_python),
            str(anarci_script),
            "-i",
            sequence,
            "-s",
            scheme,
            "-r",
            "H",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return result.stdout


def parse_anarci_output(text: str) -> list[tuple[str, int, str, str]]:
    numbered = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "H":
            continue
        if len(parts) == 3:
            chain_type, num, aa = parts
            insertion = ""
        elif len(parts) == 4:
            chain_type, num, insertion, aa = parts
        else:
            continue
        if not re.fullmatch(r"\d+", num):
            continue
        numbered.append((chain_type, int(num), insertion.strip(), aa))
    return numbered


def map_anarci_to_structure(
    numbered: list[tuple[str, int, str, str]],
    records: list[ResidueRecord],
) -> list[NumberedResidue]:
    if len(numbered) > len(records):
        raise ValueError(
            f"ANARCI returned {len(numbered)} residues but chain has {len(records)}"
        )

    mapped = []
    for idx, (chain_type, number, insertion, aa) in enumerate(numbered):
        record = records[idx]
        if aa != record.aa:
            raise ValueError(
                "ANARCI/structure sequence mismatch at chain position "
                f"{record.seq_index}: ANARCI {aa}, structure {record.aa}"
            )
        mapped.append(NumberedResidue(chain_type, number, insertion, aa, record))
    return mapped


def select_numbered_range(
    mapped: list[NumberedResidue],
    start: tuple[int, str],
    end: tuple[int, str],
) -> list[NumberedResidue]:
    labels = [(r.number, r.insertion) for r in mapped]
    try:
        start_idx = labels.index(start)
        end_idx = labels.index(end)
    except ValueError as exc:
        raise ValueError(f"ANARCI range endpoint missing: {start}..{end}") from exc
    if end_idx < start_idx:
        raise ValueError(f"ANARCI range is reversed: {start}..{end}")
    return mapped[start_idx : end_idx + 1]


def compact_ranges(indices: list[int]) -> str:
    if not indices:
        return ""
    indices = sorted(set(indices))
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}" if start == prev else f"{start}..{prev}")
        start = prev = idx
    ranges.append(f"{start}" if start == prev else f"{start}..{prev}")
    return ",".join(ranges)


def hotspot_positions(records: list[ResidueRecord], auth_numbers: list[int]) -> list[int]:
    positions = []
    missing = []
    for auth_num in auth_numbers:
        matches = [r for r in records if r.auth_num == auth_num]
        if not matches:
            missing.append(auth_num)
            continue
        positions.extend(r.seq_index for r in matches)
    if missing:
        raise ValueError(f"Target hotspot author residues missing: {missing}")
    return positions


def _heavy_atoms(residue):
    atoms = []
    for atom in residue:
        element = (atom.element or atom.name[0]).upper()
        if element == "H" or atom.name.strip().upper().startswith("H"):
            continue
        atoms.append(atom)
    return atoms


def _cdr_for_seq_index(seq_index: int, selected: dict[str, list[NumberedResidue]]):
    for cdr, residues in selected.items():
        indices = {residue.record.seq_index for residue in residues}
        if seq_index in indices:
            return cdr
    return None


def contact_hotspots(
    pdb_path: Path,
    *,
    binder_chain: str,
    target_chain: str,
    selected: dict[str, list[NumberedResidue]],
    cutoff: float,
) -> list[dict]:
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(iter(structure))
    binder_residues = {
        record.seq_index: residue
        for record, residue in _residue_objects_by_seq_index(model[binder_chain]).items()
    }
    target_residues = _residue_objects_by_seq_index(model[target_chain])

    selected_indices = {
        residue.record.seq_index
        for residues in selected.values()
        for residue in residues
    }

    rows = []
    for target_record, target_residue in target_residues.items():
        min_distance = float("inf")
        atom_pairs = 0
        cdr_counts: dict[str, int] = {}
        for binder_index in selected_indices:
            binder_residue = binder_residues[binder_index]
            cdr = _cdr_for_seq_index(binder_index, selected)
            for binder_atom in _heavy_atoms(binder_residue):
                for target_atom in _heavy_atoms(target_residue):
                    distance = float(
                        np.linalg.norm(binder_atom.coord - target_atom.coord)
                    )
                    min_distance = min(min_distance, distance)
                    if distance <= cutoff:
                        atom_pairs += 1
                        if cdr is not None:
                            cdr_counts[cdr] = cdr_counts.get(cdr, 0) + 1
        if atom_pairs:
            rows.append(
                {
                    "target_seq_index": target_record.seq_index,
                    "target_auth_label": target_record.auth_label,
                    "target_aa": target_record.aa,
                    "target_resname": target_record.resname,
                    "min_heavy_distance": min_distance,
                    "atom_pairs_le_cutoff": atom_pairs,
                    "cdr_contacts": ";".join(
                        f"{cdr}:{count}" for cdr, count in sorted(cdr_counts.items())
                    ),
                }
            )
    rows.sort(key=lambda row: (row["min_heavy_distance"], row["target_seq_index"]))
    return rows


def _residue_objects_by_seq_index(chain) -> dict[ResidueRecord, object]:
    out = {}
    seen = set()
    for residue in chain:
        hetflag, auth_num, insertion = residue.id
        if hetflag != " " or not residue.has_id("CA"):
            continue
        key = (auth_num, insertion)
        if key in seen:
            continue
        seen.add(key)
        record = ResidueRecord(
            seq_index=len(out) + 1,
            auth_num=int(auth_num),
            insertion=insertion.strip(),
            resname=residue.resname,
            aa=protein_letters_3to1.get(residue.resname, "X"),
        )
        out[record] = residue
    return out


def write_hotspot_csv(path: Path, rows: list[dict], cutoff: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_seq_index",
                "target_auth_label",
                "target_aa",
                "target_resname",
                "min_heavy_distance",
                "atom_pairs_le_cutoff",
                "cdr_contacts",
                "cutoff",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "cutoff": cutoff})


def write_mapping_csv(
    path: Path,
    selected: dict[str, list[NumberedResidue]],
    known_hits: dict[str, NumberedResidue],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "anarci_label",
                "seq_index",
                "pdb_auth_label",
                "aa",
                "variant",
            ],
        )
        writer.writeheader()
        for group, residues in selected.items():
            for residue in residues:
                writer.writerow(
                    {
                        "group": group,
                        "anarci_label": residue.anarci_label,
                        "seq_index": residue.record.seq_index,
                        "pdb_auth_label": residue.record.auth_label,
                        "aa": residue.aa,
                        "variant": "",
                    }
                )
        for variant, residue in known_hits.items():
            writer.writerow(
                {
                    "group": "known_variant",
                    "anarci_label": residue.anarci_label,
                    "seq_index": residue.record.seq_index,
                    "pdb_auth_label": residue.record.auth_label,
                    "aa": residue.aa,
                    "variant": variant,
                }
            )


def write_boltzgen_yaml(
    path: Path,
    *,
    pdb_path: Path,
    binder_chain: str,
    target_chain: str,
    design_res_index: str,
    hotspot_res_index: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_pdb = os.path.relpath(pdb_path.resolve(), path.parent.resolve())
    text = f"""# VHH72 WT / SARS-CoV-2 WT RBD benchmark.
# CDRs are ANARCI Kabat mapped to BoltzGen chain-order res_index values.
# Paper benchmark design set: H30-H31, H52-H64, H96-H100G.
# Known optimized substitutions to recover/rank: S56M, L97W, T99V.

entities:
  - file:
      path: {rel_pdb}

      include:
        - chain:
            id: {target_chain}  # SARS-CoV-2 RBD target
        - chain:
            id: {binder_chain}  # VHH72 binder

      structure_groups:
        - group:
            visibility: 1
            id: {target_chain}
        - group:
            visibility: 1
            id: {binder_chain}
        - group:
            visibility: 0
            id: {binder_chain}
            res_index: {design_res_index}

      design:
        - chain:
            id: {binder_chain}
            res_index: {design_res_index}

      binding_types:
        - chain:
            id: {target_chain}
            binding: {hotspot_res_index}
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    binder_records = chain_residues(args.pdb, args.binder_chain)
    target_records = chain_residues(args.pdb, args.target_chain)
    binder_sequence = "".join(r.aa for r in binder_records)

    anarci_output = run_anarci(
        binder_sequence,
        scheme=args.scheme,
        anarci_python=args.anarci_python,
        anarci_script=args.anarci_script,
        hmmer_dir=args.hmmer_dir,
    )
    mapped = map_anarci_to_structure(parse_anarci_output(anarci_output), binder_records)

    paper_selected = {
        cdr: select_numbered_range(mapped, start, end)
        for cdr, (start, end) in PAPER_CDRS.items()
    }
    full_kabat_selected = {
        f"full_{cdr}": select_numbered_range(mapped, start, end)
        for cdr, (start, end) in KABAT_FULL_CDRS.items()
    }

    by_label = {(r.number, r.insertion): r for r in mapped}
    known_hits = {}
    for variant, (num, ins, wt, _mut) in KNOWN_VARIANTS.items():
        residue = by_label[(num, ins)]
        if residue.aa != wt:
            raise ValueError(
                f"{variant} expected WT {wt} at H{num}{ins}, found {residue.aa}"
            )
        known_hits[variant] = residue

    design_indices = [
        residue.record.seq_index
        for residues in paper_selected.values()
        for residue in residues
    ]
    design_res_index = compact_ranges(design_indices)

    auto_hotspot_rows = contact_hotspots(
        args.pdb,
        binder_chain=args.binder_chain,
        target_chain=args.target_chain,
        selected=paper_selected,
        cutoff=args.hotspot_cutoff,
    )
    write_hotspot_csv(args.out_hotspots_csv, auto_hotspot_rows, args.hotspot_cutoff)

    if args.target_hotspots.strip().lower() == "auto":
        hotspot_positions_chain_order = [
            int(row["target_seq_index"]) for row in auto_hotspot_rows
        ]
        hotspot_auth_numbers = [
            row["target_auth_label"] for row in auto_hotspot_rows
        ]
    else:
        hotspot_auth_numbers_int = [
            int(piece)
            for piece in args.target_hotspots.replace(" ", "").split(",")
            if piece
        ]
        hotspot_positions_chain_order = hotspot_positions(
            target_records, hotspot_auth_numbers_int
        )
        hotspot_auth_numbers = [str(i) for i in hotspot_auth_numbers_int]

    hotspot_res_index = compact_ranges(hotspot_positions_chain_order)

    write_mapping_csv(
        args.out_csv,
        selected={**paper_selected, **full_kabat_selected},
        known_hits=known_hits,
    )
    write_boltzgen_yaml(
        args.out_yaml,
        pdb_path=args.pdb,
        binder_chain=args.binder_chain,
        target_chain=args.target_chain,
        design_res_index=design_res_index,
        hotspot_res_index=hotspot_res_index,
    )

    print(f"PDB: {args.pdb}")
    print(f"Binder chain: {args.binder_chain}, length {len(binder_records)}")
    print(f"Target chain: {args.target_chain}, length {len(target_records)}")
    print(f"ANARCI scheme: {args.scheme}, numbered residues {len(mapped)}")
    print("\nPaper benchmark CDR design set:")
    for cdr, residues in paper_selected.items():
        seq_range = compact_ranges([r.record.seq_index for r in residues])
        auth_range = f"{residues[0].record.auth_label}..{residues[-1].record.auth_label}"
        seq = "".join(r.aa for r in residues)
        print(f"  {cdr}: H{residues[0].number}{residues[0].insertion}"
              f"..H{residues[-1].number}{residues[-1].insertion}"
              f" | PDB A {auth_range} | BoltzGen {seq_range} | {seq}")
    print("\nKnown optimized residues:")
    for variant, residue in known_hits.items():
        print(
            f"  {variant}: {args.binder_chain}{residue.record.auth_label} "
            f"(chain position {residue.record.seq_index}, ANARCI {residue.anarci_label})"
        )
    print(f"\nBoltzGen design res_index: {design_res_index}")
    print(
        f"Target hotspot mode: {args.target_hotspots} "
        f"(heavy cutoff {args.hotspot_cutoff:.2f} A)"
    )
    print(f"Target hotspot author residues: {hotspot_auth_numbers}")
    print(f"BoltzGen target binding res_index: {hotspot_res_index}")
    print("\nRelaxed-structure target hotspots:")
    for row in auto_hotspot_rows:
        print(
            f"  E{row['target_auth_label']} {row['target_aa']} "
            f"| chain position {row['target_seq_index']} "
            f"| min_heavy={row['min_heavy_distance']:.2f} A "
            f"| atom_pairs={row['atom_pairs_le_cutoff']} "
            f"| {row['cdr_contacts']}"
        )
    print(f"Wrote YAML: {args.out_yaml}")
    print(f"Wrote CDR map: {args.out_csv}")
    print(f"Wrote hotspot map: {args.out_hotspots_csv}")


if __name__ == "__main__":
    main()
