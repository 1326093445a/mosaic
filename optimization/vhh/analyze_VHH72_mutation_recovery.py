#!/usr/bin/env python3
"""Check whether VHH72 benchmark designs recover paper mutation hits.

The paper reports mutations in Kabat/PDB author numbering. Mosaic output
sequences are plain chain-order sequences, so this script maps the paper labels
through ``VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv`` before checking designs.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_VARIANTS = ["S56M", "L97W", "T99V"]
VARIANT_RE = re.compile(r"^([A-Z])(\d+[A-Za-z]?)([A-Z])$")


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def numeric(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_cdr_map(path: Path) -> dict[str, dict]:
    by_auth = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            auth = row["pdb_auth_label"].upper()
            existing = by_auth.get(auth)
            if existing is None or row["group"] == "known_variant":
                by_auth[auth] = row
    return by_auth


def parse_variant(raw: str, cdr_by_auth: dict[str, dict]) -> dict:
    raw = raw.strip()
    match = VARIANT_RE.match(raw)
    if not match:
        raise ValueError(f"Cannot parse variant {raw!r}; expected form S56M")
    wt, auth_label, target = match.groups()
    auth_label = auth_label.upper()
    if auth_label not in cdr_by_auth:
        raise ValueError(
            f"Variant {raw!r} maps to PDB/Kabat label {auth_label}, "
            f"but that label is absent from the CDR map."
        )
    mapped = cdr_by_auth[auth_label]
    observed_wt = mapped["aa"]
    if observed_wt != wt:
        raise ValueError(
            f"Variant {raw!r} expects WT {wt} at {auth_label}, "
            f"but CDR map has {observed_wt}."
        )
    return {
        "variant": raw,
        "wt": wt,
        "target": target,
        "pdb_auth_label": auth_label,
        "anarci_label": mapped["anarci_label"],
        "seq_index": int(mapped["seq_index"]),
        "cdr_group": mapped["group"],
    }


def default_variants_from_map(cdr_map: Path) -> list[str]:
    variants = []
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["group"] == "known_variant" and row["variant"]:
                variants.append(row["variant"])
    return variants or DEFAULT_VARIANTS


def resolve_input_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    for name in ("combined_refold_ranked.csv", "refold_ranked.csv"):
        candidate = input_path / name
        if candidate.exists():
            return [candidate]
    found = sorted(input_path.rglob("refold_ranked.csv"))
    if not found:
        raise FileNotFoundError(
            f"No refold ranking CSV found under {input_path}. Expected "
            "combined_refold_ranked.csv, refold_ranked.csv, or seed*/refold_ranked.csv."
        )
    return found


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name("mutation_recovery.csv")
    return input_path / "mutation_recovery.csv"


def read_rows(csv_paths: list[Path]) -> list[dict]:
    rows = []
    for path in csv_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row.setdefault("source_csv", str(path))
                if len(csv_paths) > 1:
                    row.setdefault("run_dir", str(path.parent))
                rows.append(row)
    return rows


def sort_rows(rows: list[dict]) -> list[dict]:
    def key(row):
        rank = str(row.get("rank", "")).strip()
        has_rank = rank != ""
        return (
            not truthy(row.get("rmsd_pass")),
            not has_rank,
            numeric(rank, 1e9),
            -numeric(row.get("ipsae_min"), -1e9),
            -numeric(row.get("iptm"), -1e9),
        )

    return sorted(rows, key=key)


def annotate_rows(rows: list[dict], variants: list[dict]) -> list[dict]:
    annotated = []
    for row in rows:
        seq = row.get("sequence", "").strip()
        observed = []
        missing = []
        site_states = []
        out = dict(row)

        for variant in variants:
            idx0 = variant["seq_index"] - 1
            aa = seq[idx0] if 0 <= idx0 < len(seq) else ""
            hit = aa == variant["target"]
            out[f"has_{variant['variant']}"] = hit
            out[f"aa_at_{variant['pdb_auth_label']}"] = aa
            site_states.append(
                f"{variant['anarci_label']}/{variant['pdb_auth_label']}:"
                f"{variant['wt']}->{aa or '?'}"
            )
            if hit:
                observed.append(variant["variant"])
            else:
                missing.append(variant["variant"])

        out["intended_mutation_count"] = len(observed)
        out["intended_mutations_observed"] = ";".join(observed)
        out["intended_mutations_missing"] = ";".join(missing)
        out["intended_site_states"] = ";".join(site_states)
        annotated.append(out)
    return annotated


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], variants: list[dict], top_n: int) -> None:
    passing = [row for row in rows if truthy(row.get("rmsd_pass"))]
    ranked = [row for row in rows if str(row.get("rank", "")).strip()]
    counts = Counter(int(row["intended_mutation_count"]) for row in rows)
    passing_counts = Counter(int(row["intended_mutation_count"]) for row in passing)

    print("Verified paper-hit mapping:")
    for variant in variants:
        print(
            f"  {variant['variant']}: {variant['anarci_label']} / "
            f"PDB {variant['pdb_auth_label']} / Mosaic seq_index "
            f"{variant['seq_index']}"
        )

    print("")
    print(f"Rows analyzed: {len(rows)}")
    print(f"Passing RMSD rows: {len(passing)}")
    print(f"Ranked rows: {len(ranked)}")
    print(
        "Intended mutation count histogram: "
        + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))
    )
    if passing:
        print(
            "Passing-only histogram: "
            + ", ".join(f"{k}:{passing_counts[k]}" for k in sorted(passing_counts))
        )

    print("")
    print(f"Top {min(top_n, len(rows))} rows by RMSD-pass/rank/ipSAE:")
    for row in sort_rows(rows)[:top_n]:
        label = row.get("rank") or "unranked"
        print(
            f"  rank={label} edit={row.get('edit_count', '')} "
            f"ipsae={row.get('ipsae_min', '')} rmsd_pass={row.get('rmsd_pass', '')} "
            f"hits={row['intended_mutation_count']} "
            f"observed={row['intended_mutations_observed'] or '-'} "
            f"states={row['intended_site_states']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Run output directory or ranking CSV. Directories may contain "
            "combined_refold_ranked.csv, refold_ranked.csv, or seed*/refold_ranked.csv."
        ),
    )
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated paper/Kabat mutations to check. Defaults to known_variant rows.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cdr_by_auth = load_cdr_map(args.cdr_map)
    raw_variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else default_variants_from_map(args.cdr_map)
    )
    variants = [parse_variant(raw, cdr_by_auth) for raw in raw_variants]
    csv_paths = resolve_input_csvs(args.input)
    rows = annotate_rows(read_rows(csv_paths), variants)
    output_csv = args.output_csv or default_output_path(args.input)
    write_csv(output_csv, rows)
    print_summary(rows, variants, args.top_n)
    print("")
    print(f"Wrote annotated CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
