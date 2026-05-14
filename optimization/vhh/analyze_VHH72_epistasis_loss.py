#!/usr/bin/env python3
"""Fit a tiny MULTI-evolve-style epistasis scorer for VHH72.

This is a benchmark diagnostic. It uses the paper's VHH72 off-rate values
(`k_d`, not equilibrium `K_D`) for WT, singles, doubles, and the triple mutant.

The model is intentionally simple:

  benefit = log10(parent_off_rate / variant_off_rate)
  pairwise model = intercept + single effects + pair interaction effects

By default, the pairwise model is trained only on WT/singles/doubles and then
used to predict the triple, matching the MULTI-evolve premise that double
mutants can capture epistasis for higher-order proposals.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_OUTPUT = Path("optimization/vhh/VHH72_epistasis_loss_scores.csv")

# Lower-case k_d / off-rate from the VHH72 paper, units 10^-4 s^-1.
PAPER_OFF_RATES = {
    "WT": 91.0,
    "S56M": 31.0,
    "L97W": 18.0,
    "T99V": 7.2,
    "S56M+L97W": 4.4,
    "S56M+T99V": 4.1,
    "L97W+T99V": 3.5,
    "S56M+L97W+T99V": 2.8,
}


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def numeric(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def default_variants_from_map(cdr_map: Path) -> list[dict]:
    variants = []
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") == "known_variant" and row.get("variant"):
                variant = row["variant"].upper()
                variants.append({
                    "variant": variant,
                    "wt": variant[0],
                    "target": variant[-1],
                    "seq_index": int(row["seq_index"]),
                    "anarci_label": row["anarci_label"],
                    "pdb_auth_label": row["pdb_auth_label"],
                })
    if not variants:
        raise ValueError(f"No known_variant rows found in {cdr_map}")
    return variants


def combo_to_bits(combo: str, variants: list[dict]) -> np.ndarray:
    names = [] if combo == "WT" else combo.split("+")
    return np.asarray([v["variant"] in names for v in variants], dtype=float)


def bits_to_combo(bits: np.ndarray, variants: list[dict]) -> str:
    names = [v["variant"] for bit, v in zip(bits, variants) if bit > 0.5]
    return "+".join(names) if names else "WT"


def feature_names(variants: list[dict]) -> list[str]:
    names = ["intercept"]
    names.extend(v["variant"] for v in variants)
    for i, j in itertools.combinations(range(len(variants)), 2):
        names.append(f"{variants[i]['variant']}:{variants[j]['variant']}")
    return names


def featurize_bits(bits: np.ndarray) -> np.ndarray:
    feats = [1.0]
    feats.extend(bits.tolist())
    for i, j in itertools.combinations(range(len(bits)), 2):
        feats.append(float(bits[i] * bits[j]))
    return np.asarray(feats, dtype=float)


def observed_benefit(off_rate: float) -> float:
    return math.log10(PAPER_OFF_RATES["WT"] / off_rate)


def fit_pairwise_model(variants: list[dict], *, include_triple: bool = False) -> np.ndarray:
    rows = []
    targets = []
    for combo, off_rate in PAPER_OFF_RATES.items():
        bits = combo_to_bits(combo, variants)
        if not include_triple and int(bits.sum()) > 2:
            continue
        rows.append(featurize_bits(bits))
        targets.append(observed_benefit(off_rate))
    X = np.vstack(rows)
    y = np.asarray(targets, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_benefit(bits: np.ndarray, coef: np.ndarray) -> float:
    return float(featurize_bits(bits) @ coef)


def known_site_state(sequence: str, variants: list[dict]) -> tuple[np.ndarray, list[str]]:
    bits = []
    unknown = []
    for variant in variants:
        idx0 = variant["seq_index"] - 1
        aa = sequence[idx0] if 0 <= idx0 < len(sequence) else ""
        if aa == variant["target"]:
            bits.append(1.0)
        elif aa == variant["wt"]:
            bits.append(0.0)
        else:
            bits.append(0.0)
            unknown.append(f"{variant['variant']}@{variant['seq_index']}:{aa or '?'}")
    return np.asarray(bits, dtype=float), unknown


def publication_rows(variants: list[dict], coef_pairwise: np.ndarray, coef_saturated: np.ndarray) -> list[dict]:
    rows = []
    for combo, off_rate in PAPER_OFF_RATES.items():
        bits = combo_to_bits(combo, variants)
        pred_pair = predict_benefit(bits, coef_pairwise)
        pred_sat = predict_benefit(bits, coef_saturated)
        observed = observed_benefit(off_rate)
        rows.append(score_record(
            source="publication_grid",
            label=combo,
            combo=combo,
            bits=bits,
            unknown=[],
            variants=variants,
            pred_pairwise_benefit=pred_pair,
            pred_saturated_benefit=pred_sat,
            observed_off_rate=off_rate,
            observed_benefit_value=observed,
        ))
    return rows


def score_record(
    *,
    source: str,
    label: str,
    combo: str,
    bits: np.ndarray,
    unknown: list[str],
    variants: list[dict],
    pred_pairwise_benefit: float,
    pred_saturated_benefit: float,
    observed_off_rate: float | None = None,
    observed_benefit_value: float | None = None,
    extra: dict | None = None,
) -> dict:
    pred_off_rate = PAPER_OFF_RATES["WT"] / (10.0 ** pred_pairwise_benefit)
    row = {
        "source": source,
        "label": label,
        "known_combo": combo,
        "known_mutation_count": int(bits.sum()),
        "known_site_unknown_states": ";".join(unknown),
        "multi_evolve_pairwise_benefit_log10": pred_pairwise_benefit,
        "multi_evolve_pairwise_loss": -pred_pairwise_benefit,
        "multi_evolve_pairwise_pred_off_rate_x1e4_s": pred_off_rate,
        "multi_evolve_pairwise_pred_improvement_fold": 10.0 ** pred_pairwise_benefit,
        "saturated_fit_benefit_log10": pred_saturated_benefit,
    }
    for bit, variant in zip(bits, variants):
        row[f"has_{variant['variant']}"] = bool(bit > 0.5)
    if observed_off_rate is not None:
        row["observed_off_rate_x1e4_s"] = observed_off_rate
    if observed_benefit_value is not None:
        row["observed_benefit_log10"] = observed_benefit_value
        row["pairwise_residual_benefit_log10"] = (
            pred_pairwise_benefit - observed_benefit_value
        )
    if extra:
        row.update(extra)
    return row


def resolve_design_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    for name in ("mutation_recovery.csv", "combined_refold_ranked.csv", "refold_ranked.csv"):
        candidate = path / name
        if candidate.exists():
            return [candidate]
    return sorted(path.rglob("refold_ranked.csv"))


def sort_design_rows(rows: list[dict]) -> list[dict]:
    def key(row):
        rank = str(row.get("rank", "")).strip()
        return (
            not truthy(row.get("rmsd_pass")),
            rank == "",
            numeric(rank, 1e9),
            -numeric(row.get("ipsae_min"), -1e9),
            -numeric(row.get("iptm"), -1e9),
        )

    return sorted(rows, key=key)


def score_designs(
    input_path: Path,
    variants: list[dict],
    coef_pairwise: np.ndarray,
    coef_saturated: np.ndarray,
    max_designs: int,
) -> list[dict]:
    if max_designs <= 0:
        return []
    csv_paths = resolve_design_csvs(input_path)
    if not csv_paths:
        raise FileNotFoundError(f"No design CSVs found under {input_path}")

    raw = []
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                seq = row.get("sequence", "").strip()
                if seq:
                    raw.append({**row, "source_csv": str(csv_path)})

    rows = []
    seen = set()
    for row in sort_design_rows(raw):
        seq = row["sequence"]
        if seq in seen:
            continue
        seen.add(seq)
        bits, unknown = known_site_state(seq, variants)
        combo = bits_to_combo(bits, variants)
        pred_pair = predict_benefit(bits, coef_pairwise)
        pred_sat = predict_benefit(bits, coef_saturated)
        label = "design"
        if row.get("rank"):
            label += f"_rank_{row['rank']}"
        if row.get("edit_count"):
            label += f"_edit_{row['edit_count']}"
        if row.get("sample_idx"):
            label += f"_sample_{row['sample_idx']}"
        rows.append(score_record(
            source="design_csv",
            label=label,
            combo=combo,
            bits=bits,
            unknown=unknown,
            variants=variants,
            pred_pairwise_benefit=pred_pair,
            pred_saturated_benefit=pred_sat,
            extra={
                "sequence": seq,
                "source_csv": row.get("source_csv", ""),
                "rank": row.get("rank", ""),
                "edit_count": row.get("edit_count", ""),
                "sample_idx": row.get("sample_idx", ""),
                "ipsae_min": row.get("ipsae_min", ""),
                "iptm": row.get("iptm", ""),
                "binder_ca_rmsd_target_aligned": row.get("binder_ca_rmsd_target_aligned", ""),
                "rmsd_pass": row.get("rmsd_pass", ""),
            },
        ))
        if len(rows) >= max_designs:
            break
    return rows


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


def write_multievolve_training_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mutation", "property_value", "round"])
        writer.writeheader()
        for row in rows:
            if row["source"] != "publication_grid":
                continue
            writer.writerow({
                "mutation": "WT" if row["known_combo"] == "WT" else row["known_combo"].replace("+", "/"),
                "property_value": row["observed_benefit_log10"],
                "round": 0 if int(row["known_mutation_count"]) <= 1 else 1,
            })


def print_summary(rows: list[dict], variants: list[dict], coef_pairwise: np.ndarray) -> None:
    print("Pairwise epistasis model coefficients:")
    for name, value in zip(feature_names(variants), coef_pairwise):
        print(f"  {name}: {value:+.4f}")

    print("\nPublication grid:")
    for row in [r for r in rows if r["source"] == "publication_grid"]:
        print(
            f"  {row['known_combo']}: observed_off={float(row['observed_off_rate_x1e4_s']):.2f}, "
            f"observed_benefit={float(row['observed_benefit_log10']):+.3f}, "
            f"pairwise_pred_off={float(row['multi_evolve_pairwise_pred_off_rate_x1e4_s']):.2f}, "
            f"pairwise_loss={float(row['multi_evolve_pairwise_loss']):+.3f}, "
            f"residual={float(row['pairwise_residual_benefit_log10']):+.3f}"
        )

    design_rows = [r for r in rows if r["source"] == "design_csv"]
    if design_rows:
        print("\nTop design rows by pairwise epistasis loss:")
        for row in sorted(design_rows, key=lambda r: float(r["multi_evolve_pairwise_loss"]))[:10]:
            print(
                f"  {row['label']} combo={row['known_combo']} "
                f"loss={float(row['multi_evolve_pairwise_loss']):+.3f} "
                f"pred_off={float(row['multi_evolve_pairwise_pred_off_rate_x1e4_s']):.2f} "
                f"ipSAE={row.get('ipsae_min', '')} "
                f"RMSD={row.get('binder_ca_rmsd_target_aligned', '')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--design-input", type=Path, default=None)
    parser.add_argument("--max-designs", type=int, default=0)
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=None,
        help="Optional MULTI-evolve-format training CSV to write.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    variants = default_variants_from_map(args.cdr_map)
    coef_pairwise = fit_pairwise_model(variants, include_triple=False)
    coef_saturated = fit_pairwise_model(variants, include_triple=True)

    rows = publication_rows(variants, coef_pairwise, coef_saturated)
    if args.design_input is not None:
        rows.extend(score_designs(
            args.design_input,
            variants,
            coef_pairwise,
            coef_saturated,
            args.max_designs,
        ))

    write_csv(args.output_csv, rows)
    if args.training_csv is not None:
        write_multievolve_training_csv(args.training_csv, rows)
    print_summary(rows, variants, coef_pairwise)
    print(f"\nWrote epistasis scores: {args.output_csv}")
    if args.training_csv is not None:
        print(f"Wrote MULTI-evolve training CSV: {args.training_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
