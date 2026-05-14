#!/usr/bin/env python3
"""Find a one-antibody AlphaSeq SARS-CoV-2 RBD single-mutant benchmark.

This script audits the NaturalAntibody/ASD HuggingFace mirror and extracts the
largest obvious antibody-side DMS set for the AlphaSeq SARS-CoV-2 RBD target.
It uses the most replicated VHH sequence as the parent/background, then reports
all exact one-substitution antibody variants against the same RBD antigen.

AlphaSeq values in this mirror are treated as log-scale affinity measurements
where lower values are better. The good/bad labels are therefore based on
delta(processed_measurement) relative to the parent median.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def normalize_target(value: Any) -> str:
    return str(value or "").lower().replace("-", "_")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def hamming_single_mutation(parent: str, sequence: str) -> tuple[int, str, str] | None:
    if len(parent) != len(sequence) or parent == sequence:
        return None
    diff_pos = -1
    diff_count = 0
    for i, (a, b) in enumerate(zip(parent, sequence, strict=True)):
        if a != b:
            diff_count += 1
            if diff_count > 1:
                return None
            diff_pos = i
    if diff_count != 1:
        return None
    return diff_pos + 1, parent[diff_pos], sequence[diff_pos]


def label_delta(delta: float | None, good_delta: float, bad_delta: float) -> str:
    if delta is None:
        return "missing"
    if delta <= good_delta:
        return "good"
    if delta >= bad_delta:
        return "bad"
    return "neutral"


def load_filtered_rows(repo: str, split: str, dataset_name: str, target_name: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    stream = load_dataset(repo, split=split, streaming=True)
    for row in stream:
        if row.get("dataset") != dataset_name:
            continue
        if normalize_target(row.get("target_name")) != normalize_target(target_name):
            continue
        rows.append(dict(row))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="introvoyz041/agab-db")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name", default="alphaseq")
    parser.add_argument("--target-name", default="sars_cov2_rbd")
    parser.add_argument("--output-dir", default="vhh")
    parser.add_argument(
        "--parent-sequence",
        default=None,
        help="Override the parent/background sequence. Defaults to the most replicated VHH sequence.",
    )
    parser.add_argument(
        "--good-delta",
        type=float,
        default=-0.3,
        help="Delta processed_measurement at or below this value is labeled good.",
    )
    parser.add_argument(
        "--bad-delta",
        type=float,
        default=0.3,
        help="Delta processed_measurement at or above this value is labeled bad.",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_filtered_rows(args.repo, args.split, args.dataset_name, args.target_name)
    if not rows:
        raise RuntimeError(
            f"No rows found for dataset={args.dataset_name!r}, target={args.target_name!r}"
        )

    antigen_counts = Counter(row.get("antigen_sequence") or "" for row in rows)
    heavy_counts = Counter(row.get("heavy_sequence") or "" for row in rows)
    parent = args.parent_sequence or heavy_counts.most_common(1)[0][0]
    if not parent:
        raise RuntimeError("Could not determine a parent heavy sequence")

    per_sequence: dict[str, dict[str, Any]] = {}
    processed_values: dict[str, list[float]] = defaultdict(list)
    affinity_values: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cdrs: dict[str, tuple[Any, Any, Any]] = {}
    for row in rows:
        sequence = row.get("heavy_sequence") or ""
        if not sequence:
            continue
        processed = as_float(row.get("processed_measurement"))
        affinity = as_float(row.get("affinity"))
        if processed is not None:
            processed_values[sequence].append(processed)
        if affinity is not None:
            affinity_values[sequence].append(affinity)
        source_counts[sequence][str(row.get("source_url") or "")] += 1
        cdrs.setdefault(
            sequence,
            (row.get("heavy_cdr1"), row.get("heavy_cdr2"), row.get("heavy_cdr3")),
        )

    for sequence, count in heavy_counts.items():
        per_sequence[sequence] = {
            "count": count,
            "processed_median": median(processed_values[sequence])
            if processed_values[sequence]
            else None,
            "affinity_median": median(affinity_values[sequence])
            if affinity_values[sequence]
            else None,
            "source_url": source_counts[sequence].most_common(1)[0][0]
            if source_counts[sequence]
            else "",
        }

    parent_processed = per_sequence[parent]["processed_median"]
    parent_affinity = per_sequence[parent]["affinity_median"]
    if parent_processed is None:
        raise RuntimeError("Parent sequence has no processed_measurement values")

    single_rows: list[dict[str, Any]] = []
    for sequence, stats in per_sequence.items():
        mutation = hamming_single_mutation(parent, sequence)
        if mutation is None:
            continue
        pos, wt, mut = mutation
        processed = stats["processed_median"]
        affinity = stats["affinity_median"]
        delta = None if processed is None else processed - parent_processed
        affinity_delta = None if affinity is None or parent_affinity is None else affinity - parent_affinity
        single_rows.append(
            {
                "mutation": f"{wt}{pos}{mut}",
                "position_1based": pos,
                "wt": wt,
                "mut": mut,
                "sequence": sequence,
                "n_rows": stats["count"],
                "processed_median": processed,
                "parent_processed_median": parent_processed,
                "delta_processed": delta,
                "affinity_median": affinity,
                "parent_affinity_median": parent_affinity,
                "delta_affinity": affinity_delta,
                "label": label_delta(delta, args.good_delta, args.bad_delta),
                "source_url": stats["source_url"],
            }
        )

    single_rows.sort(
        key=lambda row: (
            row["delta_processed"] is None,
            row["delta_processed"] if row["delta_processed"] is not None else 999.0,
            row["position_1based"],
            row["mut"],
        )
    )

    by_site: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in single_rows:
        by_site[int(row["position_1based"])].append(row)

    site_rows: list[dict[str, Any]] = []
    for pos, site in sorted(by_site.items()):
        deltas = [row["delta_processed"] for row in site if row["delta_processed"] is not None]
        best = min(site, key=lambda row: row["delta_processed"] if row["delta_processed"] is not None else 999.0)
        worst = max(site, key=lambda row: row["delta_processed"] if row["delta_processed"] is not None else -999.0)
        site_rows.append(
            {
                "position_1based": pos,
                "wt": parent[pos - 1],
                "n_mutants": len(site),
                "n_good": sum(row["label"] == "good" for row in site),
                "n_bad": sum(row["label"] == "bad" for row in site),
                "n_neutral": sum(row["label"] == "neutral" for row in site),
                "best_mutation": best["mutation"],
                "best_delta_processed": best["delta_processed"],
                "worst_mutation": worst["mutation"],
                "worst_delta_processed": worst["delta_processed"],
                "median_delta_processed": median(deltas) if deltas else None,
                "min_delta_processed": min(deltas) if deltas else None,
                "max_delta_processed": max(deltas) if deltas else None,
            }
        )

    label_counts = Counter(row["label"] for row in single_rows)
    parent_cdr1, parent_cdr2, parent_cdr3 = cdrs.get(parent, (None, None, None))
    summary = {
        "repo": args.repo,
        "split": args.split,
        "dataset_name": args.dataset_name,
        "target_name": args.target_name,
        "n_filtered_rows": len(rows),
        "n_unique_antigen_sequences": len(antigen_counts),
        "top_antigen_sequence_count": antigen_counts.most_common(1)[0][1],
        "top_antigen_sequence_length": len(antigen_counts.most_common(1)[0][0]),
        "n_unique_heavy_sequences": len(heavy_counts),
        "parent_sequence": parent,
        "parent_length": len(parent),
        "parent_row_count": heavy_counts[parent],
        "parent_processed_median": parent_processed,
        "parent_affinity_median": parent_affinity,
        "parent_heavy_cdr1": parent_cdr1,
        "parent_heavy_cdr2": parent_cdr2,
        "parent_heavy_cdr3": parent_cdr3,
        "n_single_substitution_sequences": len(single_rows),
        "n_single_substitution_sites": len(site_rows),
        "label_counts": dict(label_counts),
        "good_delta_threshold": args.good_delta,
        "bad_delta_threshold": args.bad_delta,
    }

    prefix = outdir / "alphaseq_sars_cov2_rbd_top_vhh"
    (prefix.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(
        prefix.with_suffix(".single_mutants.csv"),
        single_rows,
        [
            "mutation",
            "position_1based",
            "wt",
            "mut",
            "n_rows",
            "processed_median",
            "parent_processed_median",
            "delta_processed",
            "affinity_median",
            "parent_affinity_median",
            "delta_affinity",
            "label",
            "source_url",
            "sequence",
        ],
    )
    write_csv(
        prefix.with_suffix(".site_summary.csv"),
        site_rows,
        [
            "position_1based",
            "wt",
            "n_mutants",
            "n_good",
            "n_bad",
            "n_neutral",
            "best_mutation",
            "best_delta_processed",
            "worst_mutation",
            "worst_delta_processed",
            "median_delta_processed",
            "min_delta_processed",
            "max_delta_processed",
        ],
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {prefix.with_suffix('.single_mutants.csv')}")
    print(f"wrote {prefix.with_suffix('.site_summary.csv')}")


if __name__ == "__main__":
    main()
