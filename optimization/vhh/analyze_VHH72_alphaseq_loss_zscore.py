#!/usr/bin/env python3
"""Compare VHH72 AlphaSeq affinities with Mosaic sequence-prior losses.

This analysis uses the public AlphaSeq VHH72 optimization datasets and the
precomputed VHH72 CDR single-substitution loss table.  It maps each AlphaSeq
variant back to the VHH72 WT sequence, sums covered ESM2/AbLang2/Mosaic
single-substitution loss shifts, and reports WT-relative affinity z-scores,
correlations, AUCs, Mann-Whitney tests, and top-low-loss enrichment.

AlphaSeq reports affinity as log10 Kd in nM, so lower values are stronger
binders.  Negative delta_affinity_vs_wt therefore means improved binding.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import hf_hub_download
from scipy import stats
from sklearn.metrics import roc_auc_score


DEFAULT_ASSAYS = ("YM_0549", "YM_1068")
DEFAULT_REPO = "aalphabio/open-alphaseq"
DEFAULT_TARGET_REGEX = "SARS-CoV2_RBD_\\(6LZG\\)"
DEFAULT_LOSS_CSV = Path("vhh/VHH72_cdr_substitution_loss.csv")
DEFAULT_OUTPUT_PREFIX = Path("vhh/VHH72_alphaseq_loss_zscore")


@dataclass(frozen=True)
class Mutation:
    position: int
    wt: str
    mut: str

    @property
    def label(self) -> str:
        return f"{self.wt}{self.position}{self.mut}"


def zscore(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    mean = arr.mean(skipna=True)
    std = arr.std(skipna=True, ddof=0)
    if not math.isfinite(float(std)) or float(std) == 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (arr - mean) / std


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float | None:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return None
    value = subset[x].corr(subset[y], method=method)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def auc_good_vs_bad(df: pd.DataFrame, score_col: str) -> float | None:
    subset = df[df["affinity_label"].isin(["good", "bad"])].copy()
    subset = subset[[score_col, "affinity_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset["affinity_label"].nunique() < 2:
        return None
    y = (subset["affinity_label"] == "good").astype(int).to_numpy()
    # Lower loss is predicted better, so negate the loss for an AUC score.
    score = -subset[score_col].astype(float).to_numpy()
    return float(roc_auc_score(y, score))


def auc_good_vs_bad_high_score(df: pd.DataFrame, score_col: str) -> float | None:
    subset = df[df["affinity_label"].isin(["good", "bad"])].copy()
    subset = subset[[score_col, "affinity_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset["affinity_label"].nunique() < 2:
        return None
    y = (subset["affinity_label"] == "good").astype(int).to_numpy()
    score = subset[score_col].astype(float).to_numpy()
    return float(roc_auc_score(y, score))


def mannwhitney_good_bad(df: pd.DataFrame, score_col: str) -> dict[str, float | None]:
    good = df.loc[df["affinity_label"] == "good", score_col].replace([np.inf, -np.inf], np.nan).dropna()
    bad = df.loc[df["affinity_label"] == "bad", score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(good) < 2 or len(bad) < 2:
        return {"u": None, "pvalue": None, "good_median": None, "bad_median": None}
    result = stats.mannwhitneyu(good, bad, alternative="two-sided")
    return {
        "u": float(result.statistic),
        "pvalue": float(result.pvalue),
        "good_median": float(good.median()),
        "bad_median": float(bad.median()),
    }


def load_alphaseq_assay(repo_id: str, assay: str) -> pl.DataFrame:
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename="data.parquet",
        repo_type="dataset",
        subfolder=f"data/{assay}",
    )
    return pl.read_parquet(parquet_path).with_columns(pl.lit(assay).alias("assay"))


def load_alphaseq(
    repo_id: str,
    assays: Iterable[str],
    target_regex: str,
    include_all_targets: bool,
) -> pl.DataFrame:
    frames = []
    for assay in assays:
        frame = load_alphaseq_assay(repo_id, assay)
        if not include_all_targets:
            frame = frame.filter(pl.col("matalpha_description").str.contains(target_regex))
        frames.append(frame)
    if not frames:
        raise ValueError("No assays requested")
    return pl.concat(frames, how="vertical_relaxed")


def infer_wt_sequence(frame: pl.DataFrame, explicit_wt: str | None) -> str:
    if explicit_wt:
        return explicit_wt.strip()
    wt = (
        frame.filter(pl.col("mata_description").str.to_lowercase().str.contains("wt"))
        .select("mata_sequence")
        .drop_nulls()
        .unique()
    )
    if wt.height != 1:
        raise ValueError(
            f"Expected exactly one WT sequence from AlphaSeq descriptions, observed {wt.height}"
        )
    return str(wt.item())


def load_loss_lookup(loss_csv: Path) -> tuple[dict[tuple[int, str, str], dict[str, object]], set[int]]:
    df = pd.read_csv(loss_csv)
    scan = df[df["row_type"].eq("single_substitution")].copy()
    required = {
        "seq_index",
        "wt",
        "mut",
        "cdr",
        "anarci_label",
        "esm2_delta_nll",
        "ablang2_delta_nll",
        "mosaic_sequence_delta_loss",
    }
    missing = sorted(required - set(scan.columns))
    if missing:
        raise ValueError(f"{loss_csv} is missing columns: {missing}")
    lookup: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in scan.to_dict("records"):
        position = int(row["seq_index"])
        lookup[(position, str(row["wt"]), str(row["mut"]))] = row
    return lookup, {int(v) for v in scan["seq_index"].dropna().unique()}


def compare_to_wt(sequence: str, wt_sequence: str) -> tuple[list[Mutation], int]:
    shared = min(len(sequence), len(wt_sequence))
    mutations = [
        Mutation(i + 1, wt_sequence[i], sequence[i])
        for i in range(shared)
        if sequence[i] != wt_sequence[i]
    ]
    length_delta = len(sequence) - len(wt_sequence)
    return mutations, length_delta


def score_variant(
    sequence: str,
    wt_sequence: str,
    lookup: dict[tuple[int, str, str], dict[str, object]],
    cdr_positions: set[int],
) -> dict[str, object]:
    mutations, length_delta = compare_to_wt(sequence, wt_sequence)
    covered = []
    missing = []
    cdr_mutations = 0
    esm2 = 0.0
    ablang2 = 0.0
    mosaic = 0.0
    cdrs: set[str] = set()
    anarci_labels: list[str] = []

    for mut in mutations:
        if mut.position in cdr_positions:
            cdr_mutations += 1
        entry = lookup.get((mut.position, mut.wt, mut.mut))
        if entry is None:
            missing.append(mut.label)
            continue
        covered.append(mut.label)
        cdrs.add(str(entry["cdr"]))
        anarci_labels.append(str(entry["anarci_label"]))
        esm2 += float(entry["esm2_delta_nll"])
        ablang2 += float(entry["ablang2_delta_nll"])
        mosaic += float(entry["mosaic_sequence_delta_loss"])

    missing_count = len(missing) + abs(length_delta)
    mutation_count = len(mutations) + abs(length_delta)
    covered_count = len(covered)
    denom = max(covered_count, 1)
    return {
        "sequence_length": len(sequence),
        "length_delta_vs_wt": length_delta,
        "mutation_count": mutation_count,
        "substitution_count": len(mutations),
        "cdr_mutation_count": cdr_mutations,
        "covered_mutation_count": covered_count,
        "missing_mutation_count": missing_count,
        "loss_coverage_fraction": covered_count / mutation_count if mutation_count else 1.0,
        "fully_covered_by_loss_table": missing_count == 0,
        "mutation_list": ";".join(mut.label for mut in mutations),
        "covered_mutation_list": ";".join(covered),
        "missing_mutation_list": ";".join(missing),
        "covered_cdrs": ";".join(sorted(cdrs)),
        "covered_anarci_labels": ";".join(anarci_labels),
        "esm2_delta_nll_sum": esm2 if covered_count else np.nan,
        "ablang2_delta_nll_sum": ablang2 if covered_count else np.nan,
        "mosaic_sequence_delta_loss_sum": mosaic if covered_count else np.nan,
        "esm2_delta_nll_mean": esm2 / denom if covered_count else np.nan,
        "ablang2_delta_nll_mean": ablang2 / denom if covered_count else np.nan,
        "mosaic_sequence_delta_loss_mean": mosaic / denom if covered_count else np.nan,
    }


def add_variant_scores(
    df: pd.DataFrame,
    wt_sequence: str,
    lookup: dict[tuple[int, str, str], dict[str, object]],
    cdr_positions: set[int],
) -> pd.DataFrame:
    records = [
        score_variant(str(seq), wt_sequence, lookup, cdr_positions)
        for seq in df["mata_sequence"].astype(str).tolist()
    ]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def add_affinity_columns(df: pd.DataFrame, good_threshold_log10: float) -> pd.DataFrame:
    df = df.copy()
    wt_medians = (
        df[df["is_wt_sequence"]]
        .groupby(["assay", "matalpha_description"], dropna=False)["alphaseq_affinity"]
        .median()
        .rename("wt_median_alphaseq_affinity")
        .reset_index()
    )
    df = df.merge(wt_medians, on=["assay", "matalpha_description"], how="left")
    df["delta_affinity_vs_wt"] = df["alphaseq_affinity"] - df["wt_median_alphaseq_affinity"]
    df["fold_kd_vs_wt"] = np.power(10.0, df["delta_affinity_vs_wt"].astype(float))

    df["affinity_label"] = "unlabeled"
    df.loc[df["delta_affinity_vs_wt"] <= -good_threshold_log10, "affinity_label"] = "good"
    df.loc[df["delta_affinity_vs_wt"].abs() < good_threshold_log10, "affinity_label"] = "neutral"
    df.loc[df["delta_affinity_vs_wt"] >= good_threshold_log10, "affinity_label"] = "bad"
    df.loc[df["delta_affinity_vs_wt"].isna(), "affinity_label"] = "missing_wt_or_affinity"

    group_cols = ["assay", "matalpha_description"]
    df["affinity_z_by_target"] = df.groupby(group_cols, dropna=False)["alphaseq_affinity"].transform(zscore)
    df["delta_affinity_z_by_target"] = df.groupby(group_cols, dropna=False)["delta_affinity_vs_wt"].transform(zscore)
    df["normalized_affinity_z_by_target"] = df.groupby(group_cols, dropna=False)["normalized_affinity"].transform(zscore)

    for col in [
        "esm2_delta_nll_sum",
        "ablang2_delta_nll_sum",
        "mosaic_sequence_delta_loss_sum",
        "esm2_delta_nll_mean",
        "ablang2_delta_nll_mean",
        "mosaic_sequence_delta_loss_mean",
    ]:
        df[f"{col}_z_by_target"] = df.groupby(group_cols, dropna=False)[col].transform(zscore)
        df[f"{col}_z_by_mutcount"] = df.groupby(
            group_cols + ["mutation_count"], dropna=False
        )[col].transform(zscore)
    return df


def topk_enrichment(df: pd.DataFrame, score_col: str, fractions: tuple[float, ...]) -> list[dict[str, object]]:
    measured = df[df["affinity_label"].isin(["good", "neutral", "bad"])].copy()
    measured = measured[[score_col, "affinity_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if measured.empty:
        return []
    measured = measured.sort_values(score_col, ascending=True)
    total = len(measured)
    total_good = int((measured["affinity_label"] == "good").sum())
    rows: list[dict[str, object]] = []
    for frac in fractions:
        k = max(1, int(round(total * frac)))
        top = measured.head(k)
        good = int((top["affinity_label"] == "good").sum())
        bad = int((top["affinity_label"] == "bad").sum())
        neutral = int((top["affinity_label"] == "neutral").sum())
        pvalue = None
        if 0 < total_good < total and good > 0:
            pvalue = float(stats.hypergeom.sf(good - 1, total, total_good, k))
        rows.append({
            "top_fraction": frac,
            "top_n": k,
            "good": good,
            "bad": bad,
            "neutral": neutral,
            "good_fraction": good / k,
            "bad_fraction": bad / k,
            "background_good_fraction": total_good / total,
            "good_enrichment": (good / k) / (total_good / total) if total_good else np.nan,
            "hypergeom_good_pvalue": pvalue,
        })
    return rows


def summarize_group(group: pd.DataFrame, group_name: tuple[str, str] | tuple[str]) -> dict[str, object]:
    assay = group_name[0]
    target = group_name[1] if len(group_name) > 1 else "ALL"
    scored = group[
        group["fully_covered_by_loss_table"]
        & group["alphaseq_affinity"].notna()
        & group["mosaic_sequence_delta_loss_sum"].notna()
        & (group["mutation_count"] > 0)
    ].copy()
    row: dict[str, object] = {
        "assay": assay,
        "target": target,
        "n_rows": int(len(group)),
        "n_with_affinity": int(group["alphaseq_affinity"].notna().sum()),
        "n_wt_rows": int(group["is_wt_sequence"].sum()),
        "n_fully_loss_covered": int(group["fully_covered_by_loss_table"].sum()),
        "n_scored_for_stats": int(len(scored)),
        "wt_median_alphaseq_affinity": (
            float(group["wt_median_alphaseq_affinity"].dropna().iloc[0])
            if group["wt_median_alphaseq_affinity"].notna().any()
            else None
        ),
        "good_threshold_log10_kd": (
            float(scored.attrs.get("good_threshold_log10", np.nan))
            if scored.attrs
            else None
        ),
        "good_count": int((scored["affinity_label"] == "good").sum()),
        "neutral_count": int((scored["affinity_label"] == "neutral").sum()),
        "bad_count": int((scored["affinity_label"] == "bad").sum()),
    }
    for score_col in [
        "esm2_delta_nll_sum",
        "ablang2_delta_nll_sum",
        "mosaic_sequence_delta_loss_sum",
        "esm2_delta_nll_mean",
        "ablang2_delta_nll_mean",
        "mosaic_sequence_delta_loss_mean",
    ]:
        row[f"{score_col}_spearman_vs_delta_affinity"] = safe_corr(
            scored, score_col, "delta_affinity_vs_wt", "spearman"
        )
        row[f"{score_col}_pearson_vs_delta_affinity"] = safe_corr(
            scored, score_col, "delta_affinity_vs_wt", "pearson"
        )
    row["mosaic_sum_auc_good_vs_bad_low_loss_is_good"] = auc_good_vs_bad(
        scored, "mosaic_sequence_delta_loss_sum"
    )
    row["mosaic_sum_auc_good_vs_bad_high_loss_is_good"] = auc_good_vs_bad_high_score(
        scored, "mosaic_sequence_delta_loss_sum"
    )
    mw = mannwhitney_good_bad(scored, "mosaic_sequence_delta_loss_sum")
    for key, value in mw.items():
        row[f"mosaic_sum_mannwhitney_{key}"] = value

    enrichments = topk_enrichment(scored, "mosaic_sequence_delta_loss_sum", (0.01, 0.05, 0.10))
    for enrichment in enrichments:
        prefix = f"top_{int(round(float(enrichment['top_fraction']) * 100))}pct_low_loss"
        for key, value in enrichment.items():
            if key == "top_fraction":
                continue
            row[f"{prefix}_{key}"] = value
    return row


def summarize(df: pd.DataFrame, good_threshold_log10: float) -> tuple[pd.DataFrame, dict[str, object]]:
    df.attrs["good_threshold_log10"] = good_threshold_log10
    rows = []
    for name, group in df.groupby(["assay", "matalpha_description"], dropna=False):
        group.attrs["good_threshold_log10"] = good_threshold_log10
        rows.append(summarize_group(group, name))
    for assay, group in df.groupby("assay", dropna=False):
        group.attrs["good_threshold_log10"] = good_threshold_log10
        rows.append(summarize_group(group, (assay,)))
    stats_df = pd.DataFrame(rows)

    meta = {
        "n_rows": int(len(df)),
        "assays": sorted(df["assay"].dropna().unique().tolist()),
        "targets": sorted(df["matalpha_description"].dropna().unique().tolist()),
        "good_bad_definition": {
            "affinity_unit": "log10 Kd nM",
            "lower_alphaseq_affinity": "stronger binding",
            "good": f"delta_affinity_vs_wt <= -{good_threshold_log10}",
            "bad": f"delta_affinity_vs_wt >= {good_threshold_log10}",
            "neutral": f"|delta_affinity_vs_wt| < {good_threshold_log10}",
        },
        "coverage": {
            "fully_covered_rows": int(df["fully_covered_by_loss_table"].sum()),
            "rows_with_any_loss": int(df["mosaic_sequence_delta_loss_sum"].notna().sum()),
            "median_loss_coverage_fraction": float(df["loss_coverage_fraction"].median()),
        },
        "label_counts": {
            str(k): int(v) for k, v in df["affinity_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    return stats_df, meta


def plot_loss_vs_affinity(df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = df[
        df["fully_covered_by_loss_table"]
        & df["delta_affinity_vs_wt"].notna()
        & df["mosaic_sequence_delta_loss_sum"].notna()
        & (df["mutation_count"] > 0)
    ].copy()
    if plot_df.empty:
        return
    colors = {"good": "#238b45", "neutral": "#7a869a", "bad": "#cb181d"}
    markers = {"YM_0549": "o", "YM_1068": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    for ax, score_col, title in [
        (axes[0], "mosaic_sequence_delta_loss_sum", "Summed CDR substitution loss"),
        (axes[1], "mosaic_sequence_delta_loss_mean", "Mean CDR substitution loss"),
    ]:
        for (assay, label), sub in plot_df.groupby(["assay", "affinity_label"]):
            ax.scatter(
                sub[score_col],
                sub["delta_affinity_vs_wt"],
                s=18,
                alpha=0.45,
                linewidth=0,
                c=colors.get(label, "#333333"),
                marker=markers.get(assay, "o"),
                label=f"{assay} {label}",
            )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.axhline(-0.3, color="#238b45", lw=0.8, ls="--")
        ax.axhline(0.3, color="#cb181d", lw=0.8, ls="--")
        ax.axvline(0.0, color="#333333", lw=0.8)
        ax.set_xlabel(score_col)
        ax.set_ylabel("AlphaSeq delta log10 Kd vs WT")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("VHH72 AlphaSeq affinity versus Mosaic sequence-prior loss", fontweight="bold")
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_loss_vs_affinity.png"), dpi=220)
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_loss_vs_affinity.pdf"))
    plt.close(fig)


def plot_loss_by_label(df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = df[
        df["fully_covered_by_loss_table"]
        & df["affinity_label"].isin(["good", "neutral", "bad"])
        & df["mosaic_sequence_delta_loss_sum"].notna()
        & (df["mutation_count"] > 0)
    ].copy()
    if plot_df.empty:
        return
    order = ["good", "neutral", "bad"]
    colors = ["#238b45", "#7a869a", "#cb181d"]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    data = [plot_df.loc[plot_df["affinity_label"] == label, "mosaic_sequence_delta_loss_sum"] for label in order]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.82)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#333333")
        parts[key].set_linewidth(1.0)
    for i, (label, values) in enumerate(zip(order, data), start=1):
        x = np.full(len(values), i, dtype=float)
        jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) <= 200 else np.random.default_rng(0).uniform(-0.16, 0.16, len(values))
        ax.scatter(x + jitter, values, s=7, alpha=0.18, color=colors[i - 1], linewidth=0)
        ax.text(i, ax.get_ylim()[1], f"n={len(values)}", ha="center", va="top", fontsize=9)
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Mosaic sequence delta loss sum")
    ax.set_title("Loss distribution by AlphaSeq WT-relative class", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_loss_by_affinity_label.png"), dpi=220)
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_loss_by_affinity_label.pdf"))
    plt.close(fig)


def plot_single_mutation_heatmap(df: pd.DataFrame, output_prefix: Path) -> None:
    single = df[
        df["fully_covered_by_loss_table"]
        & (df["mutation_count"] == 1)
        & df["delta_affinity_vs_wt"].notna()
        & df["mosaic_sequence_delta_loss_sum"].notna()
    ].copy()
    if single.empty:
        return
    rows = []
    for _, row in single.iterrows():
        label = str(row["covered_mutation_list"])
        if not label:
            continue
        rows.append({
            "mutation": label,
            "assay": row["assay"],
            "target": row["matalpha_description"],
            "delta_affinity_vs_wt": row["delta_affinity_vs_wt"],
            "mosaic_sequence_delta_loss_sum": row["mosaic_sequence_delta_loss_sum"],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return
    reduced = (
        table.groupby("mutation", dropna=False)
        .agg(
            delta_affinity_vs_wt=("delta_affinity_vs_wt", "median"),
            mosaic_sequence_delta_loss_sum=("mosaic_sequence_delta_loss_sum", "median"),
            n=("mutation", "size"),
        )
        .reset_index()
    )
    reduced = reduced.sort_values("delta_affinity_vs_wt", ascending=True).head(80)

    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    x = np.arange(len(reduced))
    ax.bar(
        x - 0.18,
        reduced["delta_affinity_vs_wt"],
        width=0.36,
        color="#4a90a4",
        label="median delta log10 Kd vs WT",
    )
    ax2 = ax.twinx()
    ax2.bar(
        x + 0.18,
        reduced["mosaic_sequence_delta_loss_sum"],
        width=0.36,
        color="#d95f02",
        alpha=0.78,
        label="median Mosaic delta loss",
    )
    ax.axhline(0, color="#333333", lw=0.8)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(reduced["mutation"], rotation=90, fontsize=7)
    ax.set_ylabel("AlphaSeq delta log10 Kd vs WT")
    ax2.set_ylabel("Mosaic sequence delta loss")
    ax.set_title("Best measured single substitutions and their sequence-prior losses", fontweight="bold")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_single_mutation_loss_affinity.png"), dpi=220)
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_single_mutation_loss_affinity.pdf"))
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    meta: dict[str, object],
    output_prefix: Path,
    no_plots: bool,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored_csv = output_prefix.with_name(f"{output_prefix.name}_variant_scores.csv")
    stats_csv = output_prefix.with_name(f"{output_prefix.name}_group_stats.csv")
    summary_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    top_csv = output_prefix.with_name(f"{output_prefix.name}_top_low_loss_candidates.csv")

    df.to_csv(scored_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    summary_json.write_text(json.dumps(meta, indent=2) + "\n")
    top = (
        df[df["fully_covered_by_loss_table"] & df["mosaic_sequence_delta_loss_sum"].notna()]
        .sort_values("mosaic_sequence_delta_loss_sum", ascending=True)
        .head(500)
    )
    top.to_csv(top_csv, index=False)
    print(f"wrote {scored_csv}")
    print(f"wrote {stats_csv}")
    print(f"wrote {summary_json}")
    print(f"wrote {top_csv}")
    if not no_plots:
        plot_loss_vs_affinity(df, output_prefix)
        plot_loss_by_label(df, output_prefix)
        plot_single_mutation_heatmap(df, output_prefix)
        for suffix in [
            "loss_vs_affinity",
            "loss_by_affinity_label",
            "single_mutation_loss_affinity",
        ]:
            print(f"wrote {output_prefix.with_name(f'{output_prefix.name}_{suffix}.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--assays", default=",".join(DEFAULT_ASSAYS))
    parser.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    parser.add_argument(
        "--include-all-targets",
        action="store_true",
        help="Analyze every target in each assay instead of filtering by --target-regex.",
    )
    parser.add_argument("--wt-sequence", default=None)
    parser.add_argument("--loss-csv", type=Path, default=DEFAULT_LOSS_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--good-threshold-log10",
        type=float,
        default=0.30,
        help="WT-relative log10 Kd change used for good/bad labels. 0.30 is about 2-fold.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assays = [value.strip() for value in args.assays.split(",") if value.strip()]
    pl_df = load_alphaseq(args.repo_id, assays, args.target_regex, args.include_all_targets)
    wt_sequence = infer_wt_sequence(pl_df, args.wt_sequence)
    lookup, cdr_positions = load_loss_lookup(args.loss_csv)

    rows = pl_df.filter(pl.col("mata_sequence").is_not_null()).to_dicts()
    df = pd.DataFrame.from_records(rows)
    df["is_wt_sequence"] = df["mata_sequence"].astype(str).eq(wt_sequence)
    df = add_variant_scores(df, wt_sequence, lookup, cdr_positions)
    df = add_affinity_columns(df, args.good_threshold_log10)
    stats_df, meta = summarize(df, args.good_threshold_log10)
    meta.update({
        "repo_id": args.repo_id,
        "assays": assays,
        "target_regex": None if args.include_all_targets else args.target_regex,
        "include_all_targets": bool(args.include_all_targets),
        "wt_sequence": wt_sequence,
        "wt_sequence_length": len(wt_sequence),
        "loss_csv": str(args.loss_csv),
    })
    write_outputs(df, stats_df, meta, args.output_prefix, args.no_plots)
    print("\nKey target statistics:")
    display_cols = [
        "assay",
        "target",
        "n_scored_for_stats",
        "good_count",
        "bad_count",
        "mosaic_sequence_delta_loss_sum_spearman_vs_delta_affinity",
        "mosaic_sum_auc_good_vs_bad_low_loss_is_good",
        "mosaic_sum_auc_good_vs_bad_high_loss_is_good",
        "mosaic_sum_mannwhitney_pvalue",
        "top_5pct_low_loss_good_enrichment",
    ]
    present = [col for col in display_cols if col in stats_df.columns]
    print(stats_df[present].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
