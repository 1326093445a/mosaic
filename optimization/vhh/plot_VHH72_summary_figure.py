#!/usr/bin/env python3
"""Create a compact summary figure for the VHH72 benchmark diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_PRIOR = Path("vhh/VHH72_sequence_prior_scores.csv")
DEFAULT_VANILLA_ME = Path("vhh/VHH72_vanilla_multievolve_esm2_target_sites.csv")
DEFAULT_RECOVERY = Path("VHH72_benchmark_both_200/mutation_recovery.csv")
DEFAULT_REFOLD = Path(
    "vhh/VHH72_publication_refold_templates_s5_all/"
    "publication_variant_refold_all_samples.csv"
)
DEFAULT_OUTPUT = Path("vhh/VHH72_benchmark_summary_figure")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def panel_sequence_priors(ax, sequence_prior_csv: Path):
    df = pd.read_csv(sequence_prior_csv)
    df = df[df["source"] == "publication_grid"].copy()
    df = df.sort_values("mosaic_sequence_delta_loss")

    labels = df["label"].tolist()
    values = df["mosaic_sequence_delta_loss"].astype(float).to_numpy()
    colors = ["#2a9d8f" if value <= 0 else "#d1495b" for value in values]

    y = np.arange(len(df))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.0034, 0.0098)
    ax.set_xlabel("Mosaic sequence-prior dLoss vs WT")
    ax.set_title("A. ESM2 + AbLang2 sequence prior")
    style_axes(ax)

    for yi, value in zip(y, values):
        ha = "left" if value >= 0 else "right"
        offset = 0.00025 if value >= 0 else -0.00025
        ax.text(value + offset, yi, f"{value:+.4f}", va="center", ha=ha, fontsize=7)


def panel_vanilla_multievolve(ax, vanilla_csv: Path):
    df = pd.read_csv(vanilla_csv)
    subset = df[df["is_paper_mutation"] | df["is_site_best"]].copy()
    site_order = ["H56", "H97", "H99"]
    subset["site"] = pd.Categorical(subset["anarci_label"], site_order, ordered=True)
    subset = subset.sort_values(["site", "is_paper_mutation"])

    x = np.arange(len(site_order))
    width = 0.36
    best_values = []
    paper_values = []
    best_labels = []
    paper_labels = []
    paper_ranks = []
    for site in site_order:
        site_df = subset[subset["anarci_label"] == site]
        best = site_df[site_df["is_site_best"]].iloc[0]
        paper = site_df[site_df["is_paper_mutation"]].iloc[0]
        best_values.append(float(best["vanilla_esm_loss"]))
        paper_values.append(float(paper["vanilla_esm_loss"]))
        best_labels.append(best["mutation"])
        paper_labels.append(paper["mutation"])
        paper_ranks.append(int(paper["site_rank"]))

    ax.bar(
        x - width / 2,
        best_values,
        width,
        label="site-best replacement",
        color="#457b9d",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        paper_values,
        width,
        label="paper replacement",
        color="#e76f51",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(site_order)
    ax.set_ylabel("vanilla ESM2 loss = -logratio")
    ax.set_title("B. Vanilla MULTI-evolve zero-shot")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ymax = max(best_values + paper_values)
    ax.set_ylim(0, ymax * 1.22)
    for xi, value, label in zip(x - width / 2, best_values, best_labels):
        ax.text(xi, value + ymax * 0.025, label, ha="center", va="bottom", fontsize=7)
    for xi, value, label, rank in zip(x + width / 2, paper_values, paper_labels, paper_ranks):
        ax.text(
            xi,
            value + ymax * 0.025,
            f"{label}\nrank {rank}/19",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def panel_recovery(ax, recovery_csv: Path):
    df = pd.read_csv(recovery_csv)
    counts = df["paper_hit_combo"].fillna("none").value_counts()

    categories = ["none", "L97W", "T99V", "S56M", "L97W+T99V", "multi"]
    values = []
    for category in categories:
        if category == "multi":
            values.append(int(df["paper_hit_multi"].fillna(False).astype(bool).sum()))
        else:
            values.append(int(counts.get(category, 0)))

    x = np.arange(len(categories))
    colors = ["#b8b8b8", "#2a9d8f", "#2a9d8f", "#d1495b", "#f4a261", "#f4a261"]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("ranked refold rows")
    ax.set_title("C. Standard Mosaic run recovery")
    style_axes(ax)

    for xi, value in zip(x, values):
        ax.text(xi, value + max(values) * 0.02, str(value), ha="center", va="bottom", fontsize=8)

    total = len(df)
    any_hits = int(df["paper_hit_any"].fillna(False).astype(bool).sum())
    multi_hits = int(df["paper_hit_multi"].fillna(False).astype(bool).sum())
    ax.text(
        0.98,
        0.92,
        f"n={total}\nany hit={any_hits}\nmulti-hit={multi_hits}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )


def panel_refold(ax, refold_csv: Path):
    df = pd.read_csv(refold_csv)
    df = df[df["candidate_label"].isin(["WT", "S56M+L97W+T99V"])].copy()

    marker_by_candidate = {"WT": "o", "S56M+L97W+T99V": "s"}
    color_by_mode = {
        "target_template_only": "#8d99ae",
        "target_plus_binder_template": "#2a9d8f",
    }
    label_seen = set()
    for (mode, candidate), sub in df.groupby(["template_mode", "candidate_label"]):
        label = f"{mode.replace('_', ' ')} / {candidate}"
        ax.scatter(
            sub["binder_ca_rmsd_target_aligned"],
            sub["ipsae_min"],
            s=44,
            marker=marker_by_candidate[candidate],
            color=color_by_mode.get(mode, "#444444"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=label if label not in label_seen else None,
        )
        label_seen.add(label)

    best = (
        df.sort_values("binder_ca_rmsd_target_aligned")
        .groupby(["template_mode", "candidate_label"], as_index=False)
        .first()
    )
    for _, row in best.iterrows():
        ax.annotate(
            f"{row['candidate_label']}\n{row['binder_ca_rmsd_target_aligned']:.1f}A",
            (
                row["binder_ca_rmsd_target_aligned"],
                row["ipsae_min"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axvline(4.0, color="#d1495b", linestyle="--", linewidth=1.0)
    ax.text(4.0, ax.get_ylim()[0], " 4A pose gate", color="#d1495b", fontsize=8, va="bottom")
    ax.set_xlabel("binder CA RMSD after target alignment (A)")
    ax.set_ylabel("ipSAE min")
    ax.set_title("D. Refold pose gate vs ipSAE")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_axes(ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-prior-csv", type=Path, default=DEFAULT_SEQUENCE_PRIOR)
    parser.add_argument("--vanilla-multievolve-csv", type=Path, default=DEFAULT_VANILLA_ME)
    parser.add_argument("--recovery-csv", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--refold-csv", type=Path, default=DEFAULT_REFOLD)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.0))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.14,
        wspace=0.18,
        hspace=0.36,
    )
    fig.suptitle(
        "VHH72 retrospective diagnostics: sequence priors suppress key paper mutations",
        fontsize=14,
        fontweight="bold",
    )

    panel_sequence_priors(axs[0, 0], args.sequence_prior_csv)
    panel_vanilla_multievolve(axs[0, 1], args.vanilla_multievolve_csv)
    panel_recovery(axs[1, 0], args.recovery_csv)
    panel_refold(axs[1, 1], args.refold_csv)

    footer = (
        "Takeaway: blind PLM sequence priors do not recover the VHH72 affinity mutations; "
        "RMSD/pose gating is required before ranking ipSAE."
    )
    fig.text(0.5, 0.045, footer, ha="center", va="bottom", fontsize=10)

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
