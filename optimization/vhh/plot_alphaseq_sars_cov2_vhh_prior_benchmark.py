#!/usr/bin/env python3
"""Plot the AlphaSeq SARS-CoV-2 RBD VHH sequence-prior benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCORED = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh_sequence_priors.scored.csv")
DEFAULT_SUMMARY = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh_sequence_priors.summary.json")
DEFAULT_OUTPUT = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh_prior_benchmark")
LABEL_ORDER = ["good", "neutral", "bad"]
LABEL_COLORS = {
    "good": "#2a9d8f",
    "neutral": "#8d99ae",
    "bad": "#d1495b",
}


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def panel_scatter(ax, df: pd.DataFrame, metrics: dict) -> None:
    for label in LABEL_ORDER:
        sub = df[df["label"] == label]
        ax.scatter(
            sub["delta_processed"],
            sub["mosaic_sequence_delta_loss"],
            s=17,
            alpha=0.68,
            linewidth=0,
            color=LABEL_COLORS[label],
            label=f"{label} (n={len(sub)})",
        )
    ax.axvline(-0.3, color="#333333", linewidth=0.8, linestyle="--")
    ax.axvline(0.3, color="#333333", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("AlphaSeq delta log10 Kd vs parent (lower is better)")
    ax.set_ylabel("Mosaic sequence dLoss (lower is preferred)")
    ax.set_title("A. Measured binding effect vs sequence-prior loss")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax, grid_axis="both")

    ax.text(
        0.98,
        0.05,
        (
            f"Spearman = {metrics['spearman']:+.3f}\n"
            f"AUC bad-vs-good = {metrics['auc']:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cfcfcf"),
    )


def panel_box(ax, df: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    values = [df[df["label"] == label]["mosaic_sequence_delta_loss"].dropna().to_numpy() for label in LABEL_ORDER]
    box = ax.boxplot(
        values,
        tick_labels=LABEL_ORDER,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, label in zip(box["boxes"], LABEL_ORDER):
        patch.set_facecolor(LABEL_COLORS[label])
        patch.set_alpha(0.55)

    for idx, label in enumerate(LABEL_ORDER, start=1):
        sub = df[df["label"] == label]["mosaic_sequence_delta_loss"].dropna().to_numpy()
        if len(sub) > 250:
            sub = rng.choice(sub, size=250, replace=False)
        x = rng.normal(idx, 0.045, size=len(sub))
        ax.scatter(x, sub, s=7, color=LABEL_COLORS[label], alpha=0.22, linewidth=0)
        median = float(df[df["label"] == label]["mosaic_sequence_delta_loss"].median())
        ax.text(idx, median, f"  {median:.3f}", va="center", ha="left", fontsize=8)

    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mosaic sequence dLoss")
    ax.set_title("B. Loss distribution by experimental label")
    style_axes(ax)


def panel_topk(ax, df: pd.DataFrame) -> pd.DataFrame:
    measured = df[df["label"].isin(LABEL_ORDER)].copy()
    rows = []
    groups = [("all", measured)]
    for k in (25, 50, 100, 200):
        groups.append((f"top {k}", measured.nsmallest(k, "mosaic_sequence_delta_loss")))

    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups), dtype=float)
    for label in LABEL_ORDER:
        fractions = []
        counts = []
        for name, sub in groups:
            count = int((sub["label"] == label).sum())
            counts.append(count)
            fractions.append(count / max(len(sub), 1))
            rows.append({
                "group": name,
                "label": label,
                "count": count,
                "n": int(len(sub)),
                "fraction": count / max(len(sub), 1),
            })
        ax.bar(
            x,
            fractions,
            bottom=bottoms,
            color=LABEL_COLORS[label],
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        bottoms += np.asarray(fractions)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in groups], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.set_title("C. What appears in the lowest-loss ranked set?")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax)

    for xi, (_, sub) in zip(x, groups):
        bad = int((sub["label"] == "bad").sum())
        good = int((sub["label"] == "good").sum())
        ax.text(
            xi,
            0.98,
            f"G{good}/B{bad}",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )

    return pd.DataFrame(rows)


def panel_examples(ax, df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    ax.axis("off")
    bad_liked = (
        df[df["label"] == "bad"]
        .nsmallest(6, "mosaic_sequence_delta_loss")
        [["mutation", "delta_processed", "mosaic_sequence_delta_loss"]]
        .copy()
    )
    good_penalized = (
        df[df["label"] == "good"]
        .nlargest(6, "mosaic_sequence_delta_loss")
        [["mutation", "delta_processed", "mosaic_sequence_delta_loss"]]
        .copy()
    )
    bad_liked["set"] = "bad_low_model_loss"
    good_penalized["set"] = "good_high_model_loss"
    examples = pd.concat([bad_liked, good_penalized], ignore_index=True)

    y = 0.97
    ax.text(0.00, y, "D. Discordant examples", fontsize=12, fontweight="bold", va="top")
    y -= 0.09
    ax.text(
        0.00,
        y,
        (
            "Sequence priors are useful regularizers, but here they are not a binding oracle. "
            "Several disruptive mutations receive low model loss, while several measured "
            "beneficial mutations are penalized."
        ),
        fontsize=9,
        va="top",
        wrap=True,
    )
    y -= 0.18
    ax.text(0.00, y, "Model likes, AlphaSeq says bad", fontsize=10, fontweight="bold", va="top")
    y -= 0.055
    ax.text(0.00, y, "mutation   AlphaSeq d   model dLoss", fontsize=8, family="monospace", va="top")
    y -= 0.045
    for _, row in bad_liked.iterrows():
        ax.text(
            0.00,
            y,
            f"{row['mutation']:<9} {row['delta_processed']:+8.3f}   {row['mosaic_sequence_delta_loss']:+9.3f}",
            fontsize=8,
            family="monospace",
            color=LABEL_COLORS["bad"],
            va="top",
        )
        y -= 0.040

    y -= 0.035
    ax.text(0.00, y, "Experiment says good, model penalizes", fontsize=10, fontweight="bold", va="top")
    y -= 0.055
    ax.text(0.00, y, "mutation   AlphaSeq d   model dLoss", fontsize=8, family="monospace", va="top")
    y -= 0.045
    for _, row in good_penalized.iterrows():
        ax.text(
            0.00,
            y,
            f"{row['mutation']:<9} {row['delta_processed']:+8.3f}   {row['mosaic_sequence_delta_loss']:+9.3f}",
            fontsize=8,
            family="monospace",
            color=LABEL_COLORS["good"],
            va="top",
        )
        y -= 0.040

    ax.text(
        0.64,
        0.30,
        (
            f"n mutations: {metrics['n']}\n"
            f"good/bad: {metrics['n_good']}/{metrics['n_bad']}\n"
            f"top100 good/bad: {metrics['top100_good']}/{metrics['top100_bad']}\n"
            f"median loss good: {metrics['median_good']:.3f}\n"
            f"median loss bad:  {metrics['median_bad']:.3f}"
        ),
        fontsize=9,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cfcfcf"),
    )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-csv", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scored_csv)
    df = df[df["label"].isin(LABEL_ORDER)].copy()

    binary = df[df["label"].isin(["good", "bad"])].copy()
    metrics = {
        "n": int(len(df)),
        "n_good": int((df["label"] == "good").sum()),
        "n_bad": int((df["label"] == "bad").sum()),
        "spearman": float(df["delta_processed"].corr(df["mosaic_sequence_delta_loss"], method="spearman")),
        "auc": auc_rank(
            (binary["label"] == "bad").to_numpy(),
            binary["mosaic_sequence_delta_loss"].to_numpy(),
        ),
        "median_good": float(df[df["label"] == "good"]["mosaic_sequence_delta_loss"].median()),
        "median_bad": float(df[df["label"] == "bad"]["mosaic_sequence_delta_loss"].median()),
    }
    top100 = df.nsmallest(100, "mosaic_sequence_delta_loss")
    metrics["top100_good"] = int((top100["label"] == "good").sum())
    metrics["top100_bad"] = int((top100["label"] == "bad").sum())

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.0))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.11,
        wspace=0.22,
        hspace=0.35,
    )
    fig.suptitle(
        "AlphaSeq SARS-CoV-2 RBD VHH DMS: ESM2/AbLang2 priors do not identify binding-good mutations",
        fontsize=14,
        fontweight="bold",
    )

    panel_scatter(axs[0, 0], df, metrics)
    panel_box(axs[0, 1], df)
    topk_df = panel_topk(axs[1, 0], df)
    examples_df = panel_examples(axs[1, 1], df, metrics)

    footer = (
        "Parent VHH background from AlphaSeq SARS-CoV-2 RBD; good/bad labels use "
        "delta log10 Kd thresholds -0.3/+0.3 relative to parent."
    )
    fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=9)

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metrics_path = args.output_prefix.with_suffix(".metrics.json")
    topk_path = args.output_prefix.with_suffix(".topk_composition.csv")
    examples_path = args.output_prefix.with_suffix(".discordant_examples.csv")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    topk_df.to_csv(topk_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {topk_path}")
    print(f"wrote {examples_path}")


if __name__ == "__main__":
    main()
