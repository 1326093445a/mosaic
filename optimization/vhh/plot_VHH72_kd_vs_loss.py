#!/usr/bin/env python3
"""Plot VHH72 paper off-rate versus sequence/model losses.

The VHH72 paper reports lower-case k_d / dissociation off-rate, not equilibrium
K_D.  Values here are in units of 1e-4 s^-1.  Lower off-rate is better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_PRIOR = Path("vhh/VHH72_sequence_prior_scores.csv")
DEFAULT_VANILLA_ME = Path("vhh/VHH72_vanilla_multievolve_esm2_candidate_scores.csv")
DEFAULT_OUTPUT = Path("vhh/VHH72_kd_vs_loss")

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

# Chain-order mutation strings in the vanilla MULTI-evolve diagnostic.
PAPER_TO_CHAIN_MUTATION = {
    "S56M": "S57M",
    "L97W": "L101W",
    "T99V": "T103V",
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.8)


def paper_combo_to_chain_combo(label: str) -> str:
    if label == "WT":
        return "WT"
    return "/".join(PAPER_TO_CHAIN_MUTATION[item] for item in label.split("+"))


def load_plot_table(sequence_prior_csv: Path, vanilla_csv: Path) -> pd.DataFrame:
    seq = pd.read_csv(sequence_prior_csv)
    seq = seq[seq["source"] == "publication_grid"].copy()
    seq = seq[["label", "variant_count", "mosaic_sequence_delta_loss"]]

    vanilla = pd.read_csv(vanilla_csv)
    vanilla = vanilla[["mutations", "vanilla_esm_additive_loss"]].copy()
    vanilla_by_mut = {
        str(row["mutations"]): float(row["vanilla_esm_additive_loss"])
        for _, row in vanilla.iterrows()
    }

    rows = []
    for _, row in seq.iterrows():
        label = str(row["label"])
        chain_combo = paper_combo_to_chain_combo(label)
        rows.append({
            "label": label,
            "variant_count": int(row["variant_count"]),
            "off_rate_x1e4_s": PAPER_OFF_RATES[label],
            "benefit_log10": np.log10(PAPER_OFF_RATES["WT"] / PAPER_OFF_RATES[label]),
            "mosaic_sequence_delta_loss": float(row["mosaic_sequence_delta_loss"]),
            "vanilla_esm_additive_loss": vanilla_by_mut[chain_combo],
        })

    return pd.DataFrame(rows)


def annotate_points(ax, x, y, labels):
    for xi, yi, label in zip(x, y, labels):
        dx = 5
        dy = 4
        if label == "WT":
            dy = -12
        ax.annotate(
            label,
            (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def plot_panel(ax, df: pd.DataFrame, x_col: str, title: str, x_label: str):
    colors = {
        0: "#9e9e9e",
        1: "#2a9d8f",
        2: "#457b9d",
        3: "#d1495b",
    }
    for count, sub in df.groupby("variant_count"):
        ax.scatter(
            sub[x_col],
            sub["off_rate_x1e4_s"],
            s=78,
            color=colors.get(int(count), "#444444"),
            edgecolor="black",
            linewidth=0.5,
            label=f"{int(count)} mutation(s)",
            zorder=3,
        )
    annotate_points(ax, df[x_col], df["off_rate_x1e4_s"], df["label"])
    ax.set_yscale("log")
    ax.set_ylabel("measured k_d off-rate (x1e-4 s^-1; lower better)")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    style_axes(ax)

    rho = spearman_r(df[x_col].to_numpy(), df["off_rate_x1e4_s"].to_numpy())
    ax.text(
        0.97,
        0.95,
        f"Spearman rho\nloss vs off-rate = {rho:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-prior-csv", type=Path, default=DEFAULT_SEQUENCE_PRIOR)
    parser.add_argument("--vanilla-multievolve-csv", type=Path, default=DEFAULT_VANILLA_ME)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = load_plot_table(args.sequence_prior_csv, args.vanilla_multievolve_csv)

    table_path = args.output_prefix.with_name(args.output_prefix.name + "_table.csv")
    df.to_csv(table_path, index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.7))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18, wspace=0.24)
    fig.suptitle(
        "VHH72 measured off-rate versus model losses",
        fontsize=15,
        fontweight="bold",
    )

    plot_panel(
        axs[0],
        df,
        "mosaic_sequence_delta_loss",
        "A. Mosaic sequence-prior loss",
        "ESM2 + AbLang2 dLoss vs WT (lower preferred)",
    )
    plot_panel(
        axs[1],
        df,
        "vanilla_esm_additive_loss",
        "B. Vanilla MULTI-evolve-style ESM2 loss",
        "additive ESM2 zero-shot loss (lower preferred)",
    )
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")

    fig.text(
        0.5,
        0.04,
        "If the losses were affinity-aligned, lower measured off-rate would also have lower loss. "
        "The S56M-containing variants violate this, especially for blind PLM losses.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
