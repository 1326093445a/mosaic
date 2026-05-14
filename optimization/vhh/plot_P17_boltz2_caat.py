#!/usr/bin/env python3
"""Plot P17 Boltz2 CAAT-style sensitivity summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AA_ORDER = "ARNDQEGHILKFPSTWYV"


def _read_config_aa_order(root: Path, fallback: str) -> str:
    for target in ("alpha", "jn1"):
        path = root / target / "config.json"
        if path.exists():
            try:
                aa_panel = json.loads(path.read_text()).get("aa_panel")
            except json.JSONDecodeError:
                aa_panel = None
            if aa_panel:
                return str(aa_panel)
    return fallback


def _load_target(root: Path, target: str) -> dict[str, pd.DataFrame]:
    base = root / target
    required = {
        "baseline": base / "baseline_samples.csv",
        "single": base / "single_mutation_scores.csv",
        "position": base / "position_sensitivity.csv",
        "aa": base / "aa_type_sensitivity.csv",
        "curve": base / "edit_count_curve.csv",
        "ack": base / "acknowledgement_summary.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {target} CAAT output(s): " + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in required.items()}


def _threshold_acknowledged(df: pd.DataFrame, min_delta_ipsae: float, min_abs_z: float) -> pd.Series:
    return (
        df["delta_ipsae_min_mean"].abs().ge(min_delta_ipsae)
        & df["z_delta_ipsae_min_mean"].abs().ge(min_abs_z)
    )


def _summarize_positions(single: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, group in single.groupby("position", sort=True):
        records = group.to_dict("records")
        most_sensitive = max(records, key=lambda r: r["abs_delta_ipsae_min_mean"])
        best_increase = max(records, key=lambda r: r["delta_ipsae_min_mean"])
        worst_decrease = min(records, key=lambda r: r["delta_ipsae_min_mean"])
        rows.append({
            "position": int(position),
            "author_residue": records[0]["author_residue"],
            "wt_aa": records[0]["wt_aa"],
            "n_mutations_tested": len(records),
            "n_acknowledged": int(sum(bool(r["acknowledged"]) for r in records)),
            "mean_abs_delta_ipsae_min": float(np.mean([r["abs_delta_ipsae_min_mean"] for r in records])),
            "max_abs_delta_ipsae_min": most_sensitive["abs_delta_ipsae_min_mean"],
            "most_sensitive_mutation": most_sensitive["mutation"],
            "most_sensitive_delta_ipsae": most_sensitive["delta_ipsae_min_mean"],
            "best_increase_mutation": best_increase["mutation"],
            "best_delta_ipsae": best_increase["delta_ipsae_min_mean"],
            "worst_decrease_mutation": worst_decrease["mutation"],
            "worst_delta_ipsae": worst_decrease["delta_ipsae_min_mean"],
        })
    return pd.DataFrame(rows).sort_values("max_abs_delta_ipsae_min", ascending=False)


def _summarize_aa_types(single: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mut_aa, group in single.groupby("mut_aa", sort=True):
        records = group.to_dict("records")
        most_sensitive = max(records, key=lambda r: r["abs_delta_ipsae_min_mean"])
        best_increase = max(records, key=lambda r: r["delta_ipsae_min_mean"])
        worst_decrease = min(records, key=lambda r: r["delta_ipsae_min_mean"])
        n_ack = int(sum(bool(r["acknowledged"]) for r in records))
        rows.append({
            "mut_aa": mut_aa,
            "n_positions_tested": len(records),
            "n_acknowledged": n_ack,
            "acknowledged_fraction": n_ack / len(records),
            "mean_delta_ipsae_min": float(np.mean([r["delta_ipsae_min_mean"] for r in records])),
            "mean_abs_delta_ipsae_min": float(np.mean([r["abs_delta_ipsae_min_mean"] for r in records])),
            "max_abs_delta_ipsae_min": most_sensitive["abs_delta_ipsae_min_mean"],
            "most_sensitive_mutation": most_sensitive["mutation"],
            "most_sensitive_position": most_sensitive["position"],
            "most_sensitive_author_residue": most_sensitive["author_residue"],
            "most_sensitive_delta_ipsae": most_sensitive["delta_ipsae_min_mean"],
            "best_increase_mutation": best_increase["mutation"],
            "best_increase_position": best_increase["position"],
            "best_delta_ipsae": best_increase["delta_ipsae_min_mean"],
            "worst_decrease_mutation": worst_decrease["mutation"],
            "worst_decrease_position": worst_decrease["position"],
            "worst_delta_ipsae": worst_decrease["delta_ipsae_min_mean"],
        })
    return pd.DataFrame(rows).sort_values("max_abs_delta_ipsae_min", ascending=False)


def _summarize_acknowledgement(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ["absolute", "increase"]:
        sub = curve[curve["mode"].eq(mode)].sort_values("edit_count")
        first = next((r for _, r in sub.iterrows() if bool(r["acknowledged"])), None)
        five_rows = sub[sub["edit_count"].astype(int).eq(5)]
        five = five_rows.iloc[0] if not five_rows.empty else None
        nearest = sub[sub["edit_count"].astype(int).le(5)].tail(1)
        chosen = five if five is not None else (nearest.iloc[0] if not nearest.empty else None)
        rows.append({
            "mode": mode,
            "first_acknowledged_edit_count": first["edit_count"] if first is not None else "",
            "first_acknowledged_delta_ipsae": first["delta_ipsae_min_mean"] if first is not None else "",
            "first_acknowledged_mutations": first["mutations"] if first is not None else "",
            "five_edit_available": five is not None,
            "five_or_nearest_edit_count": chosen["edit_count"] if chosen is not None else "",
            "five_or_nearest_acknowledged": chosen["acknowledged"] if chosen is not None else "",
            "five_or_nearest_delta_ipsae": chosen["delta_ipsae_min_mean"] if chosen is not None else "",
            "five_or_nearest_z_delta_ipsae": chosen["z_delta_ipsae_min_mean"] if chosen is not None else "",
            "five_or_nearest_mutations": chosen["mutations"] if chosen is not None else "",
        })
    return pd.DataFrame(rows)


def _apply_acknowledgement_cutoff(
    data: dict[str, dict[str, pd.DataFrame]],
    min_delta_ipsae: float,
    min_abs_z: float,
) -> dict[str, dict[str, pd.DataFrame]]:
    out = {}
    for target, frames in data.items():
        single = frames["single"].copy()
        single["acknowledged"] = _threshold_acknowledged(single, min_delta_ipsae, min_abs_z)
        curve = frames["curve"].copy()
        curve["acknowledged"] = _threshold_acknowledged(curve, min_delta_ipsae, min_abs_z)
        out[target] = {
            **frames,
            "single": single,
            "position": _summarize_positions(single),
            "aa": _summarize_aa_types(single),
            "curve": curve,
            "ack": _summarize_acknowledgement(curve),
        }
    return out


def _write_comparison_csv(root: Path, data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for target in ["alpha", "jn1"]:
        baseline = data[target]["baseline"]
        pos = data[target]["position"]
        aa = data[target]["aa"]
        ack = data[target]["ack"]
        inc = ack[ack["mode"].eq("increase")].iloc[0]
        absolute = ack[ack["mode"].eq("absolute")].iloc[0]
        rows.append({
            "target": target,
            "baseline_ipsae_mean": baseline["ipsae_min"].mean(),
            "baseline_ipsae_std": baseline["ipsae_min"].std(ddof=1),
            "top_positions": ";".join(
                f"{int(r.position)}:{r.most_sensitive_mutation}"
                f"({r.most_sensitive_delta_ipsae:+.3f})"
                for _, r in pos.head(6).iterrows()
            ),
            "top_aa_types": ";".join(
                f"{r.mut_aa}:{int(r.n_acknowledged)}/{int(r.n_positions_tested)}"
                for _, r in aa.head(6).iterrows()
            ),
            "increase_first_edits": inc["first_acknowledged_edit_count"],
            "increase_5edit_ack": inc["five_or_nearest_acknowledged"],
            "increase_5edit_delta": inc["five_or_nearest_delta_ipsae"],
            "absolute_first_edits": absolute["first_acknowledged_edit_count"],
            "absolute_5edit_ack": absolute["five_or_nearest_acknowledged"],
            "absolute_5edit_delta": absolute["five_or_nearest_delta_ipsae"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(root / "alpha_vs_jn1_caat_summary.csv", index=False)
    return out


def plot_comparison(
    root: Path,
    data: dict[str, dict[str, pd.DataFrame]],
    aa_order: str,
    min_delta_ipsae: float,
    min_abs_z: float,
) -> None:
    colors = {"alpha": "#2b8cbe", "jn1": "#d95f0e"}
    labels = {"alpha": "Alpha / binder", "jn1": "JN.1 / escape"}

    baseline = {}
    for target in ["alpha", "jn1"]:
        b = data[target]["baseline"]
        baseline[target] = {
            "mean": b["ipsae_min"].mean(),
            "std": b["ipsae_min"].std(ddof=1),
        }

    fig = plt.figure(figsize=(13.8, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.75, 1.25, 1.15],
        height_ratios=[1.0, 1.05],
    )

    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.arange(2)
    means = [baseline[t]["mean"] for t in ["alpha", "jn1"]]
    stds = [baseline[t]["std"] for t in ["alpha", "jn1"]]
    ax0.bar(
        xs,
        means,
        yerr=stds,
        color=[colors["alpha"], colors["jn1"]],
        capsize=5,
        width=0.62,
    )
    ax0.set_xticks(xs, ["Alpha", "JN.1"])
    ax0.set_ylabel("WT baseline ipSAE")
    ax0.set_ylim(0, 1.0)
    ax0.set_title("Starting Confidence")
    ax0.grid(axis="y", color="#dddddd")
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax0.text(
            i,
            mean + std + 0.035,
            f"{mean:.2f}\n+/-{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1 = fig.add_subplot(gs[0, 1])
    rows = []
    for target in ["alpha", "jn1"]:
        pos = data[target]["position"].copy()
        pos["signed_max_effect"] = pos["most_sensitive_delta_ipsae"]
        for _, row in pos.head(8).iterrows():
            rows.append({
                "target": target,
                "mutation": row["most_sensitive_mutation"],
                "effect": row["signed_max_effect"],
            })
    plot_df = pd.DataFrame(rows)
    plot_df["row_label"] = (
        plot_df["target"].map({"alpha": "A", "jn1": "J"})
        + ": "
        + plot_df["mutation"]
    )
    y = np.arange(len(plot_df))[::-1]
    ax1.barh(y, plot_df["effect"], color=[colors[t] for t in plot_df["target"]])
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_yticks(y, plot_df["row_label"])
    ax1.set_xlabel("Most sensitive single-mutant delta ipSAE")
    ax1.set_title("Position x AA Sensitivity")
    ax1.grid(axis="x", color="#dddddd")
    ax1.set_xlim(
        min(-0.8, plot_df["effect"].min() * 1.15),
        max(0.38, plot_df["effect"].max() * 1.15),
    )

    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(aa_order))
    width = 0.38
    for offset, target in [(-width / 2, "alpha"), (width / 2, "jn1")]:
        aa = data[target]["aa"].set_index("mut_aa").reindex(list(aa_order))
        frac = aa["acknowledged_fraction"].fillna(0).to_numpy()
        ax2.bar(
            x + offset,
            frac,
            width=width,
            color=colors[target],
            label=labels[target],
        )
    ax2.set_xticks(x, list(aa_order))
    ax2.set_ylim(0, 0.7)
    ax2.set_ylabel("Fraction acknowledged")
    ax2.set_title("AA-Type Sensitivity")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", color="#dddddd")

    ax3 = fig.add_subplot(gs[1, 0:2])
    for target in ["alpha", "jn1"]:
        curve = data[target]["curve"]
        sub = curve[curve["mode"].eq("increase")].sort_values("edit_count")
        ax3.plot(
            sub["edit_count"],
            sub["delta_ipsae_min_mean"],
            marker="o",
            color=colors[target],
            label=labels[target],
        )
        for _, row in sub.iterrows():
            if int(row["edit_count"]) == 5:
                ax3.scatter(
                    [row["edit_count"]],
                    [row["delta_ipsae_min_mean"]],
                    s=100,
                    facecolors="none",
                    edgecolors=colors[target],
                    linewidths=2,
                )
                ax3.text(
                    row["edit_count"] + 0.08,
                    row["delta_ipsae_min_mean"],
                    "5 edits",
                    color=colors[target],
                    fontsize=9,
                    va="center",
                )
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(
        min_delta_ipsae,
        color="#666666",
        linestyle="--",
        linewidth=0.9,
        label=f"raw delta {min_delta_ipsae:.2f}",
    )
    ax3.set_xlabel("Cumulative favorable edits")
    ax3.set_ylabel("Delta ipSAE vs WT baseline")
    ax3.set_title("How Many Favorable Edits Are Needed?")
    ax3.set_xticks(range(1, 8))
    ax3.grid(color="#dddddd")
    ax3.legend(frameon=False, ncol=3, fontsize=9)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    text = [
        "Readout",
        "",
        f"Alpha baseline ipSAE: {baseline['alpha']['mean']:.3f}",
        f"JN.1 baseline ipSAE: {baseline['jn1']['mean']:.3f}",
        "",
        "Alpha: high-confidence binder background",
        "strong disruptive sensitivity marks anchors",
        "",
        "JN.1: low-confidence escape background",
        "positive shifts mark rescue-like candidates",
        "",
        "5 favorable edits:",
        "",
        f"Acknowledged: |delta| >= {min_delta_ipsae:.2f}",
        f"and |z| >= {min_abs_z:.1f}",
    ]
    for target_label, target in [("Alpha", "alpha"), ("JN.1", "jn1")]:
        ack = data[target]["ack"]
        row = ack[ack["mode"].eq("increase")].iloc[0]
        text.append(
            f"{target_label}: ack={row['five_or_nearest_acknowledged']}, "
            f"delta={row['five_or_nearest_delta_ipsae']:+.3f}"
        )
    ax4.text(0, 1, "\n".join(text), va="top", fontsize=10.5, linespacing=1.25)

    fig.suptitle(
        "P17 Boltz2 CAAT-Style Sensitivity: Alpha Binder vs JN.1 Escape",
        fontsize=15,
    )
    fig.savefig(root / "P17_alpha_vs_jn1_caat_summary.png", dpi=240)
    fig.savefig(root / "P17_alpha_vs_jn1_caat_summary.pdf")


def plot_heatmaps(root: Path, data: dict[str, dict[str, pd.DataFrame]], aa_order: str) -> None:
    titles = {"alpha": "Alpha / P17 binder", "jn1": "JN.1 / P17 escape"}
    for target in ["alpha", "jn1"]:
        single = data[target]["single"]
        positions = sorted(single["position"].unique())
        mat = pd.DataFrame(index=positions, columns=list(aa_order), dtype=float)
        for _, row in single.iterrows():
            aa = row["mut_aa"]
            if aa in aa_order:
                mat.loc[int(row["position"]), aa] = row["delta_ipsae_min_mean"]
        wt_by_pos = (
            single[["position", "wt_aa"]]
            .drop_duplicates()
            .set_index("position")["wt_aa"]
            .to_dict()
        )
        for position, wt_aa in wt_by_pos.items():
            if wt_aa in aa_order:
                mat.loc[int(position), wt_aa] = 0.0

        fig, ax = plt.subplots(
            figsize=(max(8.8, 0.48 * len(aa_order) + 2.4), 8.8),
            constrained_layout=True,
        )
        values = mat.to_numpy(dtype=float)
        vmax = max(0.12, float(np.nanmax(np.abs(values))))
        im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(aa_order)), list(aa_order))
        ax.set_yticks(np.arange(len(positions)), positions)
        ax.set_xlabel("Mutant amino acid")
        ax.set_ylabel("P17 chain-order CDR position")
        ax.set_title(f"{titles[target]}: single-mutation delta ipSAE")
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("Delta ipSAE vs WT baseline")

        acknowledged = single[single["acknowledged"].astype(bool)]
        pos_to_y = {pos: idx for idx, pos in enumerate(positions)}
        aa_to_x = {aa: idx for idx, aa in enumerate(aa_order)}
        for _, row in acknowledged.iterrows():
            aa = row["mut_aa"]
            if aa in aa_to_x:
                ax.scatter(
                    aa_to_x[aa],
                    pos_to_y[int(row["position"])],
                    s=24,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.9,
                )
        for position, wt_aa in wt_by_pos.items():
            if wt_aa in aa_to_x and int(position) in pos_to_y:
                ax.scatter(
                    aa_to_x[wt_aa],
                    pos_to_y[int(position)],
                    marker="*",
                    s=78,
                    facecolors="#ffd447",
                    edgecolors="black",
                    linewidths=0.75,
                    zorder=5,
                )

        fig.savefig(root / f"P17_{target}_single_mutation_delta_ipsae_heatmap.png", dpi=240)
        fig.savefig(root / f"P17_{target}_single_mutation_delta_ipsae_heatmap.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("vhh/P17_boltz2_caat_sensitivity"))
    parser.add_argument("--aa-order", default=None)
    parser.add_argument("--min-delta-ipsae", type=float, default=0.10)
    parser.add_argument("--min-abs-z", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root
    aa_order = args.aa_order or _read_config_aa_order(root, DEFAULT_AA_ORDER)
    data = {target: _load_target(root, target) for target in ["alpha", "jn1"]}
    data = _apply_acknowledgement_cutoff(data, args.min_delta_ipsae, args.min_abs_z)
    _write_comparison_csv(root, data)
    plot_comparison(root, data, aa_order, args.min_delta_ipsae, args.min_abs_z)
    plot_heatmaps(root, data, aa_order)
    print(root / "P17_alpha_vs_jn1_caat_summary.png")
    print(root / "P17_alpha_vs_jn1_caat_summary.pdf")
    print(root / "P17_alpha_single_mutation_delta_ipsae_heatmap.png")
    print(root / "P17_jn1_single_mutation_delta_ipsae_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
