#!/usr/bin/env python3
"""Single-substitution loss scan across VHH72 design CDRs.

This is a DMS-style diagnostic over the benchmark CDR design region.  It mutates
each CDR residue to every other standard amino acid, scores each sequence with
the same ESM2/AbLang2 sequence priors used by the VHH72 Mosaic runs, and writes:

  * per-substitution losses and within-site ranks,
  * a summary for the paper mutations S56M/L97W/T99V,
  * 20x20 WT->mutant matrices of mean loss shift,
  * heatmap figures for the combined Mosaic sequence loss.

The paper reports Kabat labels (H56, H97, H99).  Mosaic sequences use chain-order
indices, so this script keeps both labels in the output.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mosaic.common import TOKENS
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.esm import ESM2PseudoLikelihood, load_esm2
from mosaic.optimizers import batched_value_eval


DEFAULT_PDB = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb")
DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_OUTPUT_PREFIX = Path("vhh/VHH72_cdr_substitution_loss")
DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"
AA_ORDER = list(TOKENS)


def read_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    structure = gemmi.read_structure(str(pdb_path))
    chain = structure[0][chain_id]
    sequence = gemmi.one_letter_code([res.name for res in chain])
    if not sequence:
        raise ValueError(f"No sequence found for chain {chain_id!r} in {pdb_path}")
    bad = sorted(set(sequence) - set(TOKENS))
    if bad:
        raise ValueError(f"Unsupported residue code(s) in chain {chain_id}: {bad}")
    return sequence


def load_cdr_positions(cdr_map: Path, groups: set[str]) -> list[dict]:
    rows = []
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") in groups:
                rows.append({
                    "cdr": row["group"],
                    "anarci_label": row["anarci_label"],
                    "seq_index": int(row["seq_index"]),
                    "pdb_auth_label": row["pdb_auth_label"],
                    "wt": row["aa"],
                })
    if not rows:
        raise ValueError(f"No CDR rows for groups {sorted(groups)} in {cdr_map}")
    return rows


def load_known_variants(cdr_map: Path) -> dict[tuple[int, str], str]:
    known = {}
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row.get("variant", "").strip().upper()
            if row.get("group") == "known_variant" and variant:
                known[(int(row["seq_index"]), variant[-1])] = variant
    return known


def mutate_sequence(sequence: str, seq_index: int, mut: str, expected_wt: str) -> str:
    idx0 = seq_index - 1
    observed = sequence[idx0]
    if observed != expected_wt:
        raise ValueError(
            f"seq_index {seq_index} expected {expected_wt}, observed {observed}"
        )
    chars = list(sequence)
    chars[idx0] = mut
    return "".join(chars)


def build_scan_rows(
    wt_sequence: str,
    cdr_positions: list[dict],
    known_variants: dict[tuple[int, str], str],
) -> list[dict]:
    rows = [{
        "row_type": "WT",
        "cdr": "",
        "anarci_label": "",
        "seq_index": "",
        "pdb_auth_label": "",
        "wt": "",
        "mut": "",
        "mutation_chain_index": "WT",
        "substitution_type": "WT",
        "paper_variant": "",
        "is_paper_variant": False,
        "sequence": wt_sequence,
    }]

    for position in cdr_positions:
        seq_index = int(position["seq_index"])
        wt = position["wt"]
        if wt_sequence[seq_index - 1] != wt:
            raise ValueError(
                f"{position['anarci_label']} seq_index {seq_index} expects {wt}, "
                f"observed {wt_sequence[seq_index - 1]}"
            )
        for mut in AA_ORDER:
            if mut == wt:
                continue
            paper_variant = known_variants.get((seq_index, mut), "")
            rows.append({
                "row_type": "single_substitution",
                "cdr": position["cdr"],
                "anarci_label": position["anarci_label"],
                "seq_index": seq_index,
                "pdb_auth_label": position["pdb_auth_label"],
                "wt": wt,
                "mut": mut,
                "mutation_chain_index": f"{wt}{seq_index}{mut}",
                "substitution_type": f"{wt}->{mut}",
                "paper_variant": paper_variant,
                "is_paper_variant": bool(paper_variant),
                "sequence": mutate_sequence(wt_sequence, seq_index, mut, wt),
            })
    return rows


def tokenize_sequences(sequences: list[str]) -> np.ndarray:
    return np.asarray([[TOKENS.index(aa) for aa in seq] for seq in sequences], dtype=np.int32)


def score_loss(loss, seq_ids: np.ndarray, *, batch_size: int, seed: int):
    values = []
    aux_chunks = []
    batch_size = max(1, int(batch_size))
    key = jax.random.key(seed)

    for start in range(0, len(seq_ids), batch_size):
        chunk = seq_ids[start:start + batch_size]
        valid = len(chunk)
        if valid < batch_size:
            pad = np.repeat(chunk[-1][None], batch_size - valid, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        xs = jax.nn.one_hot(jnp.asarray(chunk), len(TOKENS))
        vals, aux = batched_value_eval(
            loss,
            xs,
            jnp.broadcast_to(key, (xs.shape[0], *key.shape)),
        )
        vals.block_until_ready()
        values.extend(np.asarray(vals[:valid], dtype=float).tolist())
        aux_chunks.append(jax.tree.map(lambda x: np.asarray(x)[:valid], aux))

    return np.asarray(values, dtype=float), aux_chunks


def aux_vector(aux_chunks, key: str) -> np.ndarray:
    values = []
    for aux in aux_chunks:
        if key in aux:
            values.extend(np.asarray(aux[key], dtype=float).tolist())
    return np.asarray(values, dtype=float)


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    wt = df[df["row_type"] == "WT"].iloc[0]
    for column, delta_column in [
        ("esm2_nll", "esm2_delta_nll"),
        ("ablang2_nll", "ablang2_delta_nll"),
        ("mosaic_sequence_loss", "mosaic_sequence_delta_loss"),
    ]:
        if column in df:
            df[delta_column] = df[column].astype(float) - float(wt[column])
    return df


def add_rank_columns(df: pd.DataFrame, loss_column: str) -> pd.DataFrame:
    scan = df["row_type"] == "single_substitution"
    df[f"{loss_column}_site_rank"] = ""
    df[f"{loss_column}_site_percentile"] = ""
    df[f"{loss_column}_cdr_rank"] = ""
    df[f"{loss_column}_global_rank"] = ""

    for _, idx in df[scan].groupby("seq_index").groups.items():
        values = df.loc[idx, loss_column].astype(float)
        ranks = values.rank(method="min", ascending=True).astype(int)
        percentile = (ranks - 1) / max(len(ranks) - 1, 1)
        df.loc[idx, f"{loss_column}_site_rank"] = ranks.astype(str)
        df.loc[idx, f"{loss_column}_site_percentile"] = percentile.map(lambda x: f"{x:.4f}")

    for _, idx in df[scan].groupby("cdr").groups.items():
        ranks = df.loc[idx, loss_column].astype(float).rank(method="min", ascending=True).astype(int)
        df.loc[idx, f"{loss_column}_cdr_rank"] = ranks.astype(str)

    ranks = df.loc[scan, loss_column].astype(float).rank(method="min", ascending=True).astype(int)
    df.loc[scan, f"{loss_column}_global_rank"] = ranks.astype(str)
    return df


def matrix_for_column(scan_df: pd.DataFrame, column: str, agg: str) -> pd.DataFrame:
    values = pd.DataFrame(index=AA_ORDER, columns=AA_ORDER, dtype=float)
    grouped = scan_df.groupby(["wt", "mut"])[column]
    reduced = grouped.mean() if agg == "mean" else grouped.median()
    for (wt, mut), value in reduced.items():
        values.loc[wt, mut] = float(value)
    return values


def count_matrix(scan_df: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(0, index=AA_ORDER, columns=AA_ORDER, dtype=int)
    for (wt, mut), count in scan_df.groupby(["wt", "mut"]).size().items():
        values.loc[wt, mut] = int(count)
    return values


def write_matrices(scan_df: pd.DataFrame, output_prefix: Path, agg: str) -> None:
    for column in ["esm2_delta_nll", "ablang2_delta_nll", "mosaic_sequence_delta_loss"]:
        if column in scan_df:
            matrix = matrix_for_column(scan_df, column, agg)
            matrix.to_csv(output_prefix.with_name(f"{output_prefix.name}_{column}_{agg}_matrix.csv"))
    count_matrix(scan_df).to_csv(
        output_prefix.with_name(f"{output_prefix.name}_substitution_count_matrix.csv")
    )


def paper_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "paper_variant",
        "cdr",
        "anarci_label",
        "pdb_auth_label",
        "seq_index",
        "mutation_chain_index",
        "substitution_type",
        "esm2_delta_nll",
        "ablang2_delta_nll",
        "mosaic_sequence_delta_loss",
        "mosaic_sequence_delta_loss_site_rank",
        "mosaic_sequence_delta_loss_site_percentile",
        "mosaic_sequence_delta_loss_cdr_rank",
        "mosaic_sequence_delta_loss_global_rank",
    ]
    return df[df["is_paper_variant"].astype(bool)][cols].copy()


def plot_heatmap(
    matrix: pd.DataFrame,
    scan_df: pd.DataFrame,
    output_prefix: Path,
    *,
    title: str,
    value_label: str,
) -> None:
    data = matrix.to_numpy(dtype=float)
    finite = np.isfinite(data)
    max_abs = float(np.nanmax(np.abs(data[finite]))) if finite.any() else 1.0

    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        aspect="equal",
    )
    ax.set_xticks(np.arange(len(AA_ORDER)))
    ax.set_yticks(np.arange(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER)
    ax.set_yticklabels(AA_ORDER)
    ax.set_xlabel("mutant amino acid")
    ax.set_ylabel("WT amino acid")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(AA_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AA_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for _, row in scan_df[scan_df["is_paper_variant"].astype(bool)].iterrows():
        y = AA_ORDER.index(row["wt"])
        x = AA_ORDER.index(row["mut"])
        ax.scatter(
            [x],
            [y],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2.2,
        )
        ax.text(
            x,
            y,
            row["paper_variant"],
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
        )

    cbar = fig.colorbar(image, ax=ax, shrink=0.78)
    cbar.set_label(value_label)
    fig.text(
        0.5,
        0.03,
        "Cells are mean single-substitution loss shifts over VHH72 benchmark CDR positions with that WT residue. "
        "Black circles mark paper substitutions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def plot_site_rank(df: pd.DataFrame, output_prefix: Path) -> None:
    scan = df[df["row_type"] == "single_substitution"].copy()
    paper = scan[scan["is_paper_variant"].astype(bool)].copy()

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    x_positions = []
    labels = []
    for i, (seq_index, sub) in enumerate(scan.groupby("seq_index", sort=True)):
        x = np.full(len(sub), i, dtype=float)
        jitter = np.linspace(-0.22, 0.22, len(sub))
        sub = sub.sort_values("mosaic_sequence_delta_loss")
        ax.scatter(
            x + jitter,
            sub["mosaic_sequence_delta_loss"].astype(float),
            s=18,
            color="#b8b8b8",
            alpha=0.7,
            edgecolor="none",
        )
        label = f"{sub.iloc[0]['anarci_label']}\n{sub.iloc[0]['wt']}{seq_index}"
        x_positions.append(i)
        labels.append(label)

    ax.scatter(
        [x_positions[labels.index(f"{row['anarci_label']}\n{row['wt']}{row['seq_index']}")] for _, row in paper.iterrows()],
        paper["mosaic_sequence_delta_loss"].astype(float),
        s=95,
        color="#d1495b",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )
    for _, row in paper.iterrows():
        idx = labels.index(f"{row['anarci_label']}\n{row['wt']}{row['seq_index']}")
        ax.annotate(
            f"{row['paper_variant']}\nrank {row['mosaic_sequence_delta_loss_site_rank']}/19",
            (idx, float(row["mosaic_sequence_delta_loss"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mosaic sequence-prior dLoss vs WT")
    ax.set_title("VHH72 CDR single-substitution loss distribution by site", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument(
        "--cdr-groups",
        default="CDR1,CDR2,CDR3",
        help="Comma-separated CDR map groups to scan. Defaults to benchmark design CDRs.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--esm2-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--skip-esm2", action="store_true")
    parser.add_argument("--skip-ablang2", action="store_true")
    parser.add_argument("--weight-esm2", type=float, default=0.10)
    parser.add_argument("--weight-ablang2", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matrix-agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = {value.strip() for value in args.cdr_groups.split(",") if value.strip()}
    wt_sequence = read_chain_sequence(args.pdb, args.binder_chain)
    cdr_positions = load_cdr_positions(args.cdr_map, groups)
    known_variants = load_known_variants(args.cdr_map)
    rows = build_scan_rows(wt_sequence, cdr_positions, known_variants)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    seq_ids = tokenize_sequences([row["sequence"] for row in rows])

    if not args.skip_esm2:
        print(f"Loading ESM2 model {args.esm2_model}...", flush=True)
        esm2_loss = ESM2PseudoLikelihood(load_esm2(args.esm2_model))
        print(f"Scoring {len(rows)} sequences with ESM2...", flush=True)
        vals, aux = score_loss(
            esm2_loss, seq_ids, batch_size=args.batch_size, seed=args.seed
        )
        esm_pll = aux_vector(aux, "esm_pll")
        for row, val, pll in zip(rows, vals, esm_pll):
            row["esm2_nll"] = val
            row["esm2_pll"] = pll

    if not args.skip_ablang2:
        print("Loading AbLang2 model...", flush=True)
        ablang2_model, ablang2_tokenizer = load_ablang2()
        ablang2_loss = Ablang2PseudoLikelihood(
            ablang2_model,
            ablang2_tokenizer,
            heavy_len=len(wt_sequence),
            stop_grad=True,
        )
        print(f"Scoring {len(rows)} sequences with AbLang2...", flush=True)
        vals, aux = score_loss(
            ablang2_loss, seq_ids, batch_size=args.batch_size, seed=args.seed + 1
        )
        ablang2_ppl = aux_vector(aux, "ablang2_ppl")
        for row, val, ppl in zip(rows, vals, ablang2_ppl):
            row["ablang2_nll"] = val
            row["ablang2_ppl"] = ppl

    df = pd.DataFrame(rows)
    total = np.zeros(len(df), dtype=float)
    have = np.zeros(len(df), dtype=bool)
    if "esm2_nll" in df:
        total += args.weight_esm2 * df["esm2_nll"].astype(float).to_numpy()
        have |= True
    if "ablang2_nll" in df:
        total += args.weight_ablang2 * df["ablang2_nll"].astype(float).to_numpy()
        have |= True
    if have.any():
        df["mosaic_sequence_loss"] = total

    df = add_delta_columns(df)
    for column in ["esm2_delta_nll", "ablang2_delta_nll", "mosaic_sequence_delta_loss"]:
        if column in df:
            df = add_rank_columns(df, column)

    scan_csv = args.output_prefix.with_suffix(".csv")
    df.to_csv(scan_csv, index=False)
    print(f"wrote {scan_csv}")

    scan_df = df[df["row_type"] == "single_substitution"].copy()
    summary = paper_summary(df)
    summary_csv = args.output_prefix.with_name(f"{args.output_prefix.name}_paper_mutation_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")
    print("\nPaper mutation summary:")
    print(summary.to_string(index=False))

    write_matrices(scan_df, args.output_prefix, args.matrix_agg)

    if not args.no_plots:
        matrix = matrix_for_column(
            scan_df, "mosaic_sequence_delta_loss", args.matrix_agg
        )
        plot_heatmap(
            matrix,
            scan_df,
            args.output_prefix.with_name(
                f"{args.output_prefix.name}_mosaic_dloss_{args.matrix_agg}_heatmap"
            ),
            title="VHH72 CDR WT->mutant Mosaic sequence-prior dLoss",
            value_label=f"{args.matrix_agg} dLoss vs WT",
        )
        plot_site_rank(
            df,
            args.output_prefix.with_name(
                f"{args.output_prefix.name}_site_distribution"
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
