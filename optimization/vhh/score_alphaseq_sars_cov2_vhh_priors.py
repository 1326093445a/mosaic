#!/usr/bin/env python3
"""Score the AlphaSeq SARS-CoV-2 RBD VHH DMS with ESM2/AbLang2 priors.

The input is the single-mutant table produced by
find_alphaseq_sars_cov2_antibody_dms.py.  For each measured one-substitution
variant, this script computes parent-context masked-marginal probabilities:

    delta_nll = -log p(mutant_aa | parent with site masked)
              - -log p(wt_aa     | parent with site masked)

Negative delta_nll means the model prefers the mutant residue over WT at that
site. Positive delta_nll means the model penalizes the mutant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mosaic.common import TOKENS


AA_ORDER = list(TOKENS)
DEFAULT_SINGLE_MUTANTS = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh.single_mutants.csv")
DEFAULT_SUMMARY = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh.summary.json")
DEFAULT_OUTPUT_PREFIX = Path("vhh/alphaseq_sars_cov2_rbd_top_vhh_sequence_priors")


def batched(items: list[int], batch_size: int) -> Iterable[list[int]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def log_softmax_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def normalized_geomean_logprob(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    return log_softmax_np(0.5 * (np.asarray(logp_a) + np.asarray(logp_b)))


def load_parent(summary_path: Path) -> str:
    summary = json.loads(summary_path.read_text())
    parent = summary.get("parent_sequence")
    if not parent:
        raise ValueError(f"No parent_sequence in {summary_path}")
    return str(parent)


def direct_esm2_logprobs(
    sequence: str,
    positions_1based: list[int],
    model_name: str,
    device: str,
    batch_size: int,
) -> dict[int, np.ndarray]:
    import esm

    print(f"Loading ESM2 model {model_name} on {device}...", flush=True)
    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"Unknown ESM2 model {model_name!r}")
    model, alphabet = loader()
    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, base_tokens = batch_converter([("parent", sequence)])
    base_tokens = base_tokens.to(device)
    aa_indices = torch.as_tensor([alphabet.tok_to_idx[aa] for aa in AA_ORDER], device=device)

    result: dict[int, np.ndarray] = {}
    print(f"Scoring ESM2 masked marginals for {len(positions_1based)} sites...", flush=True)
    with torch.no_grad():
        for chunk in batched(positions_1based, batch_size):
            tokens = base_tokens.repeat(len(chunk), 1)
            for row_idx, position in enumerate(chunk):
                token_index = int(position)  # BOS token shifts chain position to the same 1-based index.
                tokens[row_idx, token_index] = alphabet.mask_idx
            logits = model(tokens)["logits"]
            for row_idx, position in enumerate(chunk):
                token_index = int(position)
                site_logits = logits[row_idx, token_index]
                logp = torch.log_softmax(site_logits, dim=-1)
                result[int(position)] = logp.index_select(0, aa_indices).cpu().numpy()
    return result


def direct_ablang2_logprobs(
    sequence: str,
    positions_1based: list[int],
    device: str,
    batch_size: int,
) -> dict[int, np.ndarray]:
    from ablang2.load_model import load_model

    print(f"Loading AbLang2 model on {device}...", flush=True)
    model, tokenizer, _ = load_model("ablang2-paired", device=device)
    model.eval()
    model.to(device)
    aa_to_token = tokenizer.aa_to_token
    token_ids = [aa_to_token["<"], *[aa_to_token[aa] for aa in sequence], aa_to_token[">"], aa_to_token["|"]]
    base_tokens = torch.as_tensor(token_ids, dtype=torch.long, device=device)[None]
    aa_indices = torch.as_tensor([aa_to_token[aa] for aa in AA_ORDER], dtype=torch.long, device=device)
    special_indices = torch.as_tensor(tokenizer.all_special_tokens, dtype=torch.long, device=device)

    result: dict[int, np.ndarray] = {}
    print(f"Scoring AbLang2 masked marginals for {len(positions_1based)} sites...", flush=True)
    with torch.no_grad():
        for chunk in batched(positions_1based, batch_size):
            tokens = base_tokens.repeat(len(chunk), 1)
            for row_idx, position in enumerate(chunk):
                token_index = int(position)  # initial "<" shifts chain position to the same 1-based index.
                tokens[row_idx, token_index] = aa_to_token["*"]
            logits = model(tokens)
            for row_idx, position in enumerate(chunk):
                token_index = int(position)
                site_logits = logits[row_idx, token_index].clone()
                site_logits.index_fill_(0, special_indices, -1e9)
                logp = torch.log_softmax(site_logits, dim=-1)
                result[int(position)] = logp.index_select(0, aa_indices).cpu().numpy()
    return result


def mutation_delta_nll(logp_by_pos: dict[int, np.ndarray], row: pd.Series) -> tuple[float, float, float, int]:
    position = int(row["position_1based"])
    wt = str(row["wt"])
    mut = str(row["mut"])
    values = logp_by_pos[position]
    wt_idx = AA_ORDER.index(wt)
    mut_idx = AA_ORDER.index(mut)
    wt_logp = float(values[wt_idx])
    mut_logp = float(values[mut_idx])
    delta_nll = wt_logp - mut_logp
    order = np.argsort(-values)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return delta_nll, wt_logp, mut_logp, int(ranks[mut_idx])


def auc_rank(labels: list[bool], scores: list[float]) -> float | None:
    if not labels:
        return None
    y = np.asarray(labels, dtype=bool)
    x = np.asarray(scores, dtype=float)
    ok = np.isfinite(x)
    y = y[ok]
    x = x[ok]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(x).rank(method="average").to_numpy()
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float | None:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return None
    value = subset[x].corr(subset[y], method=method)
    if value is None or math.isnan(float(value)):
        return None
    return float(value)


def summarize_model(df: pd.DataFrame, score_col: str) -> dict[str, object]:
    measured = df[df["label"].isin(["good", "bad", "neutral"])].copy()
    binary = measured[measured["label"].isin(["good", "bad"])].copy()
    topk = {}
    for k in (25, 50, 100):
        top = measured.nsmallest(min(k, len(measured)), score_col)
        topk[str(k)] = {
            "n": int(len(top)),
            "good": int((top["label"] == "good").sum()),
            "bad": int((top["label"] == "bad").sum()),
            "neutral": int((top["label"] == "neutral").sum()),
            "good_fraction": float((top["label"] == "good").mean()) if len(top) else None,
            "bad_fraction": float((top["label"] == "bad").mean()) if len(top) else None,
        }

    medians = (
        measured.groupby("label", dropna=False)[score_col]
        .median()
        .sort_index()
        .to_dict()
    )
    return {
        "spearman_delta_processed_vs_score": safe_corr(measured, "delta_processed", score_col, "spearman"),
        "pearson_delta_processed_vs_score": safe_corr(measured, "delta_processed", score_col, "pearson"),
        "auc_bad_vs_good_high_score_is_bad": auc_rank(
            (binary["label"] == "bad").tolist(),
            binary[score_col].tolist(),
        ),
        "auc_good_vs_bad_low_score_is_good": auc_rank(
            (binary["label"] == "good").tolist(),
            (-binary[score_col]).tolist(),
        ),
        "median_score_by_label": {k: float(v) for k, v in medians.items()},
        "top_lowest_loss": topk,
    }


def make_plots(df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = df[df["label"].isin(["good", "bad", "neutral"])].copy()
    colors = {"good": "#2ca25f", "neutral": "#7b8da6", "bad": "#de2d26"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["esm2_delta_nll", "ablang2_delta_nll", "mosaic_sequence_delta_loss"],
        ["ESM2", "AbLang2", "Weighted"],
    ):
        for label, group in plot_df.groupby("label"):
            ax.scatter(
                group["delta_processed"],
                group[col],
                s=14,
                alpha=0.65,
                linewidth=0,
                color=colors.get(label, "black"),
                label=label,
            )
        ax.axvline(-0.3, color="#444444", lw=0.8, ls="--")
        ax.axvline(0.3, color="#444444", lw=0.8, ls="--")
        ax.axhline(0.0, color="#444444", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("AlphaSeq delta processed")
        ax.set_ylabel("model delta NLL/loss")
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output_prefix.with_suffix(".scatter.png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".scatter.pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-mutants", type=Path, default=DEFAULT_SINGLE_MUTANTS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--esm2-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--skip-esm2", action="store_true")
    parser.add_argument("--skip-ablang2", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-esm2", type=float, default=0.10)
    parser.add_argument("--weight-ablang2", type=float, default=0.10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    parent = load_parent(args.summary_json)
    df = pd.read_csv(args.single_mutants)
    df = df[df["mut"].astype(str).isin(AA_ORDER)].copy()
    df = df[df["wt"].astype(str).isin(AA_ORDER)].copy()
    positions = sorted(int(v) for v in df["position_1based"].dropna().unique())

    if not args.skip_esm2:
        esm_logp = direct_esm2_logprobs(
            parent,
            positions,
            args.esm2_model,
            device,
            args.batch_size,
        )
        esm_values = df.apply(lambda row: mutation_delta_nll(esm_logp, row), axis=1)
        df["esm2_delta_nll"] = [v[0] for v in esm_values]
        df["esm2_wt_logprob"] = [v[1] for v in esm_values]
        df["esm2_mut_logprob"] = [v[2] for v in esm_values]
        df["esm2_mut_rank"] = [v[3] for v in esm_values]

    if not args.skip_ablang2:
        ablang_logp = direct_ablang2_logprobs(parent, positions, device, args.batch_size)
        ablang_values = df.apply(lambda row: mutation_delta_nll(ablang_logp, row), axis=1)
        df["ablang2_delta_nll"] = [v[0] for v in ablang_values]
        df["ablang2_wt_logprob"] = [v[1] for v in ablang_values]
        df["ablang2_mut_logprob"] = [v[2] for v in ablang_values]
        df["ablang2_mut_rank"] = [v[3] for v in ablang_values]

    if "esm2_delta_nll" in df.columns and "ablang2_delta_nll" in df.columns:
        df["mosaic_sequence_delta_loss"] = (
            args.weight_esm2 * df["esm2_delta_nll"]
            + args.weight_ablang2 * df["ablang2_delta_nll"]
        )
        df["combined_geomean_delta_nll"] = 0.5 * (
            df["esm2_delta_nll"] + df["ablang2_delta_nll"]
        )
    elif "esm2_delta_nll" in df.columns:
        df["mosaic_sequence_delta_loss"] = args.weight_esm2 * df["esm2_delta_nll"]
    elif "ablang2_delta_nll" in df.columns:
        df["mosaic_sequence_delta_loss"] = args.weight_ablang2 * df["ablang2_delta_nll"]

    summary: dict[str, object] = {
        "single_mutants": str(args.single_mutants),
        "summary_json": str(args.summary_json),
        "parent_sequence": parent,
        "n_scored": int(len(df)),
        "label_counts": {
            str(k): int(v) for k, v in df["label"].fillna("missing").value_counts().to_dict().items()
        },
        "device": device,
        "esm2_model": None if args.skip_esm2 else args.esm2_model,
        "weight_esm2": args.weight_esm2,
        "weight_ablang2": args.weight_ablang2,
        "models": {},
    }
    for col in ["esm2_delta_nll", "ablang2_delta_nll", "mosaic_sequence_delta_loss"]:
        if col in df.columns:
            summary["models"][col] = summarize_model(df, col)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored_csv = args.output_prefix.with_suffix(".scored.csv")
    summary_json = args.output_prefix.with_suffix(".summary.json")
    df.to_csv(scored_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plots and "mosaic_sequence_delta_loss" in df.columns:
        make_plots(df, args.output_prefix)

    print(json.dumps(summary, indent=2))
    print(f"wrote {scored_csv}")
    print(f"wrote {summary_json}")
    if not args.no_plots and "mosaic_sequence_delta_loss" in df.columns:
        print(f"wrote {args.output_prefix.with_suffix('.scatter.png')}")
        print(f"wrote {args.output_prefix.with_suffix('.scatter.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
