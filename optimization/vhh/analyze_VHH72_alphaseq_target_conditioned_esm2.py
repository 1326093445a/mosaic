#!/usr/bin/env python3
"""Test whether ESM2 improves when VHH72 is scored with RBD sequence context.

This is a target-conditioning diagnostic for AlphaSeq VHH72 variants.  It scores
observed VHH substitutions with masked-marginal ESM2 deltas in two contexts:

  1. binder only:         VHH72
  2. target conditioned:  VHH72 <eos> RBD

For each mutated VHH position, the script masks the WT VHH residue in the chosen
context and records log p(WT) and log p(mutant).  Per-variant scores are additive
sums over observed substitutions:

    delta_nll = -log p(mutant | masked context) - -log p(WT | masked context)
              = log p(WT | masked context) - log p(mutant | masked context)

Lower delta_nll means ESM2 prefers the mutant residue over WT in that context.
The target-conditioned score is experimental: ESM2 is not a trained PPI model,
but this tells us whether simply allowing attention to the RBD sequence improves
the sequence-prior signal against measured AlphaSeq affinity.
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
import torch
from huggingface_hub import hf_hub_download
from scipy import stats
from sklearn.metrics import roc_auc_score


DEFAULT_ASSAYS = ("YM_0549", "YM_1068")
DEFAULT_REPO = "aalphabio/open-alphaseq"
DEFAULT_TARGET_REGEX = "SARS-CoV2_RBD_\\(6LZG\\)"
DEFAULT_OUTPUT_PREFIX = Path("vhh/VHH72_alphaseq_target_conditioned_esm2")
DEFAULT_ESM2_MODEL = "esm2_t33_650M_UR50D"
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


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
    std = arr.std(skipna=True, ddof=0)
    if not math.isfinite(float(std)) or float(std) == 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (arr - arr.mean(skipna=True)) / std


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float | None:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return None
    value = subset[x].corr(subset[y], method=method)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def auc_score(df: pd.DataFrame, score_col: str, *, low_loss_is_good: bool) -> float | None:
    subset = df[df["affinity_label"].isin(["good", "bad"])].copy()
    subset = subset[[score_col, "affinity_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset["affinity_label"].nunique() < 2:
        return None
    y = (subset["affinity_label"] == "good").astype(int).to_numpy()
    score = subset[score_col].astype(float).to_numpy()
    if low_loss_is_good:
        score = -score
    return float(roc_auc_score(y, score))


def topk_enrichment(df: pd.DataFrame, score_col: str, frac: float) -> dict[str, float | int | None]:
    measured = df[df["affinity_label"].isin(["good", "neutral", "bad"])].copy()
    measured = measured[[score_col, "affinity_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if measured.empty:
        return {
            "top_n": 0,
            "good": 0,
            "bad": 0,
            "good_fraction": None,
            "background_good_fraction": None,
            "good_enrichment": None,
            "hypergeom_good_pvalue": None,
        }
    measured = measured.sort_values(score_col, ascending=True)
    total = len(measured)
    top_n = max(1, int(round(total * frac)))
    top = measured.head(top_n)
    good = int((top["affinity_label"] == "good").sum())
    bad = int((top["affinity_label"] == "bad").sum())
    total_good = int((measured["affinity_label"] == "good").sum())
    background = total_good / total if total else None
    pvalue = None
    if total_good and total_good < total and good:
        pvalue = float(stats.hypergeom.sf(good - 1, total, total_good, top_n))
    return {
        "top_n": top_n,
        "good": good,
        "bad": bad,
        "good_fraction": good / top_n,
        "background_good_fraction": background,
        "good_enrichment": (good / top_n) / background if background else None,
        "hypergeom_good_pvalue": pvalue,
    }


def load_alphaseq_assay(repo_id: str, assay: str) -> pl.DataFrame:
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename="data.parquet",
        repo_type="dataset",
        subfolder=f"data/{assay}",
    )
    return pl.read_parquet(parquet_path).with_columns(pl.lit(assay).alias("assay"))


def load_alphaseq(repo_id: str, assays: Iterable[str], target_regex: str) -> pl.DataFrame:
    frames = []
    for assay in assays:
        frame = load_alphaseq_assay(repo_id, assay)
        frame = frame.filter(pl.col("matalpha_description").str.contains(target_regex))
        frames.append(frame)
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


def compare_to_wt(sequence: str, wt_sequence: str) -> tuple[list[Mutation], int, bool]:
    shared = min(len(sequence), len(wt_sequence))
    mutations = [
        Mutation(i + 1, wt_sequence[i], sequence[i])
        for i in range(shared)
        if sequence[i] != wt_sequence[i]
    ]
    length_delta = len(sequence) - len(wt_sequence)
    clean_substitutions = (
        length_delta == 0
        and all(m.wt in AA_ORDER and m.mut in AA_ORDER for m in mutations)
    )
    return mutations, length_delta, clean_substitutions


def load_esm2(model_name: str, device: str):
    import esm

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"Unknown ESM2 model {model_name!r}")
    print(f"Loading {model_name} on {device}...", flush=True)
    model, alphabet = loader()
    model.eval()
    model.to(device)
    return model, alphabet


def token_idx(alphabet, aa: str) -> int:
    try:
        return int(alphabet.tok_to_idx[aa])
    except KeyError as exc:
        raise ValueError(f"ESM2 alphabet has no token for {aa!r}") from exc


def build_tokens(
    alphabet,
    binder_sequence: str,
    *,
    target_sequence: str | None,
    target_mode: str,
    linker: str,
) -> torch.Tensor:
    tokens = [int(alphabet.cls_idx)]
    tokens.extend(token_idx(alphabet, aa) for aa in binder_sequence)
    if target_sequence:
        if target_mode == "eos":
            tokens.append(int(alphabet.eos_idx))
            tokens.extend(token_idx(alphabet, aa) for aa in target_sequence)
            tokens.append(int(alphabet.eos_idx))
        elif target_mode == "linker":
            tokens.extend(token_idx(alphabet, aa) for aa in linker)
            tokens.extend(token_idx(alphabet, aa) for aa in target_sequence)
            tokens.append(int(alphabet.eos_idx))
        else:
            raise ValueError(f"Unsupported target_mode {target_mode!r}")
    else:
        tokens.append(int(alphabet.eos_idx))
    return torch.as_tensor(tokens, dtype=torch.long)[None]


def masked_logprobs(
    model,
    alphabet,
    base_tokens: torch.Tensor,
    positions: list[int],
    *,
    device: str,
    batch_size: int,
) -> dict[int, dict[str, float]]:
    aa_indices = torch.as_tensor(
        [token_idx(alphabet, aa) for aa in AA_ORDER],
        dtype=torch.long,
        device=device,
    )
    base_tokens = base_tokens.to(device)
    result: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            tokens = base_tokens.repeat(len(chunk), 1)
            for row_idx, position in enumerate(chunk):
                token_position = int(position)  # CLS at index 0, chain position is 1-based.
                tokens[row_idx, token_position] = int(alphabet.mask_idx)
            logits = model(tokens)["logits"]
            for row_idx, position in enumerate(chunk):
                logp = torch.log_softmax(logits[row_idx, int(position)], dim=-1)
                values = logp.index_select(0, aa_indices).detach().cpu().numpy()
                result[int(position)] = {
                    aa: float(value) for aa, value in zip(AA_ORDER, values)
                }
            print(
                f"  scored positions {start + 1}-{start + len(chunk)} / {len(positions)}",
                flush=True,
            )
    return result


def mutation_delta(logp_by_position: dict[int, dict[str, float]], mutation: Mutation) -> float:
    values = logp_by_position[mutation.position]
    return float(values[mutation.wt] - values[mutation.mut])


def score_variants(
    df: pd.DataFrame,
    wt_sequence: str,
    binder_logp: dict[int, dict[str, float]],
    target_logp_by_sequence: dict[str, dict[int, dict[str, float]]],
) -> pd.DataFrame:
    records = []
    for row in df.to_dict("records"):
        sequence = str(row["mata_sequence"])
        target_sequence = str(row["matalpha_sequence"])
        mutations, length_delta, clean = compare_to_wt(sequence, wt_sequence)
        covered_mutations = [
            m
            for m in mutations
            if m.position in binder_logp
            and m.position in target_logp_by_sequence[target_sequence]
            and m.wt in AA_ORDER
            and m.mut in AA_ORDER
        ]
        binder_sum = np.nan
        target_sum = np.nan
        if clean and covered_mutations:
            binder_sum = sum(mutation_delta(binder_logp, m) for m in covered_mutations)
            target_logp = target_logp_by_sequence[target_sequence]
            target_sum = sum(mutation_delta(target_logp, m) for m in covered_mutations)
        elif clean and not covered_mutations and not mutations:
            binder_sum = 0.0
            target_sum = 0.0
        count = len(covered_mutations)
        records.append({
            "sequence_length": len(sequence),
            "length_delta_vs_wt": length_delta,
            "mutation_count": len(mutations) + abs(length_delta),
            "substitution_count": len(mutations),
            "clean_substitution_only": clean,
            "covered_mutation_count": count,
            "mutation_list": ";".join(m.label for m in mutations),
            "covered_mutation_list": ";".join(m.label for m in covered_mutations),
            "esm2_binder_delta_nll_sum": binder_sum,
            "esm2_target_delta_nll_sum": target_sum,
            "esm2_target_minus_binder_delta_nll_sum": (
                target_sum - binder_sum
                if math.isfinite(float(target_sum)) and math.isfinite(float(binder_sum))
                else np.nan
            ),
            "esm2_binder_delta_nll_mean": binder_sum / count if count else np.nan,
            "esm2_target_delta_nll_mean": target_sum / count if count else np.nan,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def add_affinity_labels(df: pd.DataFrame, wt_sequence: str, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["is_wt_sequence"] = df["mata_sequence"].astype(str).eq(wt_sequence)
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
    df["affinity_label"] = "missing_wt_or_affinity"
    df.loc[df["delta_affinity_vs_wt"] <= -threshold, "affinity_label"] = "good"
    df.loc[df["delta_affinity_vs_wt"].abs() < threshold, "affinity_label"] = "neutral"
    df.loc[df["delta_affinity_vs_wt"] >= threshold, "affinity_label"] = "bad"
    group_cols = ["assay", "matalpha_description"]
    for col in [
        "alphaseq_affinity",
        "delta_affinity_vs_wt",
        "esm2_binder_delta_nll_sum",
        "esm2_target_delta_nll_sum",
        "esm2_target_minus_binder_delta_nll_sum",
    ]:
        df[f"{col}_z_by_target"] = df.groupby(group_cols, dropna=False)[col].transform(zscore)
    return df


def summarize(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for (assay, target), group in df.groupby(["assay", "matalpha_description"], dropna=False):
        scored = group[
            group["clean_substitution_only"]
            & group["alphaseq_affinity"].notna()
            & group["esm2_binder_delta_nll_sum"].notna()
            & group["esm2_target_delta_nll_sum"].notna()
            & (group["mutation_count"] > 0)
        ].copy()
        row: dict[str, object] = {
            "assay": assay,
            "target": target,
            "n_rows": int(len(group)),
            "n_scored_for_stats": int(len(scored)),
            "good_count": int((scored["affinity_label"] == "good").sum()),
            "neutral_count": int((scored["affinity_label"] == "neutral").sum()),
            "bad_count": int((scored["affinity_label"] == "bad").sum()),
            "wt_median_alphaseq_affinity": (
                float(group["wt_median_alphaseq_affinity"].dropna().iloc[0])
                if group["wt_median_alphaseq_affinity"].notna().any()
                else None
            ),
            "good_threshold_log10_kd": threshold,
        }
        for score_col in ["esm2_binder_delta_nll_sum", "esm2_target_delta_nll_sum"]:
            prefix = score_col.replace("_delta_nll_sum", "")
            row[f"{prefix}_spearman_vs_delta_affinity"] = safe_corr(
                scored, score_col, "delta_affinity_vs_wt", "spearman"
            )
            row[f"{prefix}_pearson_vs_delta_affinity"] = safe_corr(
                scored, score_col, "delta_affinity_vs_wt", "pearson"
            )
            row[f"{prefix}_auc_low_loss_good"] = auc_score(
                scored, score_col, low_loss_is_good=True
            )
            row[f"{prefix}_auc_high_loss_good"] = auc_score(
                scored, score_col, low_loss_is_good=False
            )
            enrichment = topk_enrichment(scored, score_col, 0.05)
            row[f"{prefix}_top_5pct_low_loss_good_enrichment"] = enrichment["good_enrichment"]
            row[f"{prefix}_top_5pct_low_loss_good_fraction"] = enrichment["good_fraction"]
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    meta = {
        "n_rows": int(len(df)),
        "target_context": "VHH <eos> RBD",
        "score_definition": "additive parent-context ESM2 masked-marginal delta NLL",
        "good_bad_definition": {
            "affinity_unit": "log10 Kd nM",
            "lower_alphaseq_affinity": "stronger binding",
            "good": f"delta_affinity_vs_wt <= -{threshold}",
            "bad": f"delta_affinity_vs_wt >= {threshold}",
            "neutral": f"|delta_affinity_vs_wt| < {threshold}",
        },
        "label_counts": {
            str(k): int(v) for k, v in df["affinity_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    return stats_df, meta


def plot_results(df: pd.DataFrame, stats_df: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = df[
        df["clean_substitution_only"]
        & df["delta_affinity_vs_wt"].notna()
        & df["esm2_binder_delta_nll_sum"].notna()
        & df["esm2_target_delta_nll_sum"].notna()
        & (df["mutation_count"] > 0)
    ].copy()
    if plot_df.empty:
        return
    colors = {"good": "#238b45", "neutral": "#7a869a", "bad": "#cb181d"}

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    for ax, score_col, title in [
        (axes[0], "esm2_binder_delta_nll_sum", "ESM2 binder-only"),
        (axes[1], "esm2_target_delta_nll_sum", "ESM2 target-conditioned"),
    ]:
        for label, sub in plot_df.groupby("affinity_label"):
            ax.scatter(
                sub[score_col],
                sub["delta_affinity_vs_wt"],
                s=16,
                alpha=0.45,
                linewidth=0,
                color=colors.get(label, "#333333"),
                label=label,
            )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.axhline(-0.3, color="#238b45", lw=0.8, ls="--")
        ax.axhline(0.3, color="#cb181d", lw=0.8, ls="--")
        ax.axvline(0.0, color="#333333", lw=0.8)
        ax.set_xlabel(score_col)
        ax.set_ylabel("AlphaSeq delta log10 Kd vs WT")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(stats_df))
    labels = [f"{row.assay}" for row in stats_df.itertuples()]
    axes[2].bar(
        x - 0.18,
        stats_df["esm2_binder_auc_low_loss_good"].astype(float),
        width=0.36,
        label="binder-only",
        color="#4a90a4",
    )
    axes[2].bar(
        x + 0.18,
        stats_df["esm2_target_auc_low_loss_good"].astype(float),
        width=0.36,
        label="target-conditioned",
        color="#d95f02",
    )
    axes[2].axhline(0.5, color="#333333", lw=0.8, ls="--")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("AUC, low loss predicts good")
    axes[2].set_title("Good vs bad classification")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Does target-conditioned ESM2 help on VHH72 AlphaSeq?", fontweight="bold")
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_comparison.png"), dpi=220)
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_comparison.pdf"))
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    meta: dict[str, object],
    output_prefix: Path,
    no_plots: bool,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scores_csv = output_prefix.with_name(f"{output_prefix.name}_variant_scores.csv")
    stats_csv = output_prefix.with_name(f"{output_prefix.name}_group_stats.csv")
    summary_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    df.to_csv(scores_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    summary_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {scores_csv}")
    print(f"wrote {stats_csv}")
    print(f"wrote {summary_json}")
    if not no_plots:
        plot_results(df, stats_df, output_prefix)
        print(f"wrote {output_prefix.with_name(f'{output_prefix.name}_comparison.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--assays", default=",".join(DEFAULT_ASSAYS))
    parser.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    parser.add_argument("--wt-sequence", default=None)
    parser.add_argument("--esm2-model", default=DEFAULT_ESM2_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-mode", choices=["eos", "linker"], default="eos")
    parser.add_argument("--linker", default="GGGGS")
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
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assays = [value.strip() for value in args.assays.split(",") if value.strip()]

    pl_df = load_alphaseq(args.repo_id, assays, args.target_regex)
    wt_sequence = infer_wt_sequence(pl_df, args.wt_sequence)
    rows = pl_df.filter(pl.col("mata_sequence").is_not_null()).to_dicts()
    df = pd.DataFrame.from_records(rows)
    target_sequences = sorted(df["matalpha_sequence"].dropna().astype(str).unique().tolist())
    if not target_sequences:
        raise ValueError("No target sequences found after filtering")

    all_positions: set[int] = set()
    for sequence in df["mata_sequence"].astype(str).unique().tolist():
        mutations, _, clean = compare_to_wt(sequence, wt_sequence)
        if clean:
            all_positions.update(m.position for m in mutations)
    positions = sorted(all_positions)
    if not positions:
        raise ValueError("No clean substitution positions found")
    print(f"WT length: {len(wt_sequence)}", flush=True)
    print(f"Target sequences: {len(target_sequences)}", flush=True)
    print(f"Scoring {len(positions)} mutated VHH positions: {positions[0]}-{positions[-1]}", flush=True)

    model, alphabet = load_esm2(args.esm2_model, device)
    binder_tokens = build_tokens(
        alphabet,
        wt_sequence,
        target_sequence=None,
        target_mode=args.target_mode,
        linker=args.linker,
    )
    print("Scoring binder-only context...", flush=True)
    binder_logp = masked_logprobs(
        model,
        alphabet,
        binder_tokens,
        positions,
        device=device,
        batch_size=args.batch_size,
    )

    target_logp_by_sequence = {}
    for target_sequence in target_sequences:
        context_tokens = build_tokens(
            alphabet,
            wt_sequence,
            target_sequence=target_sequence,
            target_mode=args.target_mode,
            linker=args.linker,
        )
        print(
            f"Scoring target-conditioned context, target length {len(target_sequence)}, "
            f"token length {context_tokens.shape[1]}...",
            flush=True,
        )
        target_logp_by_sequence[target_sequence] = masked_logprobs(
            model,
            alphabet,
            context_tokens,
            positions,
            device=device,
            batch_size=args.batch_size,
        )

    scored = score_variants(df, wt_sequence, binder_logp, target_logp_by_sequence)
    scored = add_affinity_labels(scored, wt_sequence, args.good_threshold_log10)
    stats_df, meta = summarize(scored, args.good_threshold_log10)
    meta.update({
        "repo_id": args.repo_id,
        "assays": assays,
        "target_regex": args.target_regex,
        "wt_sequence": wt_sequence,
        "wt_sequence_length": len(wt_sequence),
        "esm2_model": args.esm2_model,
        "device": device,
        "batch_size": args.batch_size,
        "target_mode": args.target_mode,
        "linker": args.linker if args.target_mode == "linker" else None,
        "scored_positions": positions,
        "target_sequence_count": len(target_sequences),
    })
    write_outputs(scored, stats_df, meta, args.output_prefix, args.no_plots)

    display_cols = [
        "assay",
        "target",
        "n_scored_for_stats",
        "good_count",
        "bad_count",
        "esm2_binder_spearman_vs_delta_affinity",
        "esm2_target_spearman_vs_delta_affinity",
        "esm2_binder_auc_low_loss_good",
        "esm2_target_auc_low_loss_good",
        "esm2_binder_auc_high_loss_good",
        "esm2_target_auc_high_loss_good",
        "esm2_binder_top_5pct_low_loss_good_enrichment",
        "esm2_target_top_5pct_low_loss_good_enrichment",
    ]
    print("\nTarget-conditioned ESM2 statistics:")
    print(stats_df[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
