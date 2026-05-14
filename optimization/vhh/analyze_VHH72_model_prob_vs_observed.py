#!/usr/bin/env python3
"""Compare VHH72 CDR model probabilities against observed Mosaic mutations.

This script answers a different question than the loss heatmap:

  * For each CDR site, what amino-acid distribution does ESM2 predict?
  * What amino-acid distribution does AbLang2 predict?
  * Are the paper mutations low-probability or high-probability at their sites?
  * Given those probabilities, are the mutations observed in the vanilla Mosaic
    run less or more often than expected under the sequence priors?

The probabilities are direct masked-marginal probabilities from each model on
the WT VHH72 sequence.  The combined distribution is the normalized geometric
mean of ESM2 and AbLang2 probabilities, which preserves the equal-weight rank
without applying Mosaic's arbitrary optimizer scale.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mosaic.common import TOKENS


DEFAULT_PDB = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb")
DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_RECOVERY = Path("VHH72_benchmark_both_200/mutation_recovery.csv")
DEFAULT_OUTPUT_PREFIX = Path("vhh/VHH72_model_prob_vs_observed")
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


def log_softmax_np(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_value = np.max(values)
    shifted = values - max_value
    return shifted - np.log(np.exp(shifted).sum())


def normalized_geomean_logprob(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    return log_softmax_np(0.5 * (np.asarray(logp_a) + np.asarray(logp_b)))


def entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def direct_esm2_logprobs(
    sequence: str,
    cdr_positions: list[dict],
    model_name: str,
    device: str,
) -> dict[int, np.ndarray]:
    """Masked-marginal ESM2 probabilities using the fair-esm torch model."""
    import esm

    print(f"Loading ESM2 model {model_name} on {device}...", flush=True)
    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"Unknown ESM2 model {model_name!r}")
    model, alphabet = loader()
    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("VHH72", sequence)])
    batch_tokens = batch_tokens.to(device)
    aa_indices = torch.as_tensor([alphabet.tok_to_idx[aa] for aa in AA_ORDER], device=device)

    result = {}
    with torch.no_grad():
        for position in cdr_positions:
            seq_index = int(position["seq_index"])
            token_index = seq_index  # BOS token shifts chain position to same 1-based index.
            masked = batch_tokens.clone()
            masked[0, token_index] = alphabet.mask_idx
            logits = model(masked)["logits"][0, token_index]
            logp = torch.log_softmax(logits, dim=-1)
            result[seq_index] = logp.index_select(0, aa_indices).cpu().numpy()
    return result


def direct_ablang2_logprobs(
    sequence: str,
    cdr_positions: list[dict],
    device: str,
) -> dict[int, np.ndarray]:
    """Masked-marginal AbLang2 probabilities using the native torch model."""
    from ablang2.load_model import load_model

    print(f"Loading AbLang2 model on {device}...", flush=True)
    model, tokenizer, _ = load_model("ablang2-paired", device=device)
    model.eval()
    model.to(device)
    at = tokenizer.aa_to_token
    token_ids = [at["<"], *[at[aa] for aa in sequence], at[">"], at["|"]]
    tokens = torch.as_tensor(token_ids, dtype=torch.long, device=device)[None]
    aa_indices = torch.as_tensor([at[aa] for aa in AA_ORDER], dtype=torch.long, device=device)
    special_indices = torch.as_tensor(tokenizer.all_special_tokens, dtype=torch.long, device=device)

    result = {}
    with torch.no_grad():
        for position in cdr_positions:
            seq_index = int(position["seq_index"])
            token_index = seq_index  # initial "<" shifts chain position to same 1-based index.
            masked = tokens.clone()
            masked[0, token_index] = at["*"]
            logits = model(masked)[0, token_index].clone()
            logits.index_fill_(0, special_indices, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            result[seq_index] = logp.index_select(0, aa_indices).cpu().numpy()
    return result


def observed_counts(
    recovery_csv: Path,
    cdr_positions: list[dict],
    sequence_length: int,
) -> dict[tuple[int, str], dict[str, float]]:
    if recovery_csv is None or not recovery_csv.exists():
        return {}
    df = pd.read_csv(recovery_csv)
    df = df[df["sequence"].astype(str).str.len() == sequence_length].copy()
    df_unique = df.drop_duplicates("sequence").copy()

    counts: dict[tuple[int, str], dict[str, float]] = {}
    for position in cdr_positions:
        seq_index = int(position["seq_index"])
        idx0 = seq_index - 1
        for aa in AA_ORDER:
            all_count = int((df["sequence"].str[idx0] == aa).sum())
            unique_count = int((df_unique["sequence"].str[idx0] == aa).sum())
            counts[(seq_index, aa)] = {
                "observed_count_all": all_count,
                "observed_freq_all": all_count / max(len(df), 1),
                "observed_n_all": len(df),
                "observed_count_unique": unique_count,
                "observed_freq_unique": unique_count / max(len(df_unique), 1),
                "observed_n_unique": len(df_unique),
            }
    return counts


def build_probability_table(
    sequence: str,
    cdr_positions: list[dict],
    known_variants: dict[tuple[int, str], str],
    esm_logp: dict[int, np.ndarray],
    ablang_logp: dict[int, np.ndarray],
    obs: dict[tuple[int, str], dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for position in cdr_positions:
        seq_index = int(position["seq_index"])
        wt = position["wt"]
        if sequence[seq_index - 1] != wt:
            raise ValueError(f"WT mismatch at seq_index {seq_index}")

        esm_site = np.asarray(esm_logp[seq_index], dtype=float)
        ablang_site = np.asarray(ablang_logp[seq_index], dtype=float)
        combined_site = normalized_geomean_logprob(esm_site, ablang_site)

        site_arrays = {
            "esm2": esm_site,
            "ablang2": ablang_site,
            "combined_geomean": combined_site,
        }
        ranks = {}
        for name, values in site_arrays.items():
            order = np.argsort(-values)
            ranks[name] = np.empty(len(order), dtype=int)
            ranks[name][order] = np.arange(1, len(order) + 1)

        entropy = {
            name: entropy_from_probs(np.exp(values))
            for name, values in site_arrays.items()
        }

        for aa_idx, aa in enumerate(AA_ORDER):
            paper_variant = known_variants.get((seq_index, aa), "")
            row = {
                "cdr": position["cdr"],
                "anarci_label": position["anarci_label"],
                "pdb_auth_label": position["pdb_auth_label"],
                "seq_index": seq_index,
                "wt": wt,
                "aa": aa,
                "is_wt": aa == wt,
                "paper_variant": paper_variant,
                "is_paper_variant": bool(paper_variant),
                "mutation_chain_index": f"{wt}{seq_index}{aa}" if aa != wt else "WT",
            }
            for name, values in site_arrays.items():
                prob = float(math.exp(values[aa_idx]))
                row[f"{name}_logprob"] = float(values[aa_idx])
                row[f"{name}_prob"] = prob
                row[f"{name}_rank"] = int(ranks[name][aa_idx])
                row[f"{name}_site_entropy"] = entropy[name]
                row[f"{name}_effective_aa"] = math.exp(entropy[name])
                row[f"{name}_max_prob"] = float(math.exp(values.max()))
            row.update(obs.get((seq_index, aa), {
                "observed_count_all": 0,
                "observed_freq_all": 0.0,
                "observed_n_all": 0,
                "observed_count_unique": 0,
                "observed_freq_unique": 0.0,
                "observed_n_unique": 0,
            }))
            rows.append(row)
    return pd.DataFrame(rows)


def add_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    for model in ["esm2", "ablang2", "combined_geomean"]:
        for scope in ["all", "unique"]:
            n_col = f"observed_n_{scope}"
            count_col = f"observed_count_{scope}"
            prob_col = f"{model}_prob"
            expected_col = f"{model}_expected_count_{scope}"
            ratio_col = f"{model}_observed_over_expected_{scope}"
            z_col = f"{model}_binomial_z_{scope}"
            df[expected_col] = df[prob_col] * df[n_col]
            expected = df[expected_col].astype(float)
            observed = df[count_col].astype(float)
            prob = df[prob_col].astype(float)
            n = df[n_col].astype(float)
            denom = np.sqrt(np.maximum(n * prob * (1.0 - prob), 1e-12))
            df[z_col] = (observed - expected) / denom
            df[ratio_col] = np.where(expected > 0, observed / expected, np.nan)
    return df


def site_distribution_summary(prob_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seq_index, model), sub in prob_df.groupby(["seq_index", lambda idx: 0]):
        del model
        first = sub.iloc[0]
        row = {
            "cdr": first["cdr"],
            "anarci_label": first["anarci_label"],
            "seq_index": int(first["seq_index"]),
            "wt": first["wt"],
        }
        for name in ["esm2", "ablang2", "combined_geomean"]:
            best = sub.sort_values(f"{name}_prob", ascending=False).iloc[0]
            row[f"{name}_top_aa"] = best["aa"]
            row[f"{name}_top_prob"] = best[f"{name}_prob"]
            row[f"{name}_wt_prob"] = float(sub[sub["is_wt"]][f"{name}_prob"].iloc[0])
            row[f"{name}_entropy"] = first[f"{name}_site_entropy"]
            row[f"{name}_effective_aa"] = first[f"{name}_effective_aa"]
        rows.append(row)
    return pd.DataFrame(rows)


def paper_summary(prob_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "paper_variant",
        "cdr",
        "anarci_label",
        "pdb_auth_label",
        "seq_index",
        "wt",
        "aa",
        "esm2_prob",
        "esm2_rank",
        "ablang2_prob",
        "ablang2_rank",
        "combined_geomean_prob",
        "combined_geomean_rank",
        "observed_count_all",
        "observed_freq_all",
        "observed_n_all",
        "observed_count_unique",
        "observed_freq_unique",
        "observed_n_unique",
        "esm2_expected_count_all",
        "esm2_observed_over_expected_all",
        "ablang2_expected_count_all",
        "ablang2_observed_over_expected_all",
        "combined_geomean_expected_count_all",
        "combined_geomean_observed_over_expected_all",
    ]
    return prob_df[prob_df["is_paper_variant"].astype(bool)][cols].copy()


def plot_paper_expected(summary: pd.DataFrame, output_prefix: Path) -> None:
    labels = summary["paper_variant"].tolist()
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = [
        ("ESM2 expected", summary["esm2_expected_count_all"], "#457b9d"),
        ("AbLang2 expected", summary["ablang2_expected_count_all"], "#2a9d8f"),
        ("Combined expected", summary["combined_geomean_expected_count_all"], "#f4a261"),
        ("Observed", summary["observed_count_all"], "#d1495b"),
    ]
    for i, (label, values, color) in enumerate(bars):
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count among ranked Mosaic rows")
    ax.set_title("Expected versus observed VHH72 paper mutations")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for i, (_, values, _) in enumerate(bars):
        for xi, value in zip(x + (i - 1.5) * width, values):
            ax.text(xi, float(value) + 0.6, f"{float(value):.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def plot_site_distributions(prob_df: pd.DataFrame, output_prefix: Path) -> None:
    paper = prob_df[prob_df["is_paper_variant"].astype(bool)].copy()
    n = len(paper)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(AA_ORDER))
    width = 0.35
    for ax, (_, row) in zip(axes, paper.iterrows()):
        sub = prob_df[prob_df["seq_index"] == row["seq_index"]].copy()
        ax.bar(
            x - width / 2,
            sub["esm2_prob"],
            width,
            label="ESM2",
            color="#457b9d",
            edgecolor="black",
            linewidth=0.3,
        )
        ax.bar(
            x + width / 2,
            sub["ablang2_prob"],
            width,
            label="AbLang2",
            color="#2a9d8f",
            edgecolor="black",
            linewidth=0.3,
        )
        aa_idx = AA_ORDER.index(row["aa"])
        ax.axvline(aa_idx, color="#d1495b", linestyle="--", linewidth=1.2)
        ax.text(
            aa_idx + 0.2,
            ax.get_ylim()[1] * 0.86,
            f"{row['paper_variant']}\nobs {int(row['observed_count_all'])}/{int(row['observed_n_all'])}",
            color="#d1495b",
            fontsize=8,
        )
        ax.set_ylabel("masked probability")
        ax.set_title(
            f"{row['paper_variant']} at {row['anarci_label']} "
            f"(ESM2 rank {int(row['esm2_rank'])}, AbLang2 rank {int(row['ablang2_rank'])})"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(AA_ORDER)
    axes[0].legend(frameon=False, loc="upper right")
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
    parser.add_argument("--recovery-csv", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument(
        "--cdr-groups",
        default="CDR1,CDR2,CDR3",
        help="Comma-separated CDR map groups to analyze.",
    )
    parser.add_argument("--esm2-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for direct masked probabilities. Default: cpu.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = {value.strip() for value in args.cdr_groups.split(",") if value.strip()}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    sequence = read_chain_sequence(args.pdb, args.binder_chain)
    cdr_positions = load_cdr_positions(args.cdr_map, groups)
    known_variants = load_known_variants(args.cdr_map)
    obs = observed_counts(args.recovery_csv, cdr_positions, len(sequence))

    esm_logp = direct_esm2_logprobs(sequence, cdr_positions, args.esm2_model, args.device)
    ablang_logp = direct_ablang2_logprobs(sequence, cdr_positions, args.device)

    prob_df = build_probability_table(
        sequence, cdr_positions, known_variants, esm_logp, ablang_logp, obs
    )
    prob_df = add_expected_columns(prob_df)

    prob_csv = args.output_prefix.with_name(f"{args.output_prefix.name}_probabilities.csv")
    prob_df.to_csv(prob_csv, index=False)
    print(f"wrote {prob_csv}")

    site_csv = args.output_prefix.with_name(f"{args.output_prefix.name}_site_summary.csv")
    site_distribution_summary(prob_df).to_csv(site_csv, index=False)
    print(f"wrote {site_csv}")

    paper = paper_summary(prob_df)
    paper_csv = args.output_prefix.with_name(f"{args.output_prefix.name}_paper_summary.csv")
    paper.to_csv(paper_csv, index=False)
    print(f"wrote {paper_csv}")
    print("\nPaper mutation probability summary:")
    print(paper.to_string(index=False))

    if not args.no_plots:
        plot_paper_expected(
            paper,
            args.output_prefix.with_name(f"{args.output_prefix.name}_paper_expected_vs_observed"),
        )
        plot_site_distributions(
            prob_df,
            args.output_prefix.with_name(f"{args.output_prefix.name}_paper_site_distributions"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
