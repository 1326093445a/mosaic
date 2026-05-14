#!/usr/bin/env python3
"""Run a vanilla MULTI-evolve-style sequence test on VHH72.

This intentionally does not use the VHH72 paper off-rate table.  It uses the
MULTI-evolve PLM zero-shot scoring formula:

    score(mut) = log p(mutant residue | WT sequence) - log p(WT residue | WT sequence)

Higher score is better; for Mosaic-style minimization the corresponding loss is
``-score``.  The script reports:

  * all single-mutant ESM scores for the VHH chain,
  * the paper substitutions S56M/L97W/T99V under this vanilla score,
  * the best-scoring substitution at those same positions,
  * WT, paper, and vanilla-best combinations scored additively.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch


DEFAULT_PDB = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb")
DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_OUTPUT_PREFIX = Path("vhh/VHH72_vanilla_multievolve")
DEFAULT_VARIANTS = "S56M,L97W,T99V"
DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"

AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]


def read_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    structure = gemmi.read_structure(str(pdb_path))
    chain = structure[0][chain_id]
    seq = gemmi.one_letter_code([res.name for res in chain])
    if not seq:
        raise ValueError(f"No sequence found for chain {chain_id!r} in {pdb_path}")
    bad = sorted(set(seq) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"Unsupported residue code(s) in chain {chain_id}: {bad}")
    return seq


def load_variant_map(cdr_map: Path, variants: list[str]) -> list[dict]:
    by_variant = {}
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant"):
                by_variant[row["variant"].upper()] = row

    mapped = []
    for raw in variants:
        variant = raw.strip().upper()
        if not variant:
            continue
        row = by_variant.get(variant)
        if row is None:
            raise ValueError(f"Variant {raw!r} is absent from {cdr_map}")
        mapped.append({
            "variant": variant,
            "wt": variant[0],
            "target": variant[-1],
            "seq_index": int(row["seq_index"]),
            "anarci_label": row["anarci_label"],
            "pdb_auth_label": row["pdb_auth_label"],
        })
    if not mapped:
        raise ValueError("No variants were requested")
    return mapped


def score_esm_wt_marginals(wt_sequence: str, model_name: str) -> pd.DataFrame:
    """Reproduce MULTI-evolve's ESM wt-marginal logratio scores."""
    from esm import pretrained

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"[vanilla-multievolve] loading {model_name} on {device}", flush=True)
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("VHH72", wt_sequence)])
    batch_tokens = batch_tokens.to(device)

    print("[vanilla-multievolve] scoring WT marginals", flush=True)
    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
    token_probs = token_probs.detach().cpu().numpy()[0]

    rows = []
    for pos0, wt in enumerate(wt_sequence):
        wt_idx = alphabet.tok_to_idx[wt]
        wt_logp = float(token_probs[pos0 + 1, wt_idx])
        site_rows = []
        for aa in AMINO_ACIDS:
            if aa == wt:
                continue
            aa_idx = alphabet.tok_to_idx[aa]
            mt_logp = float(token_probs[pos0 + 1, aa_idx])
            score = mt_logp - wt_logp
            site_rows.append({
                "mutation": f"{wt}{pos0 + 1}{aa}",
                "seq_index": pos0 + 1,
                "wt": wt,
                "mut": aa,
                "esm_model": model_name,
                "vanilla_esm_logratio": score,
                "vanilla_esm_loss": -score,
                "wt_logp": wt_logp,
                "mut_logp": mt_logp,
                "aa_mutation": aa,
                "aa_substitution_type": f"{wt}-{aa}",
            })
        site_rows.sort(key=lambda row: row["vanilla_esm_logratio"], reverse=True)
        for rank, row in enumerate(site_rows, start=1):
            row["site_rank"] = rank
            rows.append(row)

    df = pd.DataFrame(rows)
    df["global_rank"] = df["vanilla_esm_logratio"].rank(
        method="first", ascending=False
    ).astype(int)

    for group_col in ("aa_mutation", "aa_substitution_type"):
        z_col = f"{group_col}_z_logratio"
        df[z_col] = np.nan
        for _, idx in df.groupby(group_col).groups.items():
            values = df.loc[idx, "vanilla_esm_logratio"].astype(float)
            if len(values) >= 5 and values.std(ddof=0) > 0:
                df.loc[idx, z_col] = (values - values.mean()) / values.std(ddof=0)

    return df.sort_values(
        ["global_rank", "seq_index", "mut"], ascending=[True, True, True]
    )


def mutation_from_row(row: pd.Series) -> str:
    return f"{row['wt']}{int(row['seq_index'])}{row['mut']}"


def candidate_score(label: str, mutations: list[str], dms_by_mut: dict[str, dict]) -> dict:
    if not mutations:
        return {
            "label": label,
            "mutations": "WT",
            "num_mutations": 0,
            "vanilla_esm_additive_score": 0.0,
            "vanilla_esm_additive_loss": -0.0,
            "aa_mutation_z_sum": 0.0,
            "aa_substitution_type_z_sum": 0.0,
            "per_mutation_scores": "",
            "per_mutation_site_ranks": "",
        }

    rows = [dms_by_mut[mut] for mut in mutations]
    score = float(sum(row["vanilla_esm_logratio"] for row in rows))
    aa_mut_z = float(np.nansum([row["aa_mutation_z_logratio"] for row in rows]))
    aa_sub_z = float(np.nansum([row["aa_substitution_type_z_logratio"] for row in rows]))
    return {
        "label": label,
        "mutations": "/".join(mutations),
        "num_mutations": len(mutations),
        "vanilla_esm_additive_score": score,
        "vanilla_esm_additive_loss": -score,
        "aa_mutation_z_sum": aa_mut_z,
        "aa_substitution_type_z_sum": aa_sub_z,
        "per_mutation_scores": ";".join(
            f"{mut}:{dms_by_mut[mut]['vanilla_esm_logratio']:+.4f}"
            for mut in mutations
        ),
        "per_mutation_site_ranks": ";".join(
            f"{mut}:rank{int(dms_by_mut[mut]['site_rank'])}"
            for mut in mutations
        ),
    }


def build_candidate_scores(
    variants: list[dict],
    dms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dms_by_mut = {
        mutation_from_row(row): row.to_dict()
        for _, row in dms.iterrows()
    }

    target_indices = {variant["seq_index"] for variant in variants}
    site_rows = []
    best_by_site = {}
    for variant in variants:
        site = int(variant["seq_index"])
        site_df = dms[dms["seq_index"] == site].sort_values(
            "vanilla_esm_logratio", ascending=False
        )
        best = site_df.iloc[0]
        best_mut = mutation_from_row(best)
        paper_mut = f"{variant['wt']}{site}{variant['target']}"
        best_by_site[site] = best_mut

        for _, row in site_df.iterrows():
            mut = mutation_from_row(row)
            site_rows.append({
                "anarci_label": variant["anarci_label"],
                "pdb_auth_label": variant["pdb_auth_label"],
                "seq_index": site,
                "mutation": mut,
                "is_paper_mutation": mut == paper_mut,
                "is_site_best": mut == best_mut,
                "vanilla_esm_logratio": row["vanilla_esm_logratio"],
                "vanilla_esm_loss": row["vanilla_esm_loss"],
                "site_rank": int(row["site_rank"]),
                "global_rank": int(row["global_rank"]),
                "aa_mutation_z_logratio": row["aa_mutation_z_logratio"],
                "aa_substitution_type_z_logratio": row["aa_substitution_type_z_logratio"],
            })

    known_muts = [
        f"{variant['wt']}{int(variant['seq_index'])}{variant['target']}"
        for variant in variants
    ]
    best_muts = [best_by_site[int(variant["seq_index"])] for variant in variants]

    candidate_rows = [candidate_score("WT", [], dms_by_mut)]

    for r in range(1, len(known_muts) + 1):
        for combo in itertools.combinations(known_muts, r):
            candidate_rows.append(
                candidate_score("paper_" + "+".join(combo), list(combo), dms_by_mut)
            )

    for r in range(1, len(best_muts) + 1):
        for combo in itertools.combinations(best_muts, r):
            candidate_rows.append(
                candidate_score("vanilla_best_" + "+".join(combo), list(combo), dms_by_mut)
            )

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = candidate_df.drop_duplicates("mutations")
    candidate_df = candidate_df.sort_values(
        ["vanilla_esm_additive_loss", "num_mutations", "label"],
        ascending=[True, True, True],
    )
    candidate_df["rank_by_vanilla_esm_loss"] = np.arange(1, len(candidate_df) + 1)

    site_df = pd.DataFrame(site_rows)
    site_df = site_df.sort_values(["seq_index", "site_rank"])

    missing = sorted(target_indices - set(site_df["seq_index"]))
    if missing:
        raise RuntimeError(f"No DMS rows found for target seq_index values: {missing}")

    return site_df, candidate_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = load_variant_map(
        args.cdr_map,
        [value.strip() for value in args.variants.split(",")],
    )
    wt_sequence = read_chain_sequence(args.pdb, args.binder_chain)
    for variant in variants:
        observed = wt_sequence[variant["seq_index"] - 1]
        if observed != variant["wt"]:
            raise ValueError(
                f"{variant['variant']} expects {variant['wt']} at "
                f"seq_index {variant['seq_index']}, observed {observed}"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dms = score_esm_wt_marginals(wt_sequence, args.esm_model)
    site_df, candidate_df = build_candidate_scores(variants, dms)

    dms_csv = args.output_prefix.with_name(args.output_prefix.name + "_dms.csv")
    sites_csv = args.output_prefix.with_name(args.output_prefix.name + "_target_sites.csv")
    candidates_csv = args.output_prefix.with_name(
        args.output_prefix.name + "_candidate_scores.csv"
    )
    dms.to_csv(dms_csv, index=False)
    site_df.to_csv(sites_csv, index=False)
    candidate_df.to_csv(candidates_csv, index=False)

    print(f"[vanilla-multievolve] wrote DMS: {dms_csv}", flush=True)
    print(f"[vanilla-multievolve] wrote target-site scores: {sites_csv}", flush=True)
    print(f"[vanilla-multievolve] wrote candidate scores: {candidates_csv}", flush=True)

    print("\nTarget-site summary:", flush=True)
    summary = site_df[site_df["is_paper_mutation"] | site_df["is_site_best"]].copy()
    print(summary.to_string(index=False), flush=True)

    print("\nTop candidate combinations by vanilla ESM loss:", flush=True)
    cols = [
        "rank_by_vanilla_esm_loss",
        "label",
        "mutations",
        "vanilla_esm_additive_score",
        "vanilla_esm_additive_loss",
        "per_mutation_site_ranks",
    ]
    print(candidate_df[cols].head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
