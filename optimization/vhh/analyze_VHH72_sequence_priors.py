#!/usr/bin/env python3
"""Score VHH72 paper mutations with the sequence priors used by Mosaic.

This is a benchmark diagnostic, not a design objective. It answers:

  - Does ESM2 prefer or penalize S56M, L97W, T99V, pairs, or the triple?
  - Does AbLang2 prefer or penalize the same sequences?
  - Under the current Mosaic sequence-only terms, which publication variants
    are naturally easy or hard before any structure/refold signal is considered?
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS
from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
from mosaic.losses.esm import ESM2PseudoLikelihood, load_esm2
from mosaic.optimizers import batched_value_eval


DEFAULT_PDB = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_relaxed.pdb")
DEFAULT_CDR_MAP = Path("optimization/vhh/VHH72_WT_SARS-CoV-2_RBD_cdr_map.csv")
DEFAULT_OUTPUT = Path("optimization/vhh/VHH72_sequence_prior_scores.csv")


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def numeric(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    structure = gemmi.read_structure(str(pdb_path))
    chain = structure[0][chain_id]
    seq = gemmi.one_letter_code([res.name for res in chain])
    if not seq:
        raise ValueError(f"No sequence found for chain {chain_id!r} in {pdb_path}")
    bad = sorted(set(seq) - set(TOKENS))
    if bad:
        raise ValueError(f"Unsupported residue code(s) in chain {chain_id}: {bad}")
    return seq


def default_variants_from_map(cdr_map: Path) -> list[str]:
    variants = []
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") == "known_variant" and row.get("variant"):
                variants.append(row["variant"])
    if not variants:
        raise ValueError(f"No known_variant rows found in {cdr_map}")
    return variants


def load_variant_map(cdr_map: Path, variants: list[str]) -> list[dict]:
    by_variant = {}
    with cdr_map.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant"):
                by_variant[row["variant"].upper()] = row

    mapped = []
    for raw in variants:
        variant = raw.strip().upper()
        row = by_variant.get(variant)
        if row is None:
            raise ValueError(f"Variant {raw!r} is absent from {cdr_map}")
        wt = variant[0]
        target = variant[-1]
        mapped.append({
            "variant": variant,
            "wt": wt,
            "target": target,
            "seq_index": int(row["seq_index"]),
            "anarci_label": row["anarci_label"],
            "pdb_auth_label": row["pdb_auth_label"],
        })
    return mapped


def apply_variants(wt_sequence: str, variants: list[dict]) -> str:
    seq = list(wt_sequence)
    for variant in variants:
        idx0 = variant["seq_index"] - 1
        if idx0 < 0 or idx0 >= len(seq):
            raise ValueError(
                f"{variant['variant']} maps to seq_index {variant['seq_index']}, "
                f"outside sequence length {len(seq)}"
            )
        observed = seq[idx0]
        if observed != variant["wt"]:
            raise ValueError(
                f"{variant['variant']} expects WT {variant['wt']} at "
                f"{variant['anarci_label']} seq_index {variant['seq_index']}, "
                f"but sequence has {observed}"
            )
        seq[idx0] = variant["target"]
    return "".join(seq)


def build_publication_variant_rows(wt_sequence: str, variants: list[dict]) -> list[dict]:
    rows = [{
        "source": "publication_grid",
        "label": "WT",
        "variant_combo": "WT",
        "variant_count": 0,
        "sequence": wt_sequence,
    }]
    for size in range(1, len(variants) + 1):
        for combo in itertools.combinations(variants, size):
            label = "+".join(v["variant"] for v in combo)
            rows.append({
                "source": "publication_grid",
                "label": label,
                "variant_combo": label,
                "variant_count": size,
                "sequence": apply_variants(wt_sequence, list(combo)),
            })
    return rows


def resolve_design_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    for name in ("mutation_recovery.csv", "combined_refold_ranked.csv", "refold_ranked.csv"):
        candidate = path / name
        if candidate.exists():
            return [candidate]
    return sorted(path.rglob("refold_ranked.csv"))


def sort_design_rows(rows: list[dict]) -> list[dict]:
    def key(row):
        rank = str(row.get("rank", "")).strip()
        return (
            not truthy(row.get("rmsd_pass")),
            rank == "",
            numeric(rank, 1e9),
            -numeric(row.get("ipsae_min"), -1e9),
            -numeric(row.get("iptm"), -1e9),
        )

    return sorted(rows, key=key)


def read_design_rows(path: Path, wt_len: int, max_designs: int) -> list[dict]:
    if max_designs <= 0:
        return []
    csv_paths = resolve_design_csvs(path)
    if not csv_paths:
        raise FileNotFoundError(f"No design CSVs found under {path}")

    raw_rows = []
    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                seq = row.get("sequence", "").strip()
                if not seq:
                    continue
                if len(seq) != wt_len:
                    print(
                        f"warn: skipping sequence length {len(seq)} from {csv_path}; "
                        f"expected {wt_len}",
                        flush=True,
                    )
                    continue
                raw_rows.append({**row, "source_csv": str(csv_path)})

    rows = []
    seen = set()
    for row in sort_design_rows(raw_rows):
        seq = row["sequence"]
        if seq in seen:
            continue
        seen.add(seq)
        label_parts = ["design"]
        if row.get("rank"):
            label_parts.append(f"rank_{row['rank']}")
        if row.get("edit_count"):
            label_parts.append(f"edit_{row['edit_count']}")
        if row.get("sample_idx"):
            label_parts.append(f"sample_{row['sample_idx']}")
        rows.append({
            "source": "design_csv",
            "label": "_".join(label_parts),
            "variant_combo": "",
            "variant_count": "",
            "sequence": seq,
            "source_csv": row.get("source_csv", ""),
            "rank": row.get("rank", ""),
            "edit_count": row.get("edit_count", ""),
            "sample_idx": row.get("sample_idx", ""),
            "ipsae_min": row.get("ipsae_min", ""),
            "iptm": row.get("iptm", ""),
            "rmsd_pass": row.get("rmsd_pass", ""),
        })
        if len(rows) >= max_designs:
            break
    return rows


def tokenize_sequences(sequences: list[str]) -> np.ndarray:
    return np.asarray(
        [[TOKENS.index(aa) for aa in seq] for seq in sequences],
        dtype=np.int32,
    )


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


def add_delta(rows: list[dict], column: str, delta_column: str) -> None:
    wt = None
    for row in rows:
        if row["label"] == "WT" and row.get(column, "") != "":
            wt = float(row[column])
            break
    if wt is None or math.isnan(wt):
        return
    for row in rows:
        if row.get(column, "") == "":
            row[delta_column] = ""
        else:
            row[delta_column] = float(row[column]) - wt


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    publication_rows = [row for row in rows if row["source"] == "publication_grid"]
    print("\nPublication mutation grid:")
    for row in publication_rows:
        parts = [row["label"]]
        if row.get("esm2_delta_nll", "") != "":
            parts.append(f"ESM2_dNLL={float(row['esm2_delta_nll']):+.4f}")
        if row.get("ablang2_delta_nll", "") != "":
            parts.append(f"AbLang2_dNLL={float(row['ablang2_delta_nll']):+.4f}")
        if row.get("mosaic_sequence_delta_loss", "") != "":
            parts.append(
                f"weighted_dloss={float(row['mosaic_sequence_delta_loss']):+.4f}"
            )
        print("  " + "  ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--cdr-map", type=Path, default=DEFAULT_CDR_MAP)
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants. Defaults to known_variant rows in the CDR map.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--esm2-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--skip-esm2", action="store_true")
    parser.add_argument("--skip-ablang2", action="store_true")
    parser.add_argument("--weight-esm2", type=float, default=0.10)
    parser.add_argument("--weight-ablang2", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--design-input",
        type=Path,
        default=None,
        help="Optional design output directory or CSV to score alongside the paper grid.",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=0,
        help="Maximum unique design sequences to score from --design-input.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    wt_sequence = read_chain_sequence(args.pdb, args.binder_chain)
    raw_variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else default_variants_from_map(args.cdr_map)
    )
    variants = load_variant_map(args.cdr_map, raw_variants)
    rows = build_publication_variant_rows(wt_sequence, variants)
    if args.design_input is not None:
        rows.extend(read_design_rows(args.design_input, len(wt_sequence), args.max_designs))

    seq_ids = tokenize_sequences([row["sequence"] for row in rows])

    if not args.skip_esm2:
        print(f"Loading ESM2 model {args.esm2_model}...", flush=True)
        esm2_loss = ESM2PseudoLikelihood(load_esm2(args.esm2_model))
        print("Scoring ESM2 pseudo-likelihood...", flush=True)
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
        print("Scoring AbLang2 pseudo-likelihood...", flush=True)
        vals, aux = score_loss(
            ablang2_loss, seq_ids, batch_size=args.batch_size, seed=args.seed + 1
        )
        ablang2_ppl = aux_vector(aux, "ablang2_ppl")
        for row, val, ppl in zip(rows, vals, ablang2_ppl):
            row["ablang2_nll"] = val
            row["ablang2_ppl"] = ppl

    for row in rows:
        total = 0.0
        have = False
        if row.get("esm2_nll", "") != "":
            total += args.weight_esm2 * float(row["esm2_nll"])
            have = True
        if row.get("ablang2_nll", "") != "":
            total += args.weight_ablang2 * float(row["ablang2_nll"])
            have = True
        if have:
            row["mosaic_sequence_loss"] = total

    add_delta(rows, "esm2_nll", "esm2_delta_nll")
    add_delta(rows, "ablang2_nll", "ablang2_delta_nll")
    add_delta(rows, "mosaic_sequence_loss", "mosaic_sequence_delta_loss")

    write_csv(args.output_csv, rows)
    print_summary(rows)
    print(f"\nWrote sequence-prior scores: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
