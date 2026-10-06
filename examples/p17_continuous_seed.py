"""Continuous seeding stage: run a staged logit optimizer, emit a discrete start.

Why this exists. BindCraft and Germinal both anneal the *sequence
parameterisation* -- logits, then soft, then straight-through, then one-hot --
rather than jumping straight to discrete moves, and the research review found
that this, not geometric loss weights, is what AF-design pipelines actually
schedule (research_notes/Epitope targeting in binder design/README.md). The
confidence-search harness does discrete single substitutions from a fixed
start and has no continuous phase at all. `src/mosaic/optimizers.py` has
carried `bindcraft_design` and `colabdesign_stage` since before this project
started and section 8.1 records both as never used.

They cannot be a `--policy` for that harness: they are logits-in/logits-out
continuous optimizers, while the harness needs a cheap-loss callback plus a
separate confidence callback. So they run *before* it, as a seeding stage, and
hand over a start sequence -- the two-stage shape section 5's older pipeline
already uses with `simplex_APGM`.

The edit budget is the complication. None of these optimizers enforces a
WT-distance cap; the composite objective only carries `EditBudget` as a soft
hinge. So the relaxed result is projected onto the budget afterwards, keeping
the `budget` designable positions whose substitution the optimizer wanted most.
That projection is a real approximation and is recorded in the output.

Loads models, runs no search, writes a sequence.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

METHODS = ("apgm", "bindcraft", "colabdesign")


def project_to_budget(relaxed_logits, parent_tokens, designable_idx, budget):
    """Keep the `budget` designable substitutions the optimizer wanted most.

    Ranks candidate positions by the relaxed objective's own margin -- the
    logit it assigns its preferred residue minus the logit it assigns the
    parent's residue -- so the kept edits are the ones the continuous stage
    pushed hardest for, not an arbitrary subset. Returns tokens plus the
    per-position margins, since the ranking is the part worth auditing.
    """
    relaxed = np.asarray(relaxed_logits)
    preferred = relaxed.argmax(-1)
    margins = []
    for pos in designable_idx:
        pos = int(pos)
        if preferred[pos] == parent_tokens[pos]:
            continue
        margin = float(relaxed[pos, preferred[pos]] - relaxed[pos, parent_tokens[pos]])
        margins.append((margin, pos, int(preferred[pos])))
    margins.sort(reverse=True)
    kept = margins[:budget]
    tokens = np.array(parent_tokens, dtype=np.int32)
    for _, pos, residue in kept:
        tokens[pos] = residue
    return tokens, margins, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edit-budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None,
                        help="continuous steps; method default if omitted")
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--weight-registry", type=float, default=0.0)
    parser.add_argument("--registry-contact-distance", type=float, default=8.0)
    parser.add_argument("--weight-pose", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args(argv)

    import jax
    from mosaic.common import TOKENS
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.optimizers import bindcraft_design, colabdesign_stage, simplex_APGM
    from mosaic.structure_prediction import TargetChain
    from p17_hallucination_search import (
        APGM_MOMENTUM, APGM_SCALE, APGM_STEPSIZE, CDR_RESIDUE_INDICES_1IDX,
        HOTSPOT_TARGET_RESIDUE_INDICES_1IDX, build_composite_losses, load_structure,
        reference_binder_target_ca, reference_binder_target_ca_distances,
    )

    model, binder_seq, target_seq = load_structure()
    parent = np.array([TOKENS.index(a) for a in binder_seq], dtype=np.int32)
    mask = np.array([i + 1 in CDR_RESIDUE_INDICES_1IDX for i in range(len(parent))])
    designable_idx = np.flatnonzero(mask)
    epitope_idx = np.array(sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX))
    references = reference_binder_target_ca_distances(model)
    binder_ca, target_ca = reference_binder_target_ca(model)

    print(f"Loading frozen OpenDDE ({args.opendde_dtype}) and AbLang2...", flush=True)
    from mosaic.losses.ablang2 import load_ablang2

    opendde = OpenDDEModelAbag(compute_precision=args.opendde_dtype)
    features, _ = opendde.binder_features(
        len(parent), [TargetChain(target_seq, use_msa=False)]
    )
    ablang_model, ablang_tokenizer = load_ablang2()

    _, variable_only_loss = build_composite_losses(
        opendde=opendde, features=features, ablang2_model=ablang_model,
        ablang2_tokenizer=ablang_tokenizer, reference_distances=references,
        reference_binder_ca=binder_ca, reference_target_ca=target_ca,
        binder_seq=binder_seq, designable_idx=designable_idx,
        epitope_idx=epitope_idx, edit_budget=args.edit_budget,
        stop_grad_ablang2=False, opendde_path="full",
        pose_tolerance=0.0, opendde_sampling_steps=args.sampling_steps,
        opendde_num_samples=1, confidence_loss=None,
        pose_weight=args.weight_pose,
        registry_weight=args.weight_registry,
        registry_contact_distance=args.registry_contact_distance,
    )

    key = jax.random.key(args.seed)
    init_key, run_key = jax.random.split(key)
    n_designable = len(designable_idx)
    # BindCraft's own initialization; the other methods tolerate it too.
    x0 = 0.01 * jax.random.normal(init_key, (n_designable, 20))

    print(f"Running {args.method} over {n_designable} designable positions...",
          flush=True)
    if args.method == "bindcraft":
        logits, _ = bindcraft_design(
            loss_function=variable_only_loss, x=x0, lr=args.lr, key=run_key,
        )
    elif args.method == "colabdesign":
        logits, _ = colabdesign_stage(
            loss_function=variable_only_loss, x=x0,
            n_steps=args.steps or 120, soft_start=0.0, soft_end=1.0,
            temp_start=1.0, temp_end=0.01, hard=False, lr=args.lr, key=run_key,
        )
    else:
        logits, _ = simplex_APGM(
            loss_function=variable_only_loss,
            x=jax.nn.softmax(x0, -1), n_steps=args.steps or 200,
            stepsize=APGM_STEPSIZE, momentum=APGM_MOMENTUM, scale=APGM_SCALE,
            key=run_key,
        )

    # Scatter the designable-only result back into full-length logits.
    full = np.full((len(parent), 20), -1e4, dtype=np.float32)
    full[np.arange(len(parent)), parent] = 0.0
    full[designable_idx] = np.asarray(logits)

    tokens, margins, kept = project_to_budget(full, parent, designable_idx,
                                              args.edit_budget)
    sequence = "".join(TOKENS[i] for i in tokens)
    edits = [
        dict(position_1idx=pos + 1, parent=TOKENS[parent[pos]],
             seeded=TOKENS[res], margin=round(margin, 4))
        for margin, pos, res in kept
    ]
    hamming = int((tokens != parent).sum())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        method=args.method, seed=args.seed, edit_budget=args.edit_budget,
        weight_registry=args.weight_registry, weight_pose=args.weight_pose,
        sampling_steps=args.sampling_steps, opendde_dtype=args.opendde_dtype,
        parent_sequence=binder_seq, seed_sequence=sequence,
        hamming_from_parent=hamming, edits=edits,
        candidate_substitutions_considered=len(margins),
        interpretation=(
            "Discrete start for the population search, projected onto the edit "
            "budget from a continuous stage that does not enforce one. The "
            "projection keeps the highest-margin substitutions and is an "
            "approximation of the relaxed optimum, not the relaxed optimum."
        ),
    ), indent=2) + "\n")

    print(f"\nseed sequence ({hamming} edits from WT, budget {args.edit_budget}):")
    print(f"  {sequence}")
    for e in edits:
        print(f"    {e['parent']}{e['position_1idx']}{e['seeded']}  margin {e['margin']}")
    if len(margins) > args.edit_budget:
        print(f"  ({len(margins)} substitutions wanted, {args.edit_budget} kept)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
