"""Continuous seeding stage: run a staged logit optimizer, emit a discrete start.

Why this exists. BindCraft and Germinal both anneal the *sequence
parameterisation* -- logits, then soft, then straight-through, then one-hot --
rather than jumping straight to discrete moves, and the research review found
that this, not geometric loss weights, is what AF-design pipelines actually
schedule (docs/literature_review/README.md). The
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


def project_to_budget(relaxed, parent_tokens, designable_idx, budget):
    """Keep the `budget` designable substitutions the optimizer wanted most.

    Ranks candidate positions by the relaxed objective's own margin -- the
    logit it assigns its preferred residue minus the logit it assigns the
    parent's residue -- so the kept edits are the ones the continuous stage
    pushed hardest for, not an arbitrary subset. Returns tokens plus the
    per-position margins, since the ranking is the part worth auditing.
    """
    relaxed = np.asarray(relaxed)
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


def relaxed_array(result):
    """The relaxed sequence, whatever shape the optimizer wrapped it in.

    The three optimizers have DIFFERENT return arities, which is a real trap:
    without a `trajectory_fn`, `bindcraft_design` and `colabdesign_stage`
    return a bare (N, 20) array while `simplex_APGM` returns (x, best_x).
    Unpacking a bare array as a 2-tuple raises "too many values to unpack",
    which is how the bc_ cells failed on 2026-10-06 after completing all 125
    optimization steps. All of them are in probability space, not logits, so
    the margins in `project_to_budget` are probability differences.
    """
    if isinstance(result, tuple):
        return result[0]
    return result


def anchored_logits(parent_tokens, designable_idx, scale, noise):
    """Logits for the designable rows whose softmax concentrates on the parent.

    Pure noise initialization discards the one thing this project starts with:
    a parent sequence that the frozen predictor already scores. It also starts
    the optimizer outside its own feasible set. With 29 designable positions
    and `softmax` over 20 tokens, noise logits give the parent about 1/20 of
    the mass at every position, so the `EditBudget` expectation
    `E(s) = sum_pos (1 - p_parent)` opens at roughly 29 x 0.95 = 27.6 against a
    budget of 5 -- a hinge of `5.0 * relu(27.6 - 5) = 113`, which dwarfs the
    roughly 34 carried by every other term combined. The first many steps then
    spend their gradient walking back toward the parent.

    Adding `scale` to the parent column instead opens at `p_parent =
    e^scale / (e^scale + 19)`. At the default 5.0 that is 0.886, so
    `E(s) = 29 x 0.114 = 3.3`, inside the budget, and the hinge is inactive at
    step 0. Larger scales are more feasible but more saturated: the softmax
    Jacobian goes as `p(1 - p)`, which is 0.10 at scale 5 and 0.04 at scale 6.

    `noise` is supplied by the caller so this stays deterministic and testable.
    """
    noise = np.asarray(noise, dtype=np.float32)
    if noise.shape != (len(designable_idx), 20):
        raise ValueError(
            f"noise has shape {noise.shape}, expected ({len(designable_idx)}, 20)"
        )
    x = noise.copy()
    parent_rows = np.asarray(parent_tokens)[np.asarray(designable_idx)]
    x[np.arange(len(designable_idx)), parent_rows] += scale
    return x


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
    parser.add_argument("--init", choices=["parent", "noise"], default="parent",
                        help="anchor the start at the parent one-hot (default) "
                             "or use the pure-noise start the optimizers ship with")
    parser.add_argument("--init-logit-scale", type=float, default=5.0,
                        help="mass added to the parent column when --init=parent")
    args = parser.parse_args(argv)

    import jax
    import jax.numpy as jnp
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
    noise = 0.01 * np.asarray(jax.random.normal(init_key, (n_designable, 20)))
    if args.init == "parent":
        # Start inside the edit budget, at the sequence the predictor already
        # scores, rather than at BindCraft's de-novo noise start. See
        # `anchored_logits` for why the hinge makes that start expensive here.
        x0 = jnp.asarray(anchored_logits(
            parent, designable_idx, args.init_logit_scale, noise
        ))
    else:
        x0 = jnp.asarray(noise)

    print(f"Running {args.method} over {n_designable} designable positions...",
          flush=True)
    if args.method == "bindcraft":
        relaxed = relaxed_array(bindcraft_design(
            loss_function=variable_only_loss, x=x0, lr=args.lr, key=run_key,
        ))
    elif args.method == "colabdesign":
        relaxed = relaxed_array(colabdesign_stage(
            loss_function=variable_only_loss, x=x0,
            n_steps=args.steps or 120, soft_start=0.0, soft_end=1.0,
            temp_start=1.0, temp_end=0.01, hard=False, lr=args.lr, key=run_key,
        ))
    else:
        relaxed = relaxed_array(simplex_APGM(
            loss_function=variable_only_loss,
            x=jax.nn.softmax(x0, -1), n_steps=args.steps or 200,
            stepsize=APGM_STEPSIZE, momentum=APGM_MOMENTUM, scale=APGM_SCALE,
            key=run_key,
        ))
    relaxed = np.asarray(relaxed)
    if relaxed.shape != (n_designable, 20):
        raise ValueError(
            f"{args.method} returned shape {relaxed.shape}, expected "
            f"({n_designable}, 20); the return convention may have changed"
        )

    # Scatter the designable-only result back into full-length logits.
    # Non-designable rows are pinned to the parent residue, so argmax there
    # can only return the parent whatever scale the optimizer used.
    full = np.zeros((len(parent), 20), dtype=np.float32)
    full[np.arange(len(parent)), parent] = 1.0
    full[designable_idx] = np.asarray(relaxed)

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
        init=args.init, init_logit_scale=args.init_logit_scale,
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
