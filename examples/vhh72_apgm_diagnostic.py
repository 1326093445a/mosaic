"""Diagnostic for section 13's `nnz` finding: every real cluster sweep run
had simplex_APGM stuck at nnz=1.00 (fully collapsed to a single amino acid
per position) for all 200 iterations, and its continuous argmax was
byte-identical to WT in every run -- meaning the entire continuous
relaxation phase was a complete no-op; all real movement in the sweep came
from the discrete greedy/MCMC search phase alone. See
docs/guidance_alphaseq_testing_notes.md section 13 (nnz finding) and 13.1
(real sweep results this explains).

Two suspected causes, both addressed here:
  1. vhh72_hallucination_search.py initializes x0 as an exact one-hot
     vertex of the simplex (seq_to_one_hot(wt_seq)) -- APGM starts already
     collapsed, with nowhere to explore from at step 0.
  2. APGM_SCALE=1.2 (>1.0) is deliberately sparsity-encouraging (see that
     module's own docstring) -- combined with (1), there's no pressure to
     ever leave the vertex.

This script runs simplex_APGM ALONE (no discrete search afterward) with:
  - softened initialization: WT amino acid = --init-wt-prob (default 0.80),
    each of the other 19 amino acids = (1 - --init-wt-prob) / 19 (default
    ~0.0105), instead of an exact 1.0/0.0 one-hot vertex.
  - --scale (default 1.0, NOT the production 1.2) -- isolates whether
    softened init alone is enough, or whether scale also needs to change.
  - the exact same cheap distogram-only composite loss
    vhh72_hallucination_search.py's build_composite_losses builds --
    deliberately NOT touching the OpenDDE path in this diagnostic, so a fix
    here isn't confounded with a signal-quality change (see
    docs/guidance_alphaseq_testing_notes.md section 13.4's "not yet built"
    scoping for the deeper OpenDDE path).

Needs the full, uncropped complex (334 residues) -- OOMs on a 24GB GPU (a
single gradient transpose alone needs ~30GB, confirmed directly). Run this
on the cluster, not locally -- same memory ceiling as the original sweep
(docs/guidance_alphaseq_testing_notes.md section 9b).

Success criteria (from the reviewed diagnostic plan): initial nnz > 1,
nnz changes over iterations (does not stay pinned at any single value),
continuous argmax differs from WT at least somewhere, and APGM's own loss
improves over the run. simplex_APGM already logs nnz/loss every iteration
to stdout (src/mosaic/optimizers.py's _print_iter) -- no extra logging
needed here, just watch/grep the run's stdout or --log-file.

Usage:
    .venv/bin/python examples/vhh72_apgm_diagnostic.py \\
        --init-wt-prob 0.80 --scale 1.0 --n-steps 200 --seed 0 \\
        2>&1 | tee results/apgm_diagnostic_seed0.log
"""
import argparse

import jax
import numpy as np

import vhh72_hallucination_search as search
from mosaic.common import TOKENS
from mosaic.models.opendde import OpenDDEModelAbag
from mosaic.losses.ablang2 import load_ablang2
from mosaic.optimizers import simplex_APGM
from mosaic.structure_prediction import TargetChain


def softened_init(binder_seq: str, designable_idx: np.ndarray, wt_prob: float) -> np.ndarray:
    other_prob = (1.0 - wt_prob) / (len(TOKENS) - 1)
    x0 = np.full((len(designable_idx), len(TOKENS)), other_prob, dtype=np.float32)
    for i, pos in enumerate(designable_idx):
        x0[i, TOKENS.index(binder_seq[pos])] = wt_prob
    return x0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-wt-prob", type=float, default=0.80,
                    help="probability mass on the WT amino acid at each designable "
                         "position; remaining (1 - this) split evenly over the other "
                         "19 amino acids. 1.0 reproduces the exact one-hot vertex init "
                         "the real sweep used.")
    p.add_argument("--scale", type=float, default=1.0,
                    help="simplex_APGM scale; production sweep used 1.2 (sparsity-"
                         "encouraging). 1.0 = no extra sparsity pressure.")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"=== APGM-only diagnostic: init_wt_prob={args.init_wt_prob} "
          f"scale={args.scale} n_steps={args.n_steps} seed={args.seed} ===", flush=True)

    model, binder_seq, target_seq = search.load_structure()
    reference_distances = search.reference_binder_target_ca_distances(model)

    designable_mask = np.zeros(len(binder_seq), dtype=bool)
    for i in search.CDR_RESIDUE_INDICES_1IDX:
        designable_mask[i - 1] = True
    designable_idx = np.nonzero(designable_mask)[0]

    print("loading OpenDDE...", flush=True)
    opendde = OpenDDEModelAbag()
    features, _ = opendde.binder_features(len(binder_seq), [TargetChain(target_seq, use_msa=False)])

    print("loading AbLang2...", flush=True)
    ablang2_model, ablang2_tokenizer = load_ablang2()

    key = jax.random.key(args.seed)
    key, wt_key = jax.random.split(key)
    wt_drift = search.measure_wt_pose_drift(
        opendde=opendde, features=features, reference_distances=reference_distances,
        binder_seq=binder_seq, key=wt_key,
    )
    pose_drift_tolerance = wt_drift + search.POSE_DRIFT_MARGIN
    print(f"WT baseline distogram drift: {wt_drift:.2f}A -> tolerance = {pose_drift_tolerance:.2f}A", flush=True)

    # edit_budget/stop_grad don't matter for this diagnostic (EditBudget and
    # AbLang2 terms are held fixed at their production defaults, same loss
    # composition as the real sweep) -- only x0/scale are the variables under test.
    full_loss, variable_only_loss = search.build_composite_losses(
        opendde=opendde, features=features,
        ablang2_model=ablang2_model, ablang2_tokenizer=ablang2_tokenizer,
        reference_distances=reference_distances, binder_seq=binder_seq,
        designable_idx=designable_idx, edit_budget=5,
        stop_grad_ablang2=True, pose_drift_tolerance=pose_drift_tolerance,
    )

    x0 = softened_init(binder_seq, designable_idx, args.init_wt_prob)
    init_nnz = float((x0 > 0.01).sum(-1).mean())
    print(f"init nnz (>0.01 per position, averaged over {len(designable_idx)} positions): {init_nnz:.2f}", flush=True)

    print(f"\nrunning simplex_APGM alone ({args.n_steps} steps, scale={args.scale})...", flush=True)
    x_final, x_best = simplex_APGM(
        loss_function=variable_only_loss, x=jax.numpy.array(x0), n_steps=args.n_steps,
        stepsize=search.APGM_STEPSIZE, momentum=search.APGM_MOMENTUM, scale=args.scale, key=key,
    )

    final_full = np.asarray(variable_only_loss.sequence(x_best))
    final_argmax_seq = final_full.argmax(-1)
    final_argmax_str = "".join(TOKENS[i] for i in final_argmax_seq)
    n_diff = sum(1 for a, b in zip(binder_seq, final_argmax_str) if a != b)

    print(f"\nWT full seq:   {binder_seq}", flush=True)
    print(f"final argmax:  {final_argmax_str}", flush=True)
    print(f"n differences from WT (best_x argmax): {n_diff}", flush=True)
    print(f"\nSuccess criteria: init_nnz > 1 ({init_nnz:.2f} > 1: {init_nnz > 1}), "
          f"n_diff > 0 ({n_diff} > 0: {n_diff > 0}) -- see stdout above for "
          f"per-iteration nnz/loss (simplex_APGM's own logging) to confirm nnz "
          f"actually moves rather than snapping back to 1.00.", flush=True)


if __name__ == "__main__":
    main()
