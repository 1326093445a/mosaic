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


def relaxed_concentration(relaxed, parent_rows):
    """How far the relaxation actually moved, which the argmax cannot show.

    `project_to_budget` reports only which residue won at each position, and
    on an anchored start the parent wins at every designable position *before
    the first step*. So `candidate_substitutions_considered == 0` has two
    completely different causes -- the relaxation converged on a
    parent-preferring optimum, or it never moved at all -- and the seed JSON
    could not tell them apart. This is what distinguishes them: at the shipped
    `--init parent --init-logit-scale 1.0` start the parent carries 0.125 and
    the peak IS 0.125, so a run that still reports ~0.125 never left its
    initialisation, while one reporting ~0.98 on the parent chose it.

    On 2026-10-14 `apgm` came back with 0 wanted substitutions on the real loss
    for both seeds, after the toy grid in section 31.8 predicted 21, and
    nothing recorded could say which of the two it was. Hence these fields.

    `row_sum_max_deviation` is the audit on the rest: every method is supposed
    to return probabilities here (`bindcraft_design` softmaxes, `simplex_APGM`
    stays on the simplex, and `run_relaxation` softmaxes `colabdesign_stage`
    itself), and these numbers only mean what they say if that holds.
    """
    relaxed = np.asarray(relaxed, dtype=np.float64)
    parent_rows = np.asarray(parent_rows)
    peak = relaxed.max(-1)
    at_parent = relaxed[np.arange(len(parent_rows)), parent_rows]
    return dict(
        relaxed_peak_probability_min=round(float(peak.min()), 6),
        relaxed_peak_probability_mean=round(float(peak.mean()), 6),
        relaxed_parent_probability_mean=round(float(at_parent.mean()), 6),
        relaxed_row_sum_max_deviation=round(
            float(np.abs(relaxed.sum(-1) - 1.0).max()), 6),
    )


def relaxed_array(result):
    """The relaxed sequence, whatever shape the optimizer wrapped it in.

    The three optimizers have DIFFERENT return arities, which is a real trap:
    without a `trajectory_fn`, `bindcraft_design` and `colabdesign_stage`
    return a bare (N, 20) array while `simplex_APGM` returns (x, best_x).
    Unpacking a bare array as a 2-tuple raises "too many values to unpack",
    which is how the bc_ cells failed on 2026-10-06 after completing all 125
    optimization steps.

    They are NOT all in probability space. `bindcraft_design` applies a final
    `jax.nn.softmax` and `simplex_APGM` works on the simplex throughout, but
    `colabdesign_stage` is logits-in/logits-out by design, as its own docstring
    says, so a caller that wants probability margins out of it must apply the
    softmax itself -- this module's colabdesign path does. An earlier version
    of this docstring asserted all three were probabilities, which was wrong.
    The argmax, and therefore which substitutions `project_to_budget` keeps, is
    unaffected either way; only the recorded `margin` values change meaning.
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
    e^scale / (e^scale + 19)`. At the former default of 5.0 that is 0.886, so
    `E(s) = 29 x 0.114 = 3.3`, inside the budget, and the hinge was inactive at
    step 0. Larger scales are more feasible but more saturated: the softmax
    Jacobian goes as `p(1 - p)`, which is 0.10 at scale 5 and 0.04 at scale 6.

    WHAT THE SCALE ACTUALLY DOES, AND WHAT IT DOES NOT. The default moved from
    5.0 to 1.0 on 2026-10-13 on the strength of a gradient-weight argument: the
    softmax chain rule multiplies each coordinate's gradient by its own
    probability, so at `p_parent = 0.887` each of the 19 alternatives would get
    0.6 per cent of the parent's weight, a 148:1 ratio. That argument is
    withdrawn. It was drawn from `biohub_optimizer`, which this stage does not
    use, and it does not describe `colabdesign_stage`, whose loss sees
    `soft*softmax(z/T) + (1-soft)*z` -- at low `soft` the gradient reaches the
    raw logits unsuppressed. The data do not support it either:

    - On the CPU toy loss the scale has no effect on how many substitutions the
      relaxation wants at 125 steps (see `tests/test_continuous_seed.py`). It
      changes only how FAST a logit-space method gets there, because the
      preferred residue's logit has to cross the gap the scale opened, so time
      to flip is linear in it: with the hinge off, scale 1.0 has moved by 24
      steps and scale 5.0 has not.
    - On the cluster (2026-10-13, old code, hinge on, 125 steps, n=1 per cell)
      the scale 5.0 -> 1..4 moved colabdesign from 0 edits to 5, 2, 1, 3 and
      left bindcraft at 0, 1, 2, 2. Non-monotone, one seed, and the capped
      `edits` count hides how many substitutions were wanted.

    The 1.0 is kept because it is what ColabDesign's own wildtype start uses
    (a bare `np.eye(20)[aatype]`, no scale factor), not because it is shown to
    be better. The thing that DOES decide whether the relaxation leaves the
    parent is the `EditBudget` hinge, which is why `--weight-edit-budget`
    defaults to 0.0: `project_to_budget` enforces the budget exactly anyway.
    The real-loss comparison is the open question.

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


def run_relaxation(method, loss_function, x0, run_key, *, steps, lr,
                   logit_norm, stochastic_hard, hard_temp, hard_fraction,
                   apgm_stepsize=None, apgm_momentum=None, apgm_scale=None,
                   logit_norm_scale=None):
    """Run one optimizer and return its relaxed sequence as an (N, 20) array.

    This is the whole optimizer dispatch of the seeding stage, extracted so the
    tests exercise the SAME code `main()` runs. Before it was extracted the
    tests re-implemented the dispatch, and a re-implementation drifted: a
    "pre-fix" colabdesign path in the tests was a hybrid of the old and new
    structures, which made a step-count effect look like a hinge effect.

    Every option added on 2026-10-13 is explicit and has an off position, so a
    single factor can be varied with the rest held fixed:

    - `colabdesign` runs TWO stages -- a soft stage and a straight-through tail
      of `hard_fraction` of `steps` at `hard_temp`. The stage it replaced ran
      `hard=False` throughout, so the straight-through branch never executed.
    - `bindcraft` rescales its published 50:25:45:5 schedule to `steps`.
    - `apgm` is projected gradient on the simplex and takes none of the
      logit-space options.

    `colabdesign_stage` is logits-in/logits-out by its own docstring, so its
    result is passed through a softmax here; the other two already return
    probabilities. Only the margins `project_to_budget` records depend on
    that, not which substitutions it keeps.
    """
    import jax
    from mosaic.optimizers import (
        _apportion_stages, bindcraft_design, colabdesign_stage, simplex_APGM,
    )
    from p17_hallucination_search import (
        APGM_MOMENTUM, APGM_SCALE, APGM_STEPSIZE,
    )

    # Omitted apgm settings mean the PRODUCTION ones. They used to be required,
    # and the tests passed 0.1/0.9/1.0 -- inherited from an older test -- while
    # main() passed 0.05/0.5/1.2. On the same fixture at init scale 5.0 that
    # difference decided the result outright: switching the hinge off moved
    # simplex_APGM to 29 wanted substitutions at the test settings and left it
    # at 0 at the production ones. So the tests reported coverage of a
    # configuration nothing ships. Defaulting to production makes the two
    # impossible to diverge by omission.
    if apgm_stepsize is None:
        apgm_stepsize = APGM_STEPSIZE
    if apgm_momentum is None:
        apgm_momentum = APGM_MOMENTUM
    if apgm_scale is None:
        apgm_scale = APGM_SCALE

    shaping = dict(logit_norm=logit_norm, logit_norm_scale=logit_norm_scale,
                   stochastic_hard=stochastic_hard)
    if method == "bindcraft":
        raw = bindcraft_design(
            loss_function=loss_function, x=x0, lr=lr, key=run_key,
            n_steps=steps, hard_temp=hard_temp, **shaping,
        )
    elif method == "colabdesign":
        # The same apportionment the four-stage schedule uses, for the same
        # reason: `max(1, round(...))` overshoots. steps=4 with
        # hard_fraction=0.99 gave hard=4 and then clamped soft to 1, running 5
        # steps for a 4-step request. This was fixed in `_apportion_stages` and
        # left here, which is exactly the duplication that invited it.
        soft_steps, hard_steps = _apportion_stages(
            (1.0 - hard_fraction, hard_fraction), steps
        )
        logits = colabdesign_stage(
            loss_function=loss_function, x=x0, n_steps=soft_steps,
            soft_start=0.0, soft_end=1.0, temp_start=1.0, temp_end=1.0,
            hard=False, lr=lr, key=run_key, **shaping,
        )
        logits = colabdesign_stage(
            loss_function=loss_function, x=logits, n_steps=hard_steps,
            soft_start=1.0, soft_end=1.0, temp_start=hard_temp,
            temp_end=hard_temp, hard=True, lr=lr,
            key=jax.random.fold_in(run_key, 1), **shaping,
        )
        raw = jax.nn.softmax(logits, -1)
    elif method == "apgm":
        # `relaxed_array` takes result[0], the final iterate, so the
        # best-iterate tracking added on 2026-10-13 would be one extra forward
        # pass per step plus one at the start -- 126 evaluations on a 125-step
        # run -- for a value this function discards.
        raw = simplex_APGM(
            loss_function=loss_function, x=jax.nn.softmax(x0, -1),
            n_steps=steps, stepsize=apgm_stepsize, momentum=apgm_momentum,
            scale=apgm_scale, key=run_key, track_best=False,
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    return relaxed_array(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edit-budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600,
                        help="total continuous steps. Reaches all three methods: "
                             "bindcraft's four stages are rescaled to this total, "
                             "which they were NOT before -- the flag was dropped "
                             "on the floor and every bindcraft run was 125 steps. "
                             "Fast SeqProp (arXiv:2005.11275) reports un-normalised "
                             "softmax relaxation failing to converge in 20,000 "
                             "updates and its normalised variant converging in 200, "
                             "so 125 was in the worst regime of both.")
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--weight-registry", type=float, default=0.0)
    parser.add_argument("--registry-contact-distance", type=float, default=8.0)
    parser.add_argument("--weight-pose", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--init", choices=["parent", "noise"], default="parent",
                        help="anchor the start at the parent one-hot (default) "
                             "or use the pure-noise start the optimizers ship with")
    parser.add_argument("--init-logit-scale", type=float, default=1.0,
                        help="mass added to the parent column when --init=parent. "
                             "1.0 matches ColabDesign's own wildtype start, a bare "
                             "one-hot (np.eye(20)[aatype]); the former default was "
                             "5.0. NOT shown to be better: on the toy loss it only "
                             "changes how many steps a logit method needs once the "
                             "hinge is off, and the cluster probe was mixed. See "
                             "anchored_logits.")
    parser.add_argument("--weight-edit-budget", type=float, default=0.0,
                        help="weight on the EditBudget hinge DURING the relaxation. "
                             "0.0 by default: `project_to_budget` already enforces "
                             "the budget exactly downstream, while on a relaxed "
                             "sequence the hinge's only route to satisfaction is "
                             "driving p_parent toward 1, i.e. the saturation that "
                             "removes the gradient. CPU toy loss, 125 steps, scale "
                             "5.0, 2026-10-13: hinge 5.0 -> bindcraft/colabdesign/apgm "
                             "wanted 4/6/0 substitutions; hinge 0.0 -> 27/29/29. "
                             "REAL loss, 600 steps, scale 1.0, 2026-10-14 "
                             "(section 32): hinge 0.0 -> 15/15/0 of 29, two "
                             "replicates each. One cell, no hinge-on arm, so "
                             "that confirms the toy's prediction for this cell "
                             "and not the hinge's role: the two logit-space "
                             "methods move and apgm, which the toy put at 21 "
                             "here, does not.")
    parser.add_argument("--logit-norm", dest="logit_norm", action="store_true",
                        default=False,
                        help="re-normalise the logits every step (Fast SeqProp). "
                             "OFF by default: on a uniform start the "
                             "auto-calibrated gamma is 0 and every update is "
                             "multiplied back to zero (now refused rather than "
                             "frozen), and the option is otherwise unvalidated on "
                             "the real loss. Gamma fixes the GLOBAL deviation of "
                             "the logit block, so it rations total confidence "
                             "across positions rather than capping any one of "
                             "them. Fast SeqProp LEARNS gamma; a fixed one is not "
                             "their method.")
    parser.add_argument("--no-logit-norm", dest="logit_norm", action="store_false",
                        help="disable logit normalisation (the default)")
    parser.add_argument("--logit-norm-scale", type=float, default=None,
                        help="gamma for --logit-norm. Default calibrates to the "
                             "opening spread, which REFUSES a uniform start rather "
                             "than freezing on it; 1.0 is the usual gamma for a "
                             "normalisation layer.")
    parser.add_argument("--stochastic-hard", dest="stochastic_hard",
                        action="store_true", default=False,
                        help="sample the straight-through one-hot instead of taking "
                             "the argmax (Fast SeqProp's sampler). OFF by default: "
                             "the argmax is what ColabDesign does, and on the CPU "
                             "toy loss sampling did not help.")
    parser.add_argument("--no-stochastic-hard", dest="stochastic_hard",
                        action="store_false", help="use the argmax one-hot (default)")
    parser.add_argument("--hard-temp", type=float, default=1.0,
                        help="temperature of the straight-through stage. BindCraft "
                             "and ColabDesign both use 1e-2 here; this deliberately "
                             "does not. At 1e-2 the estimator's gradient is the "
                             "Jacobian of a saturated softmax, which is ~0, and "
                             "ColabDesign's own `lr_scale = soft*temp` then shrinks "
                             "the step 100x on top of that -- so the only stage that "
                             "shows the predictor a valid one-hot is also the one "
                             "that cannot move. 1.0 is the regime Fast SeqProp's "
                             "straight-through sampler operates in.")
    parser.add_argument("--hard-fraction", type=float, default=0.25,
                        help="share of --steps spent in the straight-through stage "
                             "on the colabdesign path (bindcraft keeps its own "
                             "50:25:45:5 proportions)")
    args = parser.parse_args(argv)
    if not 0.0 < args.hard_fraction < 1.0:
        parser.error("--hard-fraction must be in (0, 1)")
    if args.steps < 4:
        parser.error("--steps must be at least 4 to cover the staged schedules")
    if args.hard_temp <= 0:
        parser.error("--hard-temp must be positive")

    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS
    from mosaic.models.opendde import OpenDDEModelAbag
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
        edit_budget_weight=args.weight_edit_budget,
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

    print(f"Running {args.method} over {n_designable} designable positions "
          f"for {args.steps} steps (init scale {args.init_logit_scale}, "
          f"edit-budget weight {args.weight_edit_budget}, "
          f"logit_norm={args.logit_norm}, stochastic_hard={args.stochastic_hard}, "
          f"hard_temp={args.hard_temp})...", flush=True)
    relaxed = run_relaxation(
        args.method, variable_only_loss, x0, run_key,
        steps=args.steps, lr=args.lr, logit_norm=args.logit_norm,
        stochastic_hard=args.stochastic_hard, hard_temp=args.hard_temp,
        hard_fraction=args.hard_fraction, apgm_stepsize=APGM_STEPSIZE,
        apgm_momentum=APGM_MOMENTUM, apgm_scale=APGM_SCALE,
        logit_norm_scale=args.logit_norm_scale,
    )
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
    concentration = relaxed_concentration(relaxed, parent[designable_idx])
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
        steps=args.steps, lr=args.lr,
        weight_edit_budget=args.weight_edit_budget,
        logit_norm=args.logit_norm, logit_norm_scale=args.logit_norm_scale,
        stochastic_hard=args.stochastic_hard,
        hard_temp=args.hard_temp, hard_fraction=args.hard_fraction,
        parent_sequence=binder_seq, seed_sequence=sequence,
        hamming_from_parent=hamming, edits=edits,
        candidate_substitutions_considered=len(margins),
        designable_positions=int(n_designable),
        **concentration,
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
        print(f"  ({len(margins)} substitutions wanted of {n_designable} "
              f"designable, {args.edit_budget} kept)")
    print(f"  relaxed peak probability {concentration['relaxed_peak_probability_mean']} "
          f"mean, {concentration['relaxed_peak_probability_min']} min; "
          f"{concentration['relaxed_parent_probability_mean']} mean on the parent")
    if concentration["relaxed_row_sum_max_deviation"] > 1e-3:
        print(f"  WARNING: rows deviate from a probability distribution by up to "
              f"{concentration['relaxed_row_sum_max_deviation']}; the margins and "
              f"the concentration above are not probabilities")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
