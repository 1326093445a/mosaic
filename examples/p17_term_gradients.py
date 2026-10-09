"""Per-term gradient norms and pairwise cosines for the composite objective (§26.5).

WHY THIS EXISTS. §27.2 found the registry term null at weight 0.5 and §27.5
records that the null is uninterpretable until this measurement is made: a term
whose gradient is two orders of magnitude below its neighbours' is not a term
that was tried, it is a term that was present. The same number gates two more
questions as of 2026-10-08 -- which term produced §30.2's 5.17 A iRMSD gain, and
whether the `EditBudget` hinge can hold against the other terms at the real
objective's gradient scale, which §30.3's sweep showed decides whether a relaxed
optimum stays inside its budget.

WHAT IT MEASURES. At one sequence, for each term separately: the term's value,
the L2 norm of d(term)/dx restricted to the designable rows, the largest single
entry, and the cosine between every pair of term gradients. Five terms:

    contact    WEIGHT_OPENDDE_CONTACT * BinderTargetContact
    pose       pose_weight * BinderPoseRMSD
    registry   registry_weight * BinderTargetRegistry
    ablang2    WEIGHT_ABLANG2 * ClippedGradient(Ablang2PseudoLikelihood)
    edit       WEIGHT_EDIT_BUDGET * EditBudget

Weighted values are reported, because the question is about relative influence on
the search, not about the bare terms.

⚠️ THREE LIMITS, EACH OF WHICH CHANGES HOW THE NUMBERS READ.

1. The three structure terms are built as separate `opendde.build_loss` objects,
   so each gets its own forward pass. That matches how the real composite is
   constructed, but it is NOT one prediction decomposed: with stochastic
   sampling the three gradients are taken through three different predictions.
   `--key-seed` is fixed and shared so they are as comparable as the model
   allows, and `--repeats` quantifies what is left.

2. The real composite applies `ClippedGradient(..., CLIP_GRADIENT_NORM)` to the
   COMBINED OpenDDE loss, not to each structure term. These per-term norms are
   therefore pre-clipping influences. A term can dominate here and still be
   clipped back in the real run, so read the ratios as "what each term asks
   for", not "what the search received".

3. One sequence is one point. §28.3 is the standing caution that the objective is
   evaluated at a one-hot vertex and the gradient is a local object; a term that
   is quiet at the parent may not be quiet after five edits. `--sequence` takes
   any start, and the orchestrator runs it at the parent and at a winner.

Loads models, runs no search, writes JSON.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))


def designable_norm(gradient, designable_idx):
    """L2 norm over the rows the search is allowed to touch.

    The full-length gradient carries rows for fixed positions too. Those rows
    are real numbers but no move can ever use them, so including them would
    report influence the search cannot spend.
    """
    rows = np.asarray(gradient)[np.asarray(designable_idx)]
    return float(np.linalg.norm(rows))


def cosine(a, b, designable_idx):
    rows = np.asarray(designable_idx)
    u = np.asarray(a)[rows].ravel()
    v = np.asarray(b)[rows].ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return None
    return float(np.dot(u, v) / (nu * nv))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sequence", default=None,
        help="binder sequence to evaluate at; default is the reference's own",
    )
    parser.add_argument("--edit-budget", type=int, default=5)
    parser.add_argument("--weight-pose", type=float, default=1.0)
    parser.add_argument(
        "--weight-registry", type=float, default=0.5,
        help="the weight §27.2 tried and found null (default: %(default)s). The "
        "registry gradient is reported at this weight, so a null that is really "
        "a scale problem shows up as a small norm here.",
    )
    parser.add_argument("--registry-contact-distance", type=float, default=8.0)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--opendde-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="re-evaluate every term this many times with different keys, to "
        "separate a real term difference from sampling noise (default: 1)",
    )
    args = parser.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from mosaic.common import TOKENS
    from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
    from mosaic.losses.structure_prediction import (
        BinderPoseRMSD, BinderTargetContact, BinderTargetRegistry,
        reference_contact_pairs,
    )
    from mosaic.losses.transformations import ClippedGradient, EditBudget, SetPositions
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.structure_prediction import TargetChain
    from p17_hallucination_search import (
        CDR_RESIDUE_INDICES_1IDX, CLIP_GRADIENT_NORM, CONTACT_DISTANCE,
        HOTSPOT_TARGET_RESIDUE_INDICES_1IDX, OPENDDE_RECYCLING_STEPS,
        WEIGHT_ABLANG2, WEIGHT_EDIT_BUDGET, WEIGHT_OPENDDE_CONTACT,
        load_structure, reference_binder_target_ca,
    )

    model, binder_seq, target_seq = load_structure()
    reference_seq = binder_seq
    if args.sequence is not None:
        seq = args.sequence.strip().upper()
        if len(seq) != len(binder_seq):
            parser.error(
                f"--sequence has {len(seq)} residues, reference has {len(binder_seq)}"
            )
        if set(seq) - set(TOKENS[:20]):
            parser.error("--sequence has non-standard residues")
        binder_seq = seq

    tokens = np.array([TOKENS.index(a) for a in binder_seq], dtype=np.int32)
    anchor = np.array([TOKENS.index(a) for a in reference_seq], dtype=np.int32)
    mask = np.array([i + 1 in CDR_RESIDUE_INDICES_1IDX for i in range(len(tokens))])
    designable_idx = np.flatnonzero(mask)
    epitope_idx = np.array(sorted(i - 1 for i in HOTSPOT_TARGET_RESIDUE_INDICES_1IDX))
    binder_ca, target_ca = reference_binder_target_ca(model)

    print(f"Loading frozen OpenDDE ({args.opendde_dtype}) and AbLang2...", flush=True)
    opendde = OpenDDEModelAbag(compute_precision=args.opendde_dtype)
    features, _ = opendde.binder_features(
        len(tokens), [TargetChain(target_seq, use_msa=False)]
    )
    ablang_model, ablang_tokenizer = load_ablang2()

    def through_opendde(inner):
        """Wrap one structure term the way `build_composite_losses` does."""
        return ClippedGradient(
            opendde.build_loss(
                loss=inner, features=features,
                recycling_steps=OPENDDE_RECYCLING_STEPS,
                sampling_steps=args.sampling_steps,
            ),
            CLIP_GRADIENT_NORM,
        )

    terms = {}
    terms["contact"] = through_opendde(
        WEIGHT_OPENDDE_CONTACT * BinderTargetContact(
            paratope_idx=designable_idx, contact_distance=CONTACT_DISTANCE,
            epitope_idx=epitope_idx,
        )
    )
    terms["pose"] = through_opendde(
        args.weight_pose * ClippedGradient(
            BinderPoseRMSD(
                reference_binder_ca=binder_ca, reference_target_ca=target_ca,
                rmsd_tolerance=0.0,
            ),
            CLIP_GRADIENT_NORM,
        )
    )
    pairs = reference_contact_pairs(
        binder_ca, target_ca,
        contact_distance=args.registry_contact_distance,
        binder_subset=designable_idx,
    )
    if len(pairs) == 0:
        parser.error(
            "no reference contacts between designable positions and the target "
            f"at {args.registry_contact_distance} A; the registry term would be "
            "an empty objective"
        )
    terms["registry"] = through_opendde(
        args.weight_registry * BinderTargetRegistry(
            pairs=jnp.asarray(pairs),
            contact_distance=args.registry_contact_distance,
            repel_pairs=None, repel_weight=0.0,
        )
    )
    terms["ablang2"] = WEIGHT_ABLANG2 * ClippedGradient(
        Ablang2PseudoLikelihood(
            model=ablang_model, tokenizer=ablang_tokenizer,
            heavy_len=len(tokens),
            designable_positions=jnp.array(designable_idx, dtype=jnp.int32),
            stop_grad=False,
        ),
        CLIP_GRADIENT_NORM,
    )
    terms["edit"] = WEIGHT_EDIT_BUDGET * EditBudget.from_residues(
        reference_seq, designable_idx, budget=float(args.edit_budget),
    )

    wildtype = jnp.array(anchor, dtype=jnp.int32)
    designable_jnp = jnp.array(designable_idx, dtype=jnp.int32)
    x = jax.nn.one_hot(jnp.asarray(tokens), len(TOKENS))

    records = {name: dict(values=[], norms=[], max_abs=[]) for name in terms}
    gradients = {}
    for repeat in range(args.repeats):
        for name, term in terms.items():
            restricted = SetPositions(wildtype, designable_jnp, term)
            fn = eqx.filter_jit(eqx.filter_value_and_grad(restricted, has_aux=True))
            key = jax.random.key(args.key_seed + 1000 * repeat)
            (value, _aux), grad = fn(x, key=key)
            grad = np.asarray(grad)
            records[name]["values"].append(float(value))
            records[name]["norms"].append(designable_norm(grad, designable_idx))
            records[name]["max_abs"].append(
                float(np.abs(grad[designable_idx]).max())
            )
            if repeat == 0:
                gradients[name] = grad
            print(
                f"  [{repeat}] {name:<9} value {float(value):>12.5f}   "
                f"|grad| {records[name]['norms'][-1]:>11.5f}",
                flush=True,
            )

    total = sum(np.mean(records[n]["norms"]) for n in terms)
    summary = {}
    for name in terms:
        norms = np.array(records[name]["norms"])
        summary[name] = dict(
            value=float(np.mean(records[name]["values"])),
            grad_norm=float(norms.mean()),
            grad_norm_spread=(float(norms.std(ddof=1)) if len(norms) > 1 else None),
            max_abs_entry=float(np.mean(records[name]["max_abs"])),
            share_of_total_norm=(float(norms.mean() / total) if total > 0 else None),
        )

    cosines = {}
    for a, b in itertools.combinations(sorted(terms), 2):
        cosines[f"{a}|{b}"] = cosine(gradients[a], gradients[b], designable_idx)

    ranked = sorted(summary, key=lambda n: summary[n]["grad_norm"], reverse=True)
    spread = summary[ranked[0]]["grad_norm"] / max(
        summary[ranked[-1]]["grad_norm"], 1e-30
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        sequence=binder_seq,
        sequence_is_reference=binder_seq == reference_seq,
        hamming_from_reference=int((tokens != anchor).sum()),
        weights=dict(
            contact=WEIGHT_OPENDDE_CONTACT, pose=args.weight_pose,
            registry=args.weight_registry, ablang2=WEIGHT_ABLANG2,
            edit=WEIGHT_EDIT_BUDGET,
        ),
        edit_budget=args.edit_budget,
        sampling_steps=args.sampling_steps,
        opendde_dtype=args.opendde_dtype,
        key_seed=args.key_seed,
        repeats=args.repeats,
        registry_contact_pairs=int(len(pairs)),
        terms=summary,
        pairwise_cosines=cosines,
        ranked_by_gradient_norm=ranked,
        largest_over_smallest=float(spread),
        interpretation=(
            "Weighted per-term gradient norms over the designable rows at one "
            "sequence. Each structure term is built through its own "
            "opendde.build_loss, so the three were differentiated through three "
            "separate predictions rather than one decomposed prediction; the "
            "real composite also clips the COMBINED OpenDDE gradient, so these "
            "are pre-clipping influences. Read them as what each term asks for, "
            "not as what the search received."
        ),
    ), indent=2) + "\n")

    print()
    print(f"ranked by gradient norm: {' > '.join(ranked)}")
    print(f"largest / smallest: {spread:.3g}")
    for name in ranked:
        s = summary[name]
        print(f"  {name:<9} |grad| {s['grad_norm']:>11.5f}   "
              f"share {s['share_of_total_norm']:.3f}   value {s['value']:>12.5f}")
    print()
    print("pairwise cosines:")
    for pair, value in cosines.items():
        label = "n/a" if value is None else f"{value:+.3f}"
        print(f"  {pair:<22} {label}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
