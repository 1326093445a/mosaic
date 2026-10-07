"""Stage 0: measure the discretization gap on a synthetic constrained problem.

Why this exists. The continuous seeding stage's premise is that relaxed
optimization finds a better point than discrete gradient-guided search does,
and that enough of that advantage survives being rounded back to a sequence
under an edit budget. On the real objective both halves are confounded: a
seeded run that does badly could mean the relaxation found nothing, or that it
found something and `project_to_budget` destroyed it. Section 28.6 flagged that
as the open question and this separates it, on a problem where the objective is
known exactly and a run costs seconds instead of a GPU-night.

The problem is a Potts energy over 29 positions and 20 tokens -- the real
designable geometry -- with pairwise couplings, so it is not separable. A
separable objective would make rounding trivially lossless and rig the test.
Its multilinear extension is exact at the one-hot vertices, so the relaxed and
discrete objectives are the same function, and the only difference between the
two methods is how they search it.

Both sides run the project's own code, not a reimplementation:
`mosaic.search.run_gradient_search` does the discrete search with the same
`SearchConfig` the real cells use, the gradient is evaluated at a one-hot
vertex exactly as `examples/p17_confidence_search.py` does it, the relaxed
stage starts from `anchored_logits` and is rounded by the real
`project_to_budget`, and the hand-off uses the `wt`-is-the-parent,
`initial_sequences`-is-the-seed wiring that `--budget-anchor reference` sets up.

Runs on CPU, loads no model.
"""

import argparse
import importlib.util
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mosaic.search import ConfidenceScore, SearchConfig, run_gradient_search  # noqa: E402

WEIGHT_EDIT_BUDGET = 5.0  # matches examples/p17_hallucination_search.py


def _seeder():
    spec = importlib.util.spec_from_file_location(
        "p17_continuous_seed_stage0", REPO / "examples" / "p17_continuous_seed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the synthetic objective ----------------------------------------------


def make_problem(rng, length, alphabet, coupling_scale):
    """A frustrated Potts energy and a parent sequence to start from."""
    fields = rng.standard_normal((length, alphabet))
    raw = rng.standard_normal((length, length, alphabet, alphabet))
    # Symmetric in (i, a) <-> (j, b), and no self-coupling.
    couplings = coupling_scale * 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    idx = np.arange(length)
    couplings[idx, idx] = 0.0
    parent = rng.integers(0, alphabet, size=length).astype(np.int32)
    return fields, couplings, parent


def energy(tokens, fields, couplings):
    """The true objective at a sequence. Lower is better."""
    tokens = np.asarray(tokens, dtype=np.int64)
    idx = np.arange(len(tokens))
    pair = couplings[idx[:, None], idx[None, :], tokens[:, None], tokens[None, :]]
    return float(fields[idx, tokens].sum() + 0.5 * pair.sum())


def extension(simplex, fields, couplings):
    """Multilinear extension; equals `energy` at the one-hot vertices."""
    linear = float((fields * simplex).sum())
    # sum_{i,j,a,b} J[i,j,a,b] X[i,a] X[j,b], halved for the double count.
    pair = np.einsum("ijab,ia,jb->", couplings, simplex, simplex)
    return linear + 0.5 * float(pair)


def extension_gradient(simplex, fields, couplings):
    return fields + np.einsum("ijab,jb->ia", couplings, simplex)


def edit_expectation(simplex, parent):
    """`EditBudget`'s own quantity: expected substitutions from the parent."""
    return float((1.0 - simplex[np.arange(len(parent)), parent]).sum())


def hinge_and_gradient(simplex, parent, budget):
    excess = edit_expectation(simplex, parent) - budget
    if excess <= 0:
        return 0.0, np.zeros_like(simplex)
    gradient = np.zeros_like(simplex)
    gradient[np.arange(len(parent)), parent] = -WEIGHT_EDIT_BUDGET
    return WEIGHT_EDIT_BUDGET * excess, gradient


def objective_at_vertex(tokens, fields, couplings, parent, budget):
    """Loss and gradient as the real harness computes them: at a one-hot."""
    length, alphabet = fields.shape
    simplex = np.zeros((length, alphabet))
    simplex[np.arange(length), np.asarray(tokens, dtype=np.int64)] = 1.0
    hinge, hinge_gradient = hinge_and_gradient(simplex, parent, budget)
    loss = extension(simplex, fields, couplings) + hinge
    gradient = extension_gradient(simplex, fields, couplings) + hinge_gradient
    return loss, gradient


def exhaustive_optimum(fields, couplings, parent, budget):
    """The true constrained optimum, by enumeration. Feasible for budget <= 2.

    `best known` is otherwise the same discrete search with more evaluations,
    which cannot show that search saturating -- it would agree with itself by
    construction. This is the independent reference.
    """
    from itertools import combinations, product

    length, alphabet = fields.shape
    best = (energy(parent, fields, couplings), np.array(parent, dtype=np.int32))
    others = [[a for a in range(alphabet) if a != parent[i]] for i in range(length)]
    for size in range(1, budget + 1):
        for positions in combinations(range(length), size):
            for residues in product(*(others[i] for i in positions)):
                tokens = np.array(parent, dtype=np.int32)
                tokens[list(positions)] = residues
                value = energy(tokens, fields, couplings)
                if value < best[0]:
                    best = (value, tokens)
    return best[1]


def enumeration_size(length, alphabet, budget):
    from math import comb

    return sum(
        comb(length, size) * (alphabet - 1) ** size
        for size in range(1, budget + 1)
    )


# --- the relaxed stage -----------------------------------------------------


def relax(seeder, fields, couplings, parent, budget, steps, lr, rng, scale):
    """Gradient descent on the extension, from the anchored parent start."""
    length, alphabet = fields.shape
    designable = np.arange(length)
    noise = 0.01 * rng.standard_normal((length, alphabet))
    logits = seeder.anchored_logits(parent, designable, scale, noise).astype(float)
    moment = np.zeros_like(logits)
    for _ in range(steps):
        shifted = logits - logits.max(-1, keepdims=True)
        simplex = np.exp(shifted)
        simplex /= simplex.sum(-1, keepdims=True)
        _, hinge_gradient = hinge_and_gradient(simplex, parent, budget)
        g_simplex = extension_gradient(simplex, fields, couplings) + hinge_gradient
        # Softmax Jacobian.
        g_logits = simplex * (
            g_simplex - (g_simplex * simplex).sum(-1, keepdims=True)
        )
        moment = 0.9 * moment + g_logits
        logits -= lr * moment
    shifted = logits - logits.max(-1, keepdims=True)
    simplex = np.exp(shifted)
    simplex /= simplex.sum(-1, keepdims=True)
    return simplex


# --- the discrete stage ----------------------------------------------------


def discrete(fields, couplings, parent, budget, *, policy, initial, score_calls,
             seed, target_entropy=0.6):
    config = SearchConfig(
        policy=policy,
        edit_budget=budget,
        max_score_calls=score_calls,
        max_gradient_calls=score_calls,
        max_proposals=score_calls * 10,
        target_entropy=target_entropy,
        acceptance_temperature=0.02,
        seed=seed,
    )
    start = (
        None if initial is None
        else np.repeat(np.asarray(initial)[None], config.width, axis=0)
    )
    result = run_gradient_search(
        wt=parent,
        designable_mask=np.ones(len(parent), dtype=bool),
        config=config,
        initial_sequences=start,
        gradient_fn=lambda s: objective_at_vertex(
            s, fields, couplings, parent, budget
        ),
        # The harness maximizes its readout, so the score is the negated energy.
        confidence_fn=lambda s: ConfidenceScore(
            -energy(s, fields, couplings)
        ),
    )
    best = min(
        (c.sequence for c in result.evaluated),
        key=lambda s: energy(s, fields, couplings),
    )
    return np.asarray(best, dtype=np.int32)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--length", type=int, default=29,
                        help="designable positions; 29 is the real geometry")
    parser.add_argument("--alphabet", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--coupling-scale", type=float, default=0.3)
    parser.add_argument("--score-calls", type=int, default=200,
                        help="200 is what the real cells get per seed")
    parser.add_argument("--reference-multiplier", type=int, default=20,
                        help="score-call multiple for the best-known baseline")
    parser.add_argument("--relax-steps", type=int, default=400)
    parser.add_argument("--relax-lr", type=float, default=0.05)
    parser.add_argument("--init-logit-scale", type=float, default=5.0)
    parser.add_argument("--policy", default="independent",
                        choices=["independent", "population"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exhaustive", action="store_true",
                        help="reference against the true constrained optimum "
                             "instead of the same search with more calls; "
                             "only tractable for a small budget")
    args = parser.parse_args(argv)
    if args.exhaustive:
        size = enumeration_size(args.length, args.alphabet, args.budget)
        if size > 2_000_000:
            parser.error(
                f"--exhaustive would enumerate {size:,} sequences at budget "
                f"{args.budget}; lower --budget or --length"
            )
        print(f"exhaustive reference: {size:,} sequences per instance", flush=True)

    seeder = _seeder()
    arms = {}
    interiority = []
    wanted_edits = []

    for instance in range(args.instances):
        rng = np.random.default_rng(args.seed * 10_000 + instance)
        fields, couplings, parent = make_problem(
            rng, args.length, args.alphabet, args.coupling_scale
        )
        designable = np.arange(args.length)
        common = dict(policy=args.policy, seed=instance)

        def record(name, tokens):
            drift = int(np.count_nonzero(np.asarray(tokens) != parent))
            if drift > args.budget:
                raise AssertionError(f"{name} drifted {drift} > {args.budget}")
            arms.setdefault(name, []).append(energy(tokens, fields, couplings))

        record("parent", parent)

        # j = 0: all five edits spent by the discrete search. Today's grad arm.
        record("discrete j=0", discrete(
            fields, couplings, parent, args.budget, initial=None,
            score_calls=args.score_calls, **common))

        # The relaxed optimum, rounded at several hand-off points.
        simplex = relax(seeder, fields, couplings, parent, args.budget,
                        args.relax_steps, args.relax_lr, rng,
                        args.init_logit_scale)
        interiority.append(float(simplex.max(-1).mean()))
        wanted = int(np.count_nonzero(simplex.argmax(-1) != parent))
        wanted_edits.append(wanted)

        for j in (2, args.budget):
            seed_tokens, _, _ = seeder.project_to_budget(
                simplex, parent, designable, j
            )
            record(f"relax j={j}, no search", seed_tokens)
            # The hand-off: one shared budget, search starts at the seed.
            record(f"relax j={j} + discrete", discrete(
                fields, couplings, parent, args.budget, initial=seed_tokens,
                score_calls=args.score_calls, **common))

        # The reference the other arms are scored against.
        if args.exhaustive:
            record("REFERENCE exact", exhaustive_optimum(
                fields, couplings, parent, args.budget))
        else:
            record("REFERENCE discrete x%d" % args.reference_multiplier, discrete(
                fields, couplings, parent, args.budget, initial=None,
                score_calls=args.score_calls * args.reference_multiplier,
                **common))

        print(f"  instance {instance + 1}/{args.instances}", end="\r", flush=True)

    print(" " * 40, end="\r")
    print(f"{args.instances} instances, {args.length} positions, alphabet "
          f"{args.alphabet}, budget {args.budget}, {args.score_calls} score "
          f"calls, policy {args.policy}\n")
    print(f"relaxed optimum: mean max probability {st.mean(interiority):.3f} "
          f"(1.000 would be a vertex), wants "
          f"{st.mean(wanted_edits):.1f} edits against a budget of {args.budget}\n")

    baseline = st.mean(arms["parent"])
    reference_name = next(k for k in arms if k.startswith("REFERENCE"))
    best = st.mean(arms[reference_name])
    print(f"{'arm':26s} {'energy':>9s} {'+-sem':>7s} {'vs parent':>10s} "
          f"{'% of reference gain':>21s}")
    for name, values in arms.items():
        mean = st.mean(values)
        sem = st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        gain = baseline - mean
        share = 100 * gain / (baseline - best) if baseline != best else float("nan")
        print(f"{name:26s} {mean:9.3f} {sem:7.3f} {gain:+10.3f} {share:20.1f}%")

    # The number stage 0 exists to produce.
    pure = st.mean(arms[f"relax j={args.budget}, no search"])
    disc = st.mean(arms["discrete j=0"])
    hybrid = st.mean(arms["relax j=2 + discrete"])
    saturated = [
        i for i, (d, r) in enumerate(
            zip(arms["discrete j=0"], arms[reference_name])
        ) if d <= r + 1e-9
    ]
    print(f"\ndiscrete search reached the reference on "
          f"{len(saturated)}/{args.instances} instances")
    print(f"rounding gap: relax-and-round alone is "
          f"{pure - disc:+.3f} against discrete search "
          f"({'worse' if pure > disc else 'better'})")
    print(f"partial hand-off (j=2) is {hybrid - disc:+.3f} against discrete "
          f"({'worse' if hybrid > disc else 'better'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
