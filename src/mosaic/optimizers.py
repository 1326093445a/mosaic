import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PyTree
from scipy.special import log_softmax, softmax

from mosaic.common import LinearCombination, LossTerm

AbstractLoss = LossTerm | LinearCombination


def _ranking_leaf(aux, ranking_aux_name: str):
    """Locate the ranking-metric leaf in a loss's ``aux`` tree, or ``None``.

    ``ranking_aux_name`` is matched as a dict key ANYWHERE in the path, so the
    lookup is robust to nesting and to ``reduce_samples`` wrapping per-sample
    scalar metrics into loss-sorted lists (which appends a ``SequenceKey`` and
    would defeat a terminal-key match). When the metric is such a per-sample
    list, the best sample -- list index 0, since aux is sorted ascending by
    loss -- is returned.
    """
    matches = [
        (path, leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(aux)
        if any(getattr(k, "key", None) == ranking_aux_name for k in path)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][1]

    def _trailing_seq_idx(path):
        for k in reversed(path):
            if isinstance(k, jax.tree_util.SequenceKey):
                return k.idx
        return 0

    return min(matches, key=lambda pl: _trailing_seq_idx(pl[0]))[1]


def _print_iter(iter, aux, v):
    aux = eqx.filter(
        aux,
        lambda v: isinstance(v, float | str) or v.shape == (),
    )
    print(
        iter,
        f"loss: {v:0.2f}",
        " ".join(
            f"{jax.tree_util.keystr(k, simple=True, separator='.')}:{v: 0.2f}"
            for (k, v) in jax.tree_util.tree_leaves_with_path(aux)
            if hasattr(v, "item") or isinstance(v, float)
        ),
    )


@eqx.filter_jit
def _____eval_loss(loss, x, key):
    return loss(x, key=key)


def _eval_loss(loss_function, x, key):
    """Loss value only, for scoring a FEASIBLE iterate.

    `simplex_APGM` evaluates its gradient at the momentum extrapolation `v`,
    which in the non-logspace branch is not projected onto the simplex -- only
    the post-update iterate is. So the value returned alongside the gradient
    belongs to a point the optimiser never visits and which may not be a valid
    sequence, which is why `best_x` cannot simply be set to it.

    Returned RAW, including NaN and the infinities. This used to be
    `nan_to_num(value, nan=1e6)`, which made a diverged candidate score 1e6 and
    so beat any legitimate loss above 1e6 -- and, worse, made the `np.isfinite`
    guard at the selection site unreachable, because the NaN was already gone
    by the time it was tested. Selection rejects nonfinite scores; it cannot do
    that if scoring hides them.
    """
    x = np.array(x, dtype=np.float32)
    value, _aux = _____eval_loss(loss_function, x=x, key=key)
    return float(value)


def _batched_eval_loss(loss_function, xs, keys):
    """`_eval_loss` for a batch: RAW values, including NaN.

    `batched_value_eval` applies its own `nan_to_num(v, nan=1e6)`, which is
    defensible for driving an optimiser but not for SELECTING among iterates --
    it was the second place a NaN could be laundered into a score that beats a
    legitimate loss above 1e6, and it defeated the `isfinite` guard exactly as
    `_eval_loss` used to. The other `batched_value_eval` call sites are left
    alone; this one exists because selection has a different requirement from
    optimisation.
    """
    def single(x, key):
        v, _aux = loss_function(x, key=key)
        return v

    return np.asarray(jax.vmap(single)(xs, keys), dtype=np.float64)


def _eval_loss_and_grad(
    loss_function: AbstractLoss,
    x,
    key,
):
    """
    Evaluates the loss function and its gradient.

    Args:
    - loss_function: ...
    - x: soft sequence (N x 20 array with each row in the simplex)
    - key: jax random key

    Returns:
    - ((value, aux), g): value of the loss function and auxiliary information, and gradient of the loss with respect to x

    """
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan=1000000.0), aux), jnp.nan_to_num(
        g - g.mean(axis=-1, keepdims=True)
    )


@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


@eqx.filter_jit
def batched_eval(
    loss: AbstractLoss,
    xs: Float[Array, "B N K"],
    keys: jax.Array,
) -> tuple[Float[Array, " B"], PyTree, Float[Array, "B N K"]]:
    """Evaluate loss+grad for B sequences with B keys."""
    assert xs.ndim == 3, f"xs must be 3D [B, N, K], got {xs.ndim}D"

    def single(x: Float[Array, "N K"], key: jax.Array):
        (v, aux), g = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
        v = jnp.nan_to_num(v, nan=1e6)
        g = jnp.nan_to_num(g - g.mean(axis=-1, keepdims=True))
        return v, aux, g

    return jax.vmap(single)(xs, keys)


@eqx.filter_jit
def batched_value_eval(
    loss: AbstractLoss,
    xs: Float[Array, "B N K"],
    keys: jax.Array,
) -> tuple[Float[Array, "B"], PyTree]:
    """Evaluate loss values for B sequences without materializing gradients."""
    assert xs.ndim == 3, f"xs must be 3D [B, N, K], got {xs.ndim}D"

    def single(x: Float[Array, "N K"], key: jax.Array):
        v, aux = loss(x, key=key)
        return jnp.nan_to_num(v, nan=1e6), aux

    return jax.vmap(single)(xs, keys)




def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)


def _reverse_path_logprob(proposal, mutations, g, temp, alphabet_size):
    """Log-probability of the path that walks `proposal` back to its origin.

    Returns `(log_q_backward, restored)`. `restored` must equal the sequence the
    forward path started from -- that is the invariant, and it is what the old
    implementation violated.

    `mutations` is the forward path as `(position, new_token, previous_token)`.
    Each reverse step scores the move that UNDOES a forward one, i.e. setting
    the position back to `previous_token`, evaluated at the proposal-side
    gradient. The old version scored `log_p[position, new_token]` and then
    assigned `new_token`, which was a no-op because the proposal already held
    it: `prop_backward` never changed and every term was the probability of
    re-applying a mutation rather than reversing it. `detailed_balance=True`
    therefore applied a wrong Hastings correction. Fixed 2026-10-13.

    Extracted so the tests exercise this rather than a copy of it.
    """
    restored = proposal.copy()
    log_q_backward = 0.0
    for position, _new_token, previous_token in reversed(mutations):
        _p, log_p = _proposal(restored, g, temp, alphabet_size=alphabet_size)
        log_q_backward += log_p[position, previous_token]
        restored = restored.at[position].set(previous_token)
    return log_q_backward, restored


def gradient_MCMC(
    loss,
    sequence: Int[Array, " N"],
    temp=0.001,
    proposal_temp=0.01,
    max_path_length=2,
    steps=50,
    alphabet_size: int = 20,
    key: None = None,
    detailed_balance: bool = False,
    fix_loss_key: bool = True,
):
    """
    Implements the gradient-assisted MCMC sampler from "Plug & Play Directed Evolution of Proteins with
    Gradient-based Discrete MCMC." Uses first-order taylor approximation of the loss to propose mutations.

        WARNING: Fixes random seed used for loss evaluation.

    Args:
    - loss: log-probability/function to minimize
    - sequence: initial sequence
    - proposal_temp: temperature of the proposal distribution
    - temp: temperature for the loss function
    - max_path_length: maximum number of mutations per step
    - steps: number of optimization steps
    - key: jax random key
    - detailed_balance: whether to maintain detailed balance

    """

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    key_model = key
    (v_0, aux_0), g_0 = _eval_loss_and_grad(
        loss, jax.nn.one_hot(sequence, alphabet_size), key=key_model
    )
    max_draws = 1 if detailed_balance else 50
    for iter in range(steps):
        start_time = time.time()

        for i in range(max_draws):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            key, length_key = jax.random.split(key)
            path_length = jax.random.randint(
                key=length_key,
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            for _ in range(path_length):
                p, log_p = _proposal(
                    proposal, g_0, proposal_temp, alphabet_size=alphabet_size
                )
                mut_idx = jax.random.choice(
                    key=key,
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(),
                )
                key = jax.random.fold_in(key, 0)
                position, AA = np.unravel_index(mut_idx, p.shape)
                log_q_forward += log_p[position, AA]
                mutations += [(position, AA, int(proposal[position]))]
                proposal = proposal.at[position].set(AA)
            if np.all(proposal == sequence):
                print(f"\t {i}: proposal is the same as current sequence, skipping.")
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa, _prev) in mutations])
        print(f"Proposed mutations: {muts}")

        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss,
            jax.nn.one_hot(proposal, alphabet_size),
            key=key_model if fix_loss_key else key,
        )

        log_q_backward, prop_backward = _reverse_path_logprob(
            proposal, mutations, g_1, proposal_temp, alphabet_size
        )

        log_acceptance_probability = (v_0 - v_1) / temp + (
            (log_q_backward - log_q_forward) if detailed_balance else 0.0
        )

        log_acceptance_probability = min(0.0, log_acceptance_probability)

        print(
            f"iter: {iter}, accept {np.exp(log_acceptance_probability): 0.3f} {v_0: 0.3f} {v_1: 0.3f} {log_q_forward: 0.3f} {log_q_backward: 0.3f}"
        )

        print()
        if -jax.random.exponential(key=key) < log_acceptance_probability:
            sequence = proposal
            (v_0, aux_0), g_0 = (v_1, aux_1), g_1

        _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)

        key = jax.random.fold_in(key, 0)

    return sequence


def projection_simplex(V, z=1):
    """
    From https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    """
    V = np.array(V, dtype=np.float64)
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def simplex_APGM(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key=None,
    max_gradient_norm: float | None = None,
    scale=1.0,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
    logspace: bool = False,
    track_best: bool = True,
):
    """
    Accelerated projected gradient descent on the simplex.

    Args:
    - loss_function: function to minimize
    - x: initial sequence
    - n_steps: number of optimization steps
    - stepsize: step size for gradient descent
    - momentum: momentum factor
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - scale: proximal scaling factor for L2 regularization (or entropic regularization if logspace=True), set to > 1.0 to encourage sparsity
    - trajectory_fn: function to compute trajectory information, takes (aux, x) and returns any value.
    - logspace: whether to optimize in log space, which corresponds to a bregman proximal algorithm.

    returns:
    - x: final soft sequence after optimization
    - best_x: best soft sequence found during optimization
    - trajectory: list of trajectory information if `trajectory_fn` is provided, otherwise nothing.
    """

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[0])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    score_key = jax.random.fold_in(key, 7919)

    def score(candidate):
        feasible = jax.nn.softmax(candidate) if logspace else candidate
        return _eval_loss(loss_function, feasible, score_key)

    x = projection_simplex(x) if not logspace else x
    best_x = x
    best_val = score(x) if track_best else np.inf
    if not np.isfinite(best_val):
        best_val = np.inf

    x_prev = x

    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        v = jax.device_put(x + momentum * (x - x_prev))
        (value, aux), g = _eval_loss_and_grad(
            x=v if not logspace else jax.nn.softmax(v),
            loss_function=loss_function,
            key=key,
        )

        n = np.sqrt((g**2).sum())
        if n > max_gradient_norm:
            g = g * (max_gradient_norm / n)

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * g)
        else:
            x_new = projection_simplex(scale * (v - stepsize * g))

        x_prev = x
        x = x_new

        if track_best:
            candidate_val = score(x)
            if np.isfinite(candidate_val) and candidate_val < best_val:
                best_val = candidate_val
                best_x = x

        average_nnz = (
            (x > 0.01).sum(-1).mean()
            if not logspace
            else (jax.nn.softmax(x) > 0.01).sum(-1).mean()
        )

        aux = {
            "loss": value,
            "nnz": average_nnz,
            "time": (time.time() - start_time),
            "": aux,
        }
        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        _print_iter(
            _iter,
            eqx.filter(
                aux,
                lambda v: isinstance(v, float) or v.shape == (),
            ),
            value,
        )

    if logspace:
        x = jax.nn.softmax(x)
        best_x = jax.nn.softmax(best_x)

    if trajectory_fn is None:
        return x, best_x
    else:
        return x, best_x, trajectory


def batched_simplex_APGM(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "B N 20"],
    n_steps: int,
    stepsize: float,
    momentum: float = 0.0,
    key: jax.Array | None = None,
    max_gradient_norm: float | None = None,
    scale: float = 1.0,
    logspace: bool = False,
    track_best: bool = True,
) -> tuple[Float[Array, "B N 20"], Float[Array, "B N 20"]]:
    """
    Batched accelerated projected gradient descent on the simplex.
    Runs B copies of the optimization in parallel via vmap, where B = x.shape[0].

    Args:
    - loss_function: loss function (same for all designs)
    - x: initial soft sequences [B, N, 20]
    - n_steps: number of optimization steps
    - stepsize: step size (scalar or [B, 1, 1] array for per-design values)
    - momentum: momentum factor (scalar or [B, 1, 1] array)
    - key: jax random key
    - max_gradient_norm: maximum norm of the gradient
    - scale: proximal scaling factor
    - logspace: whether to optimize in log space

    returns:
    - x: final soft sequences [B, N, 20]
    - best_x: best soft sequences found during optimization [B, N, 20]
    """
    assert x.ndim == 3, f"x must be 3D [B, N, 20], got {x.ndim}D"
    B = x.shape[0]

    if max_gradient_norm is None:
        max_gradient_norm = np.sqrt(x.shape[1])

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    if not logspace:
        flat = np.array(x).reshape(-1, x.shape[-1])
        x = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

    score_keys = jax.random.split(jax.random.fold_in(key, 7919), B)

    def score(candidate):
        feasible = jax.nn.softmax(candidate, axis=-1) if logspace else candidate
        return _batched_eval_loss(loss_function, feasible, score_keys)

    best_x = x
    best_vals = (
        jnp.asarray(score(x)) if track_best else jnp.full(B, jnp.inf)
    )
    best_vals = jnp.where(jnp.isfinite(best_vals), best_vals, jnp.inf)
    x_prev = x

    for _iter in range(n_steps):
        start_time = time.time()
        v = jnp.array(x + momentum * (x - x_prev), dtype=jnp.float32)
        v_eval = jax.nn.softmax(v, axis=-1) if logspace else v

        values, auxs, grads = batched_eval(loss_function, v_eval, jax.random.split(key, B))

        norms = np.sqrt((grads**2).sum(axis=(-2, -1)))
        clip = np.where(norms > max_gradient_norm, max_gradient_norm / norms, 1.0)
        grads = grads * np.asarray(clip)[:, None, None]

        key = jax.random.fold_in(key, 0)

        if logspace:
            x_new = scale * (v - stepsize * grads)
        else:
            flat = np.array(scale * (v - stepsize * grads)).reshape(-1, x.shape[-1])
            x_new = jnp.array(projection_simplex(flat).reshape(x.shape), dtype=jnp.float32)

        x_prev = x
        x = x_new

        if track_best:
            candidate_vals = score(x)
            better = np.isfinite(candidate_vals) & (
                candidate_vals < np.array(best_vals)
            )
            best_vals = jnp.where(
                jnp.asarray(better), jnp.asarray(candidate_vals), best_vals
            )
            best_x = jnp.where(jnp.asarray(better)[:, None, None], x, best_x)

        for i in range(B):
            aux_i = jax.tree.map(lambda v: v[i], auxs)
            average_nnz = (
                (x[i] > 0.01).sum(-1).mean()
                if not logspace
                else (jax.nn.softmax(x[i]) > 0.01).sum(-1).mean()
            )
            _print_iter(
                f"{_iter}[{i}]",
                {"loss": values[i], "nnz": average_nnz, "time": time.time() - start_time, "": aux_i},
                values[i],
            )

    if logspace:
        x = jax.nn.softmax(x, axis=-1)
        best_x = jax.nn.softmax(best_x, axis=-1)

    return x, best_x


def _topb_unseen_mutations(seq, g, seen, b):
    """Pick up to b 1-hop neighbours of `seq` ranked by first-order predicted delta.

    Returns (candidates, predicted_deltas) with shapes (m, N) and (m,), m <= b.
    Returns None if every 1-hop neighbour has already been seen.
    """
    N, K = g.shape
    a0 = seq.astype(np.int64)
    delta = g - g[np.arange(N), a0][:, None]
    delta[np.arange(N), a0] = np.inf

    order = np.argsort(delta.ravel(), kind="stable")

    cands = []
    deltas = []
    for idx in order:
        d = delta.ravel()[idx]
        if not np.isfinite(d):
            break
        pos, aa = divmod(int(idx), K)
        cand = seq.copy()
        cand[pos] = aa
        if cand.tobytes() in seen:
            continue
        cands.append(cand)
        deltas.append(float(d))
        if len(cands) == b:
            break

    if not cands:
        return None
    return np.stack(cands), np.asarray(deltas)


def _topb_unseen_feasible_mutations(
    seq, g, seen, b, parent, designable_mask, budget
):
    """Like `_topb_unseen_mutations` but enforces a hard edit-distance budget against `parent`.

    A candidate is feasible iff hamming(candidate, parent) <= budget. Reverts toward parent
    (Δedits=-1) and swaps within the already-edited set (Δedits=0) are always feasible;
    only mutations that introduce a *new* deviation from parent count against the budget.
    Mutations at non-designable positions are filtered out as well.

    Returns None if no feasible unseen 1-hop neighbour exists.
    """
    N, K = g.shape
    a0 = seq.astype(np.int64)
    parent = np.asarray(parent, dtype=np.int64)
    designable_mask = np.asarray(designable_mask, dtype=bool)
    current_edits = int(((a0 != parent) & designable_mask).sum())
    slack = int(budget) - current_edits

    delta = g - g[np.arange(N), a0][:, None]
    delta[np.arange(N), a0] = np.inf
    delta[~designable_mask, :] = np.inf

    order = np.argsort(delta.ravel(), kind="stable")

    cands = []
    deltas = []
    for idx in order:
        d = delta.ravel()[idx]
        if not np.isfinite(d):
            break
        pos, aa = divmod(int(idx), K)
        was_edit = int(a0[pos] != parent[pos])
        will_be_edit = int(aa != parent[pos])
        d_edits = will_be_edit - was_edit
        if d_edits > 0 and slack <= 0:
            continue
        cand = seq.copy()
        cand[pos] = aa
        if cand.tobytes() in seen:
            continue
        cands.append(cand)
        deltas.append(float(d))
        if len(cands) == b:
            break

    if not cands:
        return None
    return np.stack(cands), np.asarray(deltas)


def batch_greedy_descent(
    loss: AbstractLoss,
    sequence: Int[Array, " N"],
    *,
    batch_size: int = 16,
    steps: int = 100,
    alphabet_size: int = 20,
    key: jax.Array | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy batch hillclimb on a discrete sequence.

    Each step: compute the gradient at the current sequence, rank all
    single-point mutations by predicted first-order delta, evaluate the
    top `batch_size` unseen candidates in parallel, and greedily accept
    the best if it improves. Stops early when the full 1-hop neighbourhood
    has been evaluated.

    Args:
    - loss: loss function (called as loss(x, key=...) returning (value, aux))
    - sequence: (N,) int starting sequence
    - batch_size: number of candidate mutations evaluated per step
    - steps: maximum number of steps
    - alphabet_size: token alphabet size
    - key: jax random key (fixed across all evals for deterministic comparison)

    Returns:
    - best_seq: best sequence found
    - best_val: loss at best sequence
    """
    sequence = np.asarray(sequence, dtype=np.int32).copy()
    assert sequence.ndim == 1, f"sequence must be 1D [N], got {sequence.ndim}D"
    B = int(batch_size)

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    x0 = jax.nn.one_hot(jnp.asarray(sequence[None]), alphabet_size)
    vals, aux0, grads = batched_eval(loss, x0, jnp.broadcast_to(key, (x0.shape[0], *key.shape)))
    v = float(np.asarray(vals)[0])
    g = np.asarray(grads)[0]
    aux = jax.tree.map(lambda a: a[0], aux0)

    _print_iter("init", {"": aux}, v)

    best_seq = sequence.copy()
    best_val = v
    seen: set[bytes] = {sequence.tobytes()}

    for it in range(steps):
        start_time = time.time()

        picked = _topb_unseen_mutations(sequence, g, seen, B)
        if picked is None:
            print(f"step {it}: neighbourhood exhausted, stopping")
            break
        cands, _ = picked
        m = cands.shape[0]

        xs = jax.nn.one_hot(jnp.asarray(cands), alphabet_size)
        vals, auxs, grads_batch = batched_eval(loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape)))
        vals_np = np.asarray(vals)

        for c in cands:
            seen.add(c.tobytes())

        best_in_batch = int(np.argmin(vals_np[:m]))
        v_best = float(vals_np[best_in_batch])

        if v_best < v:
            sequence = cands[best_in_batch].copy()
            v = v_best
            g = np.asarray(grads_batch)[best_in_batch]
            aux = jax.tree.map(lambda a: a[best_in_batch], auxs)

        if v < best_val:
            best_val = v
            best_seq = sequence.copy()

        _print_iter(
            it,
            {"": aux, "time": time.time() - start_time},
            v,
        )

    return best_seq, best_val


def edit_budgeted_greedy_descent(
    loss: AbstractLoss,
    sequence: Int[Array, "N"],
    *,
    parent: Int[Array, "N"],
    budget: int,
    designable_mask: Bool[Array, "N"] | None = None,
    batch_size: int = 16,
    steps: int = 200,
    alphabet_size: int = 20,
    key: jax.Array | None = None,
):
    """Greedy hill-climb with a hard edit-distance budget against a reference sequence.

    Each step proposes single-position mutations ranked by predicted Δloss, but filters
    to only those that keep `hamming(candidate, parent) <= budget`. Reverts (Δedits=-1)
    and swaps (Δedits=0) within the already-edited positions are always feasible; only
    mutations that introduce a NEW deviation from parent consume budget.

    Records a Pareto front {edit_count: (loss, sequence)} so the user can see the best
    sequence at each edit count from 0..budget and pick a trade-off.

    Args:
        loss: composite mosaic loss; called as loss(one_hot_seq, key=...) -> (value, aux).
        sequence: starting discrete sequence (e.g. argmax of guided-diffusion output).
        parent: reference sequence; budget is enforced as hamming(_, parent) <= budget.
        budget: maximum allowed edits from parent (e.g. 7).
        designable_mask: positions allowed to change. If None, all positions are designable.
        batch_size: number of candidate mutations evaluated per step.
        steps: maximum step budget (loop usually terminates earlier when no improving
            feasible mutation exists).
        alphabet_size: token alphabet size (default 20).
        key: jax random key (held fixed across all evals for deterministic comparison).

    Returns:
        best_seq: best feasible sequence found.
        best_val: loss at best feasible sequence.
        pareto: dict {edit_count: (loss, seq_array)} of best sequence at each edit count.
    """
    sequence = np.asarray(sequence, dtype=np.int32).copy()
    parent = np.asarray(parent, dtype=np.int32).copy()
    assert sequence.shape == parent.shape, "sequence and parent must have same shape"
    assert sequence.ndim == 1, f"sequence must be 1D [N], got {sequence.ndim}D"

    if designable_mask is None:
        designable_mask = np.ones(sequence.shape, dtype=bool)
    else:
        designable_mask = np.asarray(designable_mask, dtype=bool).copy()

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    B = max(1, int(batch_size))

    def _hamming(s):
        return int(((s != parent) & designable_mask).sum())

    def _value_scores(cands: np.ndarray) -> np.ndarray:
        """Score candidate sequences in fixed-size chunks to avoid recompiles."""
        if len(cands) == 0:
            return np.asarray([], dtype=np.float32)
        chunks = []
        for start in range(0, len(cands), B):
            chunk = cands[start:start + B]
            valid = len(chunk)
            if valid < B:
                pad = np.repeat(chunk[-1][None], B - valid, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            xs = jax.nn.one_hot(jnp.asarray(chunk), alphabet_size)
            vals, _ = batched_value_eval(
                loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape))
            )
            chunks.append(np.asarray(vals)[:valid])
        return np.concatenate(chunks)

    init_edits = _hamming(sequence)
    if init_edits > budget:
        original_edits = init_edits
        while _hamming(sequence) > budget:
            edited_positions = np.where((sequence != parent) & designable_mask)[0]
            candidates = []
            for pos in edited_positions:
                cand = sequence.copy()
                cand[pos] = parent[pos]
                candidates.append(cand)
            candidates = np.stack(candidates)
            vals = _value_scores(candidates)
            sequence = candidates[int(np.argmin(vals))].copy()

        init_edits = _hamming(sequence)
        kept_positions = np.where((sequence != parent) & designable_mask)[0] + 1
        print(
            f"warn: starting sequence had {original_edits} edits > budget {budget}; "
            f"loss-projected to {init_edits} edits before greedy descent. "
            f"kept chain-order positions: {','.join(map(str, kept_positions))}"
        )

    x0 = jax.nn.one_hot(jnp.asarray(sequence[None]), alphabet_size)
    vals, aux0, grads = batched_eval(
        loss, x0, jnp.broadcast_to(key, (x0.shape[0], *key.shape))
    )
    v = float(np.asarray(vals)[0])
    g = np.asarray(grads)[0]
    aux = jax.tree.map(lambda a: a[0], aux0)

    _print_iter("init", {"": aux, "edits": float(init_edits)}, v)

    best_seq = sequence.copy()
    best_val = v
    seen: set[bytes] = {sequence.tobytes()}

    pareto: dict[int, tuple[float, np.ndarray]] = {init_edits: (v, sequence.copy())}

    for it in range(steps):
        start_time = time.time()

        picked = _topb_unseen_feasible_mutations(
            sequence, g, seen, B, parent, designable_mask, budget
        )
        if picked is None:
            print(f"step {it}: feasible neighbourhood exhausted, stopping")
            break
        cands, _ = picked
        m = cands.shape[0]

        xs = jax.nn.one_hot(jnp.asarray(cands), alphabet_size)
        vals, auxs = batched_value_eval(
            loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape))
        )
        vals_np = np.asarray(vals)

        for c in cands:
            seen.add(c.tobytes())

        for i in range(m):
            cand_edits = _hamming(cands[i])
            cand_v = float(vals_np[i])
            existing = pareto.get(cand_edits)
            if existing is None or cand_v < existing[0]:
                pareto[cand_edits] = (cand_v, cands[i].copy())

        best_in_batch = int(np.argmin(vals_np[:m]))
        v_best = float(vals_np[best_in_batch])

        if v_best < v:
            sequence = cands[best_in_batch].copy()
            v = v_best
            aux = jax.tree.map(lambda a: a[best_in_batch], auxs)
            x_current = jax.nn.one_hot(jnp.asarray(sequence[None]), alphabet_size)
            _, _, grads = batched_eval(
                loss,
                x_current,
                jnp.broadcast_to(key, (x_current.shape[0], *key.shape)),
            )
            g = np.asarray(grads)[0]

        if v < best_val:
            best_val = v
            best_seq = sequence.copy()

        _print_iter(
            it,
            {"": aux, "edits": float(_hamming(sequence)), "time": time.time() - start_time},
            v,
        )

    assert _hamming(best_seq) <= budget, (
        f"internal error: best_seq has {_hamming(best_seq)} edits, exceeds budget {budget}"
    )
    return best_seq, best_val, pareto


def edit_budgeted_gradient_mcmc(
    loss: AbstractLoss,
    sequence: Int[Array, "N"],
    *,
    parent: Int[Array, "N"],
    budget: int,
    designable_mask: Bool[Array, "N"] | None = None,
    steps: int = 100,
    batch_size: int = 3,
    temp: float = 0.02,
    proposal_temp: float = 0.01,
    max_path_length: int = 2,
    alphabet_size: int = 20,
    key: jax.Array | None = None,
):
    """Budgeted, CDR-restricted gradient-assisted MCMC over discrete sequences.

    The state is a hard sequence. Proposals are 1..`max_path_length` mutations
    sampled from a gradient-biased proposal distribution, while rejecting any
    path that leaves the designable mask or exceeds the edit budget relative to
    `parent`. The Metropolis accept/reject step uses the actual loss value.

    Returns:
        best_seq: best accepted/evaluated feasible sequence found.
        best_val: loss at best_seq.
        pareto: dict {edit_count: (loss, seq_array)}.
    """
    sequence = np.asarray(sequence, dtype=np.int32).copy()
    parent = np.asarray(parent, dtype=np.int32).copy()
    assert sequence.shape == parent.shape, "sequence and parent must have same shape"
    assert sequence.ndim == 1, f"sequence must be 1D [N], got {sequence.ndim}D"

    if designable_mask is None:
        designable_mask = np.ones(sequence.shape, dtype=bool)
    else:
        designable_mask = np.asarray(designable_mask, dtype=bool).copy()

    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))
    rng_seed = int(np.asarray(jax.random.randint(key, (), 0, 2**31 - 1)))
    rng = np.random.default_rng(rng_seed)
    max_path_length = max(1, int(max_path_length))
    B = max(1, int(batch_size))

    def _hamming(s):
        return int(((s != parent) & designable_mask).sum())

    def _eval(seq):
        x = jax.nn.one_hot(jnp.asarray(seq[None]), alphabet_size)
        vals, auxs, grads = batched_eval(
            loss, x, jnp.broadcast_to(key, (x.shape[0], *key.shape))
        )
        return (
            float(np.asarray(vals)[0]),
            jax.tree.map(lambda a: a[0], auxs),
            np.asarray(grads)[0],
        )

    def _value_scores(cands: np.ndarray) -> np.ndarray:
        if len(cands) == 0:
            return np.asarray([], dtype=np.float32)
        chunks = []
        for start in range(0, len(cands), B):
            chunk = cands[start:start + B]
            valid = len(chunk)
            if valid < B:
                pad = np.repeat(chunk[-1][None], B - valid, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            xs = jax.nn.one_hot(jnp.asarray(chunk), alphabet_size)
            vals, _ = batched_value_eval(
                loss, xs, jnp.broadcast_to(key, (xs.shape[0], *key.shape))
            )
            chunks.append(np.asarray(vals)[:valid])
        return np.concatenate(chunks)

    init_edits = _hamming(sequence)
    if init_edits > budget:
        original_edits = init_edits
        while _hamming(sequence) > budget:
            edited_positions = np.where((sequence != parent) & designable_mask)[0]
            candidates = []
            for pos in edited_positions:
                cand = sequence.copy()
                cand[pos] = parent[pos]
                candidates.append(cand)
            candidates = np.stack(candidates)
            vals = _value_scores(candidates)
            sequence = candidates[int(np.argmin(vals))].copy()
        init_edits = _hamming(sequence)
        kept_positions = np.where((sequence != parent) & designable_mask)[0] + 1
        print(
            f"warn: MCMC starting sequence had {original_edits} edits > budget {budget}; "
            f"loss-projected to {init_edits} edits. "
            f"kept chain-order positions: {','.join(map(str, kept_positions))}"
        )

    def _sample_one_mutation(seq, g):
        N, K = g.shape
        a0 = seq.astype(np.int64)
        current_edits = _hamming(a0)
        slack = int(budget) - current_edits

        delta = g - g[np.arange(N), a0][:, None]
        delta[np.arange(N), a0] = np.inf
        delta[~designable_mask, :] = np.inf

        for pos in range(N):
            was_edit = int(a0[pos] != parent[pos])
            for aa in range(K):
                if not np.isfinite(delta[pos, aa]):
                    continue
                will_be_edit = int(aa != parent[pos])
                if will_be_edit - was_edit > 0 and slack <= 0:
                    delta[pos, aa] = np.inf

        flat = delta.ravel()
        valid = np.flatnonzero(np.isfinite(flat))
        if len(valid) == 0:
            return None
        logits = -flat[valid] / max(float(proposal_temp), 1e-8)
        probs = softmax(logits)
        choice_i = int(rng.choice(len(valid), p=probs))
        flat_idx = int(valid[choice_i])
        pos, aa = divmod(flat_idx, K)
        cand = seq.copy()
        cand[pos] = aa
        return cand, (pos, aa), float(np.log(probs[choice_i] + 1e-300))

    v, aux, g = _eval(sequence)
    _print_iter("mcmc init", {"": aux, "edits": float(init_edits)}, v)

    best_seq = sequence.copy()
    best_val = v
    pareto: dict[int, tuple[float, np.ndarray]] = {init_edits: (v, sequence.copy())}
    accepted = 0

    for it in range(int(steps)):
        start_time = time.time()
        proposal = sequence.copy()
        mutations = []
        path_len = int(rng.integers(1, max_path_length + 1))

        for _ in range(path_len):
            sampled = _sample_one_mutation(proposal, g)
            if sampled is None:
                break
            proposal, mutation, _ = sampled
            mutations.append(mutation)

        if not mutations or np.array_equal(proposal, sequence):
            print(f"mcmc {it}: no feasible nontrivial proposal, stopping")
            break

        v_prop, aux_prop, g_prop = _eval(proposal)
        prop_edits = _hamming(proposal)
        existing = pareto.get(prop_edits)
        if existing is None or v_prop < existing[0]:
            pareto[prop_edits] = (v_prop, proposal.copy())

        if temp <= 0:
            log_accept = 0.0 if v_prop < v else -np.inf
        else:
            log_accept = min(0.0, (v - v_prop) / float(temp))
        accept = np.log(rng.random() + 1e-300) < log_accept

        if accept:
            sequence = proposal.copy()
            v, aux, g = v_prop, aux_prop, g_prop
            accepted += 1

        if v_prop < best_val:
            best_val = v_prop
            best_seq = proposal.copy()

        if it == 0 or (it + 1) % 10 == 0 or accept:
            _print_iter(
                f"mcmc {it}",
                {
                    "": aux,
                    "proposal_loss": float(v_prop),
                    "accepted": float(accepted),
                    "edits": float(_hamming(sequence)),
                    "time": time.time() - start_time,
                },
                v,
            )

    assert _hamming(best_seq) <= budget, (
        f"internal error: best_seq has {_hamming(best_seq)} edits, exceeds budget {budget}"
    )
    return best_seq, best_val, pareto


def biohub_optimizer(
    *,
    loss_function,
    logits: Float[Array, "B N 20"],
    n_steps: int = 150,
    learning_rate: float = 0.1,
    temperature_min: float = 1e-2,
    mask: Float[Array, "N 20"] | None = None,
    key: jax.Array,
    verbose: bool = False,
    tail_loss_function=None,
    tail_select_steps: int = 20,
    beta: float = 0.0,
    ranking_aux_name: str = "ranking_loss"
):
    """Standard Biohub ESMFold2 optimizer (Algorithm 11 in https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1)

    Returns `(best_designs, best_loss)` sorted best-first. `best_designs` is
    argmax-ready only: improved tail iterates are stored as softmax
    probabilities, so consume it via `best_designs.argmax(-1)`.
    """
    assert 0 < tail_select_steps <= n_steps, (
        "tail_select_steps must be in (0, n_steps]; otherwise no iterate is ever selected"
    )
    logits = jnp.array(logits)

    if mask is not None:
        assert mask.ndim == 2
        mask = jnp.array(mask)[None]
        logits = jnp.where(mask > 0, logits, -1e9)

    if tail_loss_function is None:
        tail_loss_function = loss_function

    _warned_ranking = False

    def get_ranking_values(values, aux):
        nonlocal _warned_ranking
        leaf = _ranking_leaf(aux, ranking_aux_name)
        if leaf is None:
            if not _warned_ranking:
                print(f"Warning: biohub_optimizer ranking by loss value, not {ranking_aux_name}")
                _warned_ranking = True
            return values
        return leaf


    assert logits.ndim == 3, "biohub_optimizer is batched, `logits` must have shape [B, N, 20]"
    B = logits.shape[0]

    best_designs = np.asarray(logits).copy()
    best_loss = np.full(B, np.inf)
    v = np.zeros_like(logits)

    for step in range(n_steps):
        start_time = time.time()
        t = (step + 1) / n_steps
        temp = temperature_min + (1 - temperature_min) * 0.5 * (1 + np.cos(np.pi * t))

        in_tail = step >= n_steps - tail_select_steps
        lf = tail_loss_function if in_tail else loss_function

        design = jax.nn.softmax(logits / temp, axis=-1)
        values, aux, g_design = batched_eval(lf, design, jax.random.split(key, B))
        key = jax.random.fold_in(key, 0)

        if in_tail:
            ranking_values = get_ranking_values(values, aux)
            improved = best_loss > ranking_values                      
            design_np, vals_np = np.asarray(design), np.asarray(ranking_values)
            best_designs[improved] = design_np[improved]
            best_loss[improved] = vals_np[improved]

        xg = (design * g_design).sum(-1, keepdims=True)
        g_z = (design * (g_design - xg)) / temp
        if mask is not None:
            g_z = g_z * mask

        eff_L = (jnp.square(g_z).sum(-1) > 0).sum(-1)
        gnorm = jnp.linalg.norm(g_z, axis=(-1, -2))
        g_z = g_z / (gnorm[:, None, None] + 1e-7) * jnp.sqrt(eff_L)[:, None, None]

        v = beta * v + (learning_rate * temp) * g_z

        logits = logits - v

        for i in range(B):
            aux_i = jax.tree.map(lambda v: v[i], aux)
            average_nnz = (
                (design[i] > 0.01).sum(-1).mean()
            )
            _print_iter(
                f"{step}[{i}]",
                {"loss": values[i], "time": time.time() - start_time, "": aux_i, "nnz": average_nnz, "temp": temp},
                values[i],

            )
        if mask is not None:
            logits = jnp.where(mask > 0, logits, -1e9)

    order = np.argsort(best_loss)

    return best_designs[order], best_loss[order]


def _apportion_stages(counts, n_steps):
    """Rescale stage lengths to sum to EXACTLY `n_steps`, each at least 1.

    Largest-remainder apportionment. An earlier version rounded each stage and
    dumped the whole remainder on the longest one, clamped at a floor of 1 --
    and the clamp could swallow part of the correction, so the total came out
    wrong: four equal stages at `n_steps=6` ran 7 steps, silently spending more
    GPU than the caller asked for. Floor-then-distribute cannot overshoot,
    because the remainder handed out is exactly `n_steps - sum(floor)`, which is
    strictly less than the number of stages.

    Exact at `n_steps == sum(counts)`, so a published schedule passed at its own
    total comes back unchanged.
    """
    k = len(counts)
    total = sum(counts)
    if total <= 0:
        raise ValueError("the stage lengths must sum to a positive number")
    if n_steps < k:
        raise ValueError(
            f"n_steps={n_steps} cannot cover {k} stages; pass at least {k}"
        )
    exact = [c * n_steps / total for c in counts]
    scaled = [int(x) for x in exact]
    order = sorted(range(k), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for j in range(n_steps - sum(scaled)):
        scaled[order[j]] += 1
    for i in range(k):
        if scaled[i] == 0:
            donor = max(range(k), key=lambda j: scaled[j])
            scaled[donor] -= 1
            scaled[i] = 1
    if sum(scaled) != n_steps or min(scaled) < 1:
        raise AssertionError(
            f"apportioning {counts} to {n_steps} produced {scaled}"
        )
    return scaled


def _colabdesign_transform(z, soft, temp, hard, sample_key=None):
    """Map ColabDesign logits to the sequence passed to the user's loss.

    `sample_key` switches the hard branch from ColabDesign's argmax to a
    categorical SAMPLE, which is Fast SeqProp's delta operator (Linder & Seelig
    2021, arXiv:2005.11275). They report the stochastic sampler reaching better
    optima than the deterministic softmax relaxation; the gradient path is
    unchanged either way, since the straight-through estimator substitutes the
    softmax Jacobian for the discrete step whatever produced the one-hot.
    """
    soft_seq = jax.nn.softmax(z / temp)
    if sample_key is None:
        picks = soft_seq.argmax(-1)
    else:
        picks = jax.random.categorical(sample_key, z / temp, axis=-1)
    hard_seq = jax.nn.one_hot(picks, z.shape[-1])
    hard_seq = jax.lax.stop_gradient(hard_seq - soft_seq) + soft_seq
    pseudo = soft * soft_seq + (1.0 - soft) * z
    return hard * hard_seq + (1.0 - hard) * pseudo




def _colabdesign_pullback(z, g, soft, temp, hard, sample_key=None):
    """Pull a sequence gradient back to ColabDesign logits."""
    _, pullback = jax.vjp(
        lambda z: _colabdesign_transform(z, soft, temp, hard, sample_key),
        z,
    )
    (g_z,) = pullback(g)
    return g_z


def _calibrate_logit_norm_scale(x, minimum=1e-2):
    """Gamma from the opening spread of `x`, refusing a degenerate one.

    Calibrating to the start makes step 0 a no-op, so normalisation removes
    drift without moving where the caller put the iterate. Two traps, both hit
    on 2026-10-13:

    - A UNIFORM start has zero spread, so gamma is 0 and
      `_layer_norm_logits` multiplies every subsequent update back to exactly
      zero. The optimiser then cannot move at all, with a perfectly good
      gradient. ColabDesign's own un-seeded start is `np.zeros`, and this
      module's `--init noise` start has spread ~0.01, so both are affected.
    - Gamma RATIONS confidence across positions. After normalisation the
      logits have standard deviation gamma over the WHOLE (N, 20) block, not
      per row, so it does not cap any single row: concentrating the spread in
      one row of 29 reaches p=0.91 at gamma 0.218, because one entry can exceed
      the global deviation by up to about sqrt(29 * 20) = 24. What it does
      bound is the aggregate -- every row cannot be peaked at once. An earlier
      version of this docstring claimed a per-residue cap of about p=0.125 at
      that gamma; that was measured on a (1, 20) array, the one shape where the
      distinction is invisible, and is withdrawn. See
      `tests/test_continuous_seed.py::test_gamma_bounds_the_aggregate_spread_not_each_row`.

    Refusing is better than either silent failure. Pass `logit_norm_scale`
    explicitly to choose the regime deliberately -- 1.0 is the usual
    initialisation for a normalisation layer's gamma, and is what Fast SeqProp
    (arXiv:2005.11275) then LEARNS rather than fixes.
    """
    spread = float(jnp.sqrt(jnp.square(x - x.mean()).mean()))
    if not np.isfinite(spread) or spread < minimum:
        raise ValueError(
            f"logit_norm_scale could not be calibrated: the opening logits have "
            f"spread {spread:.3e}, below the {minimum:g} floor. Normalising to it "
            "would hold every iterate at that spread, so the softmax would stay "
            "uniform and the optimisation would not move. Pass logit_norm_scale "
            "explicitly (1.0 is the usual gamma for a normalisation layer)."
        )
    return spread


def _layer_norm_logits(x, scale):
    """Fast SeqProp's logit normalisation, protein variant.

    Linder & Seelig (arXiv:2005.11275) identify "overly large and
    disproportionally scaled logits" as the bottleneck of relaxed sequence
    design, in both convergence speed AND the optima reached, because the
    softmax Jacobian vanishes once any coordinate saturates. Their fix inserts
    a normalisation layer between the trainable logits and the sampler, so the
    logits are re-centred every step and cannot drift into saturation. For
    20-channel alphabets they report LAYER normalisation (one global mean and
    deviation, a shared scale) as more stable than per-channel instance
    normalisation, because 20 channels give noisy per-channel statistics.

    `scale` is their gamma. The caller passes the initial spread so that step 0
    is a no-op: normalisation then removes DRIFT without moving the start, and
    `--init-logit-scale` keeps its meaning. A fixed gamma of 1.0 would instead
    rescale the opening gap to the block's own standard deviation, which for a
    parent-anchored start AMPLIFIES it (a gap of 1.0 over 29x20 becomes ~4.6)
    and re-saturates exactly what this is meant to prevent.
    """
    mean = x.mean()
    deviation = jnp.sqrt(jnp.square(x - mean).mean())
    return scale * (x - mean) / (deviation + 1e-7)


def _seq_grad_norm(g):
    """Lifted from ColabDesign _norm_seq_grad."""
    eff_L = (jnp.square(g).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
    gn = jnp.linalg.norm(g, axis=(-1, -2), keepdims=True)
    return g * jnp.sqrt(eff_L) / (gn + 1e-7)


def colabdesign_stage(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "N 20"],
    n_steps: int,
    soft_start: float,
    soft_end: float,
    temp_start: float,
    temp_end: float,
    hard: bool,
    lr: float,
    norm_seq_grad: bool = True,
    max_gradient_norm: float | None = None,
    logit_norm: bool = False,
    logit_norm_scale: float | None = None,
    stochastic_hard: bool = False,
    key=None,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
):
    """
    One ColabDesign/AfDesign stage: n_steps of normalised SGD on a logit iterate,
    with soft ramping linearly, temp annealing quadratically, and hard fixed. Logits
    in and out, so stages chain with no softmax round-trip (the BindCraft recipe).

    Args:
    - loss_function: function to minimize
    - x: initial logits (N x K), centered per row on entry
    - n_steps: number of optimization steps
    - soft_start, soft_end: linear soft ramp (0 = raw logits, 1 = softmax)
    - temp_start, temp_end: quadratic temp anneal
    - hard: if True, blend in a straight-through one-hot for the stage
    - lr: base learning rate (ColabDesign default 0.1)
    - norm_seq_grad: normalise the gradient to sqrt(L); else clip at max_gradient_norm
    - max_gradient_norm: clip norm when norm_seq_grad is False (default sqrt(N))
    - logit_norm: re-normalise the LOGITS after every update (Fast SeqProp,
      arXiv:2005.11275). Off by default so existing callers are unchanged.
    - logit_norm_scale: gamma for that normalisation. None (default) calibrates
      it to the spread of `x` on entry, making step 0 a no-op -- see
      `_layer_norm_logits` for why a fixed 1.0 re-saturates a parent-anchored
      start instead of fixing it.
    - stochastic_hard: in the hard branch, SAMPLE the one-hot rather than take
      the argmax (Fast SeqProp's stochastic sampler). Off by default.
    - key: jax random key
    - trajectory_fn: takes (aux, x) and returns any value

    returns:
    - x: final logits
    - trajectory: list of trajectory information if `trajectory_fn` is provided
    """
    if max_gradient_norm is None:
        max_gradient_norm = float(np.sqrt(x.shape[0]))
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    x = jnp.asarray(x, dtype=jnp.float32)
    x = x - x.mean(-1, keepdims=True)
    if logit_norm and logit_norm_scale is None:
        logit_norm_scale = _calibrate_logit_norm_scale(x)
    trajectory = []

    for _iter in range(n_steps):
        start_time = time.time()
        frac = (_iter + 1) / n_steps
        soft = soft_start + (soft_end - soft_start) * frac
        temp = temp_end + (temp_start - temp_end) * (1 - frac) ** 2

        soft = jnp.asarray(soft, dtype=jnp.float32)
        temp = jnp.asarray(temp, dtype=jnp.float32)
        hard = jnp.asarray(hard, dtype=jnp.float32)
        sample_key = jax.random.fold_in(key, 1) if stochastic_hard else None
        pseudo = _colabdesign_transform(x, soft, temp, hard, sample_key)
        (value, aux), g = _eval_loss_and_grad(loss_function, pseudo, key)
        g = _colabdesign_pullback(x, g, soft, temp, hard, sample_key)
        value = float(value)
        key = jax.random.fold_in(key, 0)

        if norm_seq_grad:
            g = _seq_grad_norm(g)
        else:
            n = np.sqrt((g**2).sum())
            if n > max_gradient_norm:
                g = g * (max_gradient_norm / n)

        lr_scale = (1.0 - soft) + soft * temp
        x = x - (lr * lr_scale) * g
        if logit_norm:
            x = _layer_norm_logits(x, logit_norm_scale)

        average_nnz = float((jax.nn.softmax(x) > 0.01).sum(-1).mean())
        aux = {
            "loss": value,
            "nnz": average_nnz,
            "time": time.time() - start_time,
            "soft": float(soft),
            "temp": float(temp),
            "hard": float(hard),
            "": aux,
        }
        if trajectory_fn is not None:
            trajectory.append(trajectory_fn(aux, x))

        _print_iter(
            _iter,
            eqx.filter(aux, lambda v: isinstance(v, float) or v.shape == ()),
            value,
        )

    if trajectory_fn is None:
        return x
    return x, trajectory


def bindcraft_design(
    *,
    loss_function: AbstractLoss,
    x: Float[Array, "N 20"],
    lr: float = 0.1,
    logits_iters: tuple[int, int] = (50, 25),
    soft_iters: int = 45,
    hard_iters: int = 5,
    n_steps: int | None = None,
    hard_temp: float = 1e-2,
    logit_norm: bool = False,
    logit_norm_scale: float | None = None,
    stochastic_hard: bool = False,
    key=None,
    trajectory_fn: Callable[tuple[PyTree, Float[Array, "N 20"]], any] | None = None,
):
    """Run the default four-stage BindCraft gradient design schedule.

    The caller supplies initial logits. To match the paper initialization, use
    x = 0.01 * jax.random.normal(key, (N, K)).
    
    Parameters follow default_4stage_multimer.json in BindCraft package.
    See the "materials and methods" section here: https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1

    Args:
    - loss_function: function to minimise.
    - x: initial logits with shape (N, K).
    - lr: base learning rate.
    - logits_iters: step counts for the two logits stages.
    - soft_iters: step count for the soft stage.
    - hard_iters: step count for the hard stage.
    - n_steps: total budget. When given, the four stage lengths above are
      rescaled to sum to it, keeping BindCraft's 50:25:45:5 proportions. This
      exists because the schedule was previously reachable only by passing all
      four counts, so a caller handing over a single `--steps` was silently
      ignored and every run was 125 steps whatever it asked for.
    - hard_temp: temperature of the final hard stage. BindCraft's own value is
      1e-2, which is the default here. Note what that costs: the stage feeds a
      straight-through one-hot whose gradient is the Jacobian of
      softmax(z / hard_temp), and at 1e-2 that softmax is saturated, so the
      estimator returns almost nothing -- while `lr_scale = soft*temp` shrinks
      the step by another 100x. Raise it toward 1.0 for a hard stage that can
      actually move, which is the regime Fast SeqProp's straight-through
      sampler operates in.
    - logit_norm, logit_norm_scale, stochastic_hard: forwarded to every stage;
      see `colabdesign_stage`.
    - key: JAX random key.
    - trajectory_fn: optional callback receiving (aux, x).

    Returns:
    - pssm: final soft sequence with shape (N, K).
    - trajectory: concatenated callback results, when requested.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    n1, n2 = logits_iters
    if n_steps is not None:
        if n_steps < 4:
            raise ValueError(
                f"n_steps={n_steps} cannot cover four stages; pass at least 4"
            )
        counts = (n1, n2, soft_iters, hard_iters)
        total = sum(counts)
        if total <= 0:
            raise ValueError("the stage lengths must sum to a positive number")
        n1, n2, soft_iters, hard_iters = _apportion_stages(counts, n_steps)
    stages = [
        {
            "n_steps": n1,
            "soft_start": 0.0,
            "soft_end": 0.9,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "hard": False,
        },
        {
            "n_steps": n2,
            "soft_start": 0.9,
            "soft_end": 1.0,
            "temp_start": 1.0,
            "temp_end": 1.0,
            "hard": False,
        },
        {
            "n_steps": soft_iters,
            "soft_start": 1.0,
            "soft_end": 1.0,
            "temp_start": 1.0,
            "temp_end": 1e-2,
            "hard": False,
        },
        {
            "n_steps": hard_iters,
            "soft_start": 1.0,
            "soft_end": 1.0,
            "temp_start": hard_temp,
            "temp_end": hard_temp,
            "hard": True,
        },
    ]
    stage_keys = jax.random.split(key, len(stages))

    if logit_norm and logit_norm_scale is None:
        x0 = jnp.asarray(x, dtype=jnp.float32)
        logit_norm_scale = _calibrate_logit_norm_scale(x0 - x0.mean(-1, keepdims=True))

    trajectory = []
    for stage, k in zip(stages, stage_keys):
        out = colabdesign_stage(
            loss_function=loss_function,
            x=x,
            lr=lr,
            key=k,
            logit_norm=logit_norm,
            logit_norm_scale=logit_norm_scale,
            stochastic_hard=stochastic_hard,
            trajectory_fn=trajectory_fn,
            **stage,
        )
        x, t = out if trajectory_fn is not None else (out, [])
        trajectory += t

    pssm = jax.nn.softmax(x)
    if trajectory_fn is None:
        return pssm
    return pssm, trajectory
