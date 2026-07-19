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


def _print_iter(iter, aux, v):
    # first filter out anything that isn't a float or has number of dimensions > 0
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
            if hasattr(v, "item")
            or isinstance(v, float)
        ),
    )


# Split this up so changing optim parameters doesn't trigger re-compilation of loss function
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
    # standardize input to avoid recompilation
    x = np.array(x, dtype=np.float32)
    (v, aux), g = _____eval_loss_and_grad(loss_function, x=x, key=key)
    return (jnp.nan_to_num(v, nan=1000000.0), aux), jnp.nan_to_num(
        g - g.mean(axis=-1, keepdims=True)
    )


# more underscores == more private
@eqx.filter_jit
def _____eval_loss_and_grad(loss, x, key):
    return eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)


@eqx.filter_jit
def batched_eval(
    loss: AbstractLoss,
    xs: Float[Array, "B N K"],
    keys: jax.Array,
) -> tuple[Float[Array, "B"], PyTree, Float[Array, "B N K"]]:
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



# def _proposal(sequence, g, temp, alphabet_size: int = 20):
#     input = jax.nn.one_hot(sequence, alphabet_size)
#     g_i_x_i = (g * input).sum(-1, keepdims=True)
#     logits = -((input * g).sum() - g_i_x_i + g) / temp
#     return jax.nn.softmax(logits), jax.nn.log_softmax(logits)


# rewrite in numpy to use float64
# note: this _does not match the taylor expansion estimate (above)_ (there's an extra normalization). seems better though.
def _proposal(sequence, g, temp, alphabet_size: int = 20):
    input = np.eye(alphabet_size)[sequence]
    g_i_x_i = (g * input).sum(-1, keepdims=True)
    logits = -((input * g).sum(-1, keepdims=True) - g_i_x_i + g) / temp
    return softmax(logits, axis=-1), log_softmax(logits, axis=-1)


def gradient_MCMC(
    loss,
    sequence: Int[Array, "N"],
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
    for iter in range(steps):
        start_time = time.time()
        ### generate a proposal

        for i in range(50):
            proposal = sequence.copy()
            mutations = []
            log_q_forward = 0.0
            path_length = jax.random.randint(
                key=jax.random.key(np.random.randint(10000)),
                minval=1,
                maxval=max_path_length + 1,
                shape=(),
            )
            key = jax.random.fold_in(key, 0)
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
                mutations += [(position, AA)]
                proposal = proposal.at[position].set(AA)
            # check if proposal is same as current sequence
            if np.all(proposal == sequence):
                print(f"\t {i}: proposal is the same as current sequence, skipping.")
                # _print_iter(iter, {"": aux_0, "time": time.time() - start_time}, v_0)
                # continue
            else:
                break
        muts = ", ".join([f"{pos}:{aa}" for (pos, aa) in mutations])
        print(f"Proposed mutations: {muts}")

        ### evaluate the proposal
        (v_1, aux_1), g_1 = _eval_loss_and_grad(
            loss,
            jax.nn.one_hot(proposal, alphabet_size),
            key=key_model if fix_loss_key else key,
        )

        # next bit is to calculate the backward probability, which is only used
        # if detailed_balance is True
        prop_backward = proposal.copy()
        log_q_backward = 0.0
        for position, AA in reversed(mutations):
            p, log_p = _proposal(
                prop_backward, g_1, proposal_temp, alphabet_size=alphabet_size
            )
            log_q_backward += log_p[position, AA]
            prop_backward = prop_backward.at[position].set(AA)

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

    best_val = np.inf
    x = projection_simplex(x) if not logspace else x
    best_x = x

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

        if value < best_val and not np.isnan(value):
            best_val = value
            best_x = (
                x  # this isn't exactly right, because we evaluated loss at v, not x.
            )

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

    best_vals = jnp.full(B, jnp.inf)
    best_x = x
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

        better = (np.array(values) < np.array(best_vals)) & ~np.isnan(values)
        best_vals = jnp.where(jnp.array(better), values, best_vals)
        best_x = jnp.where(jnp.array(better)[:, None, None], x, best_x)

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
    delta[np.arange(N), a0] = np.inf  # mask no-ops

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
    slack = int(budget) - current_edits  # how many more "new edits" we can afford

    delta = g - g[np.arange(N), a0][:, None]
    delta[np.arange(N), a0] = np.inf  # mask no-ops
    # Forbid mutations at non-designable positions
    delta[~designable_mask, :] = np.inf

    order = np.argsort(delta.ravel(), kind="stable")

    cands = []
    deltas = []
    for idx in order:
        d = delta.ravel()[idx]
        if not np.isfinite(d):
            break
        pos, aa = divmod(int(idx), K)
        # Net change in edit count if we apply this single-position change
        was_edit = int(a0[pos] != parent[pos])
        will_be_edit = int(aa != parent[pos])
        d_edits = will_be_edit - was_edit  # in {-1, 0, +1}
        if d_edits > 0 and slack <= 0:
            continue  # would push us over budget
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
    sequence: Int[Array, "N"],
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

    # initial eval
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
        # Starting sequence already violates the hard edit budget. Project it
        # back by greedily choosing which diffusion edits to revert under the
        # actual polish loss, instead of dropping arbitrary positions.
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

    # initial eval
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

    # Pareto front: best sequence at each edit count 0..budget
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

        # Update Pareto front with every feasible candidate's loss
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

    # Final invariant check
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
