"""Discrete gradient proposals with independent or population retention.

The gradient callback minimizes a cheap loss; the confidence callback returns a
score to MAXIMIZE. Both callbacks must be deterministic for a sequence (e.g. use
a fixed, recorded set of model seeds). Results are cached by complete sequence.
This is an optimization heuristic, not a Metropolis-Hastings sampler.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class SearchConfig:
    policy: Literal["independent", "population"] = "independent"
    width: int = 4
    edit_budget: int = 5
    alphabet_size: int = 20
    max_score_calls: int = 100
    max_gradient_calls: int = 100
    max_proposals: int = 1000
    target_entropy: float = 0.6
    acceptance_temperature: float = 0.02
    seed: int = 0
    move_arity: int = 1
    pareto_metric: str | None = None
    pareto_ceiling: float | None = None

    def __post_init__(self):
        if self.policy not in ("independent", "population"):
            raise ValueError("policy must be independent or population")
        for name, minimum in (
            ("width", 1),
            ("edit_budget", 0),
            ("alphabet_size", 2),
            ("max_score_calls", 1),
            ("max_gradient_calls", 1),
            ("max_proposals", 0),
            ("seed", 0),
            ("move_arity", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.move_arity > 2:
            raise ValueError(
                "move_arity above 2 is not implemented: the triple move set is "
                "~2.5e7 moves and would need a pruning rule, and pruning by "
                "summed delta discards exactly the combinations a multi-position "
                "operator exists to reach"
            )
        if not 0 < self.target_entropy <= 1:
            raise ValueError("target_entropy must be in (0, 1]")
        if (
            not np.isfinite(self.acceptance_temperature)
            or self.acceptance_temperature < 0
        ):
            raise ValueError("acceptance_temperature must be finite and >= 0")
        if (self.pareto_metric is None) != (self.pareto_ceiling is None):
            raise ValueError("pareto_metric and pareto_ceiling must be supplied together")
        if self.pareto_metric is not None:
            if not isinstance(self.pareto_metric, str) or not self.pareto_metric.strip():
                raise ValueError("pareto_metric must be a nonempty metric name")
            if self.policy != "population" or self.width < 2:
                raise ValueError(
                    "Pareto retention requires population policy and width >= 2"
                )
            if not np.isfinite(self.pareto_ceiling):
                raise ValueError("pareto_ceiling must be finite")


@dataclass(frozen=True)
class ConfidenceScore:
    value: float
    metrics: dict[str, float] = field(default_factory=dict)
    constraint_violation: float = 0.0


@dataclass(frozen=True)
class Candidate:
    id: int
    sequence: tuple[int, ...]
    score: float
    metrics: dict[str, float]
    constraint_violation: float = 0.0

    @property
    def retention_key(self):
        """Lower violation first, then higher confidence; larger is better."""
        return (-self.constraint_violation, self.score)


@dataclass
class SearchResult:
    best: Candidate
    active: list[Candidate]
    evaluated: list[Candidate]
    cheap_losses: dict[int, float]
    stats: dict[str, int | float]
    stop_reason: str
    pareto_front: list[Candidate] = field(default_factory=list)


def candidate_selection_key(candidate: Candidate, config: SearchConfig):
    """Larger is better; ties select the earliest evaluation.

    Pareto mode selects the highest score satisfying the metric ceiling. If no
    candidate qualifies, it falls back to the smallest metric, then score.
    Explicit constraint violations always take priority in either mode.
    """
    prefix = (-candidate.constraint_violation,)
    if config.pareto_metric is None:
        return (*prefix, candidate.score, -candidate.id)
    metric = candidate.metrics[config.pareto_metric]
    qualified = metric <= config.pareto_ceiling
    return (
        *prefix,
        qualified,
        candidate.score if qualified else -metric,
        -metric if qualified else candidate.score,
        -candidate.id,
    )


def pareto_front(candidates: list[Candidate], metric: str) -> list[Candidate]:
    """Maximize score and minimize a metric, at the lowest constraint violation.

    Objective ties retain the earliest evaluation. The returned frontier is
    ordered by increasing metric, with strictly increasing scores.
    """
    if not candidates:
        return []
    violation = min(c.constraint_violation for c in candidates)
    ordered = sorted(
        (c for c in candidates if c.constraint_violation == violation),
        key=lambda c: (c.metrics[metric], -c.score, c.id),
    )
    front, best_score = [], -np.inf
    for candidate in ordered:
        if candidate.score > best_score:
            front.append(candidate)
            best_score = candidate.score
    return front


def _pareto_population(front: list[Candidate], metric: str, width: int):
    """Keep both extremes, then normalized crowding distance; pad small fronts.

    This uses the entire evaluated frontier, including points previously omitted
    from the active population. It makes no model calls and uses no randomness.
    """
    if len(front) > width:
        distance = np.zeros(len(front))
        distance[[0, -1]] = np.inf
        for values in (
            np.array([c.score for c in front]),
            np.array([c.metrics[metric] for c in front]),
        ):
            span = values[-1] - values[0]
            if span > 0:
                distance[1:-1] += (values[2:] - values[:-2]) / span
        indices = sorted(
            range(len(front)), key=lambda i: (-distance[i], front[i].id)
        )[:width]
        front = [front[i] for i in indices]
    chosen = sorted(front, key=lambda c: c.id)
    return [chosen[i % len(chosen)] for i in range(width)]


def _moves(sequence, wt, mask, budget, gradient):
    """One substitution/reversion, plus revert-and-add exchanges at the cap.

    Moves are (position, new_token, reverted_position_or_minus_one). An exchange
    is scored by the sum of two first-order effects at the current parent.
    """
    moves, deltas = [], []
    edited = sequence != wt
    at_cap = int(edited.sum()) >= budget
    for pos in np.flatnonzero(mask):
        for aa in range(gradient.shape[1]):
            if aa == sequence[pos]:
                continue
            delta = float(gradient[pos, aa] - gradient[pos, sequence[pos]])
            if not at_cap or edited[pos] or aa == wt[pos]:
                moves.append((int(pos), aa, -1))
                deltas.append(delta)
            elif budget > 0:
                for reverted in np.flatnonzero(edited & mask):
                    moves.append((int(pos), aa, int(reverted)))
                    deltas.append(
                        delta
                        + float(
                            gradient[reverted, wt[reverted]]
                            - gradient[reverted, sequence[reverted]]
                        )
                    )
    return moves, np.asarray(deltas, dtype=np.float64)


def _atomic_edits(sequence, wt, mask, gradient):
    """Every single substitution available at the parent, vectorised.

    Returns `(position, residue, delta, cost)` arrays. `cost` is the change in
    Hamming distance from `wt` that the edit causes: `+1` when it moves a
    wt-matching position off wt, `-1` when it reverts one, and `0` when it
    swaps one non-wt residue for another. Costs **add across distinct
    positions**, which is the whole reason multi-position feasibility is a sum
    rather than the case analysis `_moves` does at the cap.
    """
    positions = np.flatnonzero(mask)
    alphabet = gradient.shape[1]
    pos = np.repeat(positions, alphabet)
    aa = np.tile(np.arange(alphabet), len(positions))
    keep = aa != sequence[pos]
    pos, aa = pos[keep], aa[keep]
    delta = (gradient[pos, aa] - gradient[pos, sequence[pos]]).astype(np.float64)
    cost = (aa != wt[pos]).astype(np.int64) - (sequence[pos] != wt[pos]).astype(
        np.int64
    )
    return pos, aa, delta, cost


def _multi_moves(sequence, wt, mask, budget, gradient, arity):
    """Single substitutions, then every feasible position-disjoint k-tuple.

    Returns `(moves, deltas, single_count)`. A move here is a tuple of
    `(position, residue)` edits applied together, so a single substitution is
    `((pos, aa),)` and a simultaneous pair is `((p1, a1), (p2, a2))`. The first
    `single_count` entries are the singles, which keeps the saliency map a
    per-(position, residue) object whatever the arity.

    The single-move subset is **exactly** the set `_moves` generates without
    exchanges, and the exchanges `_moves` builds at the cap -- edit here,
    revert there -- are a strict subset of the pairs here, namely the ones
    whose second edit returns a position to wt. So this generalises the
    existing operator rather than sitting beside it.

    ⚠️ **A first-order ranking cannot see synergy, and this does not change
    that.** The gradient is linear, so a pair's predicted delta is the *sum* of
    its two single deltas, with no interaction term. Two edits that are each
    unfavourable alone therefore rank unfavourably together, and nothing here
    discovers a combination from the gradient.

    What a pair move does buy is that **the accept test never sees the
    intermediate state.** Where edit A alone would be rejected and A+B would be
    kept, a single-move walk cannot reach A+B because A is rejected first. A
    pair move proposes A+B directly and the realised score -- not the
    first-order ranking -- is what discovers the combination. That is the
    mechanism, and it is worth stating precisely because a gain here must not
    be reported as the gradient having found an interaction.

    Arity is capped at 2 on purpose. Over 29 designable positions and 20
    residues the pair set is about 1.5e5 moves, which softmaxes in
    milliseconds; triples would be about 2.5e7, which needs a pruning rule, and
    any shortlist by summed delta would discard precisely the combinations
    whose components look bad alone.
    """
    if arity < 1:
        raise ValueError("arity must be at least 1")
    if arity > 2:
        raise ValueError(
            "arity above 2 needs a pruning rule that is not implemented; the "
            "triple set is ~2.5e7 moves and ranking by summed delta would "
            "discard the combinations this operator exists to reach"
        )
    pos, aa, delta, cost = _atomic_edits(sequence, wt, mask, gradient)
    spent = int(np.count_nonzero(sequence != wt))

    feasible = spent + cost <= budget
    moves = [
        ((int(p), int(a)),) for p, a in zip(pos[feasible], aa[feasible])
    ]
    blocks = [delta[feasible]]
    single_count = len(moves)

    if arity >= 2 and len(pos) > 1:
        left, right = np.triu_indices(len(pos), k=1)
        ok = (pos[left] != pos[right]) & (
            spent + cost[left] + cost[right] <= budget
        )
        left, right = left[ok], right[ok]
        moves.extend(
            ((int(p1), int(a1)), (int(p2), int(a2)))
            for p1, a1, p2, a2 in zip(pos[left], aa[left], pos[right], aa[right])
        )
        blocks.append(delta[left] + delta[right])

    deltas = (
        np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)
    )
    return moves, deltas, single_count


def _move_edits(move, wt):
    """The `(position, residue)` edits a move applies, for either move form.

    Two forms coexist because `_moves` predates the multi-position operator and
    is still the arity-1 path verbatim: its order sets the RNG stream for every
    result recorded so far, and `p17_pose_diagnostics.py` compares its output
    by equality. A legacy move is `(position, residue, reverted_or_minus_one)`;
    a `_multi_moves` move is a tuple of `(position, residue)` pairs.
    """
    if isinstance(move[0], tuple):
        return [(int(p), int(a)) for p, a in move]
    position, residue, reverted = move
    edits = [(int(position), int(residue))]
    if reverted >= 0:
        edits.append((int(reverted), int(wt[reverted])))
    return edits


def _proposal_distribution(deltas, target_entropy):
    """Choose temperature for H(p)/log(number of feasible moves).

    Equal deltas give uniform proposals. If tied minima make the requested
    entropy unattainable, use uniform mass on those minima and report the
    achieved entropy. Temperature is None for the uniform distribution.
    """
    count = len(deltas)
    if count == 0:
        raise ValueError("cannot sample an empty move set")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("proposal deltas must be finite")
    if count == 1:
        return np.ones(1), 0.0, None
    shifted = deltas - deltas.min()
    scale = float(shifted.max())
    if not np.isfinite(scale):
        raise ValueError("proposal delta range overflowed")
    if scale == 0 or target_entropy == 1:
        return np.full(count, 1 / count), 1.0, None
    shifted = shifted / scale
    log_count = np.log(count)

    def distribution(beta):
        weights = np.exp(-beta * shifted)
        probs = weights / weights.sum()
        positive = probs[probs > 0]
        entropy = float(-(positive * np.log(positive)).sum() / log_count)
        return probs, entropy

    minima = shifted == 0
    minimum_entropy = float(np.log(minima.sum()) / log_count)
    if target_entropy <= minimum_entropy:
        return minima / minima.sum(), minimum_entropy, 0.0
    low, high = 0.0, 1.0
    while distribution(high)[1] > target_entropy and high < 1e12:
        high *= 2
    for _ in range(48):
        middle = (low + high) / 2
        if distribution(middle)[1] > target_entropy:
            low = middle
        else:
            high = middle
    probs, entropy = distribution(high)
    return probs, entropy, scale / high


def _competitor(active, candidate, parent_slot, policy):
    if policy == "independent":
        return parent_slot
    seen = set()
    for slot, incumbent in enumerate(active):
        if incumbent.sequence in seen:
            return slot
        seen.add(incumbent.sequence)
    distances = [
        sum(a != b for a, b in zip(c.sequence, candidate.sequence)) for c in active
    ]
    return min(
        range(len(active)), key=lambda i: (distances[i], active[i].retention_key, i)
    )


def run_gradient_search(
    *,
    wt,
    designable_mask,
    gradient_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    confidence_fn: Callable[[np.ndarray], ConfidenceScore],
    config: SearchConfig,
    initial_sequences=None,
    on_event: Callable[[dict], None] | None = None,
    on_saliency: Callable[..., None] | None = None,
) -> SearchResult:
    """Compare retention policies using the same feasible gradient proposals.

    Independent slots never exchange parents. Population offspring compete with
    the nearest active sequence; initial duplicate slots admit new alternatives,
    and the highest-scoring active candidate cannot be replaced by a worse one.
    Both policies otherwise use the same confidence-based stochastic acceptance.
    The best evaluated candidate is also retained in the result archive.

    `on_saliency`, when given, receives the full per-(position, residue)
    first-order delta set at every gradient step. The search computes it to
    build the proposal distribution and otherwise discards everything but the
    chosen move, so it cannot be recovered afterwards. Backend-agnostic: the
    callback decides whether and how to persist.

    Optional nonnegative constraint violations take priority over confidence.
    Neither duplicate-slot filling nor stochastic acceptance can increase the
    challenged incumbent's violation. All-zero violations preserve the original
    confidence-only behavior.

    With `pareto_metric`, population retention instead maximizes confidence and
    minimizes that metric. The active population samples the evaluated Pareto
    frontier by crowding distance, preserving its two extremes. There is no
    stochastic acceptance in this mode; acceptance_temperature is unused.
    The metric ceiling governs final selection, not admission to the frontier,
    so an initially out-of-ceiling population can still make progress.

    Unique confidence evaluations and unique parent gradient calls have separate
    ceilings. A proposal limit bounds duplicate revisits. Equal ceilings do not
    guarantee equal elapsed compute: returned counters/timings must be compared.
    Callbacks should synchronize accelerator work before returning. Initialization
    counts toward the confidence budget; no model work occurs for invalid inputs.
    """
    wt = np.asarray(wt)
    mask = np.asarray(designable_mask)
    if wt.ndim != 1 or not len(wt) or not np.issubdtype(wt.dtype, np.integer):
        raise ValueError("wt must be a nonempty integer vector")
    if mask.shape != wt.shape or mask.dtype != np.bool_:
        raise ValueError("designable_mask must be a boolean vector matching wt")
    if np.any((wt < 0) | (wt >= config.alphabet_size)):
        raise ValueError("wt contains a token outside the alphabet")
    wt = wt.astype(np.int32, copy=True)
    mask = mask.copy()

    def validate(sequence):
        sequence = np.asarray(sequence)
        if sequence.shape != wt.shape or not np.issubdtype(sequence.dtype, np.integer):
            raise ValueError("each sequence must be an integer vector matching wt")
        if np.any((sequence < 0) | (sequence >= config.alphabet_size)):
            raise ValueError("sequence contains a token outside the alphabet")
        if np.any(sequence[~mask] != wt[~mask]):
            raise ValueError("sequence changes a fixed position")
        if np.count_nonzero(sequence != wt) > config.edit_budget:
            raise ValueError("sequence exceeds the WT-relative edit budget")
        return tuple(int(x) for x in sequence)

    initial = (
        np.repeat(wt[None], config.width, axis=0)
        if initial_sequences is None
        else np.asarray(initial_sequences)
    )
    if initial.shape != (config.width, len(wt)):
        raise ValueError("initial_sequences must have shape (width, sequence_length)")
    initial = [validate(sequence) for sequence in initial]
    if len(set(initial)) > config.max_score_calls:
        raise ValueError("confidence budget cannot cover initialization")

    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    archive: dict[tuple[int, ...], Candidate] = {}
    gradients: dict[int, np.ndarray] = {}
    cheap_losses: dict[int, float] = {}
    stats = dict(
        score_calls=0,
        gradient_calls=0,
        proposals=0,
        cache_hits=0,
        accepted=0,
        score_seconds=0.0,
        gradient_seconds=0.0,
    )

    def emit(kind, **fields):
        if on_event is not None:
            on_event(
                dict(
                    event=kind,
                    policy=config.policy,
                    elapsed_seconds=perf_counter() - started,
                    **fields,
                )
            )

    def evaluate(sequence):
        if sequence in archive:
            stats["cache_hits"] += 1
            return archive[sequence], True
        before = perf_counter()
        score = confidence_fn(np.array(sequence, dtype=np.int32))
        value = float(score.value)
        violation = float(score.constraint_violation)
        if not np.isfinite(violation) or violation < 0:
            raise ValueError("constraint violation must be finite and nonnegative")
        metrics = {name: float(value) for name, value in score.metrics.items()}
        if not np.isfinite(value) or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("confidence callback returned a nonfinite score or metric")
        if config.pareto_metric is not None and config.pareto_metric not in metrics:
            raise ValueError(
                f"confidence callback is missing Pareto metric {config.pareto_metric!r}"
            )
        stats["score_seconds"] += perf_counter() - before
        stats["score_calls"] += 1
        candidate = Candidate(len(archive), sequence, value, metrics, violation)
        archive[sequence] = candidate
        emit(
            "evaluation",
            candidate_id=candidate.id,
            sequence=list(sequence),
            score=value,
            constraint_violation=violation,
            metrics=metrics,
            edit_count=int(np.count_nonzero(wt != sequence)),
            score_calls=stats["score_calls"],
        )
        return candidate, False

    active = [evaluate(sequence)[0] for sequence in initial]
    front = []
    if config.pareto_metric is not None:
        front = pareto_front(list(archive.values()), config.pareto_metric)
        active = _pareto_population(front, config.pareto_metric, config.width)
    emit("initialization", active_ids=[c.id for c in active])
    stop_reason = "proposal_budget"
    for attempt in range(config.max_proposals):
        if stats["score_calls"] >= config.max_score_calls:
            stop_reason = "score_budget"
            break
        parent_slot = attempt % config.width
        parent = active[parent_slot]
        sequence = np.array(parent.sequence, dtype=np.int32)
        if parent.id not in gradients:
            if stats["gradient_calls"] >= config.max_gradient_calls:
                stop_reason = "gradient_budget"
                break
            before = perf_counter()
            loss, gradient = gradient_fn(sequence.copy())
            loss, gradient = float(loss), np.asarray(gradient, dtype=np.float64)
            if gradient.shape != (len(wt), config.alphabet_size):
                raise ValueError("gradient has the wrong shape")
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise ValueError(
                    "gradient callback returned a nonfinite loss or gradient"
                )
            gradients[parent.id] = gradient.copy()
            cheap_losses[parent.id] = loss
            stats["gradient_seconds"] += perf_counter() - before
            stats["gradient_calls"] += 1
            emit(
                "gradient",
                candidate_id=parent.id,
                cheap_loss=loss,
                gradient_calls=stats["gradient_calls"],
            )
        if config.move_arity == 1:
            moves, deltas = _moves(
                sequence, wt, mask, config.edit_budget, gradients[parent.id]
            )
            saliency_moves, saliency_deltas = moves, deltas
        else:
            moves, deltas, single_count = _multi_moves(
                sequence, wt, mask, config.edit_budget,
                gradients[parent.id], config.move_arity,
            )
            saliency_moves = [
                (position, residue, -1) for ((position, residue),) in moves[:single_count]
            ]
            saliency_deltas = deltas[:single_count]
        if on_saliency is not None and saliency_moves:
            on_saliency(
                candidate_id=parent.id,
                sequence=sequence.copy(),
                moves=list(saliency_moves),
                deltas=np.asarray(saliency_deltas),
                gradient_calls=stats["gradient_calls"],
            )
        if not moves:
            stop_reason = "no_feasible_moves"
            break
        probs, entropy, temperature = _proposal_distribution(
            deltas, config.target_entropy
        )
        move_index = int(rng.choice(len(moves), p=probs))
        edits = _move_edits(moves[move_index], wt)
        proposal = sequence.copy()
        for position, residue in edits:
            proposal[position] = residue
        candidate, cache_hit = evaluate(validate(proposal))
        stats["proposals"] += 1
        if config.pareto_metric is not None:
            duplicate = any(c.id == candidate.id for c in active)
            front = pareto_front(list(archive.values()), config.pareto_metric)
            active = _pareto_population(front, config.pareto_metric, config.width)
            accept = not duplicate and any(c.id == candidate.id for c in active)
            slot, incumbent, constraint_change = None, None, None
            filling, protected = False, False
        else:
            slot = _competitor(active, candidate, parent_slot, config.policy)
            incumbent = active[slot]
            improvement = candidate.score - incumbent.score
            constraint_change = (
                candidate.constraint_violation - incumbent.constraint_violation
            )
            duplicate = config.policy == "population" and any(
                c.sequence == candidate.sequence for c in active
            )
            filling = (
                config.policy == "population"
                and sum(c.sequence == incumbent.sequence for c in active) > 1
            )
            protected = (
                config.policy == "population"
                and not filling
                and candidate.retention_key < incumbent.retention_key
                and incumbent.retention_key == max(c.retention_key for c in active)
            )
            accept = False
            if not duplicate and not protected and constraint_change <= 0:
                accept = constraint_change < 0 or filling or improvement >= 0
                if not accept and config.acceptance_temperature > 0:
                    accept = rng.random() < np.exp(
                        improvement / config.acceptance_temperature
                    )
            if accept:
                active[slot] = candidate
        if accept:
            stats["accepted"] += 1
        emit(
            "proposal",
            attempt=attempt,
            parent_slot=parent_slot,
            parent_id=parent.id,
            candidate_id=candidate.id,
            competitor_id=incumbent.id if incumbent is not None else None,
            competitor_slot=slot,
            accepted=bool(accept),
            constraint_change=constraint_change,
            cache_hit=cache_hit,
            elite_protected=bool(protected),
            duplicate=duplicate,
            filling=bool(filling),
            move=list(moves[move_index]),
            move_positions=len(edits),
            predicted_loss_delta=float(deltas[move_index]),
            entropy=entropy,
            proposal_temperature=temperature,
            feasible_moves=len(moves),
            active_ids=[c.id for c in active],
            retention_mode="pareto" if config.pareto_metric is not None else "confidence",
            pareto_front_ids=[c.id for c in front],
            score_calls=stats["score_calls"],
            gradient_calls=stats["gradient_calls"],
        )
    else:
        if stats["score_calls"] >= config.max_score_calls:
            stop_reason = "score_budget"
    stats["elapsed_seconds"] = perf_counter() - started
    best = max(archive.values(), key=lambda c: candidate_selection_key(c, config))
    emit(
        "complete",
        stop_reason=stop_reason,
        best_id=best.id,
        active_ids=[c.id for c in active],
        pareto_front_ids=[c.id for c in front],
        stats=stats.copy(),
    )
    return SearchResult(
        best, active, list(archive.values()), cheap_losses, stats, stop_reason, front
    )
