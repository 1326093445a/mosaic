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
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not 0 < self.target_entropy <= 1:
            raise ValueError("target_entropy must be in (0, 1]")
        if (
            not np.isfinite(self.acceptance_temperature)
            or self.acceptance_temperature < 0
        ):
            raise ValueError("acceptance_temperature must be finite and >= 0")


@dataclass(frozen=True)
class ConfidenceScore:
    value: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    id: int
    sequence: tuple[int, ...]
    score: float
    metrics: dict[str, float]


@dataclass
class SearchResult:
    best: Candidate
    active: list[Candidate]
    evaluated: list[Candidate]
    cheap_losses: dict[int, float]
    stats: dict[str, int | float]
    stop_reason: str


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
    # Fill initial duplicate slots before imposing local competition. Keep one
    # copy of each incumbent, including the current global elite.
    seen = set()
    for slot, incumbent in enumerate(active):
        if incumbent.sequence in seen:
            return slot
        seen.add(incumbent.sequence)
    distances = [
        sum(a != b for a, b in zip(c.sequence, candidate.sequence)) for c in active
    ]
    # At equal distance, challenge the worse incumbent first.
    return min(range(len(active)), key=lambda i: (distances[i], active[i].score, i))


def run_gradient_search(
    *,
    wt,
    designable_mask,
    gradient_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    confidence_fn: Callable[[np.ndarray], ConfidenceScore],
    config: SearchConfig,
    initial_sequences=None,
    on_event: Callable[[dict], None] | None = None,
) -> SearchResult:
    """Compare retention policies using the same feasible gradient proposals.

    Independent slots never exchange parents. Population offspring compete with
    the nearest active sequence; initial duplicate slots admit new alternatives,
    and the highest-scoring active candidate cannot be replaced by a worse one.
    Both policies otherwise use the same confidence-based stochastic acceptance.
    The best evaluated candidate is also retained in the result archive.

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
        metrics = {name: float(value) for name, value in score.metrics.items()}
        if not np.isfinite(value) or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("confidence callback returned a nonfinite score or metric")
        stats["score_seconds"] += perf_counter() - before
        stats["score_calls"] += 1
        candidate = Candidate(len(archive), sequence, value, metrics)
        archive[sequence] = candidate
        emit(
            "evaluation",
            candidate_id=candidate.id,
            sequence=list(sequence),
            score=value,
            metrics=metrics,
            edit_count=int(np.count_nonzero(wt != sequence)),
            score_calls=stats["score_calls"],
        )
        return candidate, False

    active = [evaluate(sequence)[0] for sequence in initial]
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
        moves, deltas = _moves(
            sequence, wt, mask, config.edit_budget, gradients[parent.id]
        )
        if not moves:
            stop_reason = "no_feasible_moves"
            break
        probs, entropy, temperature = _proposal_distribution(
            deltas, config.target_entropy
        )
        move_index = int(rng.choice(len(moves), p=probs))
        pos, aa, reverted = moves[move_index]
        proposal = sequence.copy()
        proposal[pos] = aa
        if reverted >= 0:
            proposal[reverted] = wt[reverted]
        candidate, cache_hit = evaluate(validate(proposal))
        stats["proposals"] += 1
        slot = _competitor(active, candidate, parent_slot, config.policy)
        incumbent = active[slot]
        improvement = candidate.score - incumbent.score
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
            and improvement < 0
            and incumbent.score == max(c.score for c in active)
        )
        accept = False
        if not duplicate and not protected:
            accept = filling or improvement >= 0
            if not accept and config.acceptance_temperature > 0:
                accept = rng.random() < np.exp(
                    improvement / config.acceptance_temperature
                )
        if accept:
            active[slot] = candidate
            stats["accepted"] += 1
        emit(
            "proposal",
            attempt=attempt,
            parent_slot=parent_slot,
            parent_id=parent.id,
            candidate_id=candidate.id,
            competitor_id=incumbent.id,
            competitor_slot=slot,
            accepted=bool(accept),
            cache_hit=cache_hit,
            elite_protected=bool(protected),
            duplicate=duplicate,
            filling=bool(filling),
            move=list(moves[move_index]),
            predicted_loss_delta=float(deltas[move_index]),
            entropy=entropy,
            proposal_temperature=temperature,
            feasible_moves=len(moves),
            active_ids=[c.id for c in active],
            score_calls=stats["score_calls"],
            gradient_calls=stats["gradient_calls"],
        )
    else:
        if stats["score_calls"] >= config.max_score_calls:
            stop_reason = "score_budget"
    stats["elapsed_seconds"] = perf_counter() - started
    best = max(archive.values(), key=lambda c: c.score)
    emit(
        "complete",
        stop_reason=stop_reason,
        best_id=best.id,
        active_ids=[c.id for c in active],
        stats=stats.copy(),
    )
    return SearchResult(
        best, active, list(archive.values()), cheap_losses, stats, stop_reason
    )
