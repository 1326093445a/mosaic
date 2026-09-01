"""Phase 2 guidance diagnostics (docs/legacy/guidance_implementation_todo.md Phase 2).

Consumes the per-step diagnostics dict returned by
`guided_partial_diffusion(..., return_diagnostics=True)`
(src/mosaic/models/boltzgen.py) and computes the metrics Phase 2 calls for:

  - cosine similarity between the guided and unguided reverse-update
    direction (is guidance pushing the trajectory in a different direction
    than the unguided prior would have gone?)
  - norm ratio between the guided and unguided *physical* per-step
    displacement (is guidance also making steps bigger, not just different
    in direction?)
  - fraction of steps with strong directional disagreement
  - stratification of the above by noise level (sigma / t_hat)
  - pairwise inter-objective conflict: cos(g_bind, g_nat), cos(g_bind, g_edit)
  - (separately) correlation of per-trajectory summary stats against final
    refolded outcome metrics (ipSAE, RMSD), once those are available

This module is pure numpy. No BoltzGen model, checkpoint, or GPU needed to
run or test it -- only `guided_partial_diffusion`'s actual diagnostics
output is GPU/model-dependent; everything downstream of that dict is not.
"""
import json
from pathlib import Path

import numpy as np


def _flatten_last_two(x: np.ndarray) -> np.ndarray:
    """(..., M, 3) -> (..., M*3), so norm/dot reduce over atoms and coords together."""
    return x.reshape(*x.shape[:-2], -1)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity between two (steps, M, 3) arrays, per step."""
    a_flat = _flatten_last_two(a)
    b_flat = _flatten_last_two(b)
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm = np.linalg.norm(a_flat, axis=-1) * np.linalg.norm(b_flat, axis=-1)
    return dot / (norm + eps)


def _norm(a: np.ndarray) -> np.ndarray:
    """L2 norm of a (steps, M, 3) array, per step."""
    return np.linalg.norm(_flatten_last_two(a), axis=-1)


def _select_designable(a: np.ndarray, atom_partial_mask: np.ndarray) -> np.ndarray:
    """Restrict a (steps, M, 3) array to designable atoms (mask > 0)."""
    return a[:, atom_partial_mask > 0, :]


def per_step_metrics(diagnostics: dict, atom_partial_mask=None) -> dict:
    """Reduce one trajectory's stacked diagnostics dict to per-step scalars.

    `diagnostics` is exactly what `guided_partial_diffusion(...,
    return_diagnostics=True)` returns: every array has a leading
    `num_steps` axis (stacked by `jax.lax.scan`), a batch axis of size 1
    (this driver never runs guided_partial_diffusion batched), and
    per-atom/coordinate trailing axes for the direction/delta/gradient
    fields.

    `atom_partial_mask`: optional `(M,)` array, the same mask passed to
    `guided_partial_diffusion` marking which atoms are designable. When
    given, also computes `*_designable` variants restricted to those atoms
    only -- see the note below on why the unmasked metrics alone are
    misleading.

    Returns a dict of 1-D `(num_steps,)` numpy arrays:
      - cos_guided_unguided, norm_ratio: computed over ALL atoms in the
        complex (framework/target included), not just the designable
        region. Because `_mask_center_normalize` zeros the merged gradient
        on every non-designable atom before it reaches `delta`
        (`boltzgen.py`), `x0_guided` is numerically identical to `x0_hat`
        there -- so `guided_direction` and `unguided_direction` agree
        near-exactly on every frozen atom regardless of what guidance is
        doing. For a small CDR inside a large complex, that frozen
        majority can dilute cosine similarity toward 1.0 and hide real
        disagreement confined to the few atoms guidance actually touches.
        Kept as whole-complex reference values, not the primary read.
      - cos_guided_unguided_designable, norm_ratio_designable: the same
        two metrics restricted to designable atoms only (`atom_partial_mask
        > 0`) -- only present when `atom_partial_mask` is passed. This is
        the metric that actually reflects what guidance is doing; prefer
        it over the whole-complex one whenever available.
      - norm_ratio uses the *_step_delta fields (already scaled by the
        real Euler step_scale*(sigma_t - t_hat) factor -- using the raw
        directions here would not correspond to actual physical
        displacement).
      - cos_bind_nat, cos_bind_edit: pairwise inter-objective conflict.
        NaN at steps where the corresponding objective wasn't provided
        (its gradient is an all-zero array, and cosine against an all-zero
        vector is undefined -- NaN keeps that explicit instead of silently
        reporting a misleading 0.0 "perfectly orthogonal").
      - t_hat, sigma_t: passthrough, for stratifying any of the above.
    """
    unguided_dir = np.asarray(diagnostics["unguided_direction"])
    guided_dir = np.asarray(diagnostics["guided_direction"])
    unguided_delta = np.asarray(diagnostics["unguided_step_delta"])
    guided_delta = np.asarray(diagnostics["guided_step_delta"])
    g_bind = np.asarray(diagnostics["g_bind"])
    g_nat = np.asarray(diagnostics["g_nat"])
    g_edit = np.asarray(diagnostics["g_edit"])
    t_hat = np.asarray(diagnostics["t_hat"]).reshape(-1)
    sigma_t = np.asarray(diagnostics["sigma_t"]).reshape(-1)

    batch_size = unguided_dir.shape[1]
    if batch_size != 1:
        raise ValueError(
            "per_step_metrics expects batch size 1 (guided_partial_diffusion's "
            f"only production usage in this codebase), got batch size {batch_size}. "
            "If batched guidance runs are added, this function needs a batch axis "
            "in its output instead of squeezing it."
        )

    def squeeze_batch(x):
        return x[:, 0]  # (num_steps, M, 3)

    unguided_dir = squeeze_batch(unguided_dir)
    guided_dir = squeeze_batch(guided_dir)
    unguided_delta = squeeze_batch(unguided_delta)
    guided_delta = squeeze_batch(guided_delta)
    g_bind = squeeze_batch(g_bind)
    g_nat = squeeze_batch(g_nat)
    g_edit = squeeze_batch(g_edit)

    cos_guided_unguided = _cosine(guided_dir, unguided_dir)
    norm_ratio = _norm(guided_delta) / (_norm(unguided_delta) + 1e-8)

    nat_active = np.any(g_nat != 0, axis=(-2, -1))
    edit_active = np.any(g_edit != 0, axis=(-2, -1))
    cos_bind_nat = np.where(nat_active, _cosine(g_bind, g_nat), np.nan)
    cos_bind_edit = np.where(edit_active, _cosine(g_bind, g_edit), np.nan)

    result = {
        "cos_guided_unguided": cos_guided_unguided,
        "norm_ratio": norm_ratio,
        "cos_bind_nat": cos_bind_nat,
        "cos_bind_edit": cos_bind_edit,
        "t_hat": t_hat,
        "sigma_t": sigma_t,
    }

    if atom_partial_mask is not None:
        atom_partial_mask = np.asarray(atom_partial_mask)
        if atom_partial_mask.ndim != 1 or atom_partial_mask.shape[0] != unguided_dir.shape[1]:
            raise ValueError(
                f"atom_partial_mask must be 1-D with shape (M,) matching the "
                f"diagnostics' atom axis ({unguided_dir.shape[1]}), got shape "
                f"{atom_partial_mask.shape}"
            )
        n_design = int(np.sum(atom_partial_mask > 0))
        if n_design == 0:
            raise ValueError("atom_partial_mask has no designable atoms (all <= 0)")
        guided_dir_d = _select_designable(guided_dir, atom_partial_mask)
        unguided_dir_d = _select_designable(unguided_dir, atom_partial_mask)
        guided_delta_d = _select_designable(guided_delta, atom_partial_mask)
        unguided_delta_d = _select_designable(unguided_delta, atom_partial_mask)
        result["cos_guided_unguided_designable"] = _cosine(guided_dir_d, unguided_dir_d)
        result["norm_ratio_designable"] = (
            _norm(guided_delta_d) / (_norm(unguided_delta_d) + 1e-8)
        )

    return result


def _nan_safe_mean(x: np.ndarray):
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else None


def summarize(
    per_step: dict,
    *,
    disagreement_cos_threshold: float = 0.0,
    n_sigma_bins: int = 4,
) -> dict:
    """Aggregate one trajectory's per-step metrics into Phase 2 summary stats.

    `disagreement_cos_threshold`: steps with cos_guided_unguided below this
    count as "strong directional disagreement". 0.0 (orthogonal or worse) is
    a conservative default; tighten it (e.g. 0.5) once real trajectories
    show what a meaningful threshold looks like -- this default is a
    starting point, not a validated cutoff.

    `n_sigma_bins`: number of noise-level bins (by t_hat quantile, high sigma
    = early/noisy steps first) to stratify disagreement by.

    Primary stats (`mean_cos_guided_unguided`, `mean_norm_ratio`,
    `frac_strong_disagreement`, and each `by_sigma_bin` entry) use the
    designable-region metrics (`per_step["cos_guided_unguided_designable"]`
    etc.) when `per_step_metrics` was called with `atom_partial_mask` --
    see that function's docstring for why the whole-complex metrics alone
    are misleading. When no mask was passed, these fall back to the
    whole-complex values (unchanged behavior). The whole-complex values are
    always additionally reported as `*_whole_complex` secondary stats.
    """
    cos_whole = per_step["cos_guided_unguided"]
    ratio_whole = per_step["norm_ratio"]
    cos = per_step.get("cos_guided_unguided_designable", cos_whole)
    ratio = per_step.get("norm_ratio_designable", ratio_whole)
    t_hat = per_step["t_hat"]
    n_steps = len(cos)

    edges = np.quantile(t_hat, np.linspace(0, 1, n_sigma_bins + 1))
    bin_idx = np.clip(np.digitize(t_hat, edges[1:-1], right=True), 0, n_sigma_bins - 1)

    by_sigma_bin = []
    for b in range(n_sigma_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        by_sigma_bin.append({
            "bin": b,
            "t_hat_range": [float(t_hat[mask].min()), float(t_hat[mask].max())],
            "n_steps": int(mask.sum()),
            "mean_cos_guided_unguided": float(np.mean(cos[mask])),
            "mean_norm_ratio": float(np.mean(ratio[mask])),
            "frac_strong_disagreement": float(
                np.mean(cos[mask] < disagreement_cos_threshold)
            ),
        })

    return {
        "n_steps": n_steps,
        "mean_cos_guided_unguided": float(np.mean(cos)),
        "mean_norm_ratio": float(np.mean(ratio)),
        "mean_cos_guided_unguided_whole_complex": float(np.mean(cos_whole)),
        "mean_norm_ratio_whole_complex": float(np.mean(ratio_whole)),
        "disagreement_cos_threshold": disagreement_cos_threshold,
        "frac_strong_disagreement": float(np.mean(cos < disagreement_cos_threshold)),
        "mean_cos_bind_nat": _nan_safe_mean(per_step["cos_bind_nat"]),
        "mean_cos_bind_edit": _nan_safe_mean(per_step["cos_bind_edit"]),
        "by_sigma_bin": by_sigma_bin,
    }


def correlate_with_outcomes(records: list[dict]) -> dict:
    """Correlate per-trajectory summary stats against final refolded outcomes.

    `records`: list of `{"summary": <summarize() output>, "outcome": {"ipsae":
    float, "rmsd": float}}`. Meant to be called once several trajectories
    have both Phase 2 summaries and post-refold metrics available (this is
    the last, outcome-dependent piece of the Phase 2 checklist -- it needs
    real refolded designs, not just diagnostics from the guided sampler, so
    it is a separate call from `summarize`, not fused into it).

    Uses both ipSAE and RMSD, never just one -- this project's own prior
    validation work found the two metrics can disagree on the same designs.
    Returns Pearson correlations of {mean_cos_guided_unguided,
    mean_norm_ratio, frac_strong_disagreement} against {ipsae, rmsd}. `None`
    for any pair where there are fewer than 3 records or the input is
    degenerate (zero variance), rather than raising or silently returning
    NaN as if it were a real correlation.
    """
    stat_keys = ["mean_cos_guided_unguided", "mean_norm_ratio", "frac_strong_disagreement"]
    outcome_keys = ["ipsae", "rmsd"]

    result = {}
    for stat_key in stat_keys:
        for outcome_key in outcome_keys:
            xs, ys = [], []
            for r in records:
                x = r["summary"].get(stat_key)
                y = r["outcome"].get(outcome_key)
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            pair_key = f"{stat_key}_vs_{outcome_key}"
            if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
                result[pair_key] = None
                continue
            corr = float(np.corrcoef(xs, ys)[0, 1])
            result[pair_key] = corr if np.isfinite(corr) else None

    result["n_records"] = len(records)
    return result


def write_report(trajectory_records: list[dict], output_path: Path) -> dict:
    """Write the full Phase 2 report (JSON) and return it.

    `trajectory_records`: list of `{"label": str, "summary": <summarize()
    output>, "outcome": {"ipsae": float, "rmsd": float} | None}`. Records
    without an outcome are included in the report but excluded from
    `correlate_with_outcomes` (which needs the outcome).
    """
    with_outcome = [r for r in trajectory_records if r.get("outcome") is not None]
    report = {
        "n_trajectories": len(trajectory_records),
        "n_trajectories_with_outcome": len(with_outcome),
        "trajectories": trajectory_records,
        "outcome_correlations": (
            correlate_with_outcomes(with_outcome) if with_outcome else None
        ),
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def format_summary_text(label: str, summary: dict) -> str:
    """One trajectory's summary as a short human-readable block for stdout/logs."""
    lines = [
        f"[guidance diagnostics] {label}: "
        f"{summary['n_steps']} steps, "
        f"mean cos(guided,unguided)={summary['mean_cos_guided_unguided']:.3f}, "
        f"mean norm_ratio={summary['mean_norm_ratio']:.3f}, "
        f"strong_disagreement={summary['frac_strong_disagreement']:.1%} "
        f"(cos < {summary['disagreement_cos_threshold']})",
        f"  whole-complex reference: "
        f"cos={summary['mean_cos_guided_unguided_whole_complex']:.3f}, "
        f"norm_ratio={summary['mean_norm_ratio_whole_complex']:.3f}",
    ]
    if summary["mean_cos_bind_nat"] is not None:
        lines.append(f"  cos(g_bind,g_nat)={summary['mean_cos_bind_nat']:.3f}")
    if summary["mean_cos_bind_edit"] is not None:
        lines.append(f"  cos(g_bind,g_edit)={summary['mean_cos_bind_edit']:.3f}")
    for b in summary["by_sigma_bin"]:
        lines.append(
            f"  sigma bin {b['bin']} (t_hat {b['t_hat_range'][0]:.3f}-"
            f"{b['t_hat_range'][1]:.3f}, n={b['n_steps']}): "
            f"cos={b['mean_cos_guided_unguided']:.3f}, "
            f"norm_ratio={b['mean_norm_ratio']:.3f}, "
            f"disagreement={b['frac_strong_disagreement']:.1%}"
        )
    return "\n".join(lines)
