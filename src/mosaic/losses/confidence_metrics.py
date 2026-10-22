import jax
import jax.numpy as jnp
import numpy as onp


def plddt(logits: jnp.ndarray) -> jnp.ndarray:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = onp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = jax.nn.softmax(logits, -1)
    predicted_lddt_ca = jnp.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


def predicted_aligned_error(
    logits: jnp.ndarray, breaks: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    aligned_confidence_probs = jax.nn.softmax(logits, axis=-1)
    return _calculate_expected_aligned_error(
        alignment_confidence_breaks=breaks,
        aligned_distance_error_probs=aligned_confidence_probs,
    )


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: jnp.ndarray, aligned_distance_error_probs: jnp.ndarray
) -> jnp.ndarray:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

    return jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1)


def _calculate_bin_centers(breaks: jnp.ndarray):
    """Bin centers from bin edges, with a catch-all bin appended.

    `breaks[-2:-1] + step` was a transcription error: it takes the SECOND to
    last center, so the catch-all duplicated the previous bin instead of
    extending past it -- breaks [0,1,2] gave [0.5,1.5,2.5,2.5] where AlphaFold
    gives [0.5,1.5,2.5,3.5]. The correct expression is already in this repo, at
    `mosaic/alphafold/common/confidence.py`:
    `np.concatenate([bin_centers, [bin_centers[-1] + step]])`.

    Fixed 2026-10-13. It biases `predicted_tm_score` and
    `_calculate_expected_aligned_error` by under-weighting the worst-error bin,
    so AF2-path confidence numbers recorded before that date are not
    comparable with ones recorded after. OpenDDE uses its own helper and is
    unaffected.
    """
    step = breaks[1] - breaks[0]

    bin_centers = breaks + step / 2
    return jnp.append(bin_centers, bin_centers[-1:] + step, axis=0)


def predicted_tm_score(
    logits: jnp.ndarray,
    breaks: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    interface: bool = False,
) -> jnp.ndarray:
    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[0]
    clipped_num_res = max(num_res, 19)

    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = jax.nn.softmax(logits, axis=-1)

    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment[per_alignment.argmax()]


def confidence_metrics(prediction_result: dict[str, any]):
    return {
        "plddt": plddt(prediction_result["predicted_lddt"]["logits"]),
        "predicted_aligned_error": jnp.squeeze(
            predicted_aligned_error(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
            )
        ),
        "iptm": predicted_tm_score(
            logits=prediction_result["predicted_aligned_error"]["logits"],
            breaks=prediction_result["predicted_aligned_error"]["breaks"],
            asym_id=prediction_result["predicted_aligned_error"]["asym_id"],
            interface=True,
        ),
    }
