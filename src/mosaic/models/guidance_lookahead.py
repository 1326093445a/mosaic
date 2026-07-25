"""Real look-ahead guidance mechanism (docs/guidance_alphaseq_testing_notes.md
section 12b) -- arXiv:2404.14743's actual "differentiate through the
denoiser's own Jacobian" argument, not a multi-step rollout (a different
technique used elsewhere in the guidance literature).

`guided_partial_diffusion`'s existing one-shot and NOS-iterative paths
(src/mosaic/models/boltzgen.py) both compute `g_bind`/`g_nat`/`g_edit` as
`jax.grad(guidance_fn)(x0_hat)` -- gradients of the guidance losses with
respect to `x0_hat` treated as a free variable, with
`x0_hat = jax.lax.stop_gradient(D(atom_coords_noisy, t_hat))` blocking any
gradient flow back through the denoiser `D` itself. Look-ahead removes that
boundary: the guidance losses are differentiated with respect to
`atom_coords_noisy`, through the *entire* denoiser forward pass. Per
Theorem 1 in arXiv:2404.14743, this confines the guidance direction to the
subspace the denoiser's own Jacobian can express -- a structural argument
for staying in-distribution, distinct from NOS's explicit penalty (section
12a). The theorem's formal guarantee is proved only under a linear-subspace
data assumption that does not hold for a real nonlinear denoiser like
BoltzGen (see docs/guidance_alphaseq_testing_notes.md section 3a) -- this is
a heuristic worth measuring, not a guaranteed fix.

This is deliberately a separate module from boltzgen.py: the piece here
(compose "denoiser forward -> guidance loss" and differentiate with respect
to the noisy input) is self-contained and independently testable with a
synthetic denoiser, unlike the merge/application logic in `step_body`, which
stays in boltzgen.py and is reused unchanged (`_merge_aux_gradients`,
`_clip_rms`) -- splitting those out too would just create a circular import
back into boltzgen.py's internals.
"""
import jax


def build_lookahead_grad_fn(denoiser_fn, guidance_fn):
    """Compose `guidance_fn` after `denoiser_fn` and differentiate the
    result with respect to `denoiser_fn`'s input.

    `denoiser_fn`: callable `(atom_coords_noisy) -> x0`, already closed over
    everything else the real denoiser call needs for this step (`t_hat`,
    `network_condition_kwargs`, a `key`) -- built fresh per step since those
    all vary step to step, unlike the existing `grad_bind`/`grad_nat`/
    `grad_edit` closures (built once, outside the scan, since they only
    close over the guidance loss itself and are called at whatever `x0`
    point the caller passes in).
    `guidance_fn`: callable `(x0) -> scalar`, the same raw guidance loss
    used to build the existing `grad_bind`/etc. closures (`jax.grad` not yet
    applied).

    Returns a callable `(atom_coords_noisy) -> gradient`, shaped like
    `atom_coords_noisy`, suitable for passing directly to
    `_merge_aux_gradients` in place of a `grad_bind`/`grad_nat`/`grad_edit`
    closure -- the merge machinery (mask/center/normalize/PCGrad) doesn't
    care which point space a closure differentiates in, only that it maps a
    coordinate array to a same-shaped gradient array.
    """
    def loss_from_noisy(atom_coords_noisy):
        return guidance_fn(denoiser_fn(atom_coords_noisy))

    return jax.grad(loss_from_noisy)
