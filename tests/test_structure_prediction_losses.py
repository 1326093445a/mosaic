from types import SimpleNamespace

import jax
import jax.numpy as jnp

from mosaic.losses.structure_prediction import DistogramIPTMProxy


def test_distogram_iptm_proxy_retains_gradient_when_reported_proxy_is_zero():
    binder_len = 2
    n_tokens = 3
    n_bins = 64
    sequence = jnp.zeros((binder_len, 20))
    bins = jnp.arange(n_bins, dtype=jnp.float32) + 0.5
    logits = jnp.zeros((n_tokens, n_tokens, n_bins))
    loss_term = DistogramIPTMProxy(contact_distance=8.0)

    def loss_from_logits(x):
        output = SimpleNamespace(distogram_logits=x, distogram_bins=bins)
        return loss_term(sequence, output, key=jax.random.key(0))

    value, aux = loss_from_logits(logits)
    grad = jax.grad(lambda x: loss_from_logits(x)[0])(logits)

    assert value > 1.0
    assert aux["distogram_iptm"] == 0.0
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0)
