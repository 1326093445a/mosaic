"""Mixed-precision boundaries and an opt-in real-checkpoint GPU smoke test."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jopendde.backend import LayerNorm, Linear

from mosaic.opendde_precision import cast_float_arrays, prepare_opendde_model


def test_cast_preserves_indices_masks_and_metadata():
    tree = dict(
        x=jnp.ones((3,), dtype=jnp.float32),
        index=jnp.array([1, 2], dtype=jnp.int32),
        mask=jnp.array([True, False]),
        count=3,
        absent=None,
    )
    cast = cast_float_arrays(tree, jnp.bfloat16)
    assert cast["x"].dtype == jnp.bfloat16
    assert cast["index"] is tree["index"]
    assert cast["mask"] is tree["mask"]
    assert cast["count"] == 3 and cast["absent"] is None


def test_fp32_preserves_original_model_and_invalid_precision_fails():
    linear = Linear(jnp.eye(3), None)
    assert prepare_opendde_model(linear, "fp32") is linear
    with pytest.raises(ValueError, match="compute dtype"):
        prepare_opendde_model(linear, "fp16")


def test_bf16_projection_accepts_float32_geometry_and_backpropagates():
    linear = Linear(jnp.eye(3), jnp.ones(3))
    bf16 = prepare_opendde_model(linear, "bf16")
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = eqx.filter_jit(bf16)(x)
    assert y.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(y, dtype=np.float32), [2.0, 3.0, 4.0])
    gradient = jax.jit(jax.grad(lambda z: bf16(z).astype(jnp.float32).sum()))(x)
    assert gradient.dtype == jnp.float32
    np.testing.assert_allclose(gradient, np.ones(3))
    assert linear.weight.dtype == jnp.float32  # No mutation of the original.


def test_bf16_normalization_uses_float32_statistics():
    norm = prepare_opendde_model(LayerNorm(jnp.ones(4), jnp.zeros(4), 1e-5), "bf16")
    x = jnp.array([128.0, 129.0, 130.0, 140.0], dtype=jnp.bfloat16)
    y = eqx.filter_jit(norm)(x)
    fp32 = np.asarray(x, dtype=np.float32)
    expected = (fp32 - fp32.mean()) / np.sqrt(fp32.var() + 1e-5)
    assert y.dtype == jnp.bfloat16
    np.testing.assert_array_equal(y, jnp.asarray(expected, dtype=jnp.bfloat16))
    gradient = jax.grad(lambda z: norm(z).astype(jnp.float32)[0])(x)
    assert np.isfinite(np.asarray(gradient, dtype=np.float32)).all()


def test_bf16_conversion_preserves_partitioned_scan_modules():
    modules = [Linear(jnp.eye(3), None), Linear(2 * jnp.eye(3), None)]
    arrays = jax.tree.map(
        lambda *x: jnp.stack(x),
        *[eqx.filter(m, eqx.is_inexact_array) for m in modules],
    )
    _, static = eqx.partition(modules[0], eqx.is_inexact_array)
    arrays, static = prepare_opendde_model((arrays, static), "bf16")

    @jax.jit
    def run(x):
        def step(value, params):
            return eqx.combine(params, static)(value), None

        return jax.lax.scan(step, x, arrays)[0]

    result = run(jnp.ones(3, dtype=jnp.bfloat16))
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result, jnp.full(3, 2))


@pytest.mark.slow
@pytest.mark.opendde_smoke
def test_real_bf16_forward_and_gradient():
    """Small synthetic complex, full checkpoint; does not establish P17 memory use."""
    from mosaic.common import TOKENS
    from mosaic.losses.structure_prediction import BinderPoseRMSD, IPTMLoss
    from mosaic.models.opendde import OpenDDEModelAbag
    from mosaic.structure_prediction import TargetChain

    model = OpenDDEModelAbag(compute_precision="bf16")
    binder, target = "AGSA", "AGSTLVKA"
    features, _ = model.binder_features(
        len(binder), [TargetChain(target, use_msa=False)]
    )
    x = jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in binder]), 20)
    ref = np.random.default_rng(2).normal(size=(len(binder) + len(target), 3)) * 5
    objective = (
        BinderPoseRMSD(ref[: len(binder)], ref[len(binder) :], rmsd_tolerance=0.0)
        + IPTMLoss()
    )
    loss = model.build_loss(
        loss=objective, features=features, recycling_steps=1, sampling_steps=2
    )
    key = jax.random.key(0)
    (value, _), gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(loss, has_aux=True)
    )(x, key=key)
    assert np.isfinite(float(value))
    assert gradient.dtype == jnp.float32
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.linalg.norm(np.asarray(gradient)) > 0
    output = model.model_output(
        PSSM=x, features=features, recycling_steps=1, sampling_steps=2, key=key
    )
    for values in (
        output.pae,
        output.pae_logits,
        output.distogram_logits,
        output.backbone_coordinates,
    ):
        assert values.dtype == jnp.float32
        assert np.isfinite(np.asarray(values)).all()
