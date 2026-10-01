"""Checkpoint-free behavioral checks for the installed padding patch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jopendde.backend import Identity, LayerNorm, Linear
from jopendde.diffusion import centre_random_augmentation, sample_diffusion
from jopendde.transformer import AttentionPairBias, AtomTransformer, _WindowedAttention

from mosaic.opendde_padding import mask_structural_inputs, occupied_tokens


class _OneAttention(eqx.Module):
    attention: AttentionPairBias

    def __call__(self, a, s, z, **kwargs):
        return self.attention(a, s, z, **kwargs)


def _averaging_attention():
    zero = Linear(jnp.zeros((1, 1)), None)
    identity = Linear(jnp.ones((1, 1)), None)
    attention = _WindowedAttention(
        linear_q=zero,
        linear_k=zero,
        linear_v=identity,
        linear_o=identity,
        linear_g=None,
        sigmoid=None,
        num_heads=1,
        c_hidden=1,
    )
    apb = AttentionPairBias(
        layernorm_a=Identity(),
        layernorm_kv=None,
        attention=attention,
        layernorm_z=LayerNorm(jnp.ones(1), jnp.zeros(1), 1e-5),
        linear_nobias_z=zero,
        linear_a_last=None,
        has_s=False,
        cross_attention_mode=False,
    )
    return AtomTransformer(
        diffusion_transformer=_OneAttention(apb), n_queries=2, n_keys=4
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_windowed_atom_attention_excludes_padding_and_its_gradient(compiled):
    model = _averaging_attention()

    def forward(x):
        return model(
            x,
            jnp.zeros_like(x),
            jnp.zeros((2, 2, 4, 1)),
            atom_mask=jnp.array([True, True, False, False]),
        )[:2]

    run = jax.jit(forward) if compiled else forward
    x = jnp.array([[1.0], [3.0], [99.0], [-41.0]])
    np.testing.assert_allclose(run(x), [[2], [2]], atol=1e-6)
    np.testing.assert_array_equal(run(x), run(x.at[2:].set(1000)))
    grad = jax.grad(lambda v: forward(v).sum())(x)
    np.testing.assert_allclose(grad, [[1], [1], [0], [0]], atol=1e-6)


def test_structural_padding_is_absent_from_pair_and_attention_inputs():
    ids = jnp.array([0, 0, 2, 3, -1])  # token 1 empty, 3/-1 out of range
    np.testing.assert_array_equal(occupied_tokens(ids, 3), [True, False, True])
    inputs = jnp.array([[1.0], [900.0], [3.0]])
    si, s, z, bias, pair_mask = mask_structural_inputs(
        ids, inputs, inputs, jnp.ones((3, 3, 1)), jnp.zeros((3, 3))
    )
    np.testing.assert_array_equal(si, [[1], [0], [3]])
    np.testing.assert_array_equal(s, si)
    assert not np.asarray(pair_mask)[1].any()
    assert not np.asarray(pair_mask)[:, 1].any()
    assert not np.asarray(z)[1].any()
    weights = jax.nn.softmax(bias, axis=-1)
    np.testing.assert_array_equal(weights[:, 1], [0, 0, 0])
    np.testing.assert_allclose(weights @ s, [[2], [2], [2]])


def test_centering_ignores_padded_coordinate_values():
    x = jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [999.0, 999.0, 999.0]])
    mask = jnp.array([True, True, False])
    a = centre_random_augmentation(x, jax.random.key(2), mask=mask)
    b = centre_random_augmentation(x[:2], jax.random.key(2))
    np.testing.assert_allclose(a[..., :2, :], b, atol=1e-6)
    np.testing.assert_array_equal(a[..., 2, :], [[0, 0, 0]])


def test_sampler_keeps_padding_zero_at_every_denoiser_call():
    seen = []

    class Dummy:
        diffusion_conditioning = SimpleNamespace(
            prepare_cache=lambda *a: jnp.array(0.0)
        )
        diffusion_transformer = SimpleNamespace(
            precompute_pair_bias=lambda *a: jnp.array(0.0)
        )
        atom_attention_encoder = SimpleNamespace(
            prepare_cache=lambda **kw: (jnp.array(0.0), jnp.array(0.0))
        )

        def __call__(self, x_noisy, **kw):
            jax.debug.callback(lambda x: seen.append(np.asarray(x)), x_noisy)
            return jnp.zeros_like(x_noisy)

    feat = SimpleNamespace(
        atom_to_token_idx=jnp.array([0, 0, 2]),
        relp=jnp.zeros((2, 2, 1)),
        ref_pos=None,
        ref_charge=None,
        ref_mask=None,
        ref_atom_name_chars=None,
        ref_element=None,
        d_lm=None,
        v_lm=None,
        pad_info=None,
    )
    output = sample_diffusion(
        Dummy(),
        feat,
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 2, 1)),
        jnp.array([2.0, 1.0, 0.0]),
        jax.random.key(1),
    )
    output.block_until_ready()
    jax.effects_barrier()
    assert len(seen) == 2
    for coords in [*seen, np.asarray(output)]:
        np.testing.assert_array_equal(coords[..., 2, :], [[0, 0, 0]])
        assert np.isfinite(coords).all()


def test_padding_patch_validates_all_files_before_writing_and_is_idempotent(tmp_path):
    path = Path(__file__).resolve().parents[1] / "patches/patch_jopendde_padding.py"
    spec = importlib.util.spec_from_file_location("padding_patch", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    # Use the installed source as a realistic, version-matched fixture.
    for filename in {site["file"] for site in patch.SITES}:
        # Source syntax varies by indentation level, so keep the real installed
        # module as the integration fixture and undo this patch if present.
        import jopendde

        source = (Path(jopendde.__file__).parent / filename).read_text()
        for site in patch.SITES:
            if site["file"] == filename:
                source = source.replace(site["replacement"], site["original"])
        (tmp_path / filename).write_text(source)
    before = {p.name: p.read_text() for p in tmp_path.iterdir()}
    broken = tmp_path / "diffusion.py"
    broken.write_text("# unsupported source\n")
    with pytest.raises(ValueError):
        patch.patch_package(tmp_path)
    assert (tmp_path / "transformer.py").read_text() == before["transformer.py"]
    broken.write_text(before["diffusion.py"])
    patch.patch_package(tmp_path)
    first = {p.name: p.read_text() for p in tmp_path.glob("*.py")}
    patch.patch_package(tmp_path)
    assert first == {p.name: p.read_text() for p in tmp_path.glob("*.py")}
    modified = tmp_path / "diffusion.py"
    damaged = modified.read_text().replace("mask=atom_mask", "mask=None", 1)
    modified.write_text(damaged)
    with pytest.raises(ValueError, match="unsupported"):
        patch.patch_package(tmp_path)
    assert modified.read_text() == damaged
