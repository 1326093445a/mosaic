"""Patches the three places where jopendde hard-forces float32, so the model
can actually run end-to-end in bfloat16 instead of being silently dragged
back to float32 no matter what the caller casts.

Why this matters (the headline number): at P17's real complex size, the
structural-token triangle attention needs a single **29.79GiB** allocation
in float32. That was confirmed by a direct RESOURCE_EXHAUSTED crash
requesting exactly 29.79GiB with XLA autotuning disabled
(XLA_FLAGS=--xla_gpu_autotune_level=0), so it is the true steady-state
requirement, not autotuner scratch space. No 24GB card can ever run that,
at any diffusion-step or recycling count -- which is exactly the OOM that
made mosaic's OpenDDE path unusable locally while native torch OpenDDE
(which runs bf16 by default via `-d bf16`) did the identical job in ~10s.
In bfloat16 that tensor is ~14.9GiB, which fits.

Why casting alone didn't work: casting all 549 model parameters AND every
feature leaf to bfloat16 still produced a float32 trunk output. Verified
with eqx.filter_eval_shape, which isolated it to
`atom_attention_encoder -> float32` while
`relative_position_encoding -> bfloat16` was already correct. Each fix
below then revealed the next site, in order.

The three sites, all the same root cause (a float32 introduced with no
regard for the surrounding dtype):

1. transformer.py, AtomAttentionEncoder.prepare_cache
   `jnp.arcsinh(ref_charge)` -- `ref_charge` is an int32 feature, and JAX
   promotes integer inputs to transcendentals to float32. That made `c_l`
   float32, and since every other tensor in prepare_cache is explicitly cast
   *to* `c_l.dtype`, the whole atom encoder output went float32 -> `s_inputs`
   -> the trunk's `z` -> the structural-token pair representation.

2. confidence.py, `_one_hot_bins`
   `.astype(jnp.float32)` on the distance-bin indicator. Surfaced as a
   `jax.lax.scan` carry dtype mismatch inside ConfidenceHead's pairformer
   stack (carry in bfloat16[318,318,64], carry out float32[318,318,64]).

3. embedders.py, RelativePositionEncoding.generate_relp
   `.astype(jnp.float32)` on the relative-position one-hot grid. This one
   feeds specifically the structural-token (621-token) branch -- model.py's
   `expand_to_structural_tokens` regenerates relp for the expanded token
   set -- i.e. it lands directly on the tensor that OOMs.

Every fix anchors to a dtype already present at that point (`ref_pos`, the
input `x`, and the consuming Linear's own weight) rather than hardcoding a
dtype. So float32 runs are bit-for-bit unchanged and this is a no-op for
them; only a caller that has deliberately cast to bf16 sees a difference.

NOTE ON NUMERICS: this makes bf16 *possible*, it does not make it
*validated*. Check a bf16 run's outputs against the native torch reference
for the same input before trusting it (for P17-vs-Alpha, no MSA, no
template, recycling=3, seed 0: native torch ipTM = 0.9006).

jopendde is an external git dependency
(pyproject.toml: jopendde = { git = "https://github.com/escalante-bio/jopendde.git" }),
not vendored in this repo, so this can't be a normal source edit here --
it patches the installed package directly. Idempotent (safe to re-run,
including after `uv sync` reinstalls a fresh copy) and fails loudly if any
installed file doesn't match the exact text this patch expects, rather than
silently mis-patching a different version.

Usage:
    .venv/bin/python patches/patch_jopendde_bf16_dtype.py
"""
import importlib.util
import sys
from pathlib import Path

MARKER = "MOSAIC PATCH"

SITES = [
    {
        "file": "transformer.py",
        "what": "AtomAttentionEncoder.prepare_cache (arcsinh on int32 ref_charge)",
        "probe": "ref_charge.astype(ref_pos.dtype)",
        "original": """        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            jnp.arcsinh(ref_charge).reshape(*batch_shape, n_atom, 1)
        )""",
        "patched": """        # MOSAIC PATCH (see patches/patch_jopendde_bf16_dtype.py): `ref_charge`
        # is int32, and jnp.arcsinh promotes integer inputs to float32 -- which
        # forced `c_l`, and via the `.astype(c_l.dtype)` casts below the entire
        # atom encoder output, to float32 even when every parameter and feature
        # had been cast to bfloat16. Anchor to `ref_pos`, the co-located float
        # feature: a no-op for fp32 runs, and it lets bf16 runs stay bf16.
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            jnp.arcsinh(ref_charge.astype(ref_pos.dtype)).reshape(
                *batch_shape, n_atom, 1
            )
        )""",
    },
    {
        "file": "triangular.py",
        "what": "autocast-style compute dtype helper for the attention core",
        "probe": "def _mosaic_attention_dtype()",
        "original": """@register_from_torch("opendde.model.modules.primitives.Attention")
@register_from_torch("opendde.model.triangular.layers.Attention")
class Attention(AbstractFromTorch):""",
        "patched": '''# MOSAIC PATCH: opt-in autocast-style compute dtype for the attention core.
# Off by default (returns None -> byte-for-byte the original fp32 behaviour).
# Set JOPENDDE_ATTENTION_DTYPE=bf16 to run the QK^T/softmax/AV core in bfloat16
# while every parameter and every other activation stays float32, mirroring
# native torch OpenDDE's torch.autocast(dtype=bfloat16) path -- which is what
# `opendde pred -d bf16` actually does. This is the single change that brings
# the model within a 24GB card's budget (measured peak 15.25GiB vs an
# unsatisfiable 29.79GiB request before).
def _mosaic_attention_dtype():
    import os

    v = os.environ.get("JOPENDDE_ATTENTION_DTYPE", "").strip().lower()
    if v in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if v in ("f16", "fp16", "float16"):
        return jnp.float16
    return None


@register_from_torch("opendde.model.modules.primitives.Attention")
@register_from_torch("opendde.model.triangular.layers.Attention")
class Attention(AbstractFromTorch):''',
    },
    {
        "file": "triangular.py",
        "what": "Attention.__call__ (autocast-style core + numpy-scalar scale)",
        "probe": "_compute_dtype = _mosaic_attention_dtype()",
        "original": """        q = q / np.sqrt(self.c_hidden)

        a = jnp.einsum("...hqc,...hkc->...hqk", q, k)

        if attn_bias is not None:
            a = a + self._maybe_add_head_dim(attn_bias, a.ndim)
        if biases is not None:
            for bias in biases:
                a = a + self._maybe_add_head_dim(bias, a.ndim)

        a = jax.nn.softmax(a, axis=-1)

        o = jnp.einsum("...hqk,...hkc->...hqc", a, v)
        o = einops.rearrange(o, "... H Q C -> ... Q H C")""",
        "patched": """        # MOSAIC PATCH (see patches/patch_jopendde_bf16_dtype.py): np.sqrt
        # returns a *strongly typed* numpy float64 scalar, and dividing a
        # bfloat16 array by one promotes it to float32. float() makes it a
        # weakly typed Python scalar that takes on the array's dtype; the value
        # is unchanged, so fp32 runs are bit-for-bit identical.
        q = q / float(np.sqrt(self.c_hidden))

        # MOSAIC PATCH: autocast-style mixed precision for the attention core
        # only. Native torch OpenDDE's `-d bf16` is torch.autocast, which keeps
        # weights/reductions in fp32 and runs only the matmuls in bf16 -- NOT a
        # pure bf16 cast. This mirrors that: the [.., N*H, N, N] score tensor
        # (the single largest allocation in the model -- f32[7452,621,621] =
        # 10.71GiB at P17's complex size) is computed in the compute dtype and
        # cast straight back, so nothing outside this block changes dtype.
        # Opt-in via JOPENDDE_ATTENTION_DTYPE and off by default.
        _compute_dtype = _mosaic_attention_dtype()
        _out_dtype = q.dtype
        if _compute_dtype is not None:
            q = q.astype(_compute_dtype)
            k = k.astype(_compute_dtype)
            v = v.astype(_compute_dtype)

        a = jnp.einsum("...hqc,...hkc->...hqk", q, k)

        if attn_bias is not None:
            bias = self._maybe_add_head_dim(attn_bias, a.ndim)
            a = a + (bias.astype(a.dtype) if _compute_dtype is not None else bias)
        if biases is not None:
            for bias in biases:
                bias = self._maybe_add_head_dim(bias, a.ndim)
                a = a + (bias.astype(a.dtype) if _compute_dtype is not None else bias)

        a = jax.nn.softmax(a, axis=-1)

        o = jnp.einsum("...hqk,...hkc->...hqc", a, v)
        if _compute_dtype is not None:
            o = o.astype(_out_dtype)
        o = einops.rearrange(o, "... H Q C -> ... Q H C")""",
    },
    {
        "file": "confidence.py",
        "what": "_one_hot_bins (distance-bin indicator)",
        "probe": "& (x[..., None] < upper_bins)).astype(x.dtype)",
        "original": """    return ((x[..., None] > lower_bins) & (x[..., None] < upper_bins)).astype(jnp.float32)""",
        "patched": """    # MOSAIC PATCH (see patches/patch_jopendde_bf16_dtype.py): was
    # .astype(jnp.float32), which broke bf16 runs with a jax.lax.scan carry
    # dtype mismatch inside ConfidenceHead's pairformer stack. Anchor to the
    # input's own dtype; a no-op when x is float32.
    return ((x[..., None] > lower_bins) & (x[..., None] < upper_bins)).astype(x.dtype)""",
    },
    {
        "file": "embedders.py",
        "what": "RelativePositionEncoding.generate_relp (structural-token relp grid)",
        "probe": ").astype(self.linear_no_bias.weight.dtype)",
        "original": """        return jnp.concatenate(
            [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(a_rel_pos.dtype), a_rel_chain],
            axis=-1,
        ).astype(jnp.float32)""",
        "patched": """        # MOSAIC PATCH (see patches/patch_jopendde_bf16_dtype.py): was
        # .astype(jnp.float32). jax.nn.one_hot defaults to float32 and this
        # grid feeds the structural-token (621-token) branch, i.e. exactly the
        # pair representation whose triangle attention OOMs. generate_relp only
        # receives integer metadata, so there is no float argument to anchor
        # to -- anchor to the Linear that consumes the grid instead.
        return jnp.concatenate(
            [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(a_rel_pos.dtype), a_rel_chain],
            axis=-1,
        ).astype(self.linear_no_bias.weight.dtype)""",
    },
]


def _purge_stale_pyc(source_path: Path) -> None:
    """Network-mounted storage (e.g. the cluster's /storage) has coarse mtime
    resolution, so Python can serve stale bytecode after an in-place source
    patch. Drop the cached .pyc so the next import recompiles."""
    cache_path = Path(importlib.util.cache_from_source(str(source_path)))
    if cache_path.exists():
        cache_path.unlink()
        print(f"  purged stale bytecode cache: {cache_path.name}")


def main() -> int:
    try:
        import jopendde
    except ImportError:
        print("ERROR: jopendde is not installed in this environment", file=sys.stderr)
        return 1

    pkg_dir = Path(jopendde.__file__).parent
    print(f"jopendde: {pkg_dir}")

    for site in SITES:
        target = pkg_dir / site["file"]
        print(f"\n[{site['file']}] {site['what']}")
        if not target.exists():
            print(f"ERROR: {target} does not exist", file=sys.stderr)
            return 1

        source = target.read_text()

        if MARKER in source and site["probe"] in source:
            print("  already patched -- checking bytecode cache")
            _purge_stale_pyc(target)
            continue

        count = source.count(site["original"])
        if count != 1:
            print(
                f"ERROR: expected exactly 1 occurrence of the target text in "
                f"{target}, found {count}. The installed jopendde does not match "
                f"what this patch was written against. Refusing to patch rather "
                f"than risk a silent mis-patch.",
                file=sys.stderr,
            )
            return 1

        target.write_text(source.replace(site["original"], site["patched"]))
        print("  patched")
        _purge_stale_pyc(target)

    print("\nall dtype-anchor sites patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
