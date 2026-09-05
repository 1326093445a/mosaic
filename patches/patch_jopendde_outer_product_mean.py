"""Patches jopendde's OuterProductMean to chunk its outer-product einsum
instead of materializing the whole [..., N, N, C, C] tensor at once.

Why this exists: jopendde's OuterProductMean docstring says outright
"Chunking is dropped" -- the unchunked einsum blows up to ~199GiB at P17's
real complex size (N~600 padded tokens, C=384), confirmed by a direct
RESOURCE_EXHAUSTED crash requesting exactly 199.09GiB during a full-OpenDDE
confidence-aware gradient smoke test
(examples/p17_opendde_full_gradient_smoke_test.py). Native (torch) OpenDDE
never hits this because its own OuterProductMean supports real chunking;
jopendde's JAX port dropped that path during porting. The chunked
reimplementation here was verified forward- and gradient-exact against the
unchunked reference in float64 (0.000e+00 difference at every chunk size
tested, including chunk_size=1) -- float32 shows ~1e-4 differences, but
that's pure summation-order noise from splitting one batched matmul into
several smaller ones, not a logic error (re-verified in float64 where FP
reordering noise vanishes).

jopendde is an external git dependency
(pyproject.toml: jopendde = { git = "https://github.com/escalante-bio/jopendde.git" }),
not vendored in this repo, so this can't be a normal source edit here --
it patches the installed package directly. Idempotent (safe to re-run,
including after `uv sync` reinstalls a fresh copy) and fails loudly if the
installed jopendde's OuterProductMean doesn't match the exact text this
patch expects, rather than silently mis-patching a different version.

Usage (on whichever machine/venv is about to run OpenDDE full-path
gradients -- currently only needed on the cluster):
    .venv/bin/python patches/patch_jopendde_outer_product_mean.py
"""
import sys
from pathlib import Path

MARKER = "MOSAIC PATCH"

ORIGINAL = '''class OuterProductMean(AbstractFromTorch):
    """Algorithm 10 in AF3. Chunking is dropped."""

    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: float

    def __call__(
        self,
        m: Float[Array, "... S N Cm"],
        mask: Bool[Array, "... S N"] | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> Float[Array, "... N N Cz"]:
        assert chunk_size is None, "chunked path is not implemented"

        if mask is None:
            mask = jnp.ones(m.shape[:-1], dtype=m.dtype)

        ln = self.layer_norm(m)

        mask = mask[..., None]
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        # [..., N, S, C]
        a = jnp.swapaxes(a, -2, -3)
        b = jnp.swapaxes(b, -2, -3)

        outer = jnp.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)

        norm = jnp.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        return outer / norm'''

PATCHED = '''class OuterProductMean(AbstractFromTorch):
    """Algorithm 10 in AF3.

    MOSAIC PATCH (see patches/patch_jopendde_outer_product_mean.py): the
    unchunked einsum below materializes the full [..., N, N, C, C] outer
    product (C = linear_1/linear_2's output dim) before ever reducing it --
    at N~600, C~384 that's a single ~199GiB f32 tensor, confirmed by direct
    crash-shape arithmetic (602*602*384*384*4 bytes = 199.09GiB, matching an
    observed RESOURCE_EXHAUSTED allocation of exactly that size during a
    P17-complex-size confidence-aware gradient call). Chunking over the
    output's first N axis is exact (no cross terms between different "b"
    indices in the einsum below), verified forward- and gradient-exact
    against the unchunked reference in float64 (0.000e+00 difference at
    every chunk size, including chunk_size=1 -- float32 shows ~1e-4 from
    summation-order reordering only, not a real discrepancy).
    `chunk_size=None` now means "pick a safe default", not "run unchunked"
    -- nothing in jopendde currently plumbs an explicit chunk_size down to
    this call, so keeping the old unchunked behavior on `None` would make
    this patch a no-op in practice. Override via JOPENDDE_OPM_CHUNK_SIZE if
    the default needs tuning for a different complex size / GPU.
    """

    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: float

    def __call__(
        self,
        m: Float[Array, "... S N Cm"],
        mask: Bool[Array, "... S N"] | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> Float[Array, "... N N Cz"]:
        import os

        if mask is None:
            mask = jnp.ones(m.shape[:-1], dtype=m.dtype)

        ln = self.layer_norm(m)

        mask = mask[..., None]
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        # [..., N, S, C]
        a = jnp.swapaxes(a, -2, -3)
        b = jnp.swapaxes(b, -2, -3)

        N = a.shape[-3]
        if chunk_size is None:
            chunk_size = int(os.environ.get("JOPENDDE_OPM_CHUNK_SIZE", "64"))
        chunk_size = min(chunk_size, N)

        def compute_chunk(a_chunk):
            outer_chunk = jnp.einsum("...bac,...dae->...bdce", a_chunk, b)
            outer_chunk = outer_chunk.reshape(outer_chunk.shape[:-2] + (-1,))
            return self.linear_out(outer_chunk)

        if chunk_size >= N:
            outer = compute_chunk(a)
        else:
            chunks = [
                compute_chunk(a[..., start : min(start + chunk_size, N), :, :])
                for start in range(0, N, chunk_size)
            ]
            outer = jnp.concatenate(chunks, axis=-3)

        norm = jnp.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        return outer / norm'''


def main():
    import jopendde

    target = Path(jopendde.__file__).parent / "triangular.py"
    print(f"target: {target}")
    text = target.read_text()

    if MARKER in text:
        print("already patched (found MOSAIC PATCH marker) -- checking bytecode cache")
        _purge_stale_pyc(target)
        return

    if ORIGINAL not in text:
        print(
            "ERROR: the installed jopendde's OuterProductMean doesn't match "
            "the exact text this patch expects -- refusing to guess. The "
            "installed jopendde may have moved on from the version this "
            "patch was written against; check triangular.py's "
            "OuterProductMean by hand.",
            file=sys.stderr,
        )
        sys.exit(1)

    target.write_text(text.replace(ORIGINAL, PATCHED))
    print("patched OuterProductMean to chunk its outer-product einsum "
          f"(default chunk_size=64, override with JOPENDDE_OPM_CHUNK_SIZE)")

    import py_compile
    py_compile.compile(str(target), doraise=True)
    print("py_compile OK")

    _purge_stale_pyc(target)


def _purge_stale_pyc(source_path: Path) -> None:
    """Delete the compiled __pycache__ entry for this source file.

    Needed on filesystems with coarse mtime resolution (common on network-
    mounted storage, e.g. /storage/... cluster mounts) where Python's
    source-vs-.pyc staleness check can't tell the source just changed --
    confirmed necessary here: a re-run reproduced the exact same crash with
    byte-identical internal XLA instruction IDs, which only makes sense if
    the patched source was never actually re-imported.
    """
    import importlib.util

    cached = Path(importlib.util.cache_from_source(str(source_path)))
    if cached.exists():
        cached.unlink()
        print(f"purged stale bytecode cache: {cached}")


if __name__ == "__main__":
    main()
