"""Patches jopendde's StructuralTokenExpander to avoid materializing a full
[N, N, Cz_out, Cz_in] gathered-weight tensor during structural-token pair
projection.

Why this exists: `_pair_project_by_role_full` gathers a distinct
LinearNoBias(c_z, c_z) weight matrix per (row_role, col_role) position pair
via `stacked_weight[role_pair_idx]`, producing an explicit
[N, N, Cz_out, Cz_in] tensor before the einsum. At P17's real complex size
(N~602 structural tokens, Cz~384) that's a single 602*602*384*384*4 bytes =
199.09GiB f32 tensor -- confirmed by direct crash-shape arithmetic, matching
an observed RESOURCE_EXHAUSTED allocation of exactly that size.

This is a DIFFERENT bug than patch_jopendde_outer_product_mean.py's target
(that one fixed the trunk's OuterProductMean, and was verified correct, but
did NOT fix this crash -- confirmed by bisection:
examples/p17_opendde_full_gradient_bisect.py showed BinderTargetContact
(needs only the trunk's distogram, no structural-token expansion) succeeds,
while BinderPoseRMSD/confidence terms (need real coordinates, which require
structural-token expansion) fail with the byte-identical crash). Both
patches are real and independently necessary; neither is a substitute for
the other.

The fix: instead of gathering per-position weight matrices into one huge
tensor, apply each of the n_roles**2 projections to the FULL `z` once each
(n_roles**2 is typically small -- a handful of residue/atom roles squared --
so this is n_roles**2 cheap [N,N,Cz]-shaped results, not one [N,N,Cz,Cz]
tensor), then select the correct candidate per position via
`jnp.take_along_axis`. Verified forward- AND gradient-exact against the
original gather-based computation in float64 (differences at machine
epsilon, ~1e-15/1e-16) using a from-scratch reimplementation mirroring the
real Linear/einsum semantics exactly (bias=None, matching "LinearNoBias").

jopendde is an external git dependency, not vendored in this repo, so this
patches the installed package directly, same mechanism as
patch_jopendde_outer_product_mean.py -- idempotent, fails loudly if the
installed jopendde doesn't match the exact text this patch expects.

Usage:
    .venv/bin/python patches/patch_jopendde_structural_token_expander.py
"""
import sys
from pathlib import Path

MARKER = "MOSAIC PATCH"

ORIGINAL = '''    def _pair_project_by_role_full(
        self, z: Float[Array, "... N N Cz"], role: Int[Array, "N"]
    ) -> Float[Array, "... N N Cz"]:
        # Stack all n_roles**2 LinearNoBias(c_z, c_z) weight matrices
        # (shape [Out, In] each, per backend.Linear convention) and gather the
        # one matching each (row_role, col_role) pair; avoids dynamic-shape
        # boolean indexing.
        stacked_weight = jnp.stack(
            [lin.weight for lin in self.pair_block_proj], axis=0
        )  # [n_roles*n_roles, Cz_out, Cz_in]
        role_pair_idx = role[:, None] * self.n_roles + role[None, :]  # [N, N]
        w = stacked_weight[role_pair_idx]  # [N, N, Cz_out, Cz_in]
        return jnp.einsum("...ijk,ijok->...ijo", z, w)'''

PATCHED = '''    def _pair_project_by_role_full(
        self, z: Float[Array, "... N N Cz"], role: Int[Array, "N"]
    ) -> Float[Array, "... N N Cz"]:
        # MOSAIC PATCH (see
        # patches/patch_jopendde_structural_token_expander.py): the original
        # gathered a distinct weight matrix PER POSITION PAIR into an
        # explicit [N, N, Cz_out, Cz_in] tensor before the einsum -- at
        # N~602, Cz~384 that's a single ~199GiB f32 tensor, confirmed by
        # direct crash-shape arithmetic (602*602*384*384*4 bytes =
        # 199.09GiB) matching an observed RESOURCE_EXHAUSTED allocation of
        # exactly that size. Instead: apply each of the n_roles**2
        # projections to z and select per position.
        #
        # SECOND PASS: the first version of this patch did that selection with
        # jnp.stack([lin(z) for lin in self.pair_block_proj]) + take_along_axis,
        # which materializes ALL n_roles**2 projections at once -- [49, N, N, Cz],
        # i.e. 49 x 592MB = 29.79GiB at N=621/Cz=384, confirmed by an OOM
        # requesting exactly 54 x f32[621,621,384]. That removed the 199GiB
        # gather but simply relocated the peak, and it is what kept mosaic's
        # OpenDDE path from running on a 24GB card. Native torch chunks this
        # projection instead (`structural_token_expansion.pair_chunk_size: 128`);
        # the port dropped that along with every other chunked path.
        #
        # So: scan over the stacked weights, accumulating only the positions
        # each role-pair owns. Peak memory is the accumulator plus ONE
        # [N, N, Cz] projection (~2 x 592MB), independent of n_roles. A
        # jax.lax.scan (NOT a Python loop) is required -- an unrolled loop lets
        # XLA fuse all 49 addends into a single wide `add` with every operand
        # live at once, reproducing the very 29.79GiB buffer this avoids.
        #
        # Exactly equivalent to the take_along_axis select: role_pair_idx lies
        # in [0, n_roles**2), so every (i, j) matches exactly one p and the
        # other n_roles**2 - 1 terms contribute a hard zero. Verified
        # bit-exact (max abs diff 0.0) against the take_along_axis formulation
        # in float64, which in turn was verified forward- and gradient-exact
        # against the original gather.
        role_pair_idx = role[:, None] * self.n_roles + role[None, :]  # [N, N]
        weights = jnp.stack(
            [lin.weight for lin in self.pair_block_proj], axis=0
        )  # [n_roles*n_roles, Cz_out, Cz_in]
        acc_dtype = jnp.result_type(z.dtype, weights.dtype)

        def _accumulate_role_pair(acc, xs):
            p, w = xs
            contrib = jnp.einsum("...i,oi->...o", z, w)
            keep = (role_pair_idx == p)[..., None]
            return acc + jnp.where(keep, contrib, 0).astype(acc_dtype), None

        acc0 = jnp.zeros(z.shape[:-1] + (weights.shape[-2],), dtype=acc_dtype)
        out, _ = jax.lax.scan(
            _accumulate_role_pair,
            acc0,
            (jnp.arange(weights.shape[0]), weights),
            unroll=1,
        )
        return out'''


def main():
    import jopendde

    target = Path(jopendde.__file__).parent / "structural_tokens.py"
    print(f"target: {target}")
    text = target.read_text()

    if MARKER in text:
        print("already patched (found MOSAIC PATCH marker) -- checking bytecode cache")
        _purge_stale_pyc(target)
        return

    if ORIGINAL not in text:
        print(
            "ERROR: the installed jopendde's _pair_project_by_role_full "
            "doesn't match the exact text this patch expects -- refusing "
            "to guess. Check structural_tokens.py's "
            "StructuralTokenExpander._pair_project_by_role_full by hand.",
            file=sys.stderr,
        )
        sys.exit(1)

    target.write_text(text.replace(ORIGINAL, PATCHED))
    print("patched StructuralTokenExpander._pair_project_by_role_full to "
          "avoid materializing the full [N,N,Cz,Cz] gathered-weight tensor")

    import py_compile
    py_compile.compile(str(target), doraise=True)
    print("py_compile OK")

    _purge_stale_pyc(target)


def _purge_stale_pyc(source_path: Path) -> None:
    """Delete the compiled __pycache__ entry for this source file.

    Needed on filesystems with coarse mtime resolution (common on network-
    mounted storage, e.g. /storage/... cluster mounts) where Python's
    source-vs-.pyc staleness check can't tell the source just changed --
    confirmed necessary here: a re-run after this patch reproduced the
    exact same crash with byte-identical internal XLA instruction IDs
    (same fusion/reduce/constant numbers down to the last digit), which
    only makes sense if the patched source was never actually re-imported.
    """
    import importlib.util

    cached = Path(importlib.util.cache_from_source(str(source_path)))
    if cached.exists():
        cached.unlink()
        print(f"purged stale bytecode cache: {cached}")


if __name__ == "__main__":
    main()
