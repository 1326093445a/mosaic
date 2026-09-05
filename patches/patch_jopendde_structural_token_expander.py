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
        # projections to the FULL z once each (small: n_roles**2 cheap
        # [N,N,Cz]-shaped results, not one [N,N,Cz,Cz] tensor), then select
        # per position. Verified forward- and gradient-exact against the
        # original gather-based computation in float64 (~1e-15 difference,
        # machine epsilon).
        role_pair_idx = role[:, None] * self.n_roles + role[None, :]  # [N, N]
        candidates = jnp.stack(
            [lin(z) for lin in self.pair_block_proj], axis=0
        )  # [n_roles*n_roles, N, N, Cz]
        idx = jnp.broadcast_to(
            role_pair_idx[None, :, :, None],
            (1,) + role_pair_idx.shape + (candidates.shape[-1],),
        )
        return jnp.take_along_axis(candidates, idx, axis=0)[0]'''


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
