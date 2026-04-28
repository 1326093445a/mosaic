# Simple transformations of loss functions
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
import jax
import equinox as eqx

from ..common import TOKENS, LinearCombination, LossTerm


class NoCys(LossTerm):
    loss: LossTerm
    """ Precompose loss with function that inserts zero probability for Cysteine (C) in the sequence logits.
        If using this loss, be sure to call `loss.sequence(jax.nn.softmax(logits))` after optimization to get the final sequence!"""

    def __call__(self, seq: Float[Array, "N 19"], *, key):
        assert seq.shape[-1] == 19

        return self.loss(self.sequence(seq), key=key)

    @staticmethod
    def sequence(seq: Float[Array, "N 19"]):
        cys_idx = TOKENS.index("C")
        # reinsert cys
        full_seq = jnp.concatenate(
            [seq[:, :cys_idx], jnp.zeros((*seq.shape[:-1], 1)), seq[:, cys_idx:]],
            axis=-1,
        )

        return full_seq


class SoftClip(LossTerm):
    """
        Soft clips a loss function using an ELU transformation.
        Useful for loss functions that might behave badly when over-optimized.
        For example, optimizing raw ESM2 psuedolikelihood often gives homopolymers
    
    Properties:
    - loss: LossTerm
    - l: lower bound
    - alpha: sharpness of clipping
    - name: name of the clipped loss in the aux dict
    """
    loss: LossTerm
    l: float =  eqx.field(converter=jnp.array)
    alpha: float = eqx.field(converter=jnp.array)
    name: str = "elu"

    def __call__(self, *args, key, **kwargs):
        v, aux = self.loss(*args, key=key, **kwargs)
        z = jax.nn.elu((v - self.l)*self.alpha)
        return z, {"": aux, self.name: z}



class ClippedLoss(LossTerm):
    """
    Clips a loss function to a range [l, u].
    Useful for loss functions that might behave badly when over-optimized.
    For example, optimizing raw ESM2 psuedolikelihood often gives homopolymers.

    Properties:
    - loss: LossTerm
    - l: lower bound
    - u: upper bound
    """

    loss: LossTerm
    l: float = eqx.field(converter=jnp.array)
    u: float = eqx.field(converter=jnp.array)
    name: str = "clipped"

    def __call__(self, *args, key, **kwargs):
        v, aux = self.loss(*args, key=key, **kwargs)
        return v.clip(self.l, self.u), {"": aux, self.name: v.clip(self.l, self.u)}


# Generic tools for fixing positions in a binder sequence
# Note: if you're finetuning an existing binder you might want to (additionally)
#  - If you're using Boltz: use a binder sequence (instead of all "X"'s) to generate features
#  - If using AF2: set the wildtype complex as the initial guess (maybe, this hasn't been tested)
#  - Add additional loss functions to constrain the design to be close to the wildtype (if you have a complex):
#    - ProteinMPNN inverse folding for the complex
#    - Some kind of distance metric on the predicted complex structure, e.g. DistogramCE
#
class SetPositions(LossTerm):
    """Precomposes loss functional with function that maps a soft sequence of ONLY VARIABLE positions to a full binder sequence to eliminate constraints/penalties.
    WARNING: Be sure to call `sequence` *after* optimization, e.g. `loss.sequence(jax.nn.softmax(logits))`."""

    wildtype: Int[Array, "N"] = eqx.field(converter=jnp.array)
    variable_positions: Int[Array, "M"] = eqx.field(converter=jnp.array)
    loss: LossTerm | LinearCombination

    def __call__(self, seq: Float[Array, "M 20"], *, key):
        assert seq.shape == (len(self.variable_positions), len(TOKENS))
        return self.loss(self.sequence(seq), key=key)

    def sequence(self, seq: Float[Array, "M 20"]):
        return (
            jax.nn.one_hot(self.wildtype, len(TOKENS))
            .at[self.variable_positions]
            .set(seq)
        )

    @staticmethod
    def from_sequence(wildtype: str, loss: LossTerm | LinearCombination):
        """Fix standard amino acids but allow variability at positions with 'X'"""
        wildtype_tokens = jnp.array([TOKENS.index(AA) if AA != "X" else -1 for AA in wildtype])
        variable_positions = jnp.array(
            [i for i, AA in enumerate(wildtype) if AA == "X"]
        )
        return SetPositions(wildtype_tokens, variable_positions, loss)


class FixedPositionsPenalty(LossTerm):
    """Penalizes deviation from target at fixed positions using L2^2 loss. Might make optimization more difficult compared to `SetPositions` above, but is simpler"""

    position_mask: Bool[Array, "N"] = eqx.field(converter=jnp.array)
    target: Float[Array, "N 20"] = eqx.field(converter=jnp.array)

    def __call__(self, seq: Float[Array, "N 20"], *, key):
        r = (((seq - self.target) ** 2).sum(-1) * self.position_mask).sum()
        return r, {"fixed_position_penalty": r}

    @staticmethod
    def from_residues(sequence_length: int, positions_and_AAs: list[tuple[int, str]]):
        position_mask = np.zeros(sequence_length, dtype=bool)
        target = np.zeros((sequence_length, len(TOKENS)))
        for idx, AA in positions_and_AAs:
            position_mask[idx] = True
            target[idx, TOKENS.index(AA)] = 1.0

        return FixedPositionsPenalty(jnp.array(position_mask), jnp.array(target))


class EditBudget(LossTerm):
    """Soft hinge penalty on edit distance from a reference sequence over a designable subset.

    The continuous relaxation of Hamming distance for soft `s` against one-hot `s_ref`
    over positions where `designable=True` is `E(s) = sum((1 - <s, s_ref>) * designable)`.
    This is linear in `s` (so convex on the simplex) and equals 0 at native, lower-bounding
    the rounded Hamming distance. The hinge `relu(E(s) - budget)` is zero within budget
    and pulls toward `s_ref` only when the budget is exceeded — exactly the soft-barrier
    behavior we want during gradient optimization or classifier-guided diffusion.
    """

    s_ref: Float[Array, "N 20"] = eqx.field(converter=jnp.array)
    designable: Bool[Array, "N"] = eqx.field(converter=jnp.array)
    budget: float = eqx.field(converter=jnp.array)

    def __call__(self, seq: Float[Array, "N 20"], *, key=None):
        deviation = (1.0 - (seq * self.s_ref).sum(-1)) * self.designable
        E = deviation.sum()
        violation = jax.nn.relu(E - self.budget)
        return violation, {"E_soft": E, "edit_violation": violation}

    @staticmethod
    def from_residues(parent_sequence: str, designable_indices, budget: float):
        """Build from a parent sequence string and a list/array of designable position indices."""
        n = len(parent_sequence)
        s_ref = np.zeros((n, len(TOKENS)), dtype=np.float32)
        for i, AA in enumerate(parent_sequence):
            if AA in TOKENS:
                s_ref[i, TOKENS.index(AA)] = 1.0
        designable = np.zeros(n, dtype=bool)
        designable[np.asarray(designable_indices)] = True
        return EditBudget(jnp.array(s_ref), jnp.array(designable), float(budget))


@jax.custom_vjp
def clip_gradient(threshold, x):
    return x


def clip_gradient_fwd(threshold, x):
    return x, (threshold,)


def clip_gradient_bwd(T, g):
    (threshold,) = T
    g = g - g.mean(axis=-1, keepdims=True)
    norm = jnp.sqrt((g**2).sum() + 1e-8)
    return (
        None,
        jax.lax.select(
            norm > threshold,
            g * (threshold / norm),
            g,
        ),
    )


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


class ClippedGradient(LossTerm):
    loss: LossTerm
    max_norm: float = eqx.field(converter=jnp.array)

    def __call__(self, sequence, *args, **kwargs):
        return self.loss(clip_gradient(self.max_norm, sequence), *args, **kwargs)


@jax.custom_vjp
def norm_gradient(x):
    return x


def norm_gradient_fwd(x):
    return x, None


def norm_gradient_bwd(_, g):
    g = g - g.mean(axis=-1, keepdims=True)
    norm = jnp.sqrt((g**2).sum() + 1e-8)
    return (
        g / norm,
    )


norm_gradient.defvjp(norm_gradient_fwd, norm_gradient_bwd)


class NormedGradient(LossTerm):
    loss: LossTerm

    def __call__(self, sequence, *args, **kwargs):
        return self.loss(norm_gradient(sequence), *args, **kwargs)
