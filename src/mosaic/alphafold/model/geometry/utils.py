"""Utils for geometry library."""

from typing import List

import jax.numpy as jnp


def unstack(value: jnp.ndarray, axis: int = -1) -> List[jnp.ndarray]:
  return [jnp.squeeze(v, axis=axis)
          for v in jnp.split(value, value.shape[axis], axis=axis)]
