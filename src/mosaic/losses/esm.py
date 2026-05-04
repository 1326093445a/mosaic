# This file incorporates portions of code from the esm2quinox library,
# created by Patrick Kidger and licensed under the Apache License, 
# Version 2.0 (the "License"); you may not use this file except in compliance 
# with the License. You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np
import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float

from esm2quinox import ESM2
from esm2quinox._esm2 import _alphabet as ESM_TOKENS
from ..common import LossTerm, TOKENS

ESM_VOCAB_SIZE = max(ESM_TOKENS.values()) + 1
ESM_BOS_TOKEN = "^"
ESM_EOS_TOKEN = "$"
ESM_MASK_TOKEN = "#"


def load_esm2(model_name: str = "esm2_t33_650M_UR50D"):
    """Load an ESM2 torch checkpoint and convert it to the JAX/Equinox wrapper."""
    print(f"[load_esm2] importing fair-esm for {model_name}", flush=True)
    try:
        import esm
        import esm2quinox
    except ImportError as exc:
        raise ImportError(
            "ESM2 loading requires the `esm` package from fair-esm. "
            "Install it with `uv pip install fair-esm` or add fair-esm to the env."
        ) from exc

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"Unknown ESM2 model '{model_name}' in esm.pretrained")

    print(f"[load_esm2] loading torch checkpoint {model_name}", flush=True)
    torch_model, _ = loader()
    torch_model.eval()
    print(f"[load_esm2] converting {model_name} from torch to JAX", flush=True)
    model = esm2quinox.from_torch(torch_model)
    print(f"[load_esm2] ready {model_name}", flush=True)
    return model


def boltz_to_esm_matrix():
    """Converts from standard tokenization (Boltz ... plus two???) to ESM2QUINOX tokenization"""
    T = np.zeros((len(TOKENS), ESM_VOCAB_SIZE))
    for i, tok in enumerate(TOKENS):
        esm_idx = ESM_TOKENS[tok]
        T[i, esm_idx] = 1
    return T

def apply_trunk(esm, x, is_pad):
    """Trunk portion of the forward pass of esm2quinox._esm2.ESM2"""
    dynamic_layers, static_layer = eqx.partition(esm.layers, eqx.is_array)

    def f(x, dynamic_layer):
        layer = eqx.combine(dynamic_layer, static_layer)
        x = layer(x, is_pad=is_pad)
        return x, None

    x, _ = jax.lax.scan(f, x, xs=dynamic_layers)
    return jax.vmap(esm.layer_norm)(x)

class ESM2PseudoLikelihood(LossTerm):
    """
    Pseudo-likelihood for the ESM-2 masked language model

    Usage:

        import esm
        import esm2quinox
        torch_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        ESM2PLL = ESM2PseudoLikelihood(esm2quinox.from_torch(torch_model))
    """
    esm: ESM2
    stop_grad: bool = True

    def __call__(self, seq_standard_tokens: Float[Array, "N 20"], *, key):
        n = seq_standard_tokens.shape[0]
        # convert from standard tokenization to ESM tokenization
        esm_toks_unpadded = seq_standard_tokens @ boltz_to_esm_matrix()
        # add cls and eos tokens
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([ESM_TOKENS[ESM_BOS_TOKEN]], ESM_VOCAB_SIZE),
                esm_toks_unpadded,
                jax.nn.one_hot([ESM_TOKENS[ESM_EOS_TOKEN]], ESM_VOCAB_SIZE),
            ]
        )

        def single_ll(index: int):
            # replace token at index with mask
            masked_tokens = esm_toks.at[index].set(
                jax.nn.one_hot(ESM_TOKENS[ESM_MASK_TOKEN], ESM_VOCAB_SIZE)
            )
            # embed and run ESM
            embedding = masked_tokens @ self.esm.embedding.weight
            # set masked token embedding to zero
            embedding = embedding.at[index].set(0.0)
            # rescale to account for masking during ESM training
            mask_ratio_train = 0.15 * 0.8
            embedding = embedding * ((1 - mask_ratio_train) / (1 - 1/(n+2)))
            # apply ESM trunk and LM head
            embedding = apply_trunk(self.esm, embedding, np.zeros(n + 2))
            return jax.nn.log_softmax(self.esm.logit_head(embedding[index]))

        masked_log_likelihoods = jax.vmap(single_ll)(jnp.arange(start = 1, stop = n+1))
        if self.stop_grad:
            masked_log_likelihoods = jax.lax.stop_gradient(masked_log_likelihoods)
        pll =  (masked_log_likelihoods * esm_toks_unpadded).sum(-1).mean()
        return -pll, {"esm_pll": pll}
