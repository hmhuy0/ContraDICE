from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from sources.utils import MLP

from jax.debug import print

class Discriminator(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        
        inputs = jnp.concatenate((observations, actions), axis=-1)
        out = MLP((*self.hidden_dims, 1),
                    activations=self.activations,
                    layer_norm=self.layer_norm)(inputs)
        sigmoid_output = jnp.squeeze(nn.sigmoid(out).clip(min=0.05, max=0.95), -1)
        return sigmoid_output

class Discriminator_state_only(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        
        inputs = observations
        out = MLP((*self.hidden_dims, 1),
                    activations=self.activations,
                    layer_norm=self.layer_norm)(inputs)
        sigmoid_output = jnp.squeeze(nn.sigmoid(out).clip(min=0.05, max=0.95), -1)
        return sigmoid_output
