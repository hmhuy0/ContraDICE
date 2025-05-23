from typing import Tuple
import jax
import jax.numpy as jnp
from sources.utils import Batch, InfoDict, Model, Params, PRNGKey
from functools import partial

@partial(jax.jit)
def update_discriminator(key: PRNGKey, discriminator: Model,
                         high_batch: Batch, low_batch: Batch) -> Tuple[Model, InfoDict]:

    def disc_loss_fn(discriminator_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        d_high = discriminator.apply({'params': discriminator_params},
                                high_batch.observations,
                                rngs={'dropout': key})
        d_low = discriminator.apply({'params': discriminator_params},
                                low_batch.observations,
                                rngs={'dropout': key})
        loss = -jnp.log(d_high).mean() - jnp.log(1-d_low).mean()
        
        return loss, {}
    
    new_discriminator, info = discriminator.apply_gradient(disc_loss_fn)
    return new_discriminator, info

