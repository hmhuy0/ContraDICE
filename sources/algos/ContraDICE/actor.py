from typing import Tuple
import jax
import jax.numpy as jnp
from sources.utils import Batch, InfoDict, Model, Params, PRNGKey
from functools import partial


@partial(jax.jit, static_argnames=['double', 'cal_log', 'args'])
def update_actor(key: PRNGKey, actor: Model, critic: Model, v: jnp.ndarray,
           obs: jnp.ndarray, actions: jnp.ndarray,
           is_good: jnp.ndarray, is_bad: jnp.ndarray, actor_temperature_Q: float, 
           double: bool, args, cal_log: bool) -> Tuple[Model, InfoDict]:
    batch_size = obs.shape[0]

    mixed_obs = obs[batch_size//2:]
    mixed_actions = actions[batch_size//2:]
    q1, q2 = critic(mixed_obs, mixed_actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    q = jax.lax.stop_gradient(q)
    if not args.adv_policy_extraction:
        q_weight = jnp.exp(actor_temperature_Q*q).clip(min=0,max=100)
        weight = jax.lax.stop_gradient(q_weight)
    else:
        adv = (q - v).clip(max=7.0)
        adv_weight = jnp.exp(adv*3)
        adv_weight = adv_weight.clip(min=0,max=100)
        weight = jax.lax.stop_gradient(adv_weight)
        
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           obs,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(actions).sum(-1)    
        expert_log_probs = log_probs[:batch_size//2]
        mixed_log_probs = log_probs[batch_size//2:]
        actor_loss = -(weight * mixed_log_probs).mean()
        if args.learn_expert_data:
            actor_loss += -(100 * expert_log_probs.mean())

        info = {}
        if cal_log:

            info.update({
                        'actor_update/expert_logp': expert_log_probs.mean(),
                        'actor_update/mixed_logp': mixed_log_probs.mean(),
                        'actor_update/actor_loss': actor_loss,
                        'hidden/good_logp': (expert_log_probs*is_good[batch_size//2:]).sum()/is_good[batch_size//2:].sum(),
                        'hidden/bad_logp': (mixed_log_probs*is_bad[batch_size//2:]).sum()/is_bad[batch_size//2:].sum(),
                        'hidden/good_weight_a': (weight*is_good[batch_size//2:]).sum()/is_good[batch_size//2:].sum(),
                        'hidden/bad_weight_a': (weight*is_bad[batch_size//2:]).sum()/is_bad[batch_size//2:].sum(),

                         })

        return actor_loss, info

    new_actor, grad_info = actor.apply_gradient(actor_loss_fn)
    return new_actor, grad_info
