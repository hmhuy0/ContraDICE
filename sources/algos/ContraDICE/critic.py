from typing import Tuple
import jax.numpy as jnp
import jax
from functools import partial

from sources.utils import Batch, InfoDict, Model, Params, PRNGKey


@partial(jax.jit, static_argnames=['alpha', 'args'])
def chi_square_loss(diff, alpha, args=None):
    loss = alpha*jnp.maximum(diff+diff**2/4,0) - (1-alpha)*diff
    return loss

@partial(jax.jit, static_argnames=['alpha', 'args'])
def total_variation_loss(diff, alpha, args=None):
    loss = alpha*jnp.maximum(diff,0) - (1-alpha)*diff
    return loss

@partial(jax.jit, static_argnames=['v_beta','args'])
def reverse_kl_loss(diff, v_beta, args=None):
    z = diff/v_beta
    if args.max_clip is not None:
        z = jnp.minimum(z, args.max_clip) # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = jnp.exp(z - max_z) - z*jnp.exp(-max_z) - jnp.exp(-max_z)  # scale by e^max_z
    return loss

@partial(jax.jit, static_argnames=['alpha', 'args'])
def spare_loss(q,v, alpha, args=None):
    weight = jax.lax.stop_gradient(jnp.where(1 + (q-v)/(2*alpha) > 0, 1, 0))
    loss = weight * (1 + (q-v)/(2*alpha))**2 + v/alpha
    return loss

@partial(jax.jit, static_argnames=['expectile'])
def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


@partial(jax.jit, static_argnames=['double', 'cal_log','args'])
def update_value(critic: Model, value: Model,
                 batch: Batch, is_good: jnp.ndarray, is_bad: jnp.ndarray,
                 double: bool, key: PRNGKey, 
                 args, cal_log: bool) -> Tuple[Model, InfoDict]:
    batch_size = batch.observations.shape[0]
    obs = batch.observations
    acts = batch.actions
        

    q1, q2 = critic(obs, acts)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        if (args.v_update == 'expectile_loss'):
            value_loss = expectile_loss(jax.lax.stop_gradient(q) - v, expectile=0.9).mean()
        elif (args.v_update == 'rkl_loss'):
            value_loss = reverse_kl_loss(jax.lax.stop_gradient(q) - v, v_beta=args.v_beta, args=args).mean()
        elif (args.v_update == 'spare_loss'):
            value_loss = spare_loss(jax.lax.stop_gradient(q), v, alpha=args.alpha, args=args).mean()
        else:
            raise ValueError(f"Invalid value update method: {args.v_update}")
        info = {}
        if cal_log:
            info.update({
                'value_update/loss': value_loss,
                'value_update/value': v.mean(),
                'value_update/q': q.mean(),
            })
        
        return value_loss, info

    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info



@partial(jax.jit, static_argnames=['double', 'cal_log','args'])
def update_critic(critic: Model, target_value: Model, batch: Batch,
                is_good: jnp.ndarray, is_bad: jnp.ndarray,
                reward_weight: jnp.ndarray,
                discount: float, double: bool, key: PRNGKey, 
                args, cal_log: bool) -> Tuple[Model, InfoDict]:
    batch_size = batch.observations.shape[0]
    
    next_v = target_value(batch.next_observations)
    next_v = discount * batch.masks * next_v.clip(max=args.good_reward_coeff/(1-discount), 
                                                  min=args.bad_reward_coeff/(1-discount))

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, batch.actions)
        info = {}
        
        def iq_loss(q, next_v):
            reward = q - next_v
            if args.exp_r:
                transformed_reward = jnp.exp( -(reward/(1-args.reward_beta)
                                               ).clip(min=-7.0,max=7.0)
                                            )
            else:
                transformed_reward = -reward
                
            reward_loss = (reward_weight * transformed_reward).mean()
            regularizer_loss = 0.5 * (reward**2).mean()
            loss = reward_loss + regularizer_loss
            loss_dict = {}
            return loss, loss_dict
            
            
        if double:
            loss1, loss_dict1 = iq_loss(q1, next_v)
            loss2, loss_dict2 = iq_loss(q2, next_v)
            critic_loss = (loss1 + loss2).mean()
            if(cal_log):
                for k, v in loss_dict1.items():
                    info[k] = (loss_dict1[k] + loss_dict2[k])/2
        else:
            critic_loss, loss_dict = iq_loss(q1, next_v)
            if(cal_log):
                for k, v in loss_dict.items():
                    info[k] = v

        return critic_loss, info

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

