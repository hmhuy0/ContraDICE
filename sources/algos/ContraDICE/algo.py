from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
from functools import partial
from collections import deque

from sources.networks import critic, policy, value, discriminator
from sources.utils import Batch, MixBatch, InfoDict, Model, PRNGKey, target_update
from .critic import update_value, update_critic
from .actor import update_actor
from .disc import update_discriminator

@partial(jax.jit, static_argnames=['double_q', 'discount', 'tau', 'actor_temperature_Q', 
                                   'args', 'cal_log'])
def _update_ContraDICE(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,target_critic: Model, 
    good_disc: Model, bad_disc: Model,
    good_batch: Batch, bad_batch: Batch, mix_batch: MixBatch, discount: float, tau: float,
    actor_temperature_Q: float, 
    double_q: bool, args, cal_log: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)

    def get_ratio(observations):
        bad_out = args.r_scale*bad_disc(observations)
        good_out = args.r_scale*good_disc(observations)
        if (args.r_alpha<1):
            frac = (1-args.r_alpha)
        else:
            frac = 1
        out =  jnp.exp(
        (jnp.log(good_out) - args.r_alpha*jnp.log(bad_out))/frac
                      )
        return jax.lax.stop_gradient(out.clip(
            max=args.good_reward_coeff,
            min=args.bad_reward_coeff
        ))
    
    combined_observations = jnp.concatenate((bad_batch.observations,good_batch.observations,
                                             mix_batch.observations),axis=0)
    combined_actions = jnp.concatenate((bad_batch.actions,good_batch.actions,
                                      mix_batch.actions),axis=0)
    combined_rewards = jnp.concatenate((bad_batch.rewards,good_batch.rewards,
                                        mix_batch.rewards),axis=0)*0
    combined_masks = jnp.concatenate((bad_batch.masks,good_batch.masks,
                                      mix_batch.masks),axis=0)
    combined_next_observations = jnp.concatenate((bad_batch.next_observations,good_batch.next_observations,
                                                 mix_batch.next_observations),axis=0)
    combined_batch = Batch(observations=combined_observations, actions=combined_actions,
                        rewards=combined_rewards, masks=combined_masks, 
                        next_observations=combined_next_observations)
    
    is_bad = jnp.concatenate((jnp.ones(bad_batch.observations.shape[0]),
                            jnp.zeros(good_batch.observations.shape[0]),
                            mix_batch.is_bad),axis=0)
    is_good = jnp.concatenate((jnp.zeros(bad_batch.observations.shape[0]),
                              jnp.ones(good_batch.observations.shape[0]),
                              mix_batch.is_good),axis=0)
    
    mix_ratio = get_ratio(mix_batch.next_observations)
    
    batch_size = combined_batch.observations.shape[0]
    t_V = jax.lax.stop_gradient(value(combined_batch.observations[2*batch_size//3:]))
    t_V = t_V.clip(max=args.good_reward_coeff/(1-discount), 
                   min=args.bad_reward_coeff/(1-discount))

    bad_reward_weight = args.bad_reward_coeff*jnp.ones(batch_size//3)
    good_reward_weight = args.good_reward_coeff*jnp.ones(batch_size//3)
    mix_reward_weight = mix_ratio

    reward_weight = jnp.concatenate((bad_reward_weight,
                                    good_reward_weight,
                                    mix_reward_weight),axis=0)


    new_value, value_info = update_value(target_critic, value, combined_batch, is_good, is_bad, 
                                            double_q, key, args, cal_log)
    
    new_actor, actor_info = update_actor(key, actor, target_critic, t_V, obs=combined_batch.observations[batch_size//3:], 
                                        actions=combined_batch.actions[batch_size//3:],
                                        is_good=is_good[batch_size//3:],is_bad=is_bad[batch_size//3:], 
                                        actor_temperature_Q=actor_temperature_Q, 
                                        double=double_q, args=args, cal_log=cal_log)
    
    new_critic, critic_info = update_critic(critic, new_value, combined_batch, is_good, is_bad, 
                                            reward_weight, discount, double_q, key, args, cal_log)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }

@partial(jax.jit)
def process_update_discriminator(rng: PRNGKey, disc: Model, 
                             high_batch: Batch, low_batch: Batch) -> Tuple[PRNGKey, Model, InfoDict]:
    key, rng = jax.random.split(rng)
    new_disc, info = update_discriminator(key, disc, high_batch, low_batch)
    return rng, new_disc, info

class ContraDICE(object):
    def __init__(self, seed: int,
                observations: jnp.ndarray,actions: jnp.ndarray,
                actor_lr: float,critic_lr: float,value_lr: float,disc_lr: float,
                hidden_dims: Sequence[int],discount: float,
                actor_temperature_Q: float,dropout_rate: float,
                layernorm: bool,tau: float, double_q: bool = True,
                opt_decay_schedule: Optional[str] = 'None',
                max_steps: Optional[int] = None,
                value_dropout_rate: Optional[float] = None,
                weight_decay: float = 0.0,
                batch_size: int = 256,
                args = None):
        
        self.tau = tau
        self.discount = discount
        self.actor_temperature_Q = actor_temperature_Q
        self.double_q = double_q
        self.args = args
        self.batch_size = batch_size

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, bad_disc_key, good_disc_key = jax.random.split(rng, 6)
        
        action_dim = actions.shape[1]

        #---- actor ----#
        use_tanh = False
        print(f'actor with tanh squash = {use_tanh}')
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=use_tanh)
        
        if opt_decay_schedule == "cosine":
            print(f"Using cosine decay schedule, weight decay {weight_decay}")
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.add_decayed_weights(weight_decay),
                                    optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            print(f"Using Adam with weight decay {weight_decay}")
            optimiser = optax.chain(
                optax.add_decayed_weights(weight_decay),
                optax.adam(learning_rate=actor_lr)
            )
        
        actor_net = Model.create(actor_def,
                                inputs=[actor_key, observations],
                                tx=optimiser)
        
        #---- critic ----#
        critic_def = critic.DoubleCritic(hidden_dims)
        critic_net = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adamw(learning_rate=critic_lr))
        
        #---- bad discriminator ----#
        bad_disc_def = discriminator.Discriminator_state_only(hidden_dims)
        bad_disc_net = Model.create(bad_disc_def,
                                    inputs=[bad_disc_key, observations],
                                    tx=optax.chain(
                                        optax.add_decayed_weights(weight_decay),
                                        optax.adam(learning_rate=disc_lr)
                                    ))
        
        #---- good discriminator ----#
        good_disc_def = discriminator.Discriminator_state_only(hidden_dims)
        good_disc_net = Model.create(good_disc_def,
                                    inputs=[good_disc_key, observations],
                                    tx=optax.chain(
                                        optax.add_decayed_weights(weight_decay),
                                        optax.adam(learning_rate=disc_lr)
                                    ))
        
        
        #---- target critic ----#
        target_critic_net = Model.create(
            critic_def, inputs=[critic_key, observations, actions])
        
        #---- value critic -----#
        value_def = value.ValueCritic(hidden_dims,
                                          layer_norm=layernorm,
                                          dropout_rate=value_dropout_rate)
        value_net = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adamw(learning_rate=value_lr))

        self.actor = actor_net
        self.critic = critic_net
        self.value = value_net
        self.target_critic = target_critic_net
        self.bad_disc = bad_disc_net
        self.good_disc = good_disc_net
        self.rng = rng
       
    def update_disc(self,high_dataset, low_dataset,disc: Model,num_steps: int,shift,scale) -> Model:
        def _update_step(carry, _):
            rng, cur_disc = carry
            key1, key2, rng = jax.random.split(rng,3)
            high_batch = high_dataset.sample(key1, batch_size=self.batch_size,shift=shift,scale=scale)
            low_batch = low_dataset.sample(key2, batch_size=self.batch_size,shift=shift,scale=scale)
            new_rng,new_disc,info = process_update_discriminator(rng,cur_disc,high_batch,low_batch)
            return (new_rng, new_disc), info
        
        (self.rng,new_disc), info = jax.lax.scan(
            _update_step, (self.rng, disc), None, length=num_steps)
        return new_disc
    
    def train_discriminator(self,high_dataset, low_dataset,disc: str,num_steps: int,shift,scale) -> None:
        if disc == 'good':
            self.good_disc = self.update_disc(high_dataset=high_dataset, low_dataset=low_dataset, 
                                              disc=self.good_disc, num_steps=num_steps, 
                                              shift=shift, scale=scale)
        elif disc == 'bad':
            self.bad_disc = self.update_disc(high_dataset=high_dataset, low_dataset=low_dataset, 
                                             disc=self.bad_disc, num_steps=num_steps,
                                             shift=shift, scale=scale)
        else:
            raise ValueError(f'Invalid discriminator: {disc}')
        
    def update(self, good_dataset: Batch, bad_dataset: Batch, mix_dataset: MixBatch,
               shift, scale, num_steps: int) -> InfoDict:
        def _update_step(carry, _):
            rng, actor, critic, value, target_critic = carry
            key, rng = jax.random.split(rng)
            good_batch = good_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
            bad_batch = bad_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
            mix_batch = mix_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
            
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_ContraDICE(
                rng=rng, actor=actor, critic=critic, value=value, target_critic=target_critic,
                good_disc=self.good_disc, bad_disc=self.bad_disc,
                good_batch=good_batch, bad_batch=bad_batch, mix_batch=mix_batch, 
                discount=self.discount, tau=self.tau, 
                actor_temperature_Q=self.actor_temperature_Q, double_q=self.double_q, args=self.args, 
                cal_log=False)
            return (new_rng, new_actor, new_critic, new_value, new_target_critic), info

        init_carry = (self.rng, self.actor, self.critic, self.value, self.target_critic)
        (self.rng, self.actor, self.critic, self.value, self.target_critic), info = jax.lax.scan(
            _update_step, init_carry, None, length=num_steps-1)

        # Final update with logging
        key, self.rng = jax.random.split(self.rng)
        good_batch = good_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
        bad_batch = bad_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
        mix_batch = mix_dataset.sample(key, batch_size=self.batch_size, shift=shift, scale=scale)
        
        self.rng, self.actor, self.critic, self.value, self.target_critic, info = _update_ContraDICE(
            rng=self.rng, actor=self.actor, critic=self.critic, value=self.value, target_critic=self.target_critic,
            good_disc=self.good_disc, bad_disc=self.bad_disc,
            good_batch=good_batch, bad_batch=bad_batch, mix_batch=mix_batch, 
            discount=self.discount, tau=self.tau, 
            actor_temperature_Q=self.actor_temperature_Q, double_q=self.double_q, args=self.args, 
            cal_log=True)
        return info
       
    def sample_actions(self,
                       observations: np.ndarray,
                       random_tempurature: float = 1.0,
                       training: bool = False) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, 
                                           self.actor.apply_fn,
                                           self.actor.params, 
                                           observations,
                                           random_tempurature, 
                                           training=training)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
