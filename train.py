import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax import config
import importlib.util  # Added for dynamic importing

import sys
sys.path.append('..')
sys.path.append('./')

import numpy as np
import sys
from absl import app, flags
from sources.parse import args
import jax
from collections import deque
import random
import gym
from pprint import pprint
import d4rl
import jax.numpy as jnp
from sources.utils import ConfigArgs, evaluate, log_info
from sources.algos.ContraDICE.algo import ContraDICE
from sources.dataset.mix_dataset import CombinedDataset, jax_combined_dataset
from sources.dataset.d4rl_dataset import get_d4rl_dataset, jax_d4rl_dataset
from sources.utils.env_wrappers import EpisodeMonitor, SinglePrecision



def make_mixed_dataset(args,max_episode_steps):
    dataset_ls = []
    is_good_ls = []
    is_bad_ls = []
    mix_dataset_name = ''
    for i,(mixed_name, mixed_size) in enumerate(zip(args.mixed_name_list, args.mixed_size_list)):
        dataset_ls.append(get_d4rl_dataset(mixed_name, int(mixed_size)*max_episode_steps))
        is_good_ls.append(args.is_good_list[i])
        is_bad_ls.append(args.is_bad_list[i])
        print(f'Loaded {mixed_name} with size {mixed_size}, shape {dataset_ls[-1].observations.shape}, ',
              f'is_good: {is_good_ls[-1]}, is_bad: {is_bad_ls[-1]}')
        task_name = mixed_name.split('-')[1]
        mix_dataset_name += f'{task_name}-{int(mixed_size)//1000}k,'
        
    mixed_dataset = CombinedDataset(dataset_ls, is_good_ls, is_bad_ls)

    print('-'*100)
    print('mixed_dataset')
    print('Total samples: ', mixed_dataset.observations.shape)
    print('Good samples: ', mixed_dataset.observations[mixed_dataset.is_good==1].shape)
    print('Bad samples: ', mixed_dataset.observations[mixed_dataset.is_bad==1].shape)
    print('-'*100)
    return mixed_dataset, mix_dataset_name

def create_expert_dataset_and_env(args,robot_name):
    env = gym.make(args.env_name)
    env = EpisodeMonitor(env)
    env = SinglePrecision(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    if (robot_name in ['halfcheetah', 'walker2d', 'hopper','ant']):
        env.max_episode_steps = 1000
    elif (robot_name in ['hammer', 'door', 'relocate']):
        env.max_episode_steps = 200
    elif (robot_name in ['pen']):
        env.max_episode_steps = 100
    elif (robot_name in ['kitchen']):
        env.max_episode_steps = 280
    else:
        raise ValueError(f'Unknown robot name: {robot_name}')
    
    expert_dataset = get_d4rl_dataset(args.env_name, args.expert_dataset_size*env.max_episode_steps)
    return env, expert_dataset, env.max_episode_steps

def make_bad_dataset(args,max_episode_steps):
    dataset_ls = []
    is_good_ls = []
    is_bad_ls = []
    bad_dataset_name = f''

    for i,(bad_name, bad_size) in enumerate(zip(args.bad_name_list, args.bad_size_list)):
        dataset_ls.append(get_d4rl_dataset(bad_name, int(bad_size)*max_episode_steps))
        is_good_ls.append(0)
        is_bad_ls.append(1)
        print(f'Loaded {bad_name} with size {bad_size}, shape {dataset_ls[-1].observations.shape}, ',
              f'is_good: {is_good_ls[-1]}, is_bad: {is_bad_ls[-1]}')
        task_name = bad_name.split('-')[1]
        bad_dataset_name += f'{task_name}-{int(bad_size)//1000}k,'
        
    bad_dataset = CombinedDataset(dataset_ls, is_good_ls, is_bad_ls)

    print('-'*100)
    print('bad_dataset')
    print('Total samples: ', bad_dataset.observations.shape)
    print('Good samples: ', bad_dataset.observations[bad_dataset.is_good==1].shape)
    print('Bad samples: ', bad_dataset.observations[bad_dataset.is_bad==1].shape)
    print('-'*100)
    return bad_dataset, bad_dataset_name


def main(_):
    print(args.env_name)
    print(args.bad_name_list)
    print(args.mixed_name_list)
    print(args.mixed_size_list)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    robot_name = args.env_name.split('-')[0]
    expert_task = args.env_name.split('-')[1]
    first_mixed_task_short_name = args.mixed_name_list[0]
    config_file = f'configs/{robot_name}-{args.bad_name_list[0][0]}.py'
    print(f'Using config file: {config_file}')
    
    try:
        spec = importlib.util.spec_from_file_location("config_module", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        env_config = config_module.get_config()
        env_config.actor_temperature_Q=env_config.actor_Q_scale/env_config.v_beta
        env_config.good_reward_coeff = env_config.r_scale
        env_config.bad_reward_coeff = -env_config.r_scale*env_config.r_alpha
    except Exception as e:
        print(f'Error loading config file: {e}')
        raise e
        
    
    version = args.env_name.split('-')[-1]
    run_name = f'[Good={args.expert_dataset_size}]_[Bad'
    for i in range(len(args.bad_name_list)):
        run_name += f'|{args.bad_name_list[i][0]}={int(args.bad_size_list[i])}'
        args.bad_name_list[i] = f'{robot_name}-{args.bad_name_list[i]}-{version}'

    run_name += ']_[Unlabeled'
    for i in range(len(args.mixed_name_list)):
        run_name += f'|{args.mixed_name_list[i][0]}={int(args.mixed_size_list[i])}'
        args.mixed_name_list[i] = f'{robot_name}-{args.mixed_name_list[i]}-{version}'
    run_name += ']'       

    print(env_config)
    
    env, expert_dataset,max_episode_steps = create_expert_dataset_and_env(args,robot_name)
    bad_dataset, _ = make_bad_dataset(args,max_episode_steps)
    bad_dataset = jax_combined_dataset(bad_dataset)

    mixed_dataset, _ = make_mixed_dataset(args,max_episode_steps)
    mixed_dataset = jax_combined_dataset(mixed_dataset)
    print('-'*100)
    print('mixed_dataset')
    print(f'Loaded mixed dataset with size {mixed_dataset.observations.shape}')
    print('-'*100)
    expert_dataset = jax_d4rl_dataset(expert_dataset)
    print('expert_dataset')
    print(f'Loaded expert dataset with size {expert_dataset.observations.shape}')
    print('-'*100)
    print('bad_dataset')
    print(f'Loaded bad dataset with size {bad_dataset.observations.shape}')
    print('-'*100)
    if (env_config.state_norm):
        print('Normalizing states')
        shift = - jnp.mean(mixed_dataset.observations, axis=0)
        scale = 1 / (jnp.std(mixed_dataset.observations, axis=0) + 1e-3)
    else:
        print('Not normalizing states')
        shift = 0
        scale = 1

    
    print('-'*100)
    print(run_name)
    print('args')
    pprint(args.flag_values_dict())
    print('env_config')
    pprint(env_config)
    print('-'*100)
    
    agent_args = ConfigArgs(
                    max_clip=args.max_clip,
                    v_beta=env_config.v_beta,
                    eval_interval=args.eval_interval,
                    v_update=args.v_update,
                    adv_policy_extraction=args.adv_policy_extraction,
                    r_scale=env_config.r_scale,
                    r_alpha=env_config.r_alpha,
                    good_reward_coeff=env_config.good_reward_coeff,
                    bad_reward_coeff=env_config.bad_reward_coeff,
                    exp_r=args.exp_r,
                    learn_expert_data=args.learn_expert_data,
                    )
    
    print(agent_args)
    
    hidden_dims = tuple([args.hidden_size]*args.num_layers)
    print(f'hidden_dims: {hidden_dims}')
    
    agent = ContraDICE(args.seed,
                observations=env.observation_space.sample()[np.newaxis],
                actions=env.action_space.sample()[np.newaxis],
                max_steps=args.max_steps,
                double_q=args.double,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                disc_lr=args.disc_lr,
                value_lr=args.value_lr,
                hidden_dims=hidden_dims,
                discount=args.discount,
                actor_temperature_Q=env_config.actor_temperature_Q,
                dropout_rate=args.dropout_rate,
                layernorm=args.layernorm,
                tau=env_config.tau,
                weight_decay=env_config.weight_decay,
                batch_size=args.batch_size,
                args=agent_args)
    

    disc_key = jax.random.PRNGKey(args.seed)
    def test_discriminator(agent, good_dataset, bad_dataset, mixed_dataset, 
                        shift, scale, key):
        print('test discriminator ratio')
        mix_batch = mixed_dataset.sample(key, batch_size=10000, shift=shift, scale=scale)
        bad_mix_disc = agent.bad_disc(mix_batch.observations)
        good_mix_disc = agent.good_disc(mix_batch.observations)

        ratio_info = {
            
            'final_ratio/hidden_bad_baddisc': round(((bad_mix_disc*mix_batch.is_bad).sum()/mix_batch.is_bad.sum()).item(), 2),
            'final_ratio/hidden_good_baddisc': round(((bad_mix_disc*mix_batch.is_good).sum()/mix_batch.is_good.sum()).item(), 2),
            'final_ratio/hidden_bad_gooddisc': round(((good_mix_disc*mix_batch.is_bad).sum()/mix_batch.is_bad.sum()).item(), 2),
            'final_ratio/hidden_good_gooddisc': round(((good_mix_disc*mix_batch.is_good).sum()/mix_batch.is_good.sum()).item(), 2),
            
        }
        pprint(ratio_info)
        print('-'*50)

  
    if True:
        print(f' training Discriminator from scratch')
        for step in range(args.num_disc_train//args.eval_interval):
            info_1 = agent.train_discriminator(high_dataset=expert_dataset, 
                                     low_dataset=mixed_dataset, 
                                     disc='good', 
                                     num_steps=args.eval_interval, 
                                     shift=shift, scale=scale)
            info_2 = agent.train_discriminator(high_dataset=bad_dataset, 
                                     low_dataset=mixed_dataset, 
                                     disc='bad', 
                                     num_steps=args.eval_interval, 
                                     shift=shift, scale=scale)
            disc_key, _ = jax.random.split(disc_key)
            print((step+1)*args.eval_interval)
            test_discriminator(agent, expert_dataset, bad_dataset, mixed_dataset, shift, scale, disc_key)
        print('finished training discriminator')
        
    print('-'*50)
    test_discriminator(agent, expert_dataset, bad_dataset, mixed_dataset, shift, scale, disc_key)
    
    # Setup new log directory and file path
    log_base_dir = args.exp_name
    bad_sizes_str = ",".join(map(str, args.bad_size_list))
    mixed_sizes_str = ",".join(map(str, args.mixed_size_list))
    
    dynamic_dir_name = f"{robot_name}/ContraDICE_{first_mixed_task_short_name}_{expert_task}"+\
        f"_[{args.expert_dataset_size}]_[{bad_sizes_str}]_[{mixed_sizes_str}]"
    full_log_dir_path = os.path.join(log_base_dir, dynamic_dir_name)
    os.makedirs(full_log_dir_path, exist_ok=True)
    log_file_path = os.path.join(full_log_dir_path, f"{args.seed}.txt")
    
    with open(log_file_path, 'w') as f_log:
        pass # Just to clear/create the file

    best_eval_returns = -np.inf
    last_20_returns = deque(maxlen=20)
    print('------- start training -------')
    
    for step in range(0, args.max_steps//args.eval_interval + 1): 
        update_info = {}
        if (step > 0):
            update_info = agent.update(expert_dataset,bad_dataset,mixed_dataset,
                                      shift=shift, scale=scale,
                                      num_steps=args.eval_interval)
        eval_stats = evaluate(agent, env, args.eval_episodes,
                            shift=shift, scale=scale)

        with open(log_file_path, 'a') as f_log:
            f_log.write(f"{eval_stats['return']:.3f}\n")
            
        last_20_returns.append(eval_stats['return'])
        print('-'*50)
        print(f"Eval in step {step*args.eval_interval} return: {eval_stats['return']:.2f}")
        if (eval_stats['return'] > best_eval_returns):
            best_eval_returns = eval_stats['return']
        update_info['eval/return'] = eval_stats['return']
        update_info['eval/last_20_return'] = np.mean(last_20_returns)
        update_info['eval/best_return'] = best_eval_returns
        for k, v in update_info.items():
            if isinstance(v, jax.Array):
                update_info[k] = round(v.item(), 3)
            elif isinstance(v, float):
                update_info[k] = round(v, 3)
        log_info(update_info, args, step*args.eval_interval)
            
    

if __name__ == '__main__':
    app.run(main)
