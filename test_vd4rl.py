import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import h5py
from pathlib import Path

from dm_env import specs
from vd4rl import dmc


def load_dataset(data_dir):
    FIRST = 0
    MID = 1
    LAST = 2

    filenames = sorted(data_dir.glob('*.hdf5'))
    total_dataset = None
    for file in filenames:
        episodes = h5py.File(file, 'r')
        episodes = {k: np.array(episodes[k]) for k in episodes.keys()}
        print(f'{file} loaded')
        for k, v in episodes.items():
            print(f'{k}: {v.shape}')
        print('='*30)
        if total_dataset is None:
            total_dataset = episodes
        else:
            for k, v in episodes.items():
                total_dataset[k] = np.concatenate([total_dataset[k], v], axis=0)

    
    
    print('from all files at ', data_dir)
    for k, v in total_dataset.items():
        print(f'{k}: {v.shape}')
    
    reward_arr = [0]
    traj_length_arr = [0]
    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminals = []

    for i in range(len(total_dataset['reward'])):
        if (total_dataset['step_type'][i] != LAST):
            observations.append(total_dataset['observation'][i])
            next_observations.append(total_dataset['observation'][i+1])
            rewards.append(total_dataset['reward'][i])
            terminals.append(total_dataset['step_type'][i]==2)
        if (total_dataset['step_type'][i] != FIRST):
            actions.append(total_dataset['action'][i])
            reward_arr[-1] += total_dataset['reward'][i]
            traj_length_arr[-1] += 1
        if (total_dataset['step_type'][i] == LAST and i<len(total_dataset['reward'])-1):
            reward_arr.append(0)
            traj_length_arr.append(0)
            
    observations = np.array(observations)
    actions = np.array(actions)
    next_observations = np.array(next_observations)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    
    print(observations.shape,np.max(observations),np.min(observations),np.mean(observations))
    print(actions.shape,np.max(actions),np.min(actions),np.mean(actions))
    print(next_observations.shape,np.max(next_observations),np.min(next_observations),np.mean(next_observations))
    print(rewards.shape,np.max(rewards),np.min(rewards),np.mean(rewards))
    print(terminals.shape,np.max(terminals),np.min(terminals),np.mean(terminals))
    print(len(reward_arr),np.mean(reward_arr))
    print(len(traj_length_arr),np.mean(traj_length_arr))
    raise

    return total_dataset

env_name = 'cheetah_run'  # 'cheetah_run', 'humanoid_walk', 'walker_walk'
dataset_name = 'expert'
frame_stack = 3
action_repeat = 2
seed = 0
distracting_mode = None

data_dir = Path(f'vd4rl/dataset/{env_name}/{dataset_name}')

print(f'env_name: {env_name}')
print(f'dataset_name: {dataset_name}')
print(f'data_dir: {data_dir}')

env = dmc.make(env_name, frame_stack,
                action_repeat, seed, distracting_mode)

load_dataset(data_dir)

action_spec = env.action_spec()

print(env.observation_spec())
print(env.action_spec())

state = env.reset()


def print_state(state,step):
    print('='*10 + f'step: {step}','='*10)
    # print all item in state
    for k, v in state._asdict().items():
        print('-'*5, k)
        if (isinstance(v, np.ndarray)):
            print(v.shape)
        else:
            print(v)
    print('='*30)
        
step = 0

class random_agent:
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def predict(self, state):
        return np.random.uniform(self.action_spec.minimum,
                             self.action_spec.maximum,
                             size=self.action_spec.shape)
        
        
def evaluate(env, model, num_episodes):
    return_arr = []
    for ep_idx in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        print(state)

        while not state.last():
            action = model.predict(state)
            state = env.step(action)
            episode_return += state.reward
            episode_length += 1
        return_arr.append(episode_return)
        print(state)
        print(state.last())
        raise
        print(f'episode {ep_idx} return: {episode_return}, length: {episode_length}')
    return np.mean(return_arr)


random_agent = random_agent(action_spec)
print('final return:', evaluate(env, random_agent, 10))