import gym
import d4rl
import numpy as np
import collections
from typing import Optional
from tqdm import tqdm
import jax.numpy as jnp
import jax
from ..utils.env_wrappers import EpisodeMonitor, SinglePrecision

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int,
               shift: float,
               scale: float) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=(self.observations[indx] + shift) * scale,
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=(self.next_observations[indx] + shift) * scale)


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 dataset_size: int = None,
                 start_idx: int = 0):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        if (dataset_size<0):
            dataset_size = len(dataset['observations'])

        print(f'data is from {start_idx} to {start_idx+dataset_size}')
        dataset['observations'] = dataset['observations'][start_idx:start_idx+dataset_size,:]
        dataset['actions'] = dataset['actions'][start_idx:start_idx+dataset_size,:]
        dataset['rewards'] = dataset['rewards'][start_idx:start_idx+dataset_size]
        dataset['next_observations'] = dataset['next_observations'][start_idx:start_idx+dataset_size,:]
        dataset['terminals'] = dataset['terminals'][start_idx:start_idx+dataset_size]

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        super().__init__(observations=dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))

class jax_d4rl_dataset():
    def __init__(self, numpy_dataset):
        for k in numpy_dataset.__dict__.keys():
            if (k=='size'):
                continue
            self.__dict__[k] = jnp.array(numpy_dataset.__dict__[k])
        self.size = numpy_dataset.size

    def sample(self,key, batch_size: int,
               shift: float,
               scale: float) -> Batch:
        indx = jax.random.randint(key=key, minval=0, maxval=self.size, shape=(batch_size,))

        return Batch(observations=(self.observations[indx] + shift) * scale,
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=(self.next_observations[indx] + shift) * scale)

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def normalize_rewards(dataset):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew
        return episode_return

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0

def get_d4rl_dataset(env_name, dataset_size, start_idx=0):
    env = gym.make(env_name)
    dataset = D4RLDataset(env,dataset_size=dataset_size, start_idx=start_idx)

    return dataset

def make_d4rl_env_and_dataset(args):
    env = gym.make(args.env_name)
    env = EpisodeMonitor(env)
    env = SinglePrecision(env)
    
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    dataset = D4RLDataset(env)
    
    if 'antmaze' in args.env_name:
        raise NotImplementedError('Antmaze environments are not supported')
    
    if ('halfcheetah' in args.env_name 
        or 'walker2d' in args.env_name
        or 'hopper' in args.env_name):
        print('Normalizing rewards of dataset')
        normalize_rewards(dataset)
    
    return env, dataset
    