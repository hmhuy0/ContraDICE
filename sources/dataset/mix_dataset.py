import gym
import d4rl
import numpy as np
import collections
from typing import Optional
from tqdm import tqdm
import jax.numpy as jnp
import jax
from .d4rl_dataset import Dataset

MixBatch = collections.namedtuple(
    'MixBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations',
     'is_good', 'is_bad'])


class MixDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 is_good: np.ndarray, is_bad: np.ndarray,
                 size: int):
        super().__init__(observations=observations, actions=actions,
                         rewards=rewards, masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations, size=size)
        self.is_good = is_good
        self.is_bad = is_bad

    def sample(self, batch_size: int,
               shift: float,
               scale: float) -> MixBatch:
        indx = np.random.randint(self.size, size=batch_size)
        return MixBatch(observations=(self.observations[indx] + shift) * scale,
                        actions=self.actions[indx],
                        rewards=self.rewards[indx],
                        masks=self.masks[indx],
                        next_observations=(self.next_observations[indx] + shift) * scale,
                        is_good=self.is_good[indx],
                        is_bad=self.is_bad[indx])

class CombinedDataset(MixDataset):
    def __init__(self,dataset_ls,is_good_ls,is_bad_ls):
        for i,(is_good,is_bad) in enumerate(zip(is_good_ls,is_bad_ls)):
            dataset_ls[i].is_good = np.full_like(dataset_ls[i].rewards, is_good)
            dataset_ls[i].is_bad = np.full_like(dataset_ls[i].rewards, is_bad)
        mixed_dataset = {}
        for k in dataset_ls[0].__dict__.keys():
            if (k=='size'):
                continue
            mixed_dataset[k] = np.concatenate([dataset.__dict__[k] for dataset in dataset_ls], axis=0)
        
        super().__init__(**mixed_dataset, size=mixed_dataset['observations'].shape[0])

# jax wrapper for combined dataset
class jax_combined_dataset():
    def __init__(self,numpy_dataset):
        for k in numpy_dataset.__dict__.keys():
            if (k=='size'):
                continue
            self.__dict__[k] = jnp.array(numpy_dataset.__dict__[k])
        self.size = numpy_dataset.size

    def sample(self,key, batch_size: int,
               shift: float,
               scale: float) -> MixBatch:
        indx = jax.random.randint(key=key, minval=0, maxval=self.size, shape=(batch_size,))

        return MixBatch(observations=(self.observations[indx] + shift) * scale,
                        actions=self.actions[indx],
                        rewards=self.rewards[indx],
                        masks=self.masks[indx],
                        next_observations=(self.next_observations[indx] + shift) * scale,
                        is_good=self.is_good[indx],
                        is_bad=self.is_bad[indx])