from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int,
             shift: float,
             scale: float,
             verbose: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            observation = (observation + shift) * scale
            action = agent.sample_actions(observation, 
                                        random_tempurature=0.0,
                                        training=False)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
