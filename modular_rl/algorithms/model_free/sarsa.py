import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from modular_rl.algorithms.base_algorithm import BaseAlgorithm


class Sarsa(BaseAlgorithm):
    """
    Basic SARSA implementation in JAX.
    """

    def train(
            self,
            max_episodes: int,
            gamma: float = 0.99
    ) -> Array:
        ep_rewards = jnp.zeros(max_episodes)

        for i in tqdm(range(max_episodes)):
            ep_reward = self.episode_rollout(gamma)
            ep_rewards = ep_rewards.at[i].add(ep_reward)

        return ep_rewards
