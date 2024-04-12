import abc

import jax.numpy as jnp
from jax import Array, random
from tqdm import tqdm


class BaseAlgorithm:

    def __init__(
            self,
            env,
            policy,
            alpha: float,
            key: int
    ) -> None:

        self.env = env
        self.key = random.PRNGKey(key)
        self.alpha = alpha
        self.policy = policy

    def train(
            self,
            max_episodes: int,
            gamma: float = 0.9999
    ) -> Array:
        ep_rewards = jnp.zeros(max_episodes)

        for i in tqdm(range(max_episodes)):
            ep_reward = self.episode_rollout(gamma)
            ep_rewards = ep_rewards.at[i].add(ep_reward)

        return ep_rewards

    @abc.abstractmethod
    def episode_rollout(
            self,
            gamma: float = 0.99
    ) -> float:
        pass

