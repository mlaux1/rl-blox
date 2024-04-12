import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from tqdm import tqdm

from modular_rl.algorithms.base_algorithm import BaseAlgorithm
from modular_rl.policy.base_policy import GreedyQPolicy, UniformRandomPolicy
from modular_rl.tools.error_functions import td_error


class QLearning(BaseAlgorithm):
    """
    Basic Q-Learning implementation in JAX.
    """

    def episode_rollout(
            self,
            gamma: float = 0.99
    ) -> float:
        """
        Performs a single episode rollout.

        :param gamma: Discount factor.
        :return: Episode reward.
        """
        ep_reward = 0
        truncated = False
        terminated = False
        observation, _ = self.env.reset()
        action = self.policy.get_action(observation)

        while not terminated and not truncated:
            # get action from policy and perform environment step
            next_observation, reward, terminated, truncated, info = (
                self.env.step(action))

            # get next action
            next_action = self.policy.get_greedy_action(next_observation)

            # update target policy
            val = self.policy.value_function.get_action_value(
                observation, action)
            next_val = self.policy.value_function.get_action_value(
                next_observation, next_action)

            error = td_error(reward, gamma, val, next_val)

            self.policy.value_function.update(
                observation, action, self.alpha * error)

            action = self.policy.get_action(next_observation)
            observation = next_observation
            ep_reward += reward

        return ep_reward





