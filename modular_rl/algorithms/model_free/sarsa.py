import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from modular_rl.algorithms.base_algorithm import BaseAlgorithm
from modular_rl.tools.error_functions import td_error


class Sarsa(BaseAlgorithm):
    """
    Basic SARSA implementation in JAX.
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
            next_action = self.policy.get_action(next_observation)

            # update target policy
            val = self.policy.value_function.get_action_value(
                observation, action)
            next_val = self.policy.value_function.get_action_value(
                next_observation, next_action)

            error = td_error(reward, gamma, val, next_val)

            self.policy.value_function.update(
                observation, action, self.alpha * error)

            action = next_action
            observation = next_observation
            ep_reward += reward

        return ep_reward

