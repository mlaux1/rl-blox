import abc
from jax import random
from modular_rl.tools.error_functions import td_error


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
        self.target_policy = policy
        self.exploration_policy = policy

    @abc.abstractmethod
    def train(
            self,
            num_episodes: int,
            gamma: float = 0.99
    ) -> None:
        pass

    def episode_rollout(
            self,
            gamma: float = 0.99):
        ep_reward = 0
        observation, _ = self.env.reset()
        action = self.target_policy.get_action(observation)

        while True:
            next_observation, reward, terminated, truncated, info = (
                self.env.step(action))

            next_action = self.target_policy.get_action(next_observation)

            val = self.target_policy.value_function.get_action_value(
                observation, action)
            next_val = self.target_policy.value_function.get_action_value(
                next_observation, next_action)

            error = td_error(reward, gamma, val, next_val)

            self.target_policy.value_function.update(
                observation, action, self.alpha * error)

            observation = next_observation
            action = next_action

            ep_reward += reward

            if terminated or truncated:
                return ep_reward
