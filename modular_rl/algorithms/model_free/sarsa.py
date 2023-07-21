import numpy as np
from modular_rl.policy.base_policy import QValueBasedPolicy


class Sarsa:
    """
    Basic Sarsa using temporal differences and tabular q-values.
    """

    def __init__(self, env, policy: QValueBasedPolicy, alpha):
        self.alpha = alpha
        self.env = env

        self.policy = policy

    def train(self, max_episodes: int, gamma=0.99) -> None:

        for _ in range(max_episodes):

            observation, _ = self.env.reset()
            action = self.policy.get_action(observation)

            while True:

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                next_action = self.policy.get_action(next_observation)

                td_error = reward + gamma * self.policy.value_function.get_action_value(next_observation, next_action) - \
                           self.policy.value_function.get_action_value(observation, action)

                self.policy.value_function.update(observation, action, self.alpha * td_error)

                observation = next_observation
                action = next_action

                if terminated or truncated:
                    break
