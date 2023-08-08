import numpy as np
import numpy.typing as npt
from modular_rl.policy.base_policy import QValueBasedPolicy
from tqdm import tqdm


class Sarsa:
    """
    Basic Sarsa using temporal differences and tabular q-values.
    """

    def __init__(self, env, policy: QValueBasedPolicy, alpha):
        self.alpha = alpha
        self.env = env

        self.target_policy = policy

    def train(self, max_episodes: int, gamma=0.99) -> npt.ArrayLike:

        ep_rewards = np.zeros(max_episodes)

        for i in tqdm(range(max_episodes)):

            observation, _ = self.env.reset()
            action = self.target_policy.get_action(observation)

            while True:

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                next_action = self.target_policy.get_action(next_observation)

                td_error = (reward +
                            gamma * self.target_policy.value_function.get_action_value(next_observation, next_action) -
                            self.target_policy.value_function.get_action_value(observation, action))

                self.target_policy.value_function.update(observation, action, self.alpha * td_error)

                observation = next_observation
                action = next_action

                ep_rewards[i] += reward

                if terminated or truncated:
                    break

        return ep_rewards
