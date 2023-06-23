import numpy as np
import numpy.typing as npt
from modular_rl.policy.base_policy import UniformRandomPolicy, GreedyQPolicy


class QLearning:
    """
    Basic Q-Learning using temporal differences and tabular q-values.
    """

    def __init__(self, env, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env

        self.exploration_policy = UniformRandomPolicy(env.observation_space, env.action_space)
        self.target_policy = GreedyQPolicy(env.observation_space, env.action_space, 0.0)

        self.q_table = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)

    def train(self, max_episodes: int, gamma=0.99) -> npt.ArrayLike:

        ep_rewards = np.zeros(max_episodes)

        for i in range(max_episodes):
            observation, _ = self.env.reset()
            while True:
                if np.random.random_sample() < self.epsilon:
                    action = self.exploration_policy.get_action(observation)
                else:
                    action = self.target_policy.get_action(observation)

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                next_action = self.target_policy.get_action(next_observation)
                td_error = gamma * self.q_table[next_observation, next_action] - self.q_table[observation, action]

                self.q_table[observation, action] += self.alpha * (reward + td_error)

                observation = next_observation

                ep_rewards[i] += reward

                if terminated or truncated:
                    # print(self.q_table)

                    break

        return ep_rewards

    def collect_episode_rollout(self):
        """
        Runs one full episode and returns the observations, actions and rewards.
        """

        observation = self.env.reset()[0]
        observations = [observation]
        actions = []
        rewards = []

        while True:
            if np.random.random_sample() < self.epsilon:
                action = self.exploration_policy.get_action(observation)
            else:
                action = self.target_policy.get_action(observation)

            next_observation, reward, terminated, truncated, info = self.env.step(action)

            observations.append(next_observation)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                print("Terminated episode")
                break

        return observations, actions, rewards


