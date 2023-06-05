import numpy as np


class QLearning:

    def __init__(self, env, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.q_table = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)

    def train(self, max_episodes: int, gamma=0.99) -> None:

        for _ in range(max_episodes):
            observation, _ = self.env.reset()
            while True:
                if np.random.random_sample() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.random.choice(np.flatnonzero(self.q_table[observation] == self.q_table[observation].max()))

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                self.q_table[observation, action] += self.alpha * (reward + gamma * self.q_table[next_observation].max() - self.q_table[observation, action])

                observation = next_observation

                if terminated or truncated:
                    print(self.q_table)
                    break


