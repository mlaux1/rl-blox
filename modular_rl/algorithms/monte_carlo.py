import numpy as np


class MonteCarlo:
    """
    Implements Monte Carlo Learning.
    """

    def __init__(self, env, epsilon):
        self.epsilon = epsilon
        self.env = env

        self.n_visits = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)
        self.total_return = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)
        self.q_table = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)

    def train(self, max_episodes: int) -> None:

        for _ in range(max_episodes):
            # collect episode
            obs, acs, rews = self.collect_episode_rollout()

            ep_return = sum(rews)

            for i in range(len(acs)):
                self.n_visits[obs[i], acs[i]] += 1
                self.total_return[obs[i], acs[i]] += ep_return
                self.q_table[obs[i], acs[i]] = self.total_return[obs[i], acs[i]] / self.n_visits[obs[i], acs[i]]

    def collect_episode_rollout(self):
        """
        Runs one full episode and returns the observations, actions and rewards.
        """

        observation = self.env.reset()[0]
        observations = [observation]
        actions = []
        rewards = []

        # print(self.q_table.shape)
        # print(observation)

        while True:
            if np.random.random_sample() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                # print(self.n_visits)
                # print(self.total_return)
                # print(self.q_table)
                # print(np.flatnonzero(self.q_table[observation] == self.q_table[observation].max()))
                action = np.random.choice(np.flatnonzero(self.q_table[observation] == self.q_table[observation].max()))

            next_observation, reward, terminated, truncated, info = self.env.step(action)

            observations.append(next_observation)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                print("Terminated episode")
                break

        return observations, actions, rewards

