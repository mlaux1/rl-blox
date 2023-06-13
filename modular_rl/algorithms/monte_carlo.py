import numpy as np

from modular_rl.policy.base_policy import UniformRandomPolicy, GreedyQPolicy


class MonteCarlo:
    """
    Implements Every-Visit On-Policy Monte Carlo Learning using Q-Values.
    """

    def __init__(self, env, epsilon):
        self.epsilon = epsilon
        self.env = env

        self.exploration_policy = UniformRandomPolicy(env.observation_space, env.action_space)
        self.target_policy = GreedyQPolicy(env.observation_space, env.action_space, 0.0)

        self.n_visits = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)
        self.total_return = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=0.0)

    def train(self, max_episodes: int) -> None:

        for _ in range(max_episodes):
            # collect episode
            obs, acs, rews = self.collect_episode_rollout()

            ep_return = sum(rews)

            for idx in zip(obs, acs):
                self.n_visits[idx] += 1
                self.total_return[idx] += ep_return
                self.target_policy.update(idx, self.total_return[idx] / self.n_visits[idx])

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

