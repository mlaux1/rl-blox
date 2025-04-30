import numpy as np

from ..blox.base_policy import GreedyQPolicy, UniformRandomPolicy


class MonteCarlo:
    """
    Implements Every-Visit ann First-Visit On-Policy Monte Carlo Learning using Q-Values.
    """

    def __init__(self, env, epsilon, update_mode="every_visit"):
        self.epsilon = epsilon
        self.env = env

        assert update_mode in [
            "every_visit",
            "first_visit",
        ], f"unknown update mode '{update_mode}'"
        self.update_mode = update_mode

        self.exploration_policy = UniformRandomPolicy(
            env.observation_space, env.action_space
        )
        self.target_policy = GreedyQPolicy(
            env.observation_space, env.action_space, 0.0
        )

        self.n_visits = np.full(
            shape=(env.observation_space.n, env.action_space.n), fill_value=0.0
        )
        self.total_return = np.full(
            shape=(env.observation_space.n, env.action_space.n), fill_value=0.0
        )

    def train(self, max_episodes: int) -> None:
        for _ in range(max_episodes):
            # collect episode
            obs, acs, rews = self.collect_episode_rollout()

            ep_return = sum(rews)

            # get the visited state action pairs
            state_action_pairs = zip(obs, acs, strict=False)

            if self.update_mode == "first_visit":
                state_action_pairs = list(set(state_action_pairs))

            for idx in state_action_pairs:
                self.n_visits[idx] += 1
                self.total_return[idx] += ep_return
                new_q_val = self.total_return[idx] / self.n_visits[idx]

                state, action = idx
                step = (
                    new_q_val
                    - self.target_policy.value_function.values[state][action]
                )
                self.target_policy.update(state, action, step)

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

            observation, reward, terminated, truncated, info = self.env.step(
                action
            )

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                print("Terminated episode")
                break

        return observations, actions, rewards
