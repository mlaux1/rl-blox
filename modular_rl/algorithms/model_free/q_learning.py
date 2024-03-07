import jax.numpy as jnp
from jax.typing import ArrayLike
from modular_rl.policy.base_policy import UniformRandomPolicy, GreedyQPolicy
from tqdm import tqdm


class QLearning:
    """
    Basic Q-Learning using temporal differences and tabular q-values.
    """

    def __init__(self, env, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env

        self.exploration_policy = UniformRandomPolicy(
            env.observation_space, env.action_space
        )
        self.target_policy = GreedyQPolicy(
            env.observation_space, env.action_space)

    def train(
            self,
            max_episodes: int,
            gamma=0.99
    ) -> ArrayLike:
        ep_rewards = jnp.zeros(max_episodes)

        for i in tqdm(range(max_episodes)):
            observation, _ = self.env.reset()
            while True:
                if np.random.random_sample() < self.epsilon:
                    action = self.exploration_policy.get_action(observation)
                else:
                    action = self.target_policy.get_action(observation)

                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )

                next_action = self.target_policy.get_action(next_observation)
                td_error = (
                    reward
                    + gamma
                    * self.target_policy.value_function.get_action_value(
                        next_observation, next_action
                    )
                    - self.target_policy.value_function.get_action_value(
                        observation, action
                    )
                )

                self.target_policy.update(observation, action, self.alpha * td_error)

                observation = next_observation

                ep_rewards[i] += reward

                if terminated or truncated:
                    break

        return ep_rewards
