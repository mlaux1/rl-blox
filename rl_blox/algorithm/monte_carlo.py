import gymnasium as gym
import jax
import jax.numpy as jnp
import tqdm
from jax.typing import ArrayLike

from ..blox.value_policy import get_epsilon_greedy_action
from ..logging.logger import LoggerBase


def train_monte_carlo(
    env: gym.Env,
    q_table: ArrayLike,
    total_timesteps: int,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    seed: int = 1,
    logger: LoggerBase | None = None,
) -> ArrayLike:
    key = jax.random.key(seed)

    observation, _ = env.reset()

    obs = jnp.empty(total_timesteps)
    acts = jnp.empty(total_timesteps)
    rews = jnp.empty(total_timesteps)

    for i in tqdm.trange(total_timesteps):
        obs[i] = observation
        acts[i] = get_epsilon_greedy_action(key, q_table, observation, epsilon)
        observation, rews[i], terminated, truncated, info = env.step(
            int(acts[i])
        )

        if terminated or truncated:
            q_table = update(q_table, rews, obs, acts)
            observation, _ = env.reset()

    def update(max_episodes: int) -> None:
        # TODO: implement this
        """
        for idx in state_action_pairs:
            self.n_visits[idx] += 1
            self.total_return[idx] += ep_return
            new_q_val = self.total_return[idx] / self.n_visits[idx]

            state, action = idx
            step = (
                new_q_val
                - self.target_policy.value_function.values[state][action]
            )
        """
