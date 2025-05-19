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
    n_visits: ArrayLike | None = None,
    epsilon: float = 0.5,
    gamma: float = 0.99,
    seed: int = 1,
    logger: LoggerBase | None = None,
) -> ArrayLike:
    key = jax.random.key(seed)

    observation, _ = env.reset()

    if n_visits is None:
        n_visits = jnp.zeros_like(q_table)

    obs, acts, rews = [], [], []

    for i in tqdm.trange(total_timesteps):
        key, action_key = jax.random.split(key)
        action = get_epsilon_greedy_action(
            action_key, q_table, observation, epsilon
        )

        obs.append(observation)
        acts.append(int(action))
        observation, reward, terminated, truncated, info = env.step(int(action))

        rews.append(reward)

        if terminated or truncated:
            q_table, n_visits = update(
                q_table,
                n_visits,
                jnp.array(rews),
                jnp.array(obs),
                jnp.array(acts),
                gamma,
            )
            observation, _ = env.reset()
            obs, acts, rews = [], [], []
    return q_table, n_visits


def update(
    q_table: ArrayLike,
    n_visits: ArrayLike,
    rewards: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ep_return = 0

    for i in range(len(rewards) - 1, -1, -1):
        obs = observations[i]
        act = actions[i]
        rew = rewards[i]

        ep_return = rew + gamma * ep_return
        n_visits = n_visits.at[obs, act].add(1)
        pred_error = ep_return - q_table[obs, act]
        q_table = q_table.at[obs, act].add(
            1.0 / n_visits[obs, act] * pred_error
        )

    return q_table, n_visits
