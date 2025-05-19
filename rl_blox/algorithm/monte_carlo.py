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

    obs_arr = jnp.empty((total_timesteps,), dtype=jnp.int32)
    act_arr = jnp.empty((total_timesteps,), dtype=jnp.int32)
    rew_arr = jnp.empty((total_timesteps,), dtype=jnp.float32)

    start_t = 0

    for i in tqdm.trange(total_timesteps):
        key, action_key = jax.random.split(key)
        action = get_epsilon_greedy_action(
            action_key, q_table, observation, epsilon
        )

        obs_arr = obs_arr.at[i].set(int(observation))
        observation, reward, terminated, truncated, info = env.step(int(action))

        act_arr = act_arr.at[i].set(int(action))
        rew_arr = rew_arr.at[i].set(float(reward))

        if terminated or truncated:
            q_table, n_visits = update(
                q_table,
                n_visits,
                rew_arr[start_t : i + 1],
                obs_arr[start_t : i + 1],
                act_arr[start_t : i + 1],
                gamma,
            )
            observation, _ = env.reset()
            start_t = i + 1
    return q_table, n_visits


@jax.jit
def update(
    q_table: ArrayLike,
    n_visits: ArrayLike,
    rewards: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ep_len = rewards.shape[0]

    def _update_body(i, state):
        q_table, n_visits, ep_return = state
        idx = ep_len - 1 - i

        obs = observations[idx]
        act = actions[idx]
        rew = rewards[idx]

        ep_return = rew + gamma * ep_return
        n_visits = n_visits.at[obs, act].add(1)
        pred_error = ep_return - q_table[obs, act]
        q_table = q_table.at[obs, act].add(
            1.0 / n_visits[obs, act] * pred_error
        )

        return (q_table, n_visits, ep_return)

    q_table, n_visits, _ = jax.lax.fori_loop(
        0, ep_len, _update_body, (q_table, n_visits, 0.0)
    )

    return q_table, n_visits
