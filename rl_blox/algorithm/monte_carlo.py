from collections import namedtuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tqdm.rich import tqdm, trange

from ..blox.value_policy import epsilon_greedy_policy
from ..logging.logger import LoggerBase


def train_monte_carlo(
    env: gym.Env,
    q_table: ArrayLike,
    total_timesteps: int,
    n_visits: ArrayLike | None = None,
    epsilon: float = 0.3,
    gamma: float = 0.99,
    seed: int = 42,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Monte-Carlo Learning.

    This function implements tabular Monte-Carlo Learning as described by
    Sutton and Barto. The algorithm uses epsilon-greedy exploration and the
    incremental implementation to approximate the expected return of state-action
    pairs. Training happens after each completed episode and is done "every-visit".

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    q_table : ArrayLike
        The Q-table of shape (num_states, num_actions), containing current Q-values.
    total_timesteps : int
        The number of time steps to train for.
    gamma : float, optional
        The discount factor.
    epsilon : float, optional
        The tradeoff for random exploration.
    n_visits : ArrayLike, optional
        The table of visits for each state action pair, only required when continuing prior training.
    seed : int, optional
        The random seed.
    logger : LoggerBase, optional
        Experiment Logger.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.


    Returns
    -------
    q_table : jax.numpy.ndarray
        The updated Q-table after training.
    n_visits : jax.numpy.ndarray
        The updated visitation table after training.
    """
    key = jax.random.key(seed)

    if logger is not None:
        logger.start_new_episode()

    observation, _ = env.reset()

    if n_visits is None:
        n_visits = jnp.zeros_like(q_table)

    obs_arr = jnp.empty((total_timesteps,), dtype=jnp.int32)
    act_arr = jnp.empty((total_timesteps,), dtype=jnp.int32)
    rew_arr = jnp.empty((total_timesteps,), dtype=jnp.float32)

    start_t = 0
    steps_per_episode = 0

    for i in trange(total_timesteps, disable=not progress_bar):
        steps_per_episode += 1
        key, action_key = jax.random.split(key)
        action = epsilon_greedy_policy(
            q_table, observation, epsilon, action_key
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
            if logger is not None:
                logger.record_stat("return", info["episode"]["r"], step=i)
                logger.stop_episode(steps_per_episode)
            observation, _ = env.reset()
            steps_per_episode = 0
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

    return namedtuple("MCResult", ["q_table", "n_visits"])(q_table, n_visits)
