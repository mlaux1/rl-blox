import gymnasium as gym
import jax
from jax.typing import ArrayLike
from tqdm.rich import trange

from ..blox.value_policy import epsilon_greedy_policy, greedy_policy
from ..logging.logger import LoggerBase
from ..util.error_functions import td_error


def train_q_learning(
    env: gym.Env,
    q_table: ArrayLike,
    learning_rate: float = 0.1,
    epsilon: float = 0.05,
    gamma: float = 0.99,
    total_timesteps: int = 100_000,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> ArrayLike:
    r"""Q-Learning.

    This function implements the tabular Q-Learning algorithm as originally
    described by Watkins in 1989. The algorithm is off-policy, uses an epsilon-
    greedy exploration strategy and the temporal-difference error to update the
    Q-tables.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_table : ArrayLike
        The Q-table of shape (num_states, num_actions), containing current Q-values.
    learning_rate : float
        The learning rate, determining how much new information overrides old.
    epsilon : float
        The tradeoff for random exploration.
    gamma : float
        The discount factor.
    total_timesteps : int
        The number of time steps to train for.
    seed : int
        The random seed.
    logger: LoggerBase, optional
        Experiment Logger.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.


    Returns
    -------
    q_table : jax.numpy.ndarray
        The updated Q-table after training.

    References
    ----------
    1.  Watkins, C.J.C.H., Dayan, P. Q-learning. Mach Learn 8, 279–292 (1992).
        https://doi.org/10.1007/BF00992698
    """

    key = jax.random.key(seed)

    if logger is not None:
        logger.start_new_episode()

    observation, _ = env.reset()
    steps_per_episode = 0

    for i in trange(total_timesteps, disable=not progress_bar):
        steps_per_episode += 1
        key, subkey1, subkey2 = jax.random.split(key, 3)

        action = epsilon_greedy_policy(q_table, observation, epsilon, subkey1)

        next_observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        next_action = greedy_policy(q_table, next_observation)

        q_table = _update_policy(
            q_table,
            observation,
            action,
            reward,
            next_observation,
            next_action,
            gamma,
            terminated,
            learning_rate,
        )

        if terminated or truncated:
            if logger is not None:
                logger.record_stat("return", info["episode"]["r"], step=i)
                logger.stop_episode(steps_per_episode)
            observation, _ = env.reset()
            steps_per_episode = 0
        else:
            observation = next_observation

    return q_table


@jax.jit
def _update_policy(
    q_table,
    observation,
    action,
    reward,
    next_observation,
    next_action,
    gamma,
    terminated,
    learning_rate,
):
    val = q_table[observation, action]
    next_val = (1 - terminated) * q_table[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table = q_table.at[observation, action].add(learning_rate * error)

    return q_table
