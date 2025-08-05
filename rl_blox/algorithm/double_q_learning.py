from collections import namedtuple

import gymnasium
import jax
from jax.typing import ArrayLike
from tqdm.rich import tqdm, trange

from ..blox.value_policy import epsilon_greedy_policy, greedy_policy
from ..logging.logger import LoggerBase
from ..util.error_functions import td_error


def train_double_q_learning(
    env: gymnasium.Env,
    q_table1: ArrayLike,
    q_table2: ArrayLike,
    learning_rate: float = 0.1,
    epsilon: float = 0.05,
    gamma: float = 0.99,
    total_timesteps: int = 10_000,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[ArrayLike, ArrayLike]:
    r"""Double Q-Learning.

    This function implements the double Q-Learning. It uses two tabular
    Q-functions and an off-policy TD-update. To select the next action, the
    sum of the two Q-values needs to be maximised.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_table1: ArrayLike
        The first Q-table of shape (num_states, num_actions), containing
        current Q-values.
    q_table2: ArrayLike
        The second Q-table of shape (num_states, num_actions), containing
        current Q-values.
    learning_rate : float
        Learning rate alpha for update of Q table.
    epsilon : float
        The tradeoff for random exploration
    gamma : float
        The discount factor, representing importance of future rewards
    total_timesteps : int
        The number of steps to train for
    seed : int, optional
        The random seed.
    logger : LoggerBase, optional
        Experiment logger.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.



    Returns
    -------
    q_table1 : jax.numpy.ndarray
        The first updated Q-table after training.
    q_table2 : jax.numpy.ndarray
        The second updated Q-table after training.

    References
    ----------
    [1] Hado van Hasselt. 2010. Double Q-learning. In Proceedings of the
        24th International Conference on Neural Information Processing Systems
        - Volume 2 (NIPS'10), Vol. 2. Curran Associates Inc., Red Hook, NY, USA,
        2613–2621.
    """
    key = jax.random.key(seed)
    observation, _ = env.reset()
    steps_per_episode = 0

    for i in trange(total_timesteps, disable=not progress_bar):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        q_table = q_table1 + q_table2
        action = epsilon_greedy_policy(q_table, observation, epsilon, subkey1)
        steps_per_episode += 1
        next_observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        val = jax.random.uniform(subkey3)
        if val < 0.5:
            q_table1 = _dql_update(
                subkey2,
                q_table1,
                q_table2,
                observation,
                action,
                reward,
                next_observation,
                gamma,
                learning_rate,
                terminated,
            )
        else:
            q_table2 = _dql_update(
                subkey2,
                q_table2,
                q_table1,
                observation,
                action,
                reward,
                next_observation,
                gamma,
                learning_rate,
                terminated,
            )

        if terminated or truncated:
            if logger is not None:
                logger.record_stat("return", info["episode"]["r"], step=i)
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()
            steps_per_episode = 0
            observation, _ = env.reset()
        else:
            observation = next_observation

    return namedtuple("DoubleQLearningResult", ["q_table1", "q_table2"])(
        q_table1, q_table2
    )


@jax.jit
def _dql_update(
    key,
    q_table1,
    q_table2,
    observation,
    action,
    reward,
    next_observation,
    gamma,
    learning_rate,
    terminated,
):
    next_action = greedy_policy(q_table1, observation)
    val = q_table1[observation, action]
    next_val = (1 - terminated) * q_table2[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table1 = q_table1.at[observation, action].add(learning_rate * error)
    return q_table1
