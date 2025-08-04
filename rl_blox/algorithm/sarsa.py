import gymnasium
import jax
from jax.typing import ArrayLike
from tqdm.rich import trange

from ..blox.value_policy import epsilon_greedy_policy
from ..logging.logger import LoggerBase
from ..util.error_functions import td_error


def train_sarsa(
    env: gymnasium.Env,
    q_table: ArrayLike,
    learning_rate: float = 0.1,
    epsilon: float = 0.05,
    gamma: float = 0.99,
    total_timesteps: int = 100_000,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> ArrayLike:
    r"""
    State-action-reward-state-action algorithm.

    This function implements the SARSA (State-Action-Reward-State-Action) update rule
    in the context of tabular reinforcement learning using JAX. The update is
    on-policy and uses the next action chosen by the current policy.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to train on.
    q_table : ArrayLike
        The Q-table of shape (num_states, num_actions), containing current Q-values.
    learning_rate : float
        The learning rate, determining how much new information overrides old.
    epsilon : float
        The tradeoff for random exploration.
    gamma : float
        The discount factor, representing the importance of future rewards.
    total_timesteps : int
        The number of total timesteps to train for.
    seed : int
        The random seed.
    logger : LoggerBase, optional
        Experiment logger.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.


    Returns
    -------
    q_table : jax.numpy.ndarray
        The updated Q-table after applying the SARSA update rule.

    References
    ----------
    1. Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*.
       Technical Report CUED/F-INFENG/TR 166, Department of Engineering, University of Cambridge.
       URL: [https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf](https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf)

    2. Sutton, R. S., & Barto, A. G. (1998). *Reinforcement Learning: An Introduction* (2nd ed.).
       MIT Press. ISBN 978-0262039246.

    3. Singh, S. P., & Sutton, R. S. (1996). *Reinforcement learning with replacing eligibility traces*.
       Machine Learning, 22(1-3), 123–158.
       DOI: [10.1007/BF00114726](https://link.springer.com/article/10.1007/BF00114726)
    """

    key = jax.random.key(seed)

    observation, _ = env.reset()

    if logger is not None:
        logger.start_new_episode()

    steps_per_episode = 0

    for i in trange(total_timesteps, disable=not progress_bar):
        # get action from policy and perform environment step
        key, subkey = jax.random.split(key)
        action = epsilon_greedy_policy(q_table, observation, epsilon, subkey)
        steps_per_episode += 1
        next_observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        # get next action
        key, subkey = jax.random.split(key)
        next_action = epsilon_greedy_policy(
            q_table, next_observation, epsilon, subkey
        )

        q_table = _update_policy(
            q_table,
            observation,
            action,
            reward,
            next_observation,
            next_action,
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
    learning_rate,
    terminated,
):
    val = q_table[observation, action]
    next_val = (1 - terminated) * q_table[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table = q_table.at[observation, action].add(learning_rate * error)

    return q_table
