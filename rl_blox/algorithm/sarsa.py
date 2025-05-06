import gymnasium
from jax import jit, random
from jax.random import PRNGKey
from jax.typing import ArrayLike
from tqdm import tqdm

from ..blox.value_policy import get_epsilon_greedy_action
from ..util.error_functions import td_error


def train_sarsa(
    key: PRNGKey,
    env: gymnasium.Env,
    q_table: ArrayLike,
    alpha: float,
    epsilon: float,
    gamma: float = 0.9999,
    total_timesteps: int = 10_000,
) -> ArrayLike:
    r"""
    State-action-reward-state-action algorithm.

    This function implements the SARSA (State-Action-Reward-State-Action) update rule
    in the context of tabular reinforcement learning using JAX. The update is
    on-policy and uses the next action chosen by the current policy.

    Parameters
    ----------
    key : PRNGKey
        The random key.
    env : gymnasium.Env
        The environment to train on.
    q_table : ArrayLike
        The Q-table of shape (num_states, num_actions), containing current Q-values.
    alpha : float
        The learning rate, determining how much new information overrides old.
    epsilon : float
        The tradeoff for random exploration.
    gamma : float
        The discount factor, representing the importance of future rewards.
    total_timesteps : int
        The number of total timesteps to train for.

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

    observation, _ = env.reset()

    for i in tqdm(range(total_timesteps)):
        # get action from policy and perform environment step
        key, subkey = random.split(key)
        action = get_epsilon_greedy_action(
            subkey, q_table, observation, epsilon
        )
        next_observation, reward, terminated, truncated, _ = env.step(
            int(action)
        )

        # get next action
        key, subkey = random.split(key)
        next_action = get_epsilon_greedy_action(
            subkey, q_table, observation, epsilon
        )

        q_table = _update_policy(
            q_table,
            observation,
            action,
            reward,
            next_observation,
            next_action,
            gamma,
            alpha,
        )

        if terminated or truncated:
            observation, _ = env.reset()

    return q_table


@jit
def _update_policy(
    q_table,
    observation,
    action,
    reward,
    next_observation,
    next_action,
    gamma,
    alpha,
):
    val = q_table[observation, action]
    next_val = q_table[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table = q_table.at[observation, action].add(alpha * error)

    return q_table
