import gymnasium
import jax.numpy as jnp
import jax.random
from jax import Array, jit, random
from jax.random import PRNGKey
from jax.typing import ArrayLike
from tqdm import tqdm

from ..blox.value_policy import get_epsilon_greedy_action, get_greedy_action
from ..util.error_functions import td_error


def double_q_learning(
    key: PRNGKey,
    env: gymnasium.Env,
    q_table1: ArrayLike,
    q_table2: ArrayLike,
    alpha: float,
    epsilon: float,
    num_episodes: int,
    gamma: float = 0.9999,
) -> Array:
    ep_rewards = jnp.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        key, subkey = random.split(key)
        q_table1, q_table2, ep_reward = _dql_episode(
            subkey, env, q_table1, q_table2, alpha, epsilon, gamma
        )
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return q_table1, q_table2, ep_rewards


def _dql_episode(
    key: PRNGKey,
    env: gymnasium.Env,
    q_table1: ArrayLike,
    q_table2: ArrayLike,
    alpha: float,
    epsilon: float,
    gamma: float = 0.9999,
) -> float:
    """Perform a single episode rollout."""
    ep_reward = 0
    truncated = False
    terminated = False
    observation, _ = env.reset()

    while not terminated and not truncated:
        key, subkey1, subkey2, subkey3 = random.split(key, 4)

        # sum the q_tables and select action
        q_table = q_table1 + q_table2
        action = get_epsilon_greedy_action(
            subkey1, q_table, observation, epsilon
        )
        # perform environment step
        next_observation, reward, terminated, truncated, _ = env.step(
            int(action)
        )

        val = jax.random.uniform(subkey3)
        if val < 0.5:
            q_table1 = _dql_update(
                key,
                q_table1,
                q_table2,
                observation,
                action,
                reward,
                next_observation,
                gamma,
                alpha,
            )
        else:
            q_table2 = _dql_update(
                key,
                q_table2,
                q_table1,
                observation,
                action,
                reward,
                next_observation,
                gamma,
                alpha,
            )

        # housekeeping
        observation = next_observation
        ep_reward += reward

    return q_table1, q_table2, ep_reward


@jit
def _dql_update(
    key,
    q_table1,
    q_table2,
    observation,
    action,
    reward,
    next_observation,
    gamma,
    alpha,
):
    next_action = get_greedy_action(key, q_table1, observation)
    val = q_table1[observation, action]
    next_val = q_table2[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table1 = q_table1.at[observation, action].add(alpha * error)
    return q_table1
