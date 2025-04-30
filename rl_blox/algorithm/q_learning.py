import gymnasium
import jax.numpy as jnp
from jax import Array, jit, random
from jax.random import PRNGKey
from jax.typing import ArrayLike
from tqdm import tqdm

from ..blox.value_policy import get_epsilon_greedy_action, get_greedy_action
from ..util.error_functions import td_error


def q_learning(
    key: PRNGKey,
    env: gymnasium.Env,
    q_table: ArrayLike,
    alpha: float,
    epsilon: float,
    num_episodes: int,
    gamma: float = 0.9999,
) -> Array:
    ep_rewards = jnp.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        key, subkey = random.split(key)
        q_table, ep_reward = _q_learning_episode(
            subkey, env, q_table, alpha, epsilon, gamma
        )
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return q_table, ep_rewards


def _q_learning_episode(
    key: PRNGKey,
    env: gymnasium.Env,
    q_table: ArrayLike,
    alpha: float,
    epsilon: float,
    gamma: float = 0.9999,
) -> float:
    """
    Performs a single episode rollout.

    :param gamma: Discount factor.
    :return: Episode reward.
    """
    ep_reward = 0
    truncated = False
    terminated = False
    observation, _ = env.reset()

    while not terminated and not truncated:
        key, subkey1, subkey2 = random.split(key, 3)

        action = get_epsilon_greedy_action(
            subkey1, q_table, observation, epsilon
        )
        # get action from policy and perform environment step
        next_observation, reward, terminated, truncated, _ = env.step(
            int(action)
        )
        # get next action
        next_action = get_greedy_action(subkey2, q_table, observation)

        # update target policy
        q_table = _q_learning_update(
            q_table,
            observation,
            action,
            reward,
            next_observation,
            next_action,
            gamma,
            alpha,
        )

        # housekeeping
        observation = next_observation
        ep_reward += reward

    return q_table, ep_reward


@jit
def _q_learning_update(
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
