import gymnasium
import jax.numpy as jnp
from jax import Array, random, jit
from jax.typing import ArrayLike
from jax.random import PRNGKey

from tqdm import tqdm

from modular_rl.tools.error_functions import td_error


def _get_greedy_action(
        key: PRNGKey,
        q_table: ArrayLike,
        observation: ArrayLike
) -> Array:
    true_indices = q_table[observation] == q_table[observation].max()
    return random.choice(key, jnp.flatnonzero(true_indices))


def _get_epsilon_greedy_action(
        key: PRNGKey,
        q_table: ArrayLike,
        observation: ArrayLike,
        epsilon: float
) -> Array:
    key, subkey = random.split(key)
    roll = random.uniform(subkey)
    if roll < epsilon:
        return random.choice(key, jnp.arange(len(q_table[observation])))
    else:
        return _get_greedy_action(key, q_table, observation)


def sarsa(
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
        ep_reward = _sarsa_episode(subkey, env, q_table, alpha, epsilon, gamma)
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return ep_rewards


def _sarsa_episode(
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

    key, subkey = random.split(key)

    action = _get_epsilon_greedy_action(subkey, q_table, observation, epsilon)

    while not terminated and not truncated:
        # get action from policy and perform environment step
        next_observation, reward, terminated, truncated, _ = env.step(int(action))
        # get next action
        key, subkey = random.split(key)
        next_action = _get_epsilon_greedy_action(subkey, q_table, observation, epsilon)

        # update target policy
        q_table = _update_policy(q_table, observation, action, reward, next_observation, next_action, gamma, alpha)

        # housekeeping
        action = next_action
        observation = next_observation
        ep_reward += reward

    return ep_reward


@jit
def _update_policy(q_table, observation, action, reward, next_observation, next_action, gamma, alpha):
    val = q_table[observation, action]
    next_val = q_table[next_observation, next_action]
    error = td_error(reward, gamma, val, next_val)
    q_table = q_table.at[observation, action].add(alpha * error)

    return q_table

