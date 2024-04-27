import gymnasium
import jax.numpy as jnp
import numpy as onp
from jax import Array, random, jit
from jax.random import PRNGKey

from tqdm import tqdm

from modular_rl.tools.error_functions import td_error


@jit
def _get_greedy_action(q_table, observation):
    return q_table[observation].argmax()


@jit
def _get_greedy_action2(key, q_table, observation):
    true_indices = q_table == q_table[observation].max()
    return random.choice(key, true_indices)


def _get_epsilon_greedy_action(key, q_table, observation, epsilon):
    roll = random.uniform(key)
    if roll < epsilon:
        return random.choice(key, jnp.arange(len(q_table[observation])))
    else:
        return _get_greedy_action(q_table, observation)


def _get_epsilon_greedy_action2(key, q_table, observation, epsilon):
    roll = random.uniform(key)
    if roll < epsilon:
        return random.choice(key, jnp.arange(len(q_table[observation])))
    else:
        return _get_greedy_action2(key, q_table, observation)


def sarsa(
        key: PRNGKey,
        env: gymnasium.Env,
        q_table,
        alpha: float,
        epsilon,
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
        env,
        q_table,
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
        # print(f"{action=}")
        next_observation, reward, terminated, truncated, _ = env.step(int(action))
        # get next action
        key, subkey = random.split(key)
        next_action = _get_epsilon_greedy_action(subkey, q_table, observation, epsilon)

        # update target policy
        val = q_table[observation, action]
        next_val = q_table[next_observation, next_action]
        error = td_error(reward, gamma, val, next_val)
        q_table = q_table.at[observation, action].add(alpha*error)
        #policy.value_function.update(observation, action, alpha * error)

        # housekeeping
        action = next_action
        observation = next_observation
        ep_reward += reward

    return ep_reward

