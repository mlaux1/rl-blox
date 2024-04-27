import gymnasium
import jax.numpy as jnp
import numpy as onp
from jax import Array, random

from tqdm import tqdm

from modular_rl.tools.error_functions import td_error


def sarsa(
        env: gymnasium.Env,
        policy,
        alpha: float,
        key: int,
        num_episodes: int,
        gamma: float = 0.9999
) -> Array:
    ep_rewards = jnp.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        ep_reward = _sarsa_episode(env, policy, alpha, gamma, key)
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return ep_rewards


def _sarsa_episode(
        env,
        policy,
        key,
        alpha: float = 0.01,
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
    action = policy.get_action(observation)

    while not terminated and not truncated:
        # get action from policy and perform environment step
        next_observation, reward, terminated, truncated, _ = env.step(action)

        # get next action
        next_action = policy.get_action(next_observation)

        # update target policy
        val = policy.value_function.get_action_value(observation, action)
        next_val = policy.value_function.get_action_value(next_observation, next_action)
        error = td_error(reward, gamma, val, next_val)
        policy.value_function.update(observation, action, alpha * error)

        # housekeeping
        action = next_action
        observation = next_observation
        ep_reward += reward

    return ep_reward

