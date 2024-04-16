import jax.numpy as jnp
import gymnasium
from jax import Array
from tqdm import tqdm

from modular_rl.tools.error_functions import td_error


def q_learning(
        env: gymnasium.Env,
        policy,
        alpha: float,
        key: int,
        num_episodes: int,
        gamma: float = 0.9999
) -> Array:
    ep_rewards = jnp.zeros(num_episodes)

    for i in tqdm(range(num_episodes)):
        ep_reward = _q_learning_episode(env, policy, key, alpha, gamma)
        ep_rewards = ep_rewards.at[i].add(ep_reward)

    return ep_rewards


def _q_learning_episode(
        env,
        policy,
        key,
        alpha: float = 0.01,
        gamma: float = 0.9999
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
        next_observation, reward, terminated, truncated, info = (
            env.step(action))

        # get next action
        next_action = policy.get_greedy_action(next_observation)

        # update target policy
        val = policy.value_function.get_action_value(
            observation, action)
        next_val = policy.value_function.get_action_value(
            next_observation, next_action)

        error = td_error(reward, gamma, val, next_val)

        policy.value_function.update(observation, action, alpha * error)

        action = policy.get_action(next_observation)
        observation = next_observation
        ep_reward += reward

    return ep_reward





