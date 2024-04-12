from typing import Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from modular_rl.policy.base_policy import BasePolicy


def generate_rollout(
    env: gym.Env, policy: BasePolicy
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    observation, _ = env.reset()
    terminated = False
    truncated = False

    obs = []
    actions = []
    rewards = []

    obs.append(observation)

    while not terminated or truncated:
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        obs.append(observation)
        actions.append(action)
        rewards.append(reward)

    return np.array(obs), np.array(actions), np.array(rewards)


def moving_average(array, rolling_length):
    return (
        np.convolve(array.flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
