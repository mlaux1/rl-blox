from typing import Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from modular_rl.policy.base_policy import BasePolicy


def generate_rollout(
    env: gym.Env, policy
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    observation, _ = env.reset()
    terminated = False
    truncated = False

    obs = []
    actions = []
    rewards = []

    obs.append(observation)

    while not terminated or truncated:
        action = policy(observation=observation)
        observation, reward, terminated, truncated, info = env.step(int(action)) # TODO: adapt to non-int actions

        obs.append(observation)
        actions.append(action)
        rewards.append(reward)

    return np.array(obs), np.array(actions), np.array(rewards)
