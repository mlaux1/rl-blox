import gymnasium as gym
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from modular_rl.algorithms.model_free.q_learning import QLearning


def generate_rollout(env, policy):
    observation, _ = env.reset()
    terminated = False
    truncated = False

    obs = []
    acts = []
    rews = []

    obs.append(observation)

    while not terminated and not truncated:
        action = policy.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        obs.append(observation)
        acts.append(action)
        rews.append(reward)

    return np.array(obs), np.array(acts), np.array(rews)


# logging.basicConfig(level=logging.DEBUG)

# train policy
train_env = gym.make("FrozenLake-v1", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

q_learning = QLearning(train_env, 0.1, 0.05)
train_returns = q_learning.train(1000)

train_env.close()

# evaluate the policy
test_env = gym.make(
    "FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"]
)

generate_rollout(test_env, q_learning.target_policy)

test_env.close()

sns.scatterplot(x=range(len(train_returns)), y=train_returns)
plt.show()
