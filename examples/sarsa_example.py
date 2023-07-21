import gymnasium as gym
import numpy as np

from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.policy.base_policy import EpsilonGreedyPolicy

train_env = gym.make("FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"])


policy = EpsilonGreedyPolicy(train_env.observation_space, train_env.action_space, epsilon=0.01)
q_learning = Sarsa(train_env, policy, alpha=0.1)

q_learning.train(100)


train_env.close()

