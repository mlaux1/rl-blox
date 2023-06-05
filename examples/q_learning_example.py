import gymnasium as gym
import numpy as np

from modular_rl.algorithms.q_learning import QLearning

train_env = gym.make("FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

q_learning = QLearning(train_env, 0.1, 0.1)

q_learning.train(100)


train_env.close()

