import gymnasium as gym
import numpy as np

from modular_rl.algorithms.model_free.sarsa import Sarsa

train_env = gym.make("FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

q_learning = Sarsa(train_env, 0.1, 0.1)

q_learning.train(100)


train_env.close()

