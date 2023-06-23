import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from modular_rl.algorithms.model_free.q_learning import QLearning


train_env = gym.make("FrozenLake-v1", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

q_learning = QLearning(train_env, 0.1, 0.01)

rewards = q_learning.train(1000000)

train_env.close()

print(rewards)

sns.lineplot(x=range(len(rewards)), y=rewards)
plt.show()


