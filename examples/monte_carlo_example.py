import gymnasium as gym

from modular_rl.algorithms.monte_carlo import MonteCarlo

train_env = gym.make("FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"])

qmc = MonteCarlo(train_env, 0.1)
qmc.train(100)

train_env.close()

