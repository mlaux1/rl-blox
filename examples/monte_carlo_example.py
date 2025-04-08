import gymnasium as gym
from rl_blox.algorithms.model_free.monte_carlo import MonteCarlo

train_env = gym.make(
    "FrozenLake-v1", render_mode="human", desc=["SFFH", "FFFF", "FFFF", "FFFG"]
)

qmc = MonteCarlo(train_env, 0.1, "first_visit")
qmc.train(10)

print(qmc.target_policy.value_function.values)

train_env.close()
