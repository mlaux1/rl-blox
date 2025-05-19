import gymnasium as gym

from rl_blox.algorithm.monte_carlo import train_monte_carlo
from rl_blox.blox.value_policy import make_q_table


def test_monte_carlo():
    env = gym.make("CliffWalking-v0")
    q_table = make_q_table(env)
    _ = train_monte_carlo(env, q_table, total_timesteps=10)
    env.close()
