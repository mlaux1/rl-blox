import gymnasium as gym

from rl_blox.algorithm.monte_carlo import MonteCarlo


def test_monte_carlo():
    env = gym.make("FrozenLake-v1")
    mc = MonteCarlo(env, 0.1, "first_visit")
    mc.train(1)
    env.close()
