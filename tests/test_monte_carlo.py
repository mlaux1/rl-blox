import gymnasium as gym
import pytest


@pytest.mark.skip
def test_monte_carlo():
    env = gym.make("FrozenLake-v1")
    mc = MonteCarlo(env, 0.1, "first_visit")
    mc.train(1)
    env.close()
