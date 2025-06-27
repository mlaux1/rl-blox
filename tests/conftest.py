import gymnasium as gym
import pytest


@pytest.fixture(scope="session")
def tabular_test_env():
    env = gym.make("CliffWalking-v1")
    yield env
    env.close()
