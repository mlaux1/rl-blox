import gymnasium as gym
import pytest


@pytest.fixture(scope="session")
def tabular_test_env():
    return gym.make("CliffWalking-v1")
