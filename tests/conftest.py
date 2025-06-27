import gymnasium as gym
import pytest


@pytest.fixture(scope="session")
def tabular_test_env():
    env = gym.make("CliffWalking-v1")
    yield env
    env.close()


@pytest.fixture(scope="session")
def cart_pole_env():
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture(scope="session")
def inverted_pendulum_env():
    env = gym.make("InvertedPendulum-v5")
    yield env
    env.close()


@pytest.fixture(scope="session")
def pendulum_env():
    env = gym.make("Pendulum-v1")
    yield env
    env.close()
