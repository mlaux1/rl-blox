import gymnasium as gym
import jax.numpy as jnp
from numpy.testing import assert_allclose

from rl_blox.algorithm.dynaq import q_learning_update, train_dynaq
from rl_blox.blox.value_policy import make_q_table


def test_dynaq(tabular_test_env):
    q_table = make_q_table(tabular_test_env)

    _ = train_dynaq(
        tabular_test_env,
        q_table,
        learning_rate=0.05,
        epsilon=0.05,
        total_timesteps=10,
    )

    tabular_test_env.close()


class OneStepEnv(gym.Env):
    """Each episode ends after one step from state 0 in state 1."""

    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 1, 1.0, True, False, {}


def test_q_learning_update_does_not_bootstrap_from_terminal_states():
    q_table = jnp.ones((2, 1))
    q_table = q_learning_update(0, 0, 1.0, 1, 1.0, 0.9, 0.5, q_table)
    assert_allclose(q_table[0, 0], 1.0)

    q_table = q_learning_update(0, 0, 1.0, 1, 0.0, 0.9, 0.5, q_table)
    assert_allclose(q_table[0, 0], 1.45)


def test_dynaq_does_not_bootstrap_from_terminal_states():
    env = OneStepEnv()

    q_table = train_dynaq(
        env,
        jnp.ones((2, 1)),
        gamma=0.9,
        learning_rate=0.5,
        epsilon=0.0,
        total_timesteps=5,
        progress_bar=False,
    )

    assert_allclose(q_table[0, 0], 1.0, rtol=1e-5)
