import gymnasium as gym
import numpy as np
from numpy.testing import assert_array_equal

from rl_blox.algorithm.monte_carlo import train_monte_carlo
from rl_blox.blox.value_policy import make_q_table


def test_monte_carlo(tabular_test_env):
    q_table = make_q_table(tabular_test_env)
    _ = train_monte_carlo(tabular_test_env, q_table, total_timesteps=10)


def test_monte_carlo_same_seed_gives_same_result():
    def train():
        env = gym.make("FrozenLake-v1", is_slippery=True)
        result = train_monte_carlo(
            env,
            make_q_table(env),
            total_timesteps=3000,
            epsilon=0.5,
            seed=3,
            progress_bar=False,
        )
        return np.asarray(result[0])

    q1, q2 = train(), train()
    assert np.any(q1 != 0)
    assert_array_equal(q1, q2)
