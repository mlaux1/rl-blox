import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

from rl_blox.algorithm.double_q_learning import (
    _dql_update,
    train_double_q_learning,
)
from rl_blox.blox.value_policy import make_q_table


def test_double_q_learning(tabular_test_env):
    q_table1 = make_q_table(tabular_test_env)
    q_table2 = make_q_table(tabular_test_env)

    train_double_q_learning(
        tabular_test_env,
        q_table1,
        q_table2,
        total_timesteps=10,
        seed=42,
    )


def test_double_q_learning_selects_next_action_at_next_observation():
    q_table1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    q_table2 = jnp.array([[0.0, 0.0], [0.0, 10.0]])

    q_table1 = _dql_update(
        jax.random.key(0),
        q_table1,
        q_table2,
        observation=0,
        action=0,
        reward=0.0,
        next_observation=1,
        gamma=1.0,
        learning_rate=1.0,
        terminated=False,
    )

    assert q_table1[0, 0] == 10.0


def test_double_q_learning_same_seed_gives_same_result():
    def train():
        env = gym.make("FrozenLake-v1", is_slippery=True)
        result = train_double_q_learning(
            env,
            make_q_table(env),
            make_q_table(env),
            epsilon=0.5,
            total_timesteps=3000,
            seed=3,
            progress_bar=False,
        )
        return np.asarray(result.q_table1 + result.q_table2)

    q1, q2 = train(), train()
    assert np.any(q1 != 0)
    assert_array_equal(q1, q2)
