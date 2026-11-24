import jax
import jax.numpy as jnp
import numpy as np

from rl_blox.blox.value_policy import epsilon_greedy_policy, greedy_policy


def test_greedy_policy_without_key_takes_first_maximum():
    q_table = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    assert greedy_policy(q_table, 0) == 1


def test_greedy_policy_breaks_ties_at_random():
    q_table = jnp.array([[0.0, 1.0, 1.0, 0.0, 1.0]])
    keys = jax.random.split(jax.random.key(0), 600)
    actions = np.array([int(greedy_policy(q_table, 0, key)) for key in keys])
    counts = np.bincount(actions, minlength=5)
    assert counts[0] == counts[3] == 0
    assert np.all(counts[[1, 2, 4]] > 150)


def test_epsilon_greedy_policy_breaks_ties_at_random():
    q_table = jnp.zeros((1, 4))
    keys = jax.random.split(jax.random.key(0), 800)
    actions = np.array(
        [int(epsilon_greedy_policy(q_table, 0, 0.0, key)) for key in keys]
    )
    assert np.all(np.bincount(actions, minlength=4) > 150)


def test_epsilon_greedy_policy_explores_with_probability_epsilon():
    q_table = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    keys = jax.random.split(jax.random.key(0), 2000)
    actions = np.array(
        [int(epsilon_greedy_policy(q_table, 0, 0.2, key)) for key in keys]
    )
    counts = np.bincount(actions, minlength=4)
    # greedy action: 0.8 + 0.2 / 4 = 0.85, each other action: 0.05
    np.testing.assert_allclose(
        counts / len(keys), [0.85, 0.05, 0.05, 0.05], atol=0.02
    )
