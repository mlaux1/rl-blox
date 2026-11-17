import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from rl_blox.blox.gae import compute_gae

GAMMA = 0.9
LMBDA = 0.5


def test_compute_gae_without_lambda_returns_td_errors():
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([0.5, 1.0, 1.5])
    next_values = jnp.array([1.0, 1.5, 2.0])
    terminations = jnp.array([0.0, 0.0, 1.0])

    advantages, returns = compute_gae(
        rewards, values, next_values, terminations, GAMMA, 0.0
    )

    td_errors = [1.0 + GAMMA * 1.0 - 0.5, 2.0 + GAMMA * 1.5 - 1.0, 3.0 - 1.5]
    assert_allclose(advantages, td_errors, rtol=1e-6)
    assert_allclose(returns, advantages + values)


def test_compute_gae_stops_at_dones():
    # The first episode is truncated after two steps
    rewards = jnp.array([1.0, 1.0, 10.0, 10.0])
    values = jnp.zeros(4)
    next_values = jnp.array([0.0, 2.0, 0.0, 0.0])
    terminations = jnp.zeros(4)
    dones = jnp.array([0.0, 1.0, 0.0, 1.0])

    advantages, _ = compute_gae(
        rewards, values, next_values, terminations, GAMMA, LMBDA, dones=dones
    )

    # The truncated step bootstraps, but nothing of the second episode leaks
    td_error = 1.0 + GAMMA * 2.0
    expected = [1.0 + GAMMA * LMBDA * td_error, td_error]
    assert_allclose(advantages[:2], expected, rtol=1e-6)
    assert_allclose(advantages[2:], [10.0 + GAMMA * LMBDA * 10.0, 10.0])


def test_compute_gae_stops_at_terminations_without_dones():
    rewards = jnp.array([1.0, 1.0, 10.0])
    values = jnp.zeros(3)
    next_values = jnp.ones(3)
    terminations = jnp.array([0.0, 1.0, 0.0])

    advantages, _ = compute_gae(
        rewards, values, next_values, terminations, GAMMA, LMBDA
    )

    expected, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminations,
        GAMMA,
        LMBDA,
        dones=terminations,
    )
    assert_allclose(advantages, expected)
    assert advantages[1] == pytest.approx(1.0)
