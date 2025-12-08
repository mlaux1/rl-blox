from collections import namedtuple

import jax
import jax.numpy as jnp


@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    next_values: jnp.ndarray,
    terminateds: jnp.ndarray,
    gamma: float = 0.99,
    lmbda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : jnp.ndarray
        Array of rewards per step.
    values : jnp.ndarray
        Array of predicted values per step.
    values : jnp.ndarray
        Array of predicted next_values per step.
    terminateds : jnp.ndarray
        Flags indicating episode termination per step.
    gamma : float, optional
        Discount factor for rewards.
    lmbda : float, optional
        Smoothing factor for bias-variance trade-off.

    Returns
    -------
    - advantages : jnp.ndarray
        Advantage estimates per step.
    - returns : jnp.ndarray
        Computed returns per step.
    """

    all_next_values = jnp.concatenate(
        [values[1:], jnp.expand_dims(next_values, axis=0)], axis=0
    )

    def calc_advantage_per_step(carry, inputs):
        gae = carry
        reward, value, next_value, terminated = inputs
        delta = reward + gamma * next_value * (1 - terminated) - value
        gae = delta + gamma * lmbda * (1 - terminated) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        calc_advantage_per_step,
        0.0,
        (rewards[::-1], values[::-1], all_next_values[::-1], terminateds[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return namedtuple("GAE", ["advantages", "returns"])(advantages, returns)
