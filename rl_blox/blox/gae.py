import jax
import jax.numpy as jnp
from collections import namedtuple

@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float=0.99,
    lmbda: float=0.95
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : jnp.ndarray
        Array of rewards per step.
    values : jnp.ndarray
        Array of predicted values per step.
    dones : jnp.ndarray
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
    def calc_advantage_per_step(carry, inputs):
        gae, next_value = carry
        reward, value, done = inputs
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lmbda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        calc_advantage_per_step,
        (0.0, 0.0),
        (rewards[::-1], values[::-1], dones[::-1])
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return namedtuple('GAE', ['advantages', 'returns'])(advantages, returns)