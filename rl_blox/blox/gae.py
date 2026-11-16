from collections import namedtuple

import jax
import jax.numpy as jnp


@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    next_values: jnp.ndarray,
    terminations: jnp.ndarray,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    dones: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : array, shape (n_steps,)
        Rewards per step.

    values : array, shape (n_steps,)
        Predicted values of the observations.

    next_values : array, shape (n_steps,)
        Predicted values of the next observations.

    terminations : array, shape (n_steps,)
        Flags indicating episode termination per step. The value of the next
        observation is not used after a termination.

    gamma : float, optional
        Discount factor for rewards.

    lmbda : float, optional
        Smoothing factor for bias-variance trade-off.

    dones : array, shape (n_steps,), optional
        Flags indicating the last step of an episode, e.g., because it
        terminated or was truncated. Advantages are not accumulated across
        these steps. Defaults to ``terminations``.

    Returns
    -------
    advantages : array, shape (n_steps,)
        Advantage estimates per step.

    returns : array, shape (n_steps,)
        Computed returns per step.
    """
    if dones is None:
        dones = terminations

    def calc_advantage_per_step(next_advantage, inputs):
        reward, value, next_value, termination, done = inputs
        delta = reward + gamma * next_value * (1 - termination) - value
        advantage = delta + gamma * lmbda * (1 - done) * next_advantage
        # Carried to the previous step and stored as this step's output
        return advantage, advantage

    _, advantages = jax.lax.scan(
        calc_advantage_per_step,
        0.0,
        (
            rewards[::-1],
            values[::-1],
            next_values[::-1],
            terminations[::-1],
            dones[::-1],
        ),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return namedtuple("GAE", ["advantages", "returns"])(advantages, returns)
