import jax.numpy as jnp


def discounted_n_step_return(
    reward: jnp.ndarray, terminated: jnp.ndarray, gamma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute the n-step return for a batch of subtrajectories.

    Estimates

    .. math::

        R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}
        + \ldots + \gamma^{n-1} r_{t+n-1}

    for a batch of rewards and termination flags with horizon :math:`n`.
    However, the return is truncated at the first termination flag in each
    sequence.

    Parameters
    ----------
    reward : jnp.ndarray, shape (batch_size, horizon)
        Rewards of subtrajectories with length ``horizon``.

    terminated : jnp.ndarray, shape (batch_size, horizon)
        Termination flags of subtrajectories with length ``horizon``.

    gamma : float
        Discount factor.

    Returns
    -------
    n_step_return : jnp.ndarray, shape (batch_size,)
        n-step truncated return.

    discount : jnp.ndarray, shape (batch_size,)
        Discount factor for the remaining steps (step n + 1) per sample.
        This is zero when the episode terminated. This can be used to
        combine the n-step return with a value function estimate for the
        remaining steps.
    """
    n_step_return = jnp.zeros(reward.shape[0], dtype=jnp.float32)
    discount = jnp.ones(reward.shape[0], dtype=jnp.float32)
    for t in range(reward.shape[1]):
        n_step_return += discount * reward[:, t]
        discount *= gamma * (1 - terminated[:, t])
    return n_step_return, discount
