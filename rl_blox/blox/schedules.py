import jax.numpy as jnp


def linear_schedule(
    total_timesteps: int,
    start: float = 1.0,
    end: float = 0.1,
    fraction: float = 0.1,
) -> jnp.ndarray:
    """Creates a linear schedule of values for given amount of time steps.

    Parameters
    ----------

    total_timesteps : int
        Number of time steps to create the schedule for.
    start : float, optional
        The starting value of the schedule.
    end : float, optional
        The final value of the schedule.
    fraction : float
        The fraction of time steps in which the schedule should complete.

    Returns
    -------

    schedule : jnp.ndarray
        The schedule of values for all time steps in JAX array form.

    """
    transition_steps = int(
        total_timesteps * fraction
    )  # Number of steps for decay
    schedule = jnp.ones(total_timesteps) * end  # Default value after decay

    schedule = schedule.at[:transition_steps].set(
        jnp.linspace(start, end, transition_steps)
    )

    return schedule
