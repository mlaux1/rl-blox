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


def linear_schedule_value(
    step: int,
    total_timesteps: int,
    start: float = 1.0,
    end: float = 0.1,
    fraction: float = 0.1,
) -> float:
    """Value of :func:`linear_schedule` at one time step.

    Computed on the host without creating the whole schedule.

    Parameters
    ----------
    step : int
        Time step.
    total_timesteps : int
        Number of time steps of the schedule.
    start : float, optional
        The starting value of the schedule.
    end : float, optional
        The final value of the schedule.
    fraction : float, optional
        The fraction of time steps in which the schedule should complete.

    Returns
    -------
    value : float
        The value of the schedule at the time step.
    """
    transition_steps = int(total_timesteps * fraction)
    if step >= transition_steps:
        return end
    if transition_steps == 1:
        return start
    return start + (end - start) * step / (transition_steps - 1)
