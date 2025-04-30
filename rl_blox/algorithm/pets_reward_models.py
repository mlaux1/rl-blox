import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

PENDULUM_MAX_TORQUE: float = 2.0


@jax.jit
def pendulum_reward(act: ArrayLike, obs: ArrayLike) -> jnp.ndarray:
    """Vectorized reward model for environment 'Pendulum-v1'.

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py

    Parameters
    ----------
    act : array, shape (..., 3)
        Action.

    obs : array, shape (..., 1)
        Observation.

    Returns
    -------
    reward : array, shape (...)
        Reward associated with taking act in obs.
    """
    act = jnp.asarray(act)  # (..., 1): torque
    obs = jnp.asarray(obs)  # (..., 3): cos(theta), sin(theta), theta_dot

    chex.assert_axis_dimension(act, axis=-1, expected=1)
    chex.assert_axis_dimension(obs, axis=-1, expected=3)

    theta = jnp.arccos(jnp.clip(obs[..., 0], -1.0, 1.0))
    theta_dot = obs[..., 2]
    act = jnp.clip(act, -PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE)[..., 0]

    costs = norm_angle(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * (act**2)
    return -costs


def norm_angle(angle: jnp.ndarray) -> jnp.ndarray:
    return ((angle + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
