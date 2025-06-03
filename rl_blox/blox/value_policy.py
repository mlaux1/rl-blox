import gymnasium
import jax.numpy as jnp
from gymnasium.spaces.utils import flatdim
from jax import Array, jit, random
from jax.typing import ArrayLike

from ..util import gymtools


def make_q_table(env: gymnasium.Env) -> Array:
    """Creates a Q-table for the given environment."""

    obs_shape = gymtools.space_shape(env.observation_space)
    act_shape = (flatdim(env.action_space),)
    q_table = jnp.zeros(
        shape=obs_shape + act_shape,
        dtype=jnp.float32,
    )
    return q_table


@jit
def greedy_policy(
    q_table: ArrayLike,
    observation: ArrayLike,
) -> jnp.ndarray:
    """Greedy policy for tabular Q-functions.

    Returns the greedy action for the given observation and tabular Q-function.

    Parameters
    ----------
    q_table : ArrayLike
        The tabular Q-function.
    observation : ArrayLike
        The observation.

    Returns
    -------
    action : jnp.ndarray
        The greedy action.
    """
    return jnp.argmax(q_table[observation])


def epsilon_greedy_policy(
    q_table: ArrayLike,
    observation: ArrayLike,
    epsilon: float,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Epsilon-greedy policy for tabular Q-functions.

    Returns the greedy action for the given observation and tabular Q-function.

    Parameters
    ----------
    q_table : ArrayLike
        The tabular Q-function.
    observation : ArrayLike
        The observation.
    epsilon : float
        The probability of selecting a random action uniformly.
    key : int
        The random key.

    Returns
    -------
    action : jnp.ndarray
        The greedy action.
    """
    key, subkey = random.split(key)
    roll = random.uniform(subkey)
    if roll < epsilon:
        return random.choice(subkey, jnp.arange(len(q_table[observation])))
    else:
        return greedy_policy(q_table, observation)
