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
    key: Array | None = None,
) -> jnp.ndarray:
    """Greedy policy for tabular Q-functions.

    Returns the greedy action for the given observation and tabular Q-function.

    Parameters
    ----------
    q_table : ArrayLike
        The tabular Q-function.
    observation : ArrayLike
        The observation.
    key : array, optional
        Random key to break ties between actions with the maximum value
        uniformly at random. Without a key, ties go to the first action.

    Returns
    -------
    action : jnp.ndarray
        The greedy action.
    """
    q = q_table[observation]
    if key is None:
        return jnp.argmax(q)
    return random.categorical(key, jnp.where(q == q.max(), 0.0, -jnp.inf))


def epsilon_greedy_policy(
    q_table: ArrayLike,
    observation: ArrayLike,
    epsilon: float,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Epsilon-greedy policy for tabular Q-functions.

    Returns a uniformly random action with probability epsilon and the greedy
    action otherwise. Ties between greedy actions are broken uniformly at
    random.

    Parameters
    ----------
    q_table : ArrayLike
        The tabular Q-function.
    observation : ArrayLike
        The observation.
    epsilon : float
        The probability of selecting a random action uniformly.
    key : array
        The random key.

    Returns
    -------
    action : jnp.ndarray
        The selected action.
    """
    explore_key, action_key = random.split(key)
    if random.uniform(explore_key) < epsilon:
        n_actions = q_table[observation].shape[-1]
        return random.randint(action_key, (), 0, n_actions)
    return greedy_policy(q_table, observation, action_key)
