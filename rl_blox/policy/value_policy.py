import gymnasium
import jax.numpy as jnp
from gymnasium.spaces.utils import flatdim
from jax import Array, jit, random
from jax.random import PRNGKey
from jax.typing import ArrayLike

from ..tools import gymtools


def make_q_table(env: gymnasium.Env) -> Array:
    """
    Creates a Q-table for the given environment.

    :param env: Environment.
    :return: Q-table.
    """
    obs_shape = gymtools.space_shape(env.observation_space)
    act_shape = (flatdim(env.action_space),)
    q_table = jnp.zeros(
        shape=obs_shape + act_shape,
        dtype=jnp.float32,
    )
    return q_table


@jit
def get_greedy_action(
    key: PRNGKey, q_table: ArrayLike, observation: ArrayLike
) -> Array:
    """
    Returns the greedy action for the given observation.

    :param key: PRNGKey.
    :param q_table: Q-table.
    :param observation: Observation.
    :return: Greedy action.
    """
    # TODO: technically correct way is commented out because it is super slow
    # true_indices = q_table[observation] == q_table[observation].max()
    # return random.choice(key, jnp.flatnonzero(true_indices))
    return jnp.argmax(q_table[observation])


def get_epsilon_greedy_action(
    key: PRNGKey, q_table: ArrayLike, observation: ArrayLike, epsilon: float
) -> Array:
    """
    Returns an epsilon-greedy action for the given observation.

    :param key: PRNGKey.
    :param q_table: Q-table.
    :param observation: Observation.
    :param epsilon: Probability of randomly sampling an action
    :return: The sampled action.
    """
    key, subkey = random.split(key)
    roll = random.uniform(subkey)
    if roll < epsilon:
        return random.choice(key, jnp.arange(len(q_table[observation])))
    else:
        return get_greedy_action(key, q_table, observation)
