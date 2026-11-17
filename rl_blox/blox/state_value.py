import jax.numpy as jnp
from flax import nnx


@nnx.jit
def greedy_value(q_net: nnx.Module, observation: jnp.ndarray) -> jnp.ndarray:
    r"""State value of the greedy policy of a Q-network.

    .. math::

        V(s) = \max_a Q(s, a)

    Parameters
    ----------
    q_net : nnx.Module
        Q-network that predicts the value of each action from a discrete
        action space.

    observation : array, shape (n_samples, n_observation_features)
        Observations.

    Returns
    -------
    value : array, shape (n_samples,)
        State values.
    """
    return jnp.max(q_net(observation), axis=-1)


@nnx.jit
def policy_value(
    q: nnx.Module, policy: nnx.Module, observation: jnp.ndarray
) -> jnp.ndarray:
    r"""State value of a policy according to a Q-function.

    .. math::

        V(s) = Q(s, \pi(s))

    Parameters
    ----------
    q : nnx.Module
        Q-function that takes the concatenation of observation and action as
        input. Clipped double Q-networks predict the minimum of both networks.

    policy : nnx.Module
        Policy. If it returns a tuple, e.g., the mean and standard deviation
        of a Gaussian policy, the first element is the action.

    observation : array, shape (n_samples, n_observation_features)
        Observations.

    Returns
    -------
    value : array, shape (n_samples,)
        State values.
    """
    action = policy(observation)
    if isinstance(action, tuple):
        action = action[0]
    return q(jnp.concatenate((observation, action), axis=-1)).squeeze(-1)
