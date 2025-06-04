import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from ..blox.function_approximator.mlp import MLP


@nnx.jit
def greedy_policy(
    q_net: MLP,
    obs: ArrayLike,
) -> int:
    """Greedy policy.

    Selects the greedy action for a given observation based on the given
    Q-Network by choosing the action that maximises the Q-Value.

    Parameters
    ----------
    q_net : MLP
        The Q-Network to be used for greedy action selection.
    obs : ArrayLike
        The observation for which to select an action.

    Returns
    -------
    action : int
        The selected greedy action.

    """
    q_vals = q_net([obs])
    return jnp.argmax(q_vals)
