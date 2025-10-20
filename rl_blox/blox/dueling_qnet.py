from collections.abc import Callable

import chex
import jax.numpy as jnp
from flax import nnx


class DuelingQNet(nnx.Module):
    """Dueling Q network.

    Instead of directly estimating Q-values for each action, the network splits
    into two streams after a shared feature extractor, one estimating the state
    value and the other the action advantages. The two streams are combined to
    aggregates Q-values as Q = V + (A − mean(A)).

    Parameters
    ----------
    n_features : int
        Number of features.

    n_outputs : int
        Number of output components.

    shared_nodes : list
        Numbers of hidden nodes of the shared layers.
    
    advantage_nodes : list
        Numbers of hidden nodes of the advantage layers.
    
    state_value_nodes : list
        Numbers of hidden nodes of the state value layers.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.
    
    References
    ----------
    .. [1] Wang, Z., Schaul, T., Hessel, M., Hado, V. H., Lanctot, M., & Nando,
       D. F. (2015, November 20). Dueling network architectures for deep
       reinforcement learning. arXiv.org. https://arxiv.org/abs/1511.06581
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    shared_nodes: list[nnx.Linear]
    """Shared hidden layers."""

    advantage_nodes: list[nnx.Linear]
    """Advantage stream hidden layers."""

    state_value_nodes: list[nnx.Linear]
    """State value stream hidden layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        shared_nodes: list[int],
        advantage_nodes: list[int],
        state_value_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        advantage_nodes.append(n_outputs)
        state_value_nodes.append(1)

        self.n_outputs = n_outputs
        self.activation = getattr(nnx, activation)

        self.hidden_layers = []
        n_in = n_features
        for n_out in shared_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out
        n_shared_out = n_out

        self.advantage_layers = []
        for n_out in advantage_nodes:
            self.advantage_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out
        
        self.state_value_layers = []
        n_in = n_shared_out

        for n_out in state_value_nodes:
            self.state_value_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        shared_output = x

        for layer in self.advantage_layers:
            x = self.activation(layer(x))
        advantage = x

        x = shared_output
        for layer in self.state_value_layers:
            x = self.activation(layer(x))
        state_value = x

        q_value = jnp.repeat(state_value, self.n_outputs, axis=1) + (advantage - jnp.mean(advantage, 0))
        return q_value
