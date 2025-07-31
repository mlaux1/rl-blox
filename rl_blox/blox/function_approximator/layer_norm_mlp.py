from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from flax import nnx

default_init = jax.nn.initializers.variance_scaling(
    scale=2, mode="fan_avg", distribution="uniform"
)


class LayerNormMLP(nnx.Module):
    """Multilayer Perceptron with layer normalization after each hidden layer.

    Parameters
    ----------
    n_features : int
        Number of features.

    n_outputs : int
        Number of output components.

    hidden_nodes : list
        Numbers of hidden nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.

    kernel_init : nnx.Initializer, optional
        Initializer for the weights of the linear layers.
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    hidden_layers: list[nnx.Linear]
    """Hidden layers."""

    layer_norms: list[nnx.LayerNorm]
    """Layer normalization layers for hidden layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
        kernel_init: nnx.Initializer = default_init,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs
        self.activation = getattr(nnx, activation)

        self.hidden_layers = []
        self.layer_norms = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(
                nnx.Linear(n_in, n_out, rngs=rngs, kernel_init=kernel_init)
            )
            self.layer_norms.append(
                nnx.LayerNorm(num_features=n_out, rngs=rngs)
            )
            n_in = n_out

        self.output_layer = nnx.Linear(
            n_in,
            n_outputs,
            rngs=rngs,
            kernel_init=default_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer, norm in zip(
            self.hidden_layers, self.layer_norms, strict=True
        ):
            x = self.activation(norm(layer(x)))
        return self.output_layer(x)
