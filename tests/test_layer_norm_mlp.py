import jax.numpy as jnp
from flax import nnx

from rl_blox.blox.function_approximator.layer_norm_mlp import LayerNormMLP


def test_layer_norm_mlp_uses_kernel_init_in_all_layers():
    net = LayerNormMLP(
        3, 2, [4, 4], "relu", nnx.Rngs(0), kernel_init=nnx.initializers.ones
    )
    for layer in [*net.hidden_layers, net.output_layer]:
        assert jnp.all(layer.kernel[...] == 1.0)
