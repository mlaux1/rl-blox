from collections import namedtuple
from collections.abc import Callable

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
from flax import nnx

from ..logging.logger import LoggerBase


class EpisodicReplayBuffer:
    def __init__(self):
        pass

    def add_sample(self, **sample):
        raise NotImplementedError()

    def sample_batch(
        self,
        horizon: int,
        include_intermediate: bool,
        rng: np.random.Generator,
    ) -> tuple[jnp.ndarray]:
        raise NotImplementedError()


mrq_kernel_init = jax.nn.initializers.variance_scaling(
    scale=2, mode="fan_avg", distribution="uniform"
)


class LayerNormMLP(nnx.Module):
    """Multilayer Perceptron.

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
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    hidden_layers: list[nnx.Linear]
    """Hidden layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
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
                nnx.Linear(n_in, n_out, rngs=rngs, kernel_init=mrq_kernel_init)
            )
            self.layer_norms.append(
                nnx.LayerNorm(num_features=n_out, rngs=rngs)
            )
            n_in = n_out

        self.output_layer = nnx.Linear(
            n_in,
            n_outputs,
            rngs=rngs,
            kernel_init=mrq_kernel_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer, norm in zip(
            self.hidden_layers, self.layer_norms, strict=True
        ):
            x = self.activation(norm(layer(x)))
        return self.output_layer(x)


def create_mrq_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
    seed: int = 0,
):
    return namedtuple("MRQState", [])()


def train_mrq(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    logger: LoggerBase | None = None,
) -> None:
    r"""Model-based Representation for Q-learning (MR.Q).

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    seed : int, optional
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int, optional
        Number of steps to execute in the environment.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    TODO

    References
    ----------
    .. [1] Fujimoto, S., D'Oro, P., Zhang, A., Tian, Y., Rabbat, M. (2025).
       Towards General-Purpose Model-Free Reinforcement Learning. In
       International Conference on Learning Representations (ICLR).
       https://openreview.net/forum?id=R1hIXdST22
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    replay_buffer = EpisodicReplayBuffer()
