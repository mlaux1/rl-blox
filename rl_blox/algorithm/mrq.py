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


class Encoder(nnx.Module):
    """Encoder for the MR.Q algorithm.

    Parameters
    ----------
    n_bins : int
        Number of bins for the two-hot encoding.

    n_state_features : int
        Number of state components.

    n_action_features : int
        Number of action components.

    zs_dim : int
        Dimension of the latent state representation.

    za_dim : int
        Dimension of the latent action representation.

    zsa_dim : int
        Dimension of the latent state-action representation.

    hidden_nodes : list
        Numbers of hidden nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.
    """

    zs: LayerNormMLP
    za: nnx.Linear
    zsa: LayerNormMLP
    model: nnx.Linear
    zs_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    zs_layer_norm: nnx.LayerNorm

    def __init__(
        self,
        n_state_features: int,
        n_action_features: int,
        n_bins: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        self.zs = LayerNormMLP(
            n_state_features,
            zs_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.za = nnx.Linear(
            n_action_features, za_dim, rngs=rngs, kernel_init=mrq_kernel_init
        )
        self.zsa = LayerNormMLP(
            zs_dim + za_dim,
            zsa_dim,
            hidden_nodes,
            activation,
        )
        self.model = nnx.Linear(
            zsa_dim, n_bins + zs_dim + 1, rngs=rngs, kernel_init=mrq_kernel_init
        )
        self.zs_dim = zs_dim
        self.activation = getattr(nnx, activation)
        self.zs_layer_norm = nnx.LayerNorm(num_features=zs_dim, rngs=rngs)

    def encode_zsa(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Encodes the state and action into latent representation.

        Parameters
        ----------
        zs : array, shape (n_samples, n_state_features)
            State representation.

        action : array, shape (n_samples, n_action_features)
            Action representation.

        Returns
        -------
        zsa : array, shape (n_samples, zsa_dim)
            Latent state-action representation.
        """
        za = self.activation(self.za(action))
        return self.zsa(jnp.concatenate((zs, za), axis=-1))

    def encode_zs(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Encodes the observation into a latent state representation.

        Parameters
        ----------
        observation : array, shape (n_samples, n_state_features)
            Observation representation.

        Returns
        -------
        zs : array, shape (n_samples, zs_dim)
            Latent state representation.
        """
        return self.activation(self.zs_layer_norm(self.zs(observation)))

    def model_all(
        self, zs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predicts the full model.

        Parameters
        ----------
        zs : array, shape (n_samples, zs_dim)
            Latent state representation.

        action : array, shape (n_samples, n_action_features)
            Action.

        Returns
        -------
        zsa : array, shape (n_samples, zsa_dim)
            Latent state-action representation.
        next_zs : array, shape (n_samples, zs_dim)
            Predicted next state representation.
        bins : array, shape (n_samples, n_bins + 1)
            Two-hot encoded bins for the next state.
        """
        zsa = self.encode_zsa(zs, action)
        dzr = self.model(zsa)
        done = dzr[:, 0]
        next_zs = dzr[:, 1 : 1 + self.zs_dim]
        reward = dzr[:, 1 + self.zs_dim :]
        return done, next_zs, reward


def create_mrq_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
    encoder_n_bins: int = 65,
    encoder_zs_dim: int = 512,
    encoder_za_dim: int = 256,
    encoder_zsa_dim: int = 512,
    encoder_hidden_nodes: list[int] | tuple[int] = (512, 512),
    encoder_activation: str = "elu",
    seed: int = 0,
):
    rngs = nnx.Rngs(seed)
    encoder = Encoder(
        n_state_features=env.observation_space.shape[0],
        n_action_features=env.action_space.shape[0],
        n_bins=encoder_n_bins,
        zs_dim=encoder_zs_dim,
        za_dim=encoder_za_dim,
        zsa_dim=encoder_zsa_dim,
        hidden_nodes=encoder_hidden_nodes,
        activation=encoder_activation,
        rngs=rngs,
    )
    return namedtuple("MRQState", ["encoder"])(encoder)


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
