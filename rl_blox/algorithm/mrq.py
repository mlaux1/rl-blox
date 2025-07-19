from collections import namedtuple
from collections.abc import Callable

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import nnx

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.preprocessing import (
    make_two_hot_bins,
    two_hot_cross_entropy_loss,
    two_hot_encoding,
)
from ..logging.logger import LoggerBase
from .ddpg import make_sample_actions
from .td3 import make_sample_target_actions


class EpisodicReplayBuffer:
    def __init__(self, buffer_size: int):
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
            rngs=rngs,
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
    policy_weight_decay: float = 1e-4,
    q_hidden_nodes: list[int] | tuple[int] = (512, 512, 512),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
    q_weight_decay: float = 1e-4,
    encoder_n_bins: int = 65,
    encoder_zs_dim: int = 512,
    encoder_za_dim: int = 256,
    encoder_zsa_dim: int = 512,
    encoder_hidden_nodes: list[int] | tuple[int] = (512, 512),
    encoder_activation: str = "elu",
    encoder_learning_rate: float = 1e-4,
    encoder_weight_decay: float = 1e-4,
    seed: int = 0,
):
    env.action_space.seed(seed)

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
    encoder_optimizer = nnx.Optimizer(
        encoder,
        optax.adamw(
            learning_rate=encoder_learning_rate,
            weight_decay=encoder_weight_decay,
        ),
    )

    q1 = LayerNormMLP(
        encoder_zsa_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q2 = LayerNormMLP(
        encoder_zsa_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q = ContinuousClippedDoubleQNet(q1, q2)
    q_optimizer = nnx.Optimizer(
        q,
        optax.adamw(
            learning_rate=q_learning_rate,
            weight_decay=q_weight_decay,
        ),
    )

    policy_net = LayerNormMLP(
        encoder_zs_dim,
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs=rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy,
        optax.adamw(
            learning_rate=policy_learning_rate,
            weight_decay=policy_weight_decay,
        ),
    )

    the_bins = make_two_hot_bins(n_bin_edges=encoder_n_bins)

    return namedtuple(
        "MRQState",
        [
            "encoder",
            "encoder_optimizer",
            "q",
            "q_optimizer",
            "policy",
            "policy_optimizer",
            "the_bins",
        ],
    )(
        encoder,
        encoder_optimizer,
        q,
        q_optimizer,
        policy,
        policy_optimizer,
        the_bins,
    )


def train_mrq(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    encoder: Encoder,
    encoder_optimizer: nnx.Optimizer,
    policy: DeterministicTanhPolicy,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    the_bins: jnp.ndarray,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    learning_starts: int = 10_000,
    logger: LoggerBase | None = None,
) -> None:
    r"""Model-based Representation for Q-learning (MR.Q).

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    encoder : Encoder
        Encoder for the MR.Q algorithm.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

    policy : DeterministicTanhPolicy
        Policy for the MR.Q algorithm. Maps the latent state representation
        to actions in the environment.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy.

    q : ContinuousClippedDoubleQNet
        Action-value function approximator for the MR.Q algorithm. Maps the
        latent state-action representation to the expected value of the
        state-action pair.

    q_optimizer : nnx.Optimizer
        Optimizer for the action-value function approximator.

    the_bins : jnp.ndarray
        Bin edges for the two-hot encoding of the reward predicted by the model.

    seed : int, optional
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int, optional
        Number of steps to execute in the environment.

    buffer_size : int, optional
        Size of the replay buffer.

    gamma : float, optional
        Discount factor.

    exploration_noise : float, optional
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.

    target_policy_noise : float, optional
        Exploration noise in action space for target policy smoothing.

    noise_clip : float, optional
        Maximum absolute value of the exploration noise for sampling target
        actions for the critic update. Will be scaled by half of the range
        of the action space.

    learning_starts : int, optional
        Learning starts after this number of random steps was taken in the
        environment.

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

    replay_buffer = EpisodicReplayBuffer(buffer_size)

    _sample_actions = make_sample_actions(env.action_space, exploration_noise)
    _sample_target_actions = make_sample_target_actions(
        env.action_space, target_policy_noise, noise_clip
    )

    encoder_target = nnx.clone(encoder)
    policy_target = nnx.clone(policy)
    q_target = nnx.clone(q)

    # TODO track reward scale and target reward scale
