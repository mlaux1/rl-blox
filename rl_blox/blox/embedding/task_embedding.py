from collections import namedtuple
from collections.abc import Callable

import gymnasium as gym
import jax.numpy as jnp
import optax
from flax import nnx

from ..double_qnet import ContinuousClippedDoubleQNet
from ..function_approximator.mlp import MLP
from ..function_approximator.layer_norm_mlp import LayerNormMLP, default_init
from ..function_approximator.policy_head import DeterministicTanhPolicy
from ..multitask import TaskSelectionMixin
from ..preprocessing import make_two_hot_bins
from .model_based_encoder import DeterministicPolicyWithEncoder

default_task_embedding_init = nnx.initializers.normal(stddev=1.0)


def embedding_renorm(embedding, max_norm):
    """Renormalizes the embedding to have a maximum norm of max_norm.

    This mimics the behavior of torch.nn.Embedding with max_norm.

    Parameters
    ----------
    embedding : nnx.Embed
        The embedding to renormalize.

    max_norm : float
        The maximum norm to enforce.
    """
    norm = jnp.linalg.norm(embedding.embedding.value, axis=1)[:, jnp.newaxis]
    scale = max_norm / (norm + 1e-7)
    embedding.embedding.value = jnp.where(
        norm > max_norm,
        embedding.embedding.value * scale,
        embedding.embedding.value,
    )


def concatenate_embedding(x, embedding):
    if x.ndim == 1:
        embedding = embedding.squeeze()
    elif x.ndim == 2:
        embedding = jnp.tile(embedding, (x.shape[0], 1))
    return jnp.concatenate((x, embedding), axis=-1)


class MTMLPQNetwork(nnx.Module, TaskSelectionMixin):
    """Q network for multitask setting with standard MLP.

    The Q network automatically learns an embedding of the tasks.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.

    task_embedding_dim : int
        Number of features in task embedding.

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

    max_task_embedding_norm : float
        Maximum norm for the task embedding.
    """

    _task_embedding: nnx.Embed
    """Embedding layer for the task."""

    _q : MLP
    """Q network."""

    max_task_embedding_norm: float
    """Maximum norm for the task embedding."""

    def __init__(
        self,
        n_tasks: int,
        task_embedding_dim: int,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
        max_task_embedding_norm: float = 1.0,
    ):
        super().__init__()
        self._task_embedding = nnx.Embed(
            num_embeddings=n_tasks,
            features=task_embedding_dim,
            rngs=rngs,
            embedding_init=default_task_embedding_init,
        )
        self.max_task_embedding_norm = max_task_embedding_norm
        embedding_renorm(self._task_embedding, max_norm=max_task_embedding_norm)
        self._q = MLP(
            n_features=n_features + task_embedding_dim,
            n_outputs=n_outputs,
            hidden_nodes=hidden_nodes,
            activation=activation,
            rngs=rngs,
        )

    def select_task(self, task_id: int) -> None:
        """Selects the task.

        Parameters
        ----------
        task_id : int
            Index of the task to select.
        """
        super().select_task(task_id)
        embedding_renorm(
            self._task_embedding, max_norm=self.max_task_embedding_norm
        )

    def task_embedding(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the input with task embedding."""
        embedding = self._task_embedding(jnp.array([self.task_id]))
        return concatenate_embedding(x, embedding)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._q(self.task_embedding(x))


class ModelBasedMTEncoder(nnx.Module, TaskSelectionMixin):
    r"""Encoder for the MR.Q algorithm in multitask setting.

    The encoder automatically learns an embedding of the tasks.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.

    task_embedding_dim : int
        Dimension of the task embedding.

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

    max_task_embedding_norm : float
        Maximum norm for the task embedding.
    """

    _task_embedding: nnx.Embed
    """Embedding layer for the task."""

    zs: LayerNormMLP
    """Maps observations to latent state representations (nonlinear)."""

    za: nnx.Linear
    """Maps actions to latent action representations (linear)."""

    zsa: LayerNormMLP
    """Maps zs and za to latent state-action representations (nonlinear)."""

    model: nnx.Linear
    """Maps zsa to done flag, next latent state (zs), and reward (linear)."""

    zs_dim: int
    """Dimension of the latent state representation."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    zs_layer_norm: nnx.LayerNorm
    """Layer normalization for the latent state representation."""

    max_task_embedding_norm: float
    """Maximum norm for the task embedding."""

    def __init__(
        self,
        n_tasks: int,
        task_embedding_dim: int,
        n_state_features: int,
        n_action_features: int,
        n_bins: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
        max_task_embedding_norm: float = 1.0,
    ):
        super().__init__()
        self._task_embedding = nnx.Embed(
            num_embeddings=n_tasks,
            features=task_embedding_dim,
            rngs=rngs,
            embedding_init=default_task_embedding_init,
        )
        self.max_task_embedding_norm = max_task_embedding_norm
        embedding_renorm(self._task_embedding, max_norm=max_task_embedding_norm)

        self.zs = LayerNormMLP(
            n_state_features + task_embedding_dim,
            zs_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.za = nnx.Linear(
            n_action_features + task_embedding_dim,
            za_dim,
            rngs=rngs,
            kernel_init=default_init,
        )
        self.zsa = LayerNormMLP(
            zs_dim + za_dim + task_embedding_dim,
            zsa_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.model = nnx.Linear(
            zsa_dim + task_embedding_dim,
            n_bins + zs_dim + 1,
            rngs=rngs,
            kernel_init=default_init,
        )
        self.zs_dim = zs_dim
        self.activation = getattr(nnx, activation)
        self.zs_layer_norm = nnx.LayerNorm(num_features=zs_dim, rngs=rngs)

    def select_task(self, task_id: int) -> None:
        """Selects the task.

        Parameters
        ----------
        task_id : int
            Index of the task to select.
        """
        super().select_task(task_id)
        embedding_renorm(
            self._task_embedding, max_norm=self.max_task_embedding_norm
        )

    def task_embedding(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the input with task embedding."""
        embedding = self._task_embedding(jnp.array([self.task_id]))
        return concatenate_embedding(x, embedding)

    def encode_zsa(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Encodes the state and action into latent representation.

        Parameters
        ----------
        zs : array, shape (n_samples, zs_dim + task_embedding_dim)
            State representation.

        action : array, shape (n_samples, n_action_features)
            Action representation.

        Returns
        -------
        zsa : array, shape (n_samples, zsa_dim + task_embedding_dim)
            Latent state-action representation.
        """
        # Difference to original implementation! The original implementation
        # scales actions to [-1, 1]. We do not scale the actions here.
        za = self.activation(self.za(self.task_embedding(action)))
        return self.task_embedding(self.zsa(jnp.concatenate((zs, za), axis=-1)))

    def encode_zs(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Encodes the observation into a latent state representation.

        Parameters
        ----------
        observation : array, shape (n_samples, n_state_features)
            Observation representation.

        Returns
        -------
        zs : array, shape (n_samples, zs_dim + task_embedding_dim)
            Latent state representation.
        """
        return self.task_embedding(
            self.activation(
                self.zs_layer_norm(self.zs(self.task_embedding(observation)))
            )
        )

    def model_head(
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
        done : array, shape (n_samples,)
            Flag indicating whether the episode is done.
        next_zs : array, shape (n_samples, zs_dim)
            Predicted next state representation.
        reward : array, shape (n_samples, n_bins)
            Two-hot encoded reward.
        """
        zsa = self.encode_zsa(zs, action)
        dzr = self.model(zsa)
        done = dzr[:, 0]
        next_zs = self.task_embedding(dzr[:, 1 : 1 + self.zs_dim])
        reward = dzr[:, 1 + self.zs_dim :]
        return done, next_zs, reward


def create_model_based_mt_encoder_and_policy(
    n_tasks: int,
    task_embedding_dim: int,
    n_state_features: int,
    n_action_features: int,
    action_space: gym.spaces.Box,
    policy_hidden_nodes: list[int] | tuple[int] = (512, 512),
    policy_activation: str = "relu",
    encoder_n_bins: int = 65,
    encoder_zs_dim: int = 512,
    encoder_za_dim: int = 256,
    encoder_zsa_dim: int = 512,
    encoder_hidden_nodes: list[int] | tuple[int] = (512, 512),
    encoder_activation: str = "elu",
    rngs: nnx.Rngs | None = None,
) -> DeterministicPolicyWithEncoder:
    """Creates a model-based encoder."""
    if rngs is None:
        rngs = nnx.Rngs(0)
    encoder = ModelBasedMTEncoder(
        n_tasks=n_tasks,
        task_embedding_dim=task_embedding_dim,
        n_state_features=n_state_features,
        n_action_features=n_action_features,
        n_bins=encoder_n_bins,
        zs_dim=encoder_zs_dim,
        za_dim=encoder_za_dim,
        zsa_dim=encoder_zsa_dim,
        hidden_nodes=encoder_hidden_nodes,
        activation=encoder_activation,
        rngs=rngs,
    )
    policy_net = LayerNormMLP(
        encoder_zs_dim + task_embedding_dim,
        action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs=rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, action_space)
    return DeterministicPolicyWithEncoder(encoder, policy)


def create_mt_mrq_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    n_tasks: int,
    task_embedding_dim: int = 96,
    policy_hidden_nodes: list[int] | tuple[int] = (512, 512),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    policy_weight_decay: float = 1e-4,
    q_hidden_nodes: list[int] | tuple[int] = (512, 512, 512),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
    q_weight_decay: float = 1e-4,
    q_grad_clipping: float = 20.0,
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
    policy_with_encoder = create_model_based_mt_encoder_and_policy(
        n_tasks=n_tasks,
        task_embedding_dim=task_embedding_dim,
        n_state_features=env.observation_space.shape[0],
        n_action_features=env.action_space.shape[0],
        action_space=env.action_space,
        policy_hidden_nodes=policy_hidden_nodes,
        policy_activation=policy_activation,
        encoder_n_bins=encoder_n_bins,
        encoder_zs_dim=encoder_zs_dim,
        encoder_za_dim=encoder_za_dim,
        encoder_zsa_dim=encoder_zsa_dim,
        encoder_hidden_nodes=encoder_hidden_nodes,
        encoder_activation=encoder_activation,
        rngs=rngs,
    )
    encoder_optimizer = nnx.Optimizer(
        policy_with_encoder.encoder,
        optax.adamw(
            learning_rate=encoder_learning_rate,
            weight_decay=encoder_weight_decay,
        ),
        wrt=nnx.Param,
    )
    policy_optimizer = nnx.Optimizer(
        policy_with_encoder.policy,
        optax.adamw(
            learning_rate=policy_learning_rate,
            weight_decay=policy_weight_decay,
        ),
        wrt=nnx.Param,
    )

    q1 = LayerNormMLP(
        encoder_zsa_dim + task_embedding_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q2 = LayerNormMLP(
        encoder_zsa_dim + task_embedding_dim,
        1,
        q_hidden_nodes,
        q_activation,
        rngs=rngs,
    )
    q = ContinuousClippedDoubleQNet(q1, q2)
    q_optimizer = nnx.Optimizer(
        q,
        optax.chain(
            optax.clip_by_global_norm(q_grad_clipping),
            optax.adamw(
                learning_rate=q_learning_rate,
                weight_decay=q_weight_decay,
            ),
        ),
        wrt=nnx.Param,
    )

    the_bins = make_two_hot_bins(n_bin_edges=encoder_n_bins)

    return namedtuple(
        "MTMRQState",
        [
            "policy_with_encoder",
            "encoder_optimizer",
            "policy_optimizer",
            "q",
            "q_optimizer",
            "the_bins",
        ],
    )(
        policy_with_encoder,
        encoder_optimizer,
        policy_optimizer,
        q,
        q_optimizer,
        the_bins,
    )
