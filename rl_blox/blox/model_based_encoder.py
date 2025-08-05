from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from ..blox.function_approximator.layer_norm_mlp import (
    LayerNormMLP,
    default_init,
)
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy


class ModelBasedEncoder(nnx.Module):
    r"""Encoder for the MR.Q algorithm.

    The state embedding vector :math:`\boldsymbol{z}_s` is obtained as an
    intermediate component by training end-to-end with the state-action encoder.
    MR.Q handles different input modalities by swapping the architecture of
    the state encoder. Since :math:`\boldsymbol{z}_s` is a vector, the
    remaining networks are independent of the observation space and use
    feedforward networks. Note that in this implementation, we can only handle
    observations / states represented by real vectors.

    Given the transition :math:`(o, a, r, d, o')` consisting of observation,
    action, reward, done flag (1 - terminated), and next observation
    respectively, the encoder predicts

    .. math::

        \boldsymbol{z}_s &= f(o)\\
        \boldsymbol{z}_{sa} &= g(\boldsymbol{z}_s, a)\\
        (\tilde{d}, \boldsymbol{z}_{s'}, \tilde{r})
        &= \boldsymbol{w}^T \boldsymbol{z}_{sa} + \boldsymbol{b}


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

    References
    ----------
    .. [1] Fujimoto, S., D'Oro, P., Zhang, A., Tian, Y., Rabbat, M. (2025).
       Towards General-Purpose Model-Free Reinforcement Learning. In
       International Conference on Learning Representations (ICLR).
       https://openreview.net/forum?id=R1hIXdST22
    """

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
            n_action_features, za_dim, rngs=rngs, kernel_init=default_init
        )
        self.zsa = LayerNormMLP(
            zs_dim + za_dim,
            zsa_dim,
            hidden_nodes,
            activation,
            rngs=rngs,
        )
        self.model = nnx.Linear(
            zsa_dim, n_bins + zs_dim + 1, rngs=rngs, kernel_init=default_init
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
        # Difference to original implementation! The original implementation
        # scales actions to [-1, 1]. We do not scale the actions here.
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
        next_zs = dzr[:, 1 : 1 + self.zs_dim]
        reward = dzr[:, 1 + self.zs_dim :]
        return done, next_zs, reward


class DeterministicPolicyWithEncoder(nnx.Module):
    """Combines encoder and deterministic policy."""

    encoder: ModelBasedEncoder
    policy: DeterministicTanhPolicy

    def __init__(
        self, encoder: ModelBasedEncoder, policy: DeterministicTanhPolicy
    ):
        self.encoder = encoder
        self.policy = policy

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.policy(self.encoder.encode_zs(observation))
