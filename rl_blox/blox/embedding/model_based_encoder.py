from collections.abc import Callable
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from ...blox.function_approximator.layer_norm_mlp import (
    LayerNormMLP,
    default_init,
)
from ...blox.function_approximator.policy_head import DeterministicTanhPolicy
from ...blox.losses import masked_mse_loss
from ...blox.preprocessing import two_hot_cross_entropy_loss, two_hot_decoding


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


def create_model_based_encoder_and_policy(
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
    encoder = ModelBasedEncoder(
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
        encoder_zs_dim,
        action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs=rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, action_space)
    return DeterministicPolicyWithEncoder(encoder, policy)


@partial(
    nnx.jit,
    static_argnames=(
        "encoder_horizon",
        "dynamics_weight",
        "reward_weight",
        "done_weight",
        "target_delay",
        "batch_size",
    ),
)
def update_model_based_encoder(
    encoder: ModelBasedEncoder,
    encoder_target: ModelBasedEncoder,
    encoder_optimizer: nnx.Optimizer,
    the_bins: jnp.ndarray,
    encoder_horizon: int,
    dynamics_weight: float,
    reward_weight: float,
    done_weight: float,
    target_delay: int,
    batch_size: int,
    batches: tuple[jnp.ndarray],
    environment_terminates: bool,
) -> jnp.ndarray:
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0, 0, 0, 0, 0))
    def update(args, batch):
        encoder, encoder_target, encoder_optimizer, the_bins = args
        (
            loss,
            (dynamics_loss, reward_loss, done_loss, reward_mse),
        ), grads = nnx.value_and_grad(
            model_based_encoder_loss, argnums=0, has_aux=True
        )(
            encoder,
            encoder_target,
            the_bins,
            batch,
            encoder_horizon,
            dynamics_weight,
            reward_weight,
            done_weight,
            environment_terminates,
        )
        encoder_optimizer.update(encoder, grads)
        return (
            (encoder, encoder_target, encoder_optimizer, the_bins),
            loss,
            dynamics_loss,
            reward_loss,
            done_loss,
            reward_mse,
        )

    # resize batches to (target_delay, batch_size, ...)
    batches = jax.tree_util.tree_map(
        lambda x: x.reshape(target_delay, batch_size, *x.shape[1:]),
        batches,
    )
    _, loss, dynamics_loss, reward_loss, done_loss, reward_mse = update(
        (encoder, encoder_target, encoder_optimizer, the_bins),
        batches,
    )
    losses = jnp.vstack(
        (loss, dynamics_loss, reward_loss, done_loss, reward_mse)
    )
    return jnp.mean(losses, axis=1)


def model_based_encoder_loss(
    encoder: ModelBasedEncoder,
    encoder_target: ModelBasedEncoder,
    the_bins: jnp.ndarray,
    batch: tuple[jnp.ndarray],
    encoder_horizon: int,
    dynamics_weight: float,
    reward_weight: float,
    done_weight: float,
    environment_terminates: bool,
) -> tuple[float, tuple[float, float, float, float]]:
    r"""Loss for encoder.

    The encoder loss is based on unrolling the dynamics of the learned model
    over a short horizon. Given a subsequence of an episode
    :math:`(o_0, a_0, r_1, d_1, s_1, \ldots, r_H, d_H, s_H)` with the encoder
    horizon :math:`H`, the model is unrolled by encoding the initial observation
    :math:`\tilde{\boldsymbol{z}}_s^0 = f(o_0)`, then by repeatedly applying the
    state-action encoder :math:`g` and linear MDP predictor:

    .. math::

        (\tilde{d}^t, \boldsymbol{z}_{s}^t, \tilde{r}^t)
        = \boldsymbol{w}^T g(\boldsymbol{z}_s^{t-1}, a^{t-1}) + \boldsymbol{b}

    The final loss is summed over the unrolled model and balanced by
    corresponding hyperparameters:

    .. math::

        \mathcal{L} (f, g, \boldsymbol{w}, \boldsymbol{b})
        = \sum_{t=1}^H
        \lambda_{Reward} \mathcal{L}_{Reward}(\tilde{r}^t)
        + \lambda_{Dynamics} \mathcal{L}_{Dynamics}(\boldsymbol{z}_s^t)
        + \lambda_{Terminal} \mathcal{L}_{Terminal}(\tilde{d}^t)

    The reward loss is :func:`~.blox.preprocessing.two_hot_cross_entropy_loss`.
    The dynamics loss is a mean squared error (MSE) loss between the predicted
    latent state and the latent representation of the observed state. The
    terminal loss is an MSE loss between the observed and predicted flag.

    Parameters
    ----------
    encoder : ModelBasedEncoder
        Encoder for model-based representation learning.

    encoder_target : ModelBasedEncoder
        Target encoder.

    the_bins : array, shape (n_bin_endges,)
        Bin edges for two-hot encoding.

    batch : tuple
        Batch sampled from replay buffer.

    encoder_horizon : int
        Horizon :math:`H` for dynamics unrolling.

    dynamics_weight : float
        Weight for the dynamics loss.

    reward_weight : float
        Weight for the reward loss.

    done_weight : float
        Weight for the done loss.

    environment_terminates : bool
        Flag that indicates if the environment terminates. If it does not,
        we will not use the done loss component.

    Returns
    -------
    loss : float
        Total loss for the encoder.

    loss_components : tuple
        Individual components of the loss: dynamics_loss, reward_loss,
        done_loss, reward_mse.
    """
    flat_next_observation = batch.next_observation.reshape(
        -1, *batch.next_observation.shape[2:]
    )
    flat_next_zs = jax.lax.stop_gradient(
        encoder_target.zs(flat_next_observation)
    )
    next_zs = flat_next_zs.reshape(
        list(batch.next_observation.shape[:2]) + [-1]
    )
    pred_zs_t = encoder.encode_zs(batch.observation[:, 0])
    not_done = 1 - batch.terminated
    # in subtrajectories with termination mask, mask out losses
    # after termination
    prev_not_done = jnp.ones_like(not_done[:, 0])

    @nnx.scan(
        in_axes=(nnx.Carry, None, None, None, None, None, None, 0),
        out_axes=(nnx.Carry, 0, 0, 0, 0),
    )
    def model_rollout(
        zs_t_and_prev_not_done,
        encoder,
        the_bins,
        batch,
        next_zs,
        not_done,
        environment_terminates,
        t,
    ):
        pred_zs_t, prev_not_done = zs_t_and_prev_not_done

        pred_done_t, pred_zs_t, pred_reward_logits_t = encoder.model_head(
            pred_zs_t, batch.action[:, t]
        )

        target_zs_t = next_zs[:, t]
        target_reward_t = batch.reward[:, t]
        target_done_t = batch.terminated[:, t]
        dynamics_loss = masked_mse_loss(pred_zs_t, target_zs_t, prev_not_done)
        reward_loss = jnp.mean(
            two_hot_cross_entropy_loss(
                the_bins, pred_reward_logits_t, target_reward_t
            )
            * prev_not_done
        )
        pred_reward_t = two_hot_decoding(
            the_bins, jax.nn.softmax(pred_reward_logits_t)
        )
        reward_mse = masked_mse_loss(
            pred_reward_t, target_reward_t, prev_not_done
        )
        done_loss = jnp.where(
            environment_terminates,
            masked_mse_loss(pred_done_t, target_done_t, prev_not_done),
            0.0,
        )

        # Update termination mask
        prev_not_done = not_done[:, t] * prev_not_done

        return (
            (pred_zs_t, prev_not_done),
            dynamics_loss,
            reward_loss,
            done_loss,
            reward_mse,
        )

    _, dynamics_loss, reward_loss, done_loss, reward_mse = model_rollout(
        (pred_zs_t, prev_not_done),
        encoder,
        the_bins,
        batch,
        next_zs,
        not_done,
        environment_terminates,
        jnp.arange(encoder_horizon),
    )

    dynamics_loss = jnp.sum(dynamics_loss)
    reward_loss = jnp.sum(reward_loss)
    done_loss = jnp.sum(done_loss)
    reward_mse = jnp.sum(reward_mse)

    total_loss = (
        dynamics_weight * dynamics_loss
        + reward_weight * reward_loss
        + done_weight * done_loss
    )

    return total_loss, (dynamics_loss, reward_loss, done_loss, reward_mse)
