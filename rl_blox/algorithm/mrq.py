from collections import namedtuple
from collections.abc import Callable
from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax import nnx

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.layer_norm_mlp import (
    LayerNormMLP,
    default_init,
)
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.losses import huber_loss, masked_mse_loss
from ..blox.preprocessing import (
    make_two_hot_bins,
    two_hot_cross_entropy_loss,
    two_hot_decoding,
)
from ..blox.replay_buffer import SubtrajectoryReplayBufferPER, lap_priority
from ..blox.return_estimates import discounted_n_step_return
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import make_sample_actions
from .td3 import make_sample_target_actions


class Encoder(nnx.Module):
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

    encoder: Encoder
    policy: DeterministicTanhPolicy

    def __init__(self, encoder: Encoder, policy: DeterministicTanhPolicy):
        self.encoder = encoder
        self.policy = policy

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.policy(self.encoder.encode_zs(observation))


@partial(
    nnx.jit,
    static_argnames=(
        "encoder_horizon",
        "dynamics_weight",
        "reward_weight",
        "done_weight",
        "environment_terminates",
        "target_delay",
        "batch_size",
    ),
)
def update_encoder(
    encoder: Encoder,
    encoder_target: Encoder,
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
        ), grads = nnx.value_and_grad(encoder_loss, argnums=0, has_aux=True)(
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


def encoder_loss(
    encoder: Encoder,
    encoder_target: Encoder,
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
    encoder : Encoder
        Encoder for model-based representation learning.

    encoder_target : Encoder
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


@partial(
    nnx.jit,
    static_argnames=(
        "gamma",
        "reward_scale",
        "target_reward_scale",
        "activation_weight",
    ),
)
def update_critic_and_policy(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    policy: DeterministicTanhPolicy,
    policy_optimizer: nnx.Optimizer,
    encoder: Encoder,
    encoder_target: Encoder,
    gamma: float,
    activation_weight: float,
    next_action: jnp.ndarray,
    batch: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    reward_scale: float,
    target_reward_scale: float,
) -> tuple[float, float, tuple[float, float], float, jnp.ndarray]:
    """Update the critic and policy network."""
    (q_loss, (zs, q_mean, max_abs_td_error)), grads = nnx.value_and_grad(
        mrq_loss, argnums=0, has_aux=True
    )(
        q,
        q_target,
        encoder,
        encoder_target,
        next_action,
        batch,
        gamma,
        reward_scale,
        target_reward_scale,
    )
    q_optimizer.update(q, grads)

    (policy_loss, policy_loss_components), grads = nnx.value_and_grad(
        mrq_policy_loss, argnums=0, has_aux=True
    )(
        policy,
        q,
        encoder,
        zs,
        activation_weight,
    )
    policy_optimizer.update(policy, grads)

    return q_loss, policy_loss, policy_loss_components, q_mean, max_abs_td_error


def mrq_loss(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    encoder: Encoder,
    encoder_target: Encoder,
    next_action: jnp.ndarray,
    batch: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    gamma: float,
    reward_scale: float,
    target_reward_scale: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, float, jnp.ndarray]]:
    """Compute the MR.Q critic loss."""
    observation, action, reward, next_observation, terminated, _ = batch

    n_step_return, discount = discounted_n_step_return(
        reward, terminated, gamma
    )

    next_zs = jax.lax.stop_gradient(encoder_target.encode_zs(next_observation))
    next_zsa = jax.lax.stop_gradient(
        encoder_target.encode_zsa(next_zs, next_action)
    )
    q_next = jax.lax.stop_gradient(q_target(next_zsa).squeeze())
    q_target_value = (
        n_step_return + discount * q_next * target_reward_scale
    ) / reward_scale

    zs = jax.lax.stop_gradient(encoder.encode_zs(observation))
    zsa = jax.lax.stop_gradient(encoder.encode_zsa(zs, action))

    q1_predicted = q.q1(zsa).squeeze()
    q2_predicted = q.q2(zsa).squeeze()

    td_error1 = jnp.abs(q1_predicted - q_target_value)
    td_error2 = jnp.abs(q2_predicted - q_target_value)

    max_abs_td_error = jnp.maximum(td_error1, td_error2)

    value_loss = (
        huber_loss(td_error1, 1.0).mean() + huber_loss(td_error2, 1.0).mean()
    )

    q_mean = jnp.minimum(q1_predicted, q2_predicted).mean()
    return value_loss, (zs, q_mean, max_abs_td_error)


def mrq_policy_loss(
    policy: DeterministicTanhPolicy,
    q: nnx.Module,
    encoder: Encoder,
    zs: jnp.ndarray,
    activation_weight: float,
) -> tuple[float, tuple[float, float]]:
    """Compute the policy loss for MR.Q.

    Parameters
    ----------
    policy : DeterministicTanhPolicy
        The policy network.

    q : nnx.Module
        The Q-value network used to evaluate the policy.

    encoder : Encoder
        The encoder network to encode the state-action pairs.

    zs : jnp.ndarray
        The latent state representation of the current observation.

    activation_weight : float
        Weight for the regularization term on the policy activation.

    Returns
    -------
    policy_loss : float
        The computed policy loss.

    loss_components : tuple
        A tuple containing the DPG loss and the policy regularization term.
    """
    activation = policy.policy_net(zs)
    action = policy.scale_output(activation)
    zsa = encoder.encode_zsa(zs, action)
    # - to perform gradient ascent with a minimizer
    dpg_loss = -q(zsa).mean()
    policy_regularization = jnp.square(activation).mean()
    policy_loss = dpg_loss + activation_weight * policy_regularization
    return policy_loss, (dpg_loss, policy_regularization)


def create_mrq_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
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
        wrt=nnx.Param,
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
        optax.chain(
            optax.clip_by_global_norm(q_grad_clipping),
            optax.adamw(
                learning_rate=q_learning_rate,
                weight_decay=q_weight_decay,
            ),
        ),
        wrt=nnx.Param,
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
        wrt=nnx.Param,
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
    target_delay: int = 250,
    batch_size: int = 256,
    exploration_noise: float = 0.2,
    target_policy_noise: float = 0.2,
    noise_clip: float = 0.3,
    lap_min_priority: float = 1.0,
    lap_alpha: float = 0.4,
    learning_starts: int = 10_000,
    encoder_horizon: int = 5,
    q_horizon: int = 3,
    dynamics_weight: float = 1.0,
    reward_weight: float = 0.1,
    done_weight: float = 0.1,
    activation_weight: float = 1e-5,
    logger: LoggerBase | None = None,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    SubtrajectoryReplayBufferPER,
]:
    r"""Model-based Representation for Q-learning (MR.Q).

    MR.Q is an attempt to find a unifying model-free reinforcement learning
    algorithm that can address a diverse class of domains and problem settings
    with model-based representation learning. The state representation and
    state-action representation are trained such that a linear model can predict
    from it if the episode is terminated, the next latent state, and the reward.

    MR.Q is an extension of TD3 (see :func:`.td3.train_td3`) with LAP
    (see :func:`.td3_lap.train_td3_lap`) and is similar to TD7
    (see :func:`.td7.train_td7`). TD7 learns encoders for states and
    state-action pairs, but uses a different loss and uses the embeddings
    together with the original states and actions unlike MR.Q.

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
        to actions in the environment, :math:`\pi(\boldsymbol{z}_s) = a`.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy.

    q : ContinuousClippedDoubleQNet
        Action-value function approximator for the MR.Q algorithm. Maps the
        latent state-action representation to the expected value of the
        state-action pair, :math:`Q(\boldsymbol{z}_{sa})`.

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

    target_delay : int, optional
        Delayed target net updates. The target nets are updated every
        ``target_delay`` steps.

    batch_size : int, optional
        Size of a batch during gradient computation.

    exploration_noise : float, optional
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.

    target_policy_noise : float, optional
        Exploration noise in action space for target policy smoothing.

    noise_clip : float, optional
        Maximum absolute value of the exploration noise for sampling target
        actions for the critic update. Will be scaled by half of the range
        of the action space.

    lap_alpha : float, optional
        Constant for probability smoothing in LAP.

    lap_min_priority : float, optional
        Minimum priority in LAP.

    learning_starts : int, optional
        Learning starts after this number of random steps was taken in the
        environment.

    encoder_horizon : int, optional
        Horizon for encoder training.

    q_horizon : int, optional
        Horizon for Q training.

    dynamics_weight : float, optional
        Weight for the dynamics loss in the encoder training.

    reward_weight : float, optional
        Weight for the reward loss in the encoder training.

    done_weight : float, optional
        Weight for the done loss in the encoder training.

    activation_weight : float, optional
        Weight for the activation regularization in the policy training.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    encoder : Encoder
        Encoder for the MR.Q algorithm.

    encoder_target : Encoder
        Target encoder for the MR.Q algorithm.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

    policy : DeterministicTanhPolicy
        Policy for the MR.Q algorithm.

    policy_target : DeterministicTanhPolicy
        Target policy for the MR.Q algorithm.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy.

    q : ContinuousClippedDoubleQNet
        Action-value function approximator for the MR.Q algorithm.

    q_target : ContinuousClippedDoubleQNet
        Target action-value function approximator for the MR.Q algorithm.

    q_optimizer : nnx.Optimizer
        Optimizer for the action-value function approximator.

    replay_buffer : SubtrajectoryReplayBufferPER
        Episodic replay buffer for the MR.Q algorithm.

    Notes
    -----

    Logging

    * ``reward scale`` - mean absolute reward in replay buffer
    * ``encoder loss`` - value of the loss function for ``encoder``
    * ``dynamics loss`` - value of the loss function for the dynamics model
    * ``reward loss`` - value of the loss function for the reward model
    * ``done loss`` - value of the loss function for the done model
    * ``reward mse`` - mean squared error of the reward model
    * ``q loss`` - value of the loss function for ``q``
    * ``q mean`` - mean Q value of batch used to update the critic
    * ``policy loss`` - value of the loss function for the actor
    * ``dpg loss`` - value of the DPG loss for the actor
    * ``policy regularization`` - value of the policy regularization term
    * ``return`` - return of the episode

    Checkpointing

    * ``q`` - clipped double Q network, critic
    * ``q_target`` - target network for the critic
    * ``policy_with_encoder_target`` - target policy, actor
    * ``encoder`` - encoder for the state representation
    * ``policy`` - target network for the actor

    References
    ----------
    .. [1] Fujimoto, S., D'Oro, P., Zhang, A., Tian, Y., Rabbat, M. (2025).
       Towards General-Purpose Model-Free Reinforcement Learning. In
       International Conference on Learning Representations (ICLR).
       https://openreview.net/forum?id=R1hIXdST22

    See Also
    --------
    .td3.train_td3
        TD3 algorithm.
    .td3_lap.train_td3_lap
        TD3 with LAP.
    .td7.train_td7
        TD7 algorithm, which is similar to MR.Q but uses a different encoder.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    replay_buffer = SubtrajectoryReplayBufferPER(
        buffer_size,
        horizon=max(encoder_horizon, q_horizon),
    )

    encoder_target = nnx.clone(encoder)
    policy_target = nnx.clone(policy)
    q_target = nnx.clone(q)

    policy_with_encoder = DeterministicPolicyWithEncoder(encoder, policy)
    policy_with_encoder_target = DeterministicPolicyWithEncoder(
        encoder_target, policy_target
    )

    _sample_actions = nnx.cached_partial(
        make_sample_actions(env.action_space, exploration_noise),
        policy_with_encoder,
    )
    _sample_target_actions = nnx.cached_partial(
        make_sample_target_actions(
            env.action_space, target_policy_noise, noise_clip
        ),
        policy_with_encoder_target,
    )

    epoch = 0

    _update_encoder = nnx.cached_partial(
        update_encoder,
        policy_with_encoder.encoder,
        policy_with_encoder_target.encoder,
        encoder_optimizer,
        the_bins,
        encoder_horizon,
        dynamics_weight,
        reward_weight,
        done_weight,
        target_delay,
        batch_size,
    )
    _update_critic_and_policy = nnx.cached_partial(
        update_critic_and_policy,
        q,
        q_target,
        q_optimizer,
        policy_with_encoder.policy,
        policy_optimizer,
        policy_with_encoder.encoder,
        policy_with_encoder_target.encoder,
        gamma,
        activation_weight,
    )

    reward_scale = 1.0
    target_reward_scale = 0.0

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    for global_step in tqdm.trange(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(_sample_actions(jnp.asarray(obs), action_key))

        next_obs, reward, terminated, truncated, info = env.step(action)
        steps_per_episode += 1
        accumulated_reward += reward

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        if global_step >= learning_starts:
            epoch += 1
            if epoch % target_delay == 0:
                hard_target_net_update(
                    policy_with_encoder, policy_with_encoder_target
                )
                hard_target_net_update(q, q_target)

                target_reward_scale = reward_scale
                reward_scale = replay_buffer.reward_scale()

                replay_buffer.reset_max_priority()

                batches = replay_buffer.sample_batch(
                    batch_size * target_delay, encoder_horizon, True, rng
                )
                losses = _update_encoder(
                    batches,
                    replay_buffer.environment_terminates,
                )
                if logger is not None:
                    log_step = global_step + 1
                    logger.record_stat(
                        "reward scale", reward_scale, step=log_step
                    )
                    keys = [
                        "encoder loss",
                        "dynamics loss",
                        "reward loss",
                        "done loss",
                        "reward mse",
                    ]
                    for k, v in zip(keys, losses, strict=False):
                        logger.record_stat(k, v, step=log_step)
                    logger.record_epoch(
                        "policy_with_encoder_target", policy_with_encoder_target
                    )
                    logger.record_epoch("q_target", q_target)
                    logger.record_epoch("encoder", policy_with_encoder.encoder)

            batch = replay_buffer.sample_batch(
                batch_size, q_horizon, False, rng
            )
            # policy smoothing: sample next actions from target policy
            key, sampling_key = jax.random.split(key, 2)
            next_actions = _sample_target_actions(
                batch.next_observation, sampling_key
            )

            (
                q_loss_value,
                policy_loss,
                (dpg_loss, policy_regularization),
                q_mean,
                max_abs_td_error,
            ) = _update_critic_and_policy(
                next_actions,
                batch,
                reward_scale,
                target_reward_scale,
            )
            replay_buffer.update_priority(
                lap_priority(max_abs_td_error, lap_min_priority, lap_alpha)
            )
            if logger is not None:
                logger.record_stat("q loss", q_loss_value, step=global_step + 1)
                logger.record_stat("q mean", q_mean, step=global_step + 1)
                logger.record_stat(
                    "policy loss", policy_loss, step=global_step + 1
                )
                logger.record_stat("dpg loss", dpg_loss, step=global_step + 1)
                logger.record_stat(
                    "policy regularization",
                    policy_regularization,
                    step=global_step + 1,
                )
                logger.record_epoch("q", q)
                logger.record_epoch("policy", policy_with_encoder.policy)

        if terminated or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=global_step + 1
                )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()

            steps_per_episode = 0
            accumulated_reward = 0.0
        else:
            obs = next_obs

    return namedtuple(
        "MRQResult",
        [
            "encoder",
            "encoder_target",
            "encoder_optimizer",
            "policy",
            "policy_target",
            "policy_optimizer",
            "q",
            "q_target",
            "q_optimizer",
            "replay_buffer",
        ],
    )(
        encoder,
        encoder_target,
        encoder_optimizer,
        policy,
        policy_target,
        policy_optimizer,
        q,
        q_target,
        q_optimizer,
        replay_buffer,
    )
