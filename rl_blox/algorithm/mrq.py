from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import nnx
from tqdm.rich import trange

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.embedding.model_based_encoder import (
    DeterministicPolicyWithEncoder,
    ModelBasedEncoder,
    create_model_based_encoder_and_policy,
    update_model_based_encoder,
)
from ..blox.function_approximator.layer_norm_mlp import LayerNormMLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.losses import huber_loss
from ..blox.preprocessing import make_two_hot_bins
from ..blox.replay_buffer import SubtrajectoryReplayBufferPER, lap_priority
from ..blox.return_estimates import discounted_n_step_return
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import make_sample_actions
from .td3 import make_sample_target_actions


@partial(
    nnx.jit,
    static_argnames=(
        "gamma",
        "activation_weight",
    ),
)
def update_critic_and_policy(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    policy: DeterministicTanhPolicy,
    policy_optimizer: nnx.Optimizer,
    encoder: ModelBasedEncoder,
    encoder_target: ModelBasedEncoder,
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
    encoder: ModelBasedEncoder,
    encoder_target: ModelBasedEncoder,
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
    encoder: ModelBasedEncoder,
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

    encoder : ModelBasedEncoder
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
    policy_with_encoder = create_model_based_encoder_and_policy(
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

    the_bins = make_two_hot_bins(n_bin_edges=encoder_n_bins)

    return namedtuple(
        "MRQState",
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


def train_mrq(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_with_encoder: DeterministicPolicyWithEncoder,
    encoder_optimizer: nnx.Optimizer,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    the_bins: jnp.ndarray,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    total_episodes: int | None = None,
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
    replay_buffer: SubtrajectoryReplayBufferPER | None = None,
    policy_with_encoder_target: DeterministicPolicyWithEncoder | None = None,
    q_target: ContinuousClippedDoubleQNet | None = None,
    logger: LoggerBase | None = None,
    global_step: int = 0,
    progress_bar: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
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
    See :class:`~.blox.embedding.model_based_encoder.ModelBasedEncoder` for
    more details.

    MR.Q is an extension of TD3 (see :func:`.td3.train_td3`) with LAP
    (see :func:`.td3_lap.train_td3_lap`) and is similar to TD7
    (see :func:`.td7.train_td7`). TD7 learns encoders for states and
    state-action pairs, but uses a different loss and uses the embeddings
    together with the original states and actions unlike MR.Q.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    policy_with_encoder : DeterministicPolicyWithEncoder
        Policy and encoder for the MR.Q algorithm.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

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

    total_episodes : int, optional
        Total episodes for training. This is an alternative termination
        criterion for training. Set it to None to use ``total_timesteps`` or
        set it to a positive integer to overwrite the step criterion.

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

    replay_buffer : SubtrajectoryReplayBufferPER, optional
        Episodic replay buffer for the MR.Q algorithm.

    policy_with_encoder_target : DeterministicPolicyWithEncoder, optional
        Target policy and encoder for the MR.Q algorithm.

    q_target : ContinuousClippedDoubleQNet, optional
        Target action-value function approximator for the MR.Q algorithm.

    logger : LoggerBase, optional
        Experiment logger.

    global_step : int, optional
        Global step to start training from. If not set, will start from 0.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    policy_with_encoder : DeterministicPolicyWithEncoder
        Policy and encoder for the MR.Q algorithm.

    policy_with_encoder : DeterministicPolicyWithEncoder
        Target policy and encoder.

    encoder_optimizer : nnx.Optimizer
        Optimizer for the encoder.

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

    * ``policy_with_encoder`` - actor, policy and encoder
    * ``policy_with_encoder_target`` - target policy, actor
    * ``q`` - clipped double Q network, critic
    * ``q_target`` - target network for the critic

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

    if replay_buffer is None:
        replay_buffer = SubtrajectoryReplayBufferPER(
            buffer_size,
            horizon=max(encoder_horizon, q_horizon),
        )

    if policy_with_encoder_target is None:
        policy_with_encoder_target = nnx.clone(policy_with_encoder)
    if q_target is None:
        q_target = nnx.clone(q)

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

    epoch = max(0, global_step - learning_starts)

    _update_encoder = nnx.cached_partial(
        update_model_based_encoder,
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
    if len(replay_buffer) > 0:
        reward_scale = replay_buffer.reward_scale()
        target_reward_scale = reward_scale

    episode_idx = 0
    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    for global_step in trange(
        global_step, total_timesteps, disable=not progress_bar
    ):
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
                logger.record_epoch("policy_with_encoder", policy_with_encoder)

        if terminated or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=global_step + 1
                )
                logger.stop_episode(steps_per_episode)
            episode_idx += 1
            if total_episodes is not None and episode_idx >= total_episodes:
                break
            if logger is not None:
                logger.start_new_episode()
            obs, _ = env.reset()
            steps_per_episode = 0
            accumulated_reward = 0.0
        else:
            obs = next_obs

    return namedtuple(
        "MRQResult",
        [
            "policy_with_encoder",
            "policy_with_encoder_target",
            "encoder_optimizer",
            "policy_optimizer",
            "q",
            "q_target",
            "q_optimizer",
            "replay_buffer",
        ],
    )(
        policy_with_encoder,
        policy_with_encoder_target,
        encoder_optimizer,
        policy_optimizer,
        q,
        q_target,
        q_optimizer,
        replay_buffer,
    )
