from collections import namedtuple
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm.rich import trange

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.losses import td3_loss
from ..blox.replay_buffer import ReplayBuffer
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import ddpg_update_actor, make_sample_actions
from .dqn import train_step_with_loss


def sample_target_actions(
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    action_scale: jnp.ndarray,
    exploration_noise: float,
    noise_clip: float,
    policy: DeterministicTanhPolicy,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    r"""Sample target actions with truncated Gaussian noise.

    Given a deterministic policy :math:`\pi(o) = a`, we will generate an action

    .. math::

        a = \texttt{clip}(
        \pi(o) + \texttt{clip}(\epsilon, -c, c),
        a_{low}, a_{high})

    with added clipped noise :math:`\texttt{clip}(\epsilon, -c, c)`
    (:math:`c` is ``action_scale * noise_clip``) sampled through
    :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)` (standard deviation
    ``action_scale * exploration_noise``) and clipped to
    the action range :math:`\left[a_{low}, a_{high}\right]` (parameters
    ``action_low`` and ``action_high``).

    Parameters
    ----------
    action_low : array, shape (n_action_dims,)
        Lower bound on actions.

    action_high : array, shape (n_action_dims,)
        Upper bound on actions.

    action_scale : array, shape (n_action_dims,)
        Scale of action dimensions.

    exploration_noise : float
        Scaling factor for exploration noise.

    noise_clip : float
        Scaling factor for noise clipping.

    policy : DeterministicTanhPolicy
        Deterministic policy.

    obs : array, shape (n_observations_dims,)
        Observation.

    key : array
        Key for PRNG.

    Returns
    -------
    action : array, shape (n_action_dims,)
        Exploration action.
    """
    action = policy(obs)
    eps = (
        exploration_noise * action_scale * jax.random.normal(key, action.shape)
    )
    scaled_noise_clip = action_scale * noise_clip
    clipped_eps = jnp.clip(eps, -scaled_noise_clip, scaled_noise_clip)
    return jnp.clip(action + clipped_eps, action_low, action_high)


def make_sample_target_actions(
    action_space: gym.spaces.Box,
    exploration_noise: float,
    noise_clip: float,
) -> Callable[[nnx.Module, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    action_scale = 0.5 * (action_space.high - action_space.low)
    return nnx.jit(
        partial(
            sample_target_actions,
            action_space.low,
            action_space.high,
            action_scale,
            exploration_noise,
            noise_clip,
        )
    )


def create_td3_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 3e-4,
    seed: int = 0,
) -> namedtuple:
    """Create components for TD3 algorithm with default configuration."""
    env.action_space.seed(seed)

    policy_net = MLP(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        nnx.Rngs(seed),
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate), wrt=nnx.Param
    )

    q1 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )
    q2 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed + 1),
    )
    q = ContinuousClippedDoubleQNet(q1, q2)
    q_optimizer = nnx.Optimizer(
        q, optax.adam(learning_rate=q_learning_rate), wrt=nnx.Param
    )

    return namedtuple(
        "TD3State",
        [
            "policy",
            "policy_optimizer",
            "q",
            "q_optimizer",
        ],
    )(policy, policy_optimizer, q, q_optimizer)


def train_td3(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    total_episodes: int | None = None,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.2,
    noise_clip: float = 0.5,
    learning_starts: int = 25_000,
    replay_buffer: ReplayBuffer | None = None,
    policy_target: nnx.Module | None = None,
    q_target: ContinuousClippedDoubleQNet | None = None,
    logger: LoggerBase | None = None,
    global_step: int = 0,
    progress_bar: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    ContinuousClippedDoubleQNet,
    ContinuousClippedDoubleQNet,
    nnx.Optimizer,
    ReplayBuffer,
]:
    r"""Twin Delayed DDPG (TD3).

    TD3 [1]_ extends DDPG with three techniques to improve performance:

    1. Clipped Double Q-Learning to mitigate overestimation bias of the value
       (see :class:`~.blox.double_qnet.ContinuousClippedDoubleQNet`)
    2. Delayed policy updates, controlled by the parameter ``policy_delay``
    3. Target policy smoothing, i.e., sampling from the behavior policy with
       clipped noise (parameter ``noise_clip``) for the critic update.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    policy : nnx.Module
        Deterministic policy network.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy network.

    q : ContinuousClippedDoubleQNet
        Clipped double Q network.

    q_optimizer: nnx.Optimizer
        Optimizer for q.

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

    tau : float, optional
        Learning rate for polyak averaging of target policy and value function.

    policy_delay : int, optional
        Delayed policy updates. The policy is updated every ``policy_delay``
        steps.

    batch_size : int, optional
        Size of a batch during gradient computation.

    gradient_steps : int, optional
        Number of gradient steps during one training phase.

    exploration_noise : float, optional
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.

    noise_clip : float, optional
        Maximum absolute value of the exploration noise for sampling target
        actions for the critic update. Will be scaled by half of the range
        of the action space.

    learning_starts : int, optional
        Learning starts after this number of random steps was taken in the
        environment.

    replay_buffer : ReplayBuffer
        Replay buffer.

    policy_target : nnx.Module, optional
        Target policy. Only has to be set if we want to continue training
        from an old state.

    q_target : ContinuousDoubleQNet, optional
        Target network. Only has to be set if we want to continue training
        from an old state.

    global_step : int, optional
        Global step to start training from. If not set, will start from 0.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    policy : nnx.Module
        Final policy.

    policy_target : nnx.Module
        Target policy.

    policy_optimizer : nnx.Optimizer
        Policy optimizer.

    q : ContinuousClippedDoubleQNet
        Final state-action value function.

    q_target : ContinuousClippedDoubleQNet
        Target network.

    q_optimizer : nnx.Optimizer
        Optimizer for Q network.

    replay_buffer : ReplayBuffer
        Replay buffer.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Notes
    -----

    Parameters

    * :math:`\pi(o) = a` with weights :math:`\theta^{\pi}` - deterministic
      target ``policy``, maps observations to actions
    * :math:`\pi'` with weights :math:`\theta^{\pi'}` - policy target network
      (``policy_target``), initialized as a copy of ``policy``
    * :math:`Q(o, a)` with weights :math:`\theta^{Q}` - critic network
      ``q``, composed of two q networks :math:`Q_i(o, a)` with index i
      (see :class:`~.blox.double_qnet.ContinuousClippedDoubleQNet`)
    * :math:`Q'(o, a)` with weights :math:`\theta^{Q'}` - target network
      ``q_target``, initialized as a copy of ``q``
    * :math:`R` - ``replay_buffer``

    Algorithm

    * Randomly sample ``learning_starts`` actions and record transitions in
      :math:`R`
    * For each step :math:`t`

      * Sample action with behavior policy in :func:`.ddpg.sample_actions`
      * Take a step in the environment ``env`` and observe result
      * Store transition :math:`(o_t, a_t, r_t, o_{t+1}, d_{t+1})` in :math:`R`,
        where :math:`d` indicates if a terminal state was reached
      * Sample mini-batch of ``batch_size`` transitions from :math:`R` to
        update the networks
      * Sample next target actions :math:`\tilde{a}_{i+1}` based on :math:`o_i`
        for the mini-batch with :func:`sample_target_actions` (target policy
        smoothing)
      * Update critic with :func:`~.blox.losses.td3_loss`
      * If ``t % policy_delay == 0`` (delayed policy update)

        * Update actor with :func:`.ddpg.ddpg_update_actor`
        * Update target networks :math:`Q', \pi'` with
          :func:`~.blox.target_net.soft_target_net_update`

    Logging

    * ``q loss`` - value of the loss function for ``q``
    * ``q mean`` - mean Q value of batch used to update the critic
    * ``policy loss`` - value of the loss function for the actor

    Checkpointing

    * ``q`` - clipped double Q network, critic
    * ``policy`` - target policy, actor
    * ``q_target`` - target network for the critic
    * ``policy_target`` - target network for the actor

    References
    ----------
    .. [1] Fujimoto, S., Hoof, H., Meger, D. (2018). Addressing Function
       Approximation Error in Actor-Critic Methods. Proceedings of the 35th
       International Conference on Machine Learning, in Proceedings of Machine
       Learning Research 80:1587-1596 Available from
       https://proceedings.mlr.press/v80/fujimoto18a.html.

    See Also
    --------
    .ddpg.train_ddpg
        DDPG without the extensions of TD3.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    chex.assert_scalar_in(tau, 0.0, 1.0)

    env.observation_space.dtype = np.float32
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(buffer_size)

    _sample_actions = make_sample_actions(env.action_space, exploration_noise)
    _sample_target_actions = make_sample_target_actions(
        env.action_space, exploration_noise, noise_clip
    )

    train_step = partial(train_step_with_loss, td3_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

    episode_idx = 0
    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q_target is None:
        q_target = nnx.clone(q)

    for global_step in trange(
        global_step, total_timesteps, disable=not progress_bar
    ):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, termination, truncated, info = env.step(action)
        steps_per_episode += 1
        accumulated_reward += reward

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        if global_step >= learning_starts:
            for _ in range(gradient_steps):
                batch = replay_buffer.sample_batch(batch_size, rng)

                # policy smoothing: sample next actions from target policy
                key, sampling_key = jax.random.split(key, 2)
                next_actions = _sample_target_actions(
                    policy_target, batch.next_observation, sampling_key
                )
                q_loss_value, q_mean = train_step(
                    q_optimizer,
                    q,
                    q_target,
                    next_actions,
                    batch,
                    gamma,
                )
                stats = {"q loss": q_loss_value, "q mean": q_mean}
                updated_modules = {"q": q}

                if global_step % policy_delay == 0:
                    policy_loss_value = ddpg_update_actor(
                        policy, policy_optimizer, q, batch.observation
                    )
                    soft_target_net_update(policy, policy_target, tau)
                    soft_target_net_update(q, q_target, tau)

                    stats["policy loss"] = policy_loss_value
                    updated_modules.update(
                        {
                            "policy": policy,
                            "policy_target": policy_target,
                            "q_target": q_target,
                        }
                    )

                if logger is not None:
                    for k, v in stats.items():
                        logger.record_stat(k, v, step=global_step + 1)
                    for k, v in updated_modules.items():
                        logger.record_epoch(k, v, step=global_step + 1)

        if termination or truncated:
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
        "TD3Result",
        [
            "policy",
            "policy_target",
            "policy_optimizer",
            "q",
            "q_target",
            "q_optimizer",
            "replay_buffer",
        ],
    )(
        policy,
        policy_target,
        policy_optimizer,
        q,
        q_target,
        q_optimizer,
        replay_buffer,
    )
