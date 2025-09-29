from collections import namedtuple
from collections.abc import Callable
from functools import partial

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.losses import ddpg_loss, deterministic_policy_gradient_loss
from ..blox.replay_buffer import ReplayBuffer
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase
from .dqn import train_step_with_loss


@nnx.jit
def ddpg_update_actor(
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: nnx.Module,
    observation: jnp.ndarray,
) -> float:
    r"""DDPG actor update.

    Uses ``policy_optimizer`` to update ``policy`` with the
    :func:`~.blox.losses.deterministic_policy_gradient_loss`.

    Parameters
    ----------
    policy : nnx.Module
        Policy network for deterministic target policy :math:`\pi`.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy network.

    q : nnx.Module
        Critic network.

    observation : jnp.ndarray (batch_size,) + observation_space.shape
        Mini-batch of observations.

    Returns
    -------
    actor_loss_value : float
        Loss value.

    See Also
    --------
    .blox.losses.deterministic_policy_gradient_loss
        The loss function used during the optimization step.
    """
    actor_loss_value, grads = nnx.value_and_grad(
        deterministic_policy_gradient_loss, argnums=2
    )(q, observation, policy)
    policy_optimizer.update(policy, grads)
    return actor_loss_value


def sample_actions(
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    action_scale: jnp.ndarray,
    exploration_noise: float,
    policy: DeterministicTanhPolicy,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    r"""Sample actions with deterministic policy and Gaussian action noise.

    Given a deterministic policy :math:`\pi(o) = a`, we will generate an action

    .. math::

        a = \texttt{clip}(\pi(o) + \epsilon, a_{low}, a_{high})

    with added noise :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`
    (standard deviation ``action_scale * exploration_noise``) and clipped to
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
    exploring_action = action + eps
    return jnp.clip(exploring_action, action_low, action_high)


def make_sample_actions(
    action_space: gym.spaces.Box,
    exploration_noise: float,
) -> Callable[[nnx.Module, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    action_scale = 0.5 * (action_space.high - action_space.low)
    return nnx.jit(
        partial(
            sample_actions,
            action_space.low,
            action_space.high,
            action_scale,
            exploration_noise,
        )
    )


def create_ddpg_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 1e-3,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 1e-3,
    seed: int = 0,
) -> namedtuple:
    """Create components for DDPG algorithm with default configuration."""
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

    q = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )
    q_optimizer = nnx.Optimizer(
        q, optax.adam(learning_rate=q_learning_rate), wrt=nnx.Param
    )

    return namedtuple(
        "DDPGState",
        [
            "policy",
            "policy_optimizer",
            "q",
            "q_optimizer",
        ],
    )(policy, policy_optimizer, q, q_optimizer)


def train_ddpg(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: nnx.Module,
    q_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    total_episodes: int | None = None,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.1,
    learning_starts: int = 25_000,
    replay_buffer: ReplayBuffer | None = None,
    policy_target: nnx.Optimizer | None = None,
    q_target: nnx.Optimizer | None = None,
    logger: LoggerBase | None = None,
    global_step: int = 0,
    progress_bar: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    ReplayBuffer,
]:
    r"""Deep Deterministic Policy Gradients (DDPG).

    This is an off-policy actor-critic algorithm with a deterministic policy.
    The critic approximates the action-value function. The actor will maximize
    action values based on the approximation of the action-value function.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    policy : nnx.Module
        Deterministic policy network.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy network.

    q : nnx.Module
        Q network.

    q_optimizer : nnx.Optimizer
        Optimizer for the Q network.

    seed : int
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int
        Number of steps to execute in the environment.

    total_episodes : int, optional
        Total episodes for training. This is an alternative termination
        criterion for training. Set it to None to use ``total_timesteps`` or
        set it to a positive integer to overwrite the step criterion.

    buffer_size : int
        Size of the replay buffer.

    gamma : float
        Discount factor.

    tau : float
        Learning rate for polyak averaging of target policy and value function.

    batch_size : int
        Size of a batch during gradient computation.

    gradient_steps : int
        Number of gradient steps during one training phase.

    exploration_noise : float
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.

    learning_starts : int
        Learning starts after this number of random steps was taken in the
        environment.

    replay_buffer : ReplayBuffer
        Replay buffer.

    policy_target : nnx.Module
        Target policy. Only has to be set if we want to continue training
        from an old state.

    q_target : nnx.Module
        Target network. Only has to be set if we want to continue training
        from an old state.

    logger : LoggerBase, optional
        Experiment logger.

    global_step : int, optional
        Global step to start training from. If not set, will start from 0.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    policy : nnx.Module
        Final policy.
    policy_target : nnx.Module
        Target policy.
    policy_optimizer : nnx.Optimizer
        Policy optimizer.
    q : nnx.Module
        Final state-action value function.
    q_target : nnx.Module
        Target network.
    q_optimizer : nnx.Optimizer
        Optimizer for Q network.
    replay_buffer : ReplayBuffer
        Replay buffer.

    Notes
    -----

    DDPG [2]_ extends Deterministic Policy Gradients [1]_ to use neural
    networks as function approximators with target networks for the policy
    :math:`\pi` and the action-value function :math:`Q`.

    Parameters

    * :math:`\pi(o) = a` with weights :math:`\theta^{\pi}` - deterministic
      target ``policy``, maps observations to actions
    * :math:`\pi'` with weights :math:`\theta^{\pi'}` - policy target network
      (``policy_target``), initialized as a copy of ``policy``
    * :math:`Q(s, a)` with weights :math:`\theta^Q` - critic network ``q``
    * :math:`Q'(s, a)` with weights :math:`\theta^{Q'}` - Q target network
      ``q_target``, initialized as a copy of ``q``
    * :math:`R` - ``replay_buffer``

    Algorithm

    * Randomly sample ``learning_starts`` actions and record transitions in
      :math:`R`
    * For each step

      * Sample action with behavior policy in :func:`sample_actions`
      * Take a step in the environment ``env`` and observe result
      * Store transition :math:`(o_t, a_t, r_t, o_{t+1}, d_{t+1})` in :math:`R`,
        where :math:`d` indicates if a terminal state was reached
      * Sample mini-batch of ``batch_size`` transitions from :math:`R` to
        update the networks
      * Update critic with :func:`~.blox.losses.ddpg_loss`
      * Update actor with :func:`ddpg_update_actor`
      * Update target networks :math:`Q', \pi'` with
        :func:`~.blox.target_net.soft_target_net_update`

    Logging

    * ``q loss`` - value of the loss function for the critic
    * ``q mean`` - mean Q value of batch used to update the critic
    * ``policy loss`` - value of the loss function for the actor

    Checkpointing

    * ``q`` - critic
    * ``policy`` - target policy
    * ``q_target`` - target network for critic
    * ``policy_target`` - target network for policy

    References
    ----------
    .. [1] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D. &
       Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms.
       In Proceedings of the 31st International Conference on Machine Learning,
       PMLR 32(1):387-395. https://proceedings.mlr.press/v32/silver14.html

    .. [2] Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T.,
       Tassa, Y., Silver, D. & Wierstra, D. (2016). Continuous control with
       deep reinforcement learning. In 4th International Conference on Learning
       Representations, {ICLR} 2016, San Juan, Puerto Rico, May 2-4, 2016,
       Conference Track Proceedings. http://arxiv.org/abs/1509.02971
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

    train_step = partial(train_step_with_loss, ddpg_loss)
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

                q_loss_value, q_mean = train_step(
                    q_optimizer,
                    q,
                    q_target,
                    policy_target,
                    batch,
                    gamma,
                )
                actor_loss_value = ddpg_update_actor(
                    policy, policy_optimizer, q, batch.observation
                )
                soft_target_net_update(policy, policy_target, tau)
                soft_target_net_update(q, q_target, tau)

                if logger is not None:
                    stats = {
                        "q loss": q_loss_value,
                        "q mean": q_mean,
                        "policy loss": actor_loss_value,
                    }
                    for k, v in stats.items():
                        logger.record_stat(k, v, step=global_step + 1)
                    updated_modules = {
                        "q": q,
                        "q_target": q_target,
                        "policy": policy,
                        "policy_target": policy_target,
                    }
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
        "DDPGResult",
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
