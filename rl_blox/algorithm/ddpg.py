from collections import namedtuple
from functools import partial

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax import nnx

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.losses import (
    deterministic_policy_gradient_loss,
    mse_action_value_loss,
)
from ..blox.replay_buffer import ReplayBuffer
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase


@nnx.jit
def ddpg_update_critic(
    policy_target: nnx.Module,
    q: nnx.Module,
    q_target: nnx.Module,
    q_optimizer: nnx.Optimizer,
    gamma: float,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    terminations: jnp.ndarray,
) -> float:
    r"""DDPG critic update.

    Uses the bootstrap estimate

    .. math::

        r_{t+1} + \gamma Q(o_{t+1}, \pi(o_{t+1}))

    based on the target network of :math:`Q,\pi` as a target value for the
    Q network update with a mean squared error loss.

    Parameters
    ----------
    policy_target : nnx.Module
        Target network of policy.

    q : nnx.Module
        Action-value function.

    q_target : nnx.Module
        Target network of q.

    q_optimizer : nnx.Optimizer
        Optimizer of q.

    gamma : float
        Discount factor of discounted infinite horizon return model.

    observations : array
        Observations :math:`o_t`.

    actions : array
        Actions :math:`a_t`.

    next_observations : array
        Next observations :math:`o_{t+1}`.

    rewards : array
        Rewards :math:`r_{t+1}`.

    terminations : array
        Indicates if a terminal state was reached in this step.

    Returns
    -------
    q_loss_value : float
        Loss value.

    See also
    --------
    mse_action_value_loss
        The mean squared error loss.
    """
    chex.assert_equal_shape_prefix((observations, actions), prefix_len=1)
    chex.assert_equal_shape((observations, next_observations))
    chex.assert_equal_shape_prefix((observations, rewards), prefix_len=1)
    chex.assert_equal_shape_prefix((observations, terminations), prefix_len=1)
    chex.assert_equal_shape((rewards, terminations))

    q_bootstrap = q_deterministic_bootstrap_estimate(
        policy_target, rewards, terminations, gamma, q_target, next_observations
    )

    q_loss_value, grads = nnx.value_and_grad(mse_action_value_loss, argnums=3)(
        observations, actions, q_bootstrap, q
    )
    q_optimizer.update(grads)

    return q_loss_value


def q_deterministic_bootstrap_estimate(
    policy: nnx.Module,
    rewards: jnp.ndarray,
    terminations: jnp.ndarray,
    gamma: float,
    q: nnx.Module,
    next_observations: jnp.ndarray,
) -> jnp.ndarray:
    r"""Bootstrap estimate of action-value function with deterministic policy.

    .. math::

        \mathbb{E}\left[R(o_t)\right]
        \approx
        r_{t+1} + \gamma Q(o_{t+1}, \pi(o_{t+1}))

    Parameters
    ----------
    policy : nnx.Module
        Deterministic policy for action selection.

    rewards : array
        Observed reward.

    terminations : array
        Indicates if a terminal state was reached in this step.

    gamma : float
        Discount factor.

    q : nnx.Module
        Action-value function.

    next_observations : array
        Next observations.

    Returns
    -------
    q_bootstrap : array
        Bootstrap estimate of action-value function.
    """
    next_actions = policy(next_observations)
    obs_act = jnp.concatenate((next_observations, next_actions), axis=-1)
    return rewards + (1 - terminations) * gamma * q(obs_act).squeeze()


@nnx.jit
def ddpg_update_actor(
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: nnx.Module,
    observation: jnp.ndarray,
) -> float:
    """DDPG actor update.

    See also
    --------
    .blox.losses.deterministic_policy_gradient_loss
        The loss function used during the optimization step.
    """
    actor_loss_value, grads = nnx.value_and_grad(
        deterministic_policy_gradient_loss, argnums=2
    )(q, observation, policy)
    policy_optimizer.update(grads)
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
    """Sample actions with deterministic policy and Gaussian action noise.

    Actions will be clipped to [action_low, action_high].
    """
    action = policy(obs)
    exploring_action = jax.random.multivariate_normal(
        key, action, jnp.diag(action_scale * exploration_noise)
    )
    return jnp.clip(exploring_action, action_low, action_high)


def create_ddpg_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 3e-4,
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
        policy, optax.adam(learning_rate=policy_learning_rate)
    )

    q = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )
    q_optimizer = nnx.Optimizer(q, optax.adam(learning_rate=q_learning_rate))

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
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.1,
    learning_starts: int = 25_000,
    policy_target: nnx.Optimizer | None = None,
    q_target: nnx.Optimizer | None = None,
    logger: LoggerBase | None = None,
) -> tuple[
    nnx.Module, nnx.Module, nnx.Optimizer, nnx.Module, nnx.Module, nnx.Optimizer
]:
    r"""Deep Deterministic Policy Gradients (DDPG).

    This is an off-policy actor-critic algorithm with a deterministic policy.
    The critic approximates the action-value function. The actor will maximize
    action values based on the approximation of the action-value function.

    DDPG [2]_ extends Deterministic Policy Gradients [1]_ to use neural
    networks as function approximators with target networks for the policy
    :math:`\pi` and the action-value function :math:`Q`.

    The deterministic target policy :math:`\mu(o) = a` that maps observations
    to actions has to be passed as the parameter `policy`. The function
    :func:`sample_actions` adds Gaussian noise to the target policy to create
    the behavior policy that is used during training for exploration.
    :func:`ddpg_update_actor` will update the policy with the
    `policy_optimizer` to minimize the :func:`deterministic_policy_value_loss`.

    The action-value function has to be passed as the parameter `q`. We use
    :func:`ddpg_update_critic` to update the action-value function with the
    `q_optimizer` and an `mse_action_value_loss` between the current
    prediction of `q` and the target value generated by
    :func:`q_deterministic_bootstrap_estimate` using the target networks of
    `policy` and `q`. The target networks will be updated with Polyak
    averaging in :func:`soft_target_net_update`.

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

    policy_target : nnx.Module
        Target policy. Only has to be set if we want to continue training
        from an old state.

    q_target : nnx.Module
        Target network. Only has to be set if we want to continue training
        from an old state.

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
    q : nnx.Module
        Final state-action value function.
    q_target : nnx.Module
        Target network.
    q_optimizer : nnx.Optimizer
        Optimizer for Q network.

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
    rb = ReplayBuffer(buffer_size)

    action_scale = 0.5 * (env.action_space.high - env.action_space.low)
    _sample_actions = nnx.jit(
        partial(
            sample_actions,
            env.action_space.low,
            env.action_space.high,
            action_scale,
            exploration_noise,
        )
    )

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q_target is None:
        q_target = nnx.clone(q)

    for global_step in tqdm.trange(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, termination, truncated, info = env.step(action)
        steps_per_episode += 1

        rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        if global_step >= learning_starts:
            for _ in range(gradient_steps):
                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminations,
                ) = rb.sample_batch(batch_size, rng)

                q_loss_value = ddpg_update_critic(
                    policy_target,
                    q,
                    q_target,
                    q_optimizer,
                    gamma,
                    observations,
                    actions,
                    next_observations,
                    rewards,
                    terminations,
                )
                actor_loss_value = ddpg_update_actor(
                    policy, policy_optimizer, q, observations
                )
                soft_target_net_update(policy, policy_target, tau)
                soft_target_net_update(q, q_target, tau)

                if logger is not None:
                    logger.record_stat(
                        "q loss", q_loss_value, step=global_step + 1
                    )
                    logger.record_epoch("q", q, step=global_step + 1)
                    logger.record_stat(
                        "policy loss", actor_loss_value, step=global_step + 1
                    )
                    logger.record_epoch("policy", policy, step=global_step + 1)
                    logger.record_epoch(
                        "policy_target", policy_target, step=global_step + 1
                    )
                    logger.record_epoch(
                        "q_target", q_target, step=global_step + 1
                    )

        if termination or truncated:
            if logger is not None:
                if "episode" in info:
                    logger.record_stat(
                        "return", info["episode"]["r"], step=global_step + 1
                    )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()
            steps_per_episode = 0
        else:
            obs = next_obs

    return policy, policy_target, policy_optimizer, q, q_target, q_optimizer
