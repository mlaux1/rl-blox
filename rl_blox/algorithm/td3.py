from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax import nnx

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..logging.logger import LoggerBase
from .ddpg import (
    ReplayBuffer,
    ddpg_update_actor,
    mse_action_value_loss,
    sample_actions,
    soft_target_net_update,
)


@nnx.jit
def td3_update_critic(
    q1: nnx.Module,
    q1_target: nnx.Module,
    q1_optimizer: nnx.Optimizer,
    q2: nnx.Module,
    q2_target: nnx.Module,
    q2_optimizer: nnx.Optimizer,
    gamma: float,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    next_actions: jnp.ndarray,
    rewards: jnp.ndarray,
    terminations: jnp.ndarray,
) -> tuple[float, float]:
    r"""TD3 critic update.

    Parameters
    ----------
    q1 : nnx.Module
        Action-value function.

    q1_target : nnx.Module
        Target network of q.

    q1_optimizer : nnx.Optimizer
        Optimizer of q.

    q2 : nnx.Module
        Action-value function.

    q2_target : nnx.Module
        Target network of q.

    q2_optimizer : nnx.Optimizer
        Optimizer of q.

    gamma : float
        Discount factor of discounted infinite horizon return model.

    observations : array
        Observations :math:`o_t`.

    actions : array
        Actions :math:`a_t`.

    next_observations : array
        Next observations :math:`o_{t+1}`.

    next_actions : array
        Sampled target actions :math:`a_{t+1}`.

    rewards : array
        Rewards :math:`r_{t+1}`.

    terminations : array
        Indicates if a terminal state was reached in this step.

    Returns
    -------
    q1_loss_value : float
        Loss value.

    q2_loss_value : float
        Loss value.

    See also
    --------
    mse_action_value_loss
        The mean squared error loss.
    """
    q_bootstrap = double_q_deterministic_bootstrap_estimate(
        rewards,
        terminations,
        gamma,
        q1_target,
        q2_target,
        next_observations,
        next_actions,
    )

    q1_loss_value, grads = nnx.value_and_grad(mse_action_value_loss, argnums=3)(
        observations, actions, q_bootstrap, q1
    )
    q1_optimizer.update(grads)

    q2_loss_value, grads = nnx.value_and_grad(mse_action_value_loss, argnums=3)(
        observations, actions, q_bootstrap, q2
    )
    q2_optimizer.update(grads)

    return q1_loss_value, q2_loss_value


def double_q_deterministic_bootstrap_estimate(
    rewards: jnp.ndarray,
    terminations: jnp.ndarray,
    gamma: float,
    q1: nnx.Module,
    q2: nnx.Module,
    next_observations: jnp.ndarray,
    next_actions: jnp.ndarray,
) -> jnp.ndarray:
    r"""Bootstrap estimate of action-value function with deterministic policy.

    .. math::

        \mathbb{E}\left[R(o_t)\right]
        \approx
        r_{t+1}
        + \gamma \min(Q_1(o_{t+1}, a_{t+1}), Q_2(o_{t+1}, a_{t+1}))

    Parameters
    ----------
    rewards : array
        Observed reward.

    terminations : array
        Indicates if a terminal state was reached in this step.

    gamma : float
        Discount factor.

    q1 : nnx.Module
        Action-value function.

    q2 : nnx.Module
        Action-value function.

    next_observations : array
        Next observations.

    next_actions : array
        Sampled target actions :math:`a_{t+1}`.

    Returns
    -------
    double_q_bootstrap : array
        Double Q bootstrap estimate of action-value function.
    """
    obs_act = jnp.concatenate((next_observations, next_actions), axis=-1)
    q_value = jnp.minimum(q1(obs_act).squeeze(), q2(obs_act).squeeze())
    return rewards + (1 - terminations) * gamma * q_value


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
    """Sample target actions with truncated Gaussian noise.

    Actions will be clipped to [action_low, action_high].
    """
    action = policy(obs)
    noise = jax.random.multivariate_normal(
        key,
        jnp.zeros_like(action[0]),
        jnp.diag(action_scale * exploration_noise),
        shape=(action.shape[0],),
    )
    scaled_noise_clip = action_scale * noise_clip
    clipped_noise = jnp.clip(noise, -scaled_noise_clip, scaled_noise_clip)
    return jnp.clip(action + clipped_noise, action_low, action_high)


def create_td3_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 1e-3,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 1e-3,
    seed: int = 0,
) -> namedtuple:
    """Create components for SAC algorithm with default configuration."""
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

    q1 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )
    q1_optimizer = nnx.Optimizer(q1, optax.adam(learning_rate=q_learning_rate))

    q2 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed + 1),
    )
    q2_optimizer = nnx.Optimizer(q2, optax.adam(learning_rate=q_learning_rate))

    return namedtuple(
        "TD3State",
        [
            "policy",
            "policy_optimizer",
            "q1",
            "q1_optimizer",
            "q2",
            "q2_optimizer",
        ],
    )(policy, policy_optimizer, q1, q1_optimizer, q2, q2_optimizer)


def train_td3(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q1: nnx.Module,
    q1_optimizer: nnx.Optimizer,
    q2: nnx.Module,
    q2_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.2,
    noise_clip: float = 0.5,
    learning_starts: int = 25_000,
    policy_target: nnx.Optimizer | None = None,
    q1_target: nnx.Optimizer | None = None,
    q2_target: nnx.Optimizer | None = None,
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
]:
    """Twin Delayed DDPG (TD3).

    TD3 extends DDPG with three techniques to improve performance:

    1. Clipped Double Q-Learning to mitigate overestimation bias of the value
    2. Delayed policy updates, controlled by the parameter `policy_delay`
    3. Target policy smoothing, i.e., sampling from the behavior policy with
       clipped noise (parameter `noise_clip`) for the critic update.

    As TD3 is based on DDPG, it uses :func:`.ddpg.ddpg_update_actor` to update
    the policy and several details are the same.

    To update the critic, we use :func:`td3_update_critic`. The target values
    for the action value function are generated by
    :func:`double_q_deterministic_bootstrap_estimate` based on the minimum of
    the two target networks of the action-value functions `q1` and `q2`
    (clipped double Q-learning) and next actions samples with
    :func:`sample_target_actions` (target policy smoothing). The policy and
    the target networks are only updated after `policy_delay` steps (delayed
    policy updates).

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    policy : nnx.Module
        Deterministic policy network.

    policy_optimizer : nnx.Optimizer
        Optimizer for the policy network.

    q1 : nnx.Module
        First Q network.

    q1_optimizer: nnx.Optimizer
        Optimizer for q1.

    q2 : nnx.Module
        Second Q network.

    q2_optimizer : nnx.Optimizer
        Optimizer for q2.

    seed : int, optional
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int, optional
        Number of steps to execute in the environment.

    buffer_size : int, optional
        Size of the replay buffer.

    gamma : float, optional
        Discount factor.

    tau : float, optional
        Learning rate for polyak averaging of target policy and value function.

    policy_delay : int, optional
        Delayed policy updates. The policy is updated every `policy_delay`
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

    policy_target : nnx.Module, optional
        Target policy. Only has to be set if we want to continue training
        from an old state.

    q1_target : nnx.Module, optional
        Target network. Only has to be set if we want to continue training
        from an old state.

    q2_target : nnx.Module, optional
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

    q1 : nnx.Module
        Final state-action value function.

    q1_target : nnx.Module
        Target network.

    q1_optimizer
        Optimizer for Q network.

    q2 : nnx.Module
        Final state-action value function.

    q2_target : nnx.Module
        Target network.

    q2_optimizer : nnx.Optimizer
        Optimizer for Q network.

    References
    ----------
    .. [1] Fujimoto, S., Hoof, H. &amp; Meger, D.. (2018). Addressing Function
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
    _sample_target_actions = nnx.jit(
        partial(
            sample_target_actions,
            env.action_space.low,
            env.action_space.high,
            action_scale,
            exploration_noise,
            noise_clip,
        )
    )

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q1_target is None:
        q1_target = nnx.clone(q1)
    if q2_target is None:
        q2_target = nnx.clone(q2)

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

                # policy smoothing: sample next actions from target policy
                key, sampling_key = jax.random.split(key, 2)
                next_actions = _sample_target_actions(
                    policy_target, next_observations, sampling_key
                )
                q1_loss_value, q2_loss_value = td3_update_critic(
                    q1,
                    q1_target,
                    q1_optimizer,
                    q2,
                    q2_target,
                    q2_optimizer,
                    gamma,
                    observations,
                    actions,
                    next_observations,
                    next_actions,
                    rewards,
                    terminations,
                )

                if logger is not None:
                    logger.record_stat(
                        "q1 loss", q1_loss_value, step=global_step + 1
                    )
                    logger.record_epoch("q1", q1, step=global_step + 1)
                    logger.record_stat(
                        "q2 loss", q2_loss_value, step=global_step + 1
                    )
                    logger.record_epoch("q2", q2, step=global_step + 1)

                if global_step % policy_delay == 0:
                    actor_loss_value = ddpg_update_actor(
                        policy, policy_optimizer, q1, observations
                    )
                    soft_target_net_update(policy, policy_target, tau)
                    soft_target_net_update(q1, q1_target, tau)
                    soft_target_net_update(q2, q2_target, tau)
                    if logger is not None:
                        logger.record_stat(
                            "policy loss",
                            actor_loss_value,
                            step=global_step + 1,
                        )
                    logger.record_epoch("policy", policy, step=global_step + 1)
                    logger.record_epoch(
                        "policy_target", policy_target, step=global_step + 1
                    )
                    logger.record_epoch(
                        "q_target", q1_target, step=global_step + 1
                    )
                    logger.record_epoch(
                        "q_target", q2_target, step=global_step + 1
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

    return (
        policy,
        policy_target,
        policy_optimizer,
        q1,
        q1_target,
        q1_optimizer,
        q2,
        q2_target,
        q2_optimizer,
    )
