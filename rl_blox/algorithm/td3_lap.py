from collections import namedtuple
from functools import partial

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm.rich import trange

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.losses import td3_lap_loss
from ..blox.replay_buffer import LAP, lap_priority
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import ddpg_update_actor, make_sample_actions
from .dqn import train_step_with_loss
from .td3 import make_sample_target_actions


def train_td3_lap(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    lap_alpha: float = 0.4,
    lap_min_priority: float = 1.0,
    learning_starts: int = 25_000,
    replay_buffer: LAP | None = None,
    policy_target: nnx.Module | None = None,
    q_target: ContinuousClippedDoubleQNet | None = None,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    ContinuousClippedDoubleQNet,
    ContinuousClippedDoubleQNet,
    nnx.Optimizer,
    LAP,
]:
    r"""TD3 with Loss-Adjusted Prioritized Experience Replay (LAP).

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

    replay_buffer : ReplayBuffer
        Replay buffer.

    policy_target : nnx.Module, optional
        Target policy. Only has to be set if we want to continue training
        from an old state.

    q_target : ContinuousDoubleQNet, optional
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
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    See Also
    --------
    .td3.train_td3
        TD3 without LAP.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    chex.assert_scalar_in(tau, 0.0, 1.0)

    env.observation_space.dtype = np.float32
    if replay_buffer is None:
        replay_buffer = LAP(buffer_size)

    _sample_actions = make_sample_actions(env.action_space, exploration_noise)
    _sample_target_actions = make_sample_target_actions(
        env.action_space, target_policy_noise, noise_clip
    )

    train_step = partial(train_step_with_loss, td3_lap_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma", "min_priority"))(
        train_step
    )

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q_target is None:
        q_target = nnx.clone(q)

    for global_step in trange(total_timesteps, disable=not progress_bar):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, termination, truncated, info = env.step(action)
        steps_per_episode += 1

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
                q_loss_value, (q_mean, max_abs_td_error) = train_step(
                    q_optimizer,
                    q,
                    q_target,
                    next_actions,
                    batch,
                    gamma,
                    lap_min_priority,
                )
                priority = lap_priority(
                    max_abs_td_error, lap_min_priority, lap_alpha
                )
                replay_buffer.update_priority(priority)

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

        if global_step % 250 == 0:  # hardcoded
            replay_buffer.reset_max_priority()

        if termination or truncated:
            if logger is not None:
                if "episode" in info:
                    logger.record_stat(
                        "return",
                        float(info["episode"]["r"]),
                        step=global_step + 1,
                    )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()
            steps_per_episode = 0
        else:
            obs = next_obs

    return namedtuple(
        "TD3LAPResult",
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
