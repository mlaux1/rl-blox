from collections import namedtuple
from functools import partial

import gymnasium
import jax
import numpy as np
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.mlp import MLP
from ..blox.losses import ddqn_per_loss
from ..blox.q_policy import greedy_policy
from ..blox.replay_buffer import PrioritizedReplayBuffer, per_priority
from ..blox.schedules import linear_schedule
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .dqn import train_step_with_loss


def train_ddqn_per(
    q_net: MLP,
    env: gymnasium.Env,
    replay_buffer: PrioritizedReplayBuffer,
    optimizer: nnx.Optimizer,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    batch_size: int = 64,
    total_timesteps: int = 1e4,
    total_episodes: int | None = None,
    gamma: float = 0.99,
    update_frequency: int = 4,
    target_update_frequency: int = 1000,
    learning_starts: int = 0,
    q_target_net: MLP | None = None,
    seed: int = 1,
    logger: LoggerBase | None = None,
    global_step: int = 0,
    progress_bar: bool = True,
) -> tuple[MLP, MLP, nnx.Optimizer]:
    r"""Prioritized Experience Replay

    Prioritized experience replay (PER) has two variants of priority, i.e., 
    proportional and rank-based prioritization. This implementation uses 
    the more common proportional variant as described in Schaul et al.(2016)
    [1]_.

    Transitions are sampled based on their priority, which is computed from
    their TD‑error. Those with larger errors are replayed more frequently, 
    in contrast to uniform sampling used in double DQN. New transitions, whose
    TD‑error is not yet known, are initialized with the maximum priority to
    ensure they are experienced at least once. When a transition is replayed,
    its TD‑error is updated accordingly. To correct for the bias introduced by
    this non‑uniform sampling, importance‑sampling weights are applied to the
    loss.    

    This implementation aims to be as close as possible to the original algorithm
    described in the paper while remaining not overly engineered towards a
    specific environment. For example, this implementation uses the same linear
    schedule to anneal beta from 0.4 to 1.0, but does not impose any architecture
    on the used Q-net or requires a specific preprocessing of observations as is
    done in the original paper to solve the Atari use case.

    Parameters
    ----------
    q_net : MLP
        The Q-network to be optimised.
    env: gymnasium
        The environment to train the Q-network on.
    replay_buffer : PrioritizedReplayBuffer
        The replay buffer used for storing collected transitions and it's corresponding priority.
    optimizer : nnx.Optimizer
        The optimiser for the Q-Network.
    per_alpha: float
        Initial value of prioritization exponent :math:`\alpha`.
        In proportional variant, :math:`\alpha` is a contant. In rank-based 
        variant, :math:`\alpha` decreases to 0.
    per_beta: float
        Initial value of importance-sampling correction exponent :math:`\beta`.
        In proportional variant, :math:`\alpha` anneal to 1. In rank-based 
        variant, :math:`\alpha` is constant 0.
    update_frequency : int, optional
        The number of time steps after which the Q-net is updated.
    target_update_frequency : int, optional
        The number of time steps after which the target net is updated.
    learning_starts : int
        Learning starts after this number of random steps was taken in the
        environment.
    batch_size : int, optional
        Batch size for updates.
    total_timesteps : int
        The number of environment sets to train for.
    total_episodes : int, optional
        Total episodes for training. This is an alternative termination
        criterion for training. Set it to None to use ``total_timesteps`` or
        set it to a positive integer to overwrite the step criterion.
    gamma : float
        The discount factor.
    q_target_net : MLP, optional
        The target Q-network. Only needed when continuing prior training.
    seed : int
        The random seed, which can be set to reproduce results.
    logger : LoggerBase, optional
        Experiment Logger.
    global_step : int, optional
        Global step to start training from. If not set, will start from 0.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    q_net : MLP
        The trained Q-network.
    optimizer : nnx.Optimizer
        The Q-net optimiser.
    q_target_net : MLP
        The current target Q-network (required for continuing training).

    References
    ----------
    .. [1] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    if logger is not None:
        logger.start_new_episode()

    # intialise the target network
    if q_target_net is None:
        q_target_net = nnx.clone(q_net)

    train_step = partial(train_step_with_loss, ddqn_per_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

    # initialise episode
    obs, _ = env.reset(seed=seed)

    epsilon = linear_schedule(total_timesteps)
    beta = linear_schedule(total_timesteps, start=per_beta, end=1.0, fraction=1.0)

    # TODO: rank-based PER
    # alpha = linear_schedule(total_timesteps, start=0.5, end=0.0, fraction=1.0)
    # beta = 0.0

    key, subkey = jax.random.split(key)
    epsilon_rolls = jax.random.uniform(subkey, (total_timesteps,))

    episode = 1
    accumulated_reward = 0.0

    for step in trange(global_step, total_timesteps, disable=not progress_bar):
        if step < learning_starts or epsilon_rolls[step] < epsilon[step]:
            action = env.action_space.sample()
        else:
            action = greedy_policy(q_net, obs)

        next_obs, reward, terminated, truncated, info = env.step(int(action))
        accumulated_reward += reward
        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=terminated,
        )

        if step > batch_size:
            if step % update_frequency == 0:
                transition_batch, is_ratio = replay_buffer.sample_batch(batch_size, rng, beta[step])

                weighted_loss, (q_mean, abs_td_error) = train_step(
                    optimizer, q_net, q_target_net, transition_batch, gamma, is_ratio
                )
                if logger is not None:
                    logger.record_stat(
                        "weighted loss", weighted_loss, step=step + 1, episode=episode
                    )
                    logger.record_stat(
                        "abs td error", abs_td_error, step=step + 1, episode=episode
                    )
                    logger.record_stat(
                        "q mean", q_mean, step=step + 1, episode=episode
                    )
                    logger.record_epoch(
                        "q", q_net, step=step + 1, episode=episode
                    )
                priority = per_priority(
                    abs_td_error, alpha=per_alpha, epsion=1e-6
                )
                replay_buffer.update_priority(priority)

            if step % target_update_frequency == 0:
                hard_target_net_update(q_net, q_target_net)

        # housekeeping
        if terminated or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=step + 1, episode=episode
                )
            obs, _ = env.reset()
            accumulated_reward = 0.0
            if total_episodes is not None and episode >= total_episodes:
                break
            episode += 1
        else:
            obs = next_obs

    return namedtuple("PERResult", ["q_net", "q_target_net", "optimizer"])(
        q_net, q_target_net, optimizer
    )
