from collections import namedtuple
from functools import partial

import gymnasium
import jax
import numpy as np
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.mlp import MLP
from ..blox.losses import nature_dqn_loss
from ..blox.q_policy import greedy_policy
from ..blox.replay_buffer import ReplayBuffer
from ..blox.schedules import linear_schedule
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .dqn import train_step_with_loss


def train_nature_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    replay_buffer: ReplayBuffer,
    optimizer: nnx.Optimizer,
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
    """Deep Q Learning with Experience Replay

    Implements the most common version of DQN with experience replay as described
    in Mnih et al. (2015) [1]_, which is an off-policy value-based RL algorithm.
    It uses a neural network to approximate the Q-function and samples
    minibatches from the replay buffer to calculate updates as well as target
    networks that are copied regularly from the current Q-network.

    This implementation aims to be as close as possible to the original algorithm
    described in the paper while remaining not overly engineered towards a
    specific environment. For example, this implementation uses the same linear
    schedule to decrease epsilon from 1.0 to 0.1 over the first ten percent of
    training steps, but does not impose any architecture on the used Q-net or
    requires a specific preprocessing of observations as is done in the original
    paper to solve the Atari use case.

    Parameters
    ----------
    q_net : MLP
        The Q-network to be optimised.
    env: gymnasium
        The environment to train the Q-network on.
    replay_buffer : ReplayBuffer
        The replay buffer used for storing collected transitions.
    optimizer : nnx.Optimizer
        The optimiser for the Q-Network.
    batch_size : int, optional
        Batch size for updates.
    update_frequency : int, optional
        The number of time steps after which the Q-net is updated.
    target_update_frequency : int, optional
        The number of time steps after which the target net is updated.
    total_timesteps : int
        The number of environment sets to train for.
    total_episodes : int, optional
        Total episodes for training. This is an alternative termination
        criterion for training. Set it to None to use ``total_timesteps`` or
        set it to a positive integer to overwrite the step criterion.
    gamma : float
        The discount factor.
    learning_starts : int
        Learning starts after this number of random steps was taken in the
        environment.
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
    .. [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control
       through deep reinforcement learning. Nature 518, 529–533 (2015).
       https://doi.org/10.1038/nature14236
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    # intialise the target network
    if q_target_net is None:
        q_target_net = nnx.clone(q_net)

    train_step = partial(train_step_with_loss, nature_dqn_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

    # initialise episode
    obs, _ = env.reset(seed=seed)

    epsilon = linear_schedule(total_timesteps)

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
                transition_batch = replay_buffer.sample_batch(batch_size, rng)
                q_loss, q_mean = train_step(
                    optimizer, q_net, q_target_net, transition_batch, gamma
                )
                if logger is not None:
                    logger.record_stat(
                        "q loss", q_loss, step=step + 1, episode=episode
                    )
                    logger.record_stat(
                        "q mean", q_mean, step=step + 1, episode=episode
                    )
                    logger.record_epoch(
                        "q", q_net, step=step + 1, episode=episode
                    )

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

    return namedtuple(
        "NatureDQNResult", ["q_net", "q_target_net", "optimizer"]
    )(q_net, q_target_net, optimizer)
