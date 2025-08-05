from collections import namedtuple
from functools import partial

import gymnasium
import jax
import numpy as np
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.mlp import MLP
from ..blox.losses import dqn_loss
from ..blox.q_policy import greedy_policy
from ..blox.replay_buffer import ReplayBuffer
from ..blox.schedules import linear_schedule
from ..logging.logger import LoggerBase


def train_step_with_loss(
    loss, optimizer: nnx.Optimizer, q: nnx.Module, *args, **kwargs
) -> tuple[float, float]:
    """Performs a single training step to optimize a Q-network.

    Parameters
    ----------
    loss : callable
        The loss function to be optimized, which should return a tuple with at
        least the loss value as the first element. The first argument of the
        loss should be the Q-network that will be optimized.
    optimizer : nnx.Optimizer
        The optimizer to be used.
    q : nnx.Module
        The deep Q-network to be optimized. It should be a Flax module that
        implements the forward pass.
    *args : tuple
        Arguments to be passed to the loss function.
    **kwargs : dict
        Keyword arguments to be passed to the loss function.

    Returns
    -------
    value : tuple
        Values returned by the loss function, typically including the loss
        value as the first element and possibly as the only element.
    """
    grad_fn = nnx.value_and_grad(loss, argnums=0, has_aux=True)
    value, grad = grad_fn(q, *args, **kwargs)
    optimizer.update(q, grad)
    return value


def train_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    replay_buffer: ReplayBuffer,
    optimizer: nnx.Optimizer,
    batch_size: int = 64,
    total_timesteps: int = 1e4,
    gamma: float = 0.99,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[MLP, nnx.Optimizer]:
    """Deep Q Learning with Experience Replay

    Implements the most basic version of DQN with experience replay as described
    in Mnih et al. (2013) [1]_, which is an off-policy value-based RL algorithm.
    It uses a neural network to approximate the Q-function and samples
    minibatches from the replay buffer to calculate updates.

    This implementation aims to be as close as possible to the original
    algorithm described in the paper while remaining not overly engineered
    towards a specific environment. For example, this implementation uses the
    same linear schedule to decrease epsilon from 1.0 to 0.1 over the first ten
    percent of training steps, but does not impose any architecture on the used
    Q-net or requires a specific preprocessing of observations as is done in
    the original paper to solve the Atari use case.

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
    total_timesteps : int
        The number of environment sets to train for.
    gamma : float
        The discount factor.
    seed : int
        The random seed, which can be set to reproduce results.
    logger : LoggerBase
        Logger for experiment tracking.
    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.


    Returns
    -------
    q_net : MLP
        The trained Q-network.
    optimizer : nnx.Optimizer
        The Q-net optimiser.

    References
    ----------
    .. [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I.,
       Wierstra, D., Riedmiller, M. (2013). Playing Atari with Deep
       Reinforcement Learning. https://arxiv.org/abs/1312.5602
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    if logger is not None:
        logger.start_new_episode()

    train_step = partial(train_step_with_loss, dqn_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

    # initialise episode
    obs, _ = env.reset(seed=seed)

    epsilon = linear_schedule(total_timesteps)

    key, subkey = jax.random.split(key)
    epsilon_rolls = jax.random.uniform(subkey, (total_timesteps,))

    episode = 1
    accumulated_reward = 0.0

    for step in trange(total_timesteps, disable=not progress_bar):
        if epsilon_rolls[step] < epsilon[step]:
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

        # sample minibatch from replay buffer
        if step > batch_size:
            transition_batch = replay_buffer.sample_batch(batch_size, rng)
            q_loss, q_mean = train_step(
                optimizer, q_net, transition_batch, gamma
            )
            if logger is not None:
                logger.record_stat(
                    "q loss", q_loss, step=step + 1, episode=episode
                )
                logger.record_stat(
                    "q mean", q_mean, step=step + 1, episode=episode
                )
                logger.record_epoch("q", q_net, step=step + 1, episode=episode)

        # housekeeping
        if terminated or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=step + 1, episode=episode
                )
            obs, _ = env.reset()
            accumulated_reward = 0.0
            episode += 1
        else:
            obs = next_obs

    return namedtuple("DQNResult", ["q_net", "optimizer"])(q_net, optimizer)
