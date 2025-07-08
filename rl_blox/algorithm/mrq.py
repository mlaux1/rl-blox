import dataclasses
from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax import nnx

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import make_sample_actions
from .td3 import make_sample_target_actions


def train_mrq(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    logger: LoggerBase | None = None,
) -> None:
    r"""Model-based Representation for Q-learning (MR.Q).

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    TODO

    References
    ----------
    .. [1] Fujimoto, S., D'Oro, P., Zhang, A., Tian, Y., Rabbat, M. (2025).
       Towards General-Purpose Model-Free Reinforcement Learning. In
       International Conference on Learning Representations (ICLR).
       https://openreview.net/forum?id=R1hIXdST22
    """
