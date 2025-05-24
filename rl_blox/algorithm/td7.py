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
    mse_continuous_action_value_loss,
)
from ..blox.replay_buffer import ReplayBuffer
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase


def train_td7(
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
    """TD7.

    References
    ----------
    Fujimoto, S., Chang, W.D., Smith, E., Gu, S., Precup, D., Meger, D. (2023).
    For SALE: State-Action Representation Learning for Deep Reinforcement
    Learning. In Advances in Neural Information Processing Systems 36, pp.
    61573-61624. Available from
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html
    """
