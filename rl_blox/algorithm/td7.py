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


def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """TODO"""
    return x / jnp.maximum(np.mean(jnp.abs(x), axis=-1), eps)


class Embedding(nnx.Module):
    """SALE: state-action learned embedding.

    Although the embeddings are learned by considering the dynamics of the
    environment, their purpose is solely to improve the input to the value
    function and policy, and not to serve as a world model for planning or
    estimating rollouts.
    """

    state_embedding: nnx.Module
    state_action_embedding: nnx.Module

    def __init__(self, state_embedding: nnx.Module, state_action_embedding: nnx.Module):
        self.state_embedding = state_embedding
        self.state_action_embedding = state_action_embedding

    def __call__(self, state, action):
        zs = self.state_embedding(state)
        zs_action = jnp.concatenate((zs, action), axis=-1)
        zsa = self.state_action_embedding(zs_action)
        return zs, zsa


class Actor(nnx.Module):
    """TODO"""
    policy_net: nnx.Module
    l0: nnx.Linear

    def __init__(self, policy_net: nnx.Module, n_state_features: int, hidden_nodes: int, rngs: nnx.Rngs):
        self.policy_net = policy_net
        self.l0 = nnx.Linear(n_state_features, hidden_nodes, rngs=rngs)

    def __call__(self, state: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        state_hidden = avg_l1_norm(self.l0(state))
        state_hidden_zs = jnp.concatenate((state_hidden, zs), axis=-1)
        return self.policy_net(state_hidden_zs)


class Critic(nnx.Module):
    """TODO"""
    # see Figure 1
    # https://github.com/sfujim/TD7/blob/main/TD7.py#L115


def state_action_embedding_loss(
    embedding: Embedding,
    observation,
    action,
    next_observation,
):
    r"""Loss of state-action embedding.

    The encoders are jointly trained using the mean squared error (MSE) between
    the state-action embedding :math:`z^{sa}` and the embedding of the next
    state :math:`z^{s'}:

    .. math:

        \mathcal{L} = \frac{1}{N} \sum_i \left(
        z^{s_i,a_i} - \texttt{sg}(z^{s_i'}) \right)^2,

    where :math:`\texttt{sg}(\cdot)` denotes the stop-gradient operation.

    The embeddings are designed to model the underlying structure of the
    environment. However, they may not encompass all relevant information
    needed by the value function and policy, such as features related to the
    reward, current policy, or task horizon.
    """
    _, zsa = embedding(observation, action)
    zsp = jax.lax.stop_gradient(embedding.state_embedding(next_observation))
    return optax.squared_error(predictions=zsa, targets=zsp)


def create_td3_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    n_embedding_dimensions = 100,
    state_embedding_hidden_nodes: list[int] | tuple[int] = (256, 256),
    state_action_embedding_hidden_nodes: list[int] | tuple[int] = (256, 256),
    embedding_activation: str = "relu",
    embedding_learning_rate: float = 1e-3,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 1e-3,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 1e-3,
    seed: int = 0,
) -> namedtuple:
    """Create components for TD3 algorithm with default configuration."""
    env.action_space.seed(seed)

    rngs = nnx.Rngs(seed)

    state_embedding = MLP(
        env.observation_space.shape[0],
        n_embedding_dimensions,
        state_embedding_hidden_nodes,
        embedding_activation,
        rngs,
    )
    state_action_embedding = MLP(
        n_embedding_dimensions + env.action_space.shape[0],
        n_embedding_dimensions,
        state_action_embedding_hidden_nodes,
        embedding_activation,
        rngs,
    )
    embedding = Embedding(state_embedding, state_action_embedding)
    embedding_optimizer = nnx.Optimizer(
        embedding, optax.adam(learning_rate=embedding_learning_rate)
    )

    policy_net = MLP(
        n_embedding_dimensions + 100,  # TODO configurable (see below too)
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    actor = Actor(
        policy_net,
        env.observation_space.shape[0],
        100,  # TODO configurable
        rngs,
    )
    actor_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate)
    )

    n_q_inputs = 2 * n_embedding_dimensions + env.observation_space.shape[0] + env.action_space.shape[0]
    q = MLP(
        n_q_inputs,
        1,
        q_hidden_nodes,
        q_activation,
        rngs,
    )
    q_optimizer = nnx.Optimizer(q, optax.adam(learning_rate=q_learning_rate))

    return namedtuple(
        "TD7State",
        [
            "embedding",
            "embedding_optimizer",
            "policy",
            "policy_optimizer",
            "q",
            "q_optimizer",
        ],
    )(embedding, embedding_optimizer, policy, policy_optimizer, q, q_optimizer)


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
    r"""TD7.

    TD7 [1]_ is an extension of TD3 with the following tricks:

    * SALE: state-action learned embeddings, a method that learns embeddings
      jointly over both state and action by modeling the dynamics of the
      environment in latent space
    * checkpoints: similar to representation learning, early stopping and
      checkpoints are used to enhance the performance of a model
    * prioritized experience replay

    The offline version of TD7 uses an additional behavior cloning loss. This
    is the reason why the algorithm is called TD7: TD3 + 4 additions.

    Notes
    -----

    The objecctive of SALE is to discover learned embeddings
    :math:`z^{sa}, z^s` which capture relevant structure in the observation
    space, as well as the transition dynamics of the environment. SALE defines
    a pair of encoders :math:`(f, g)`:

    .. math::

        z^s := f(s), \quad z^{sa} := g(z^s, a).

    The embeddings are split into state and state-action components so that the
    encoders can be trained with a dynamics prediction loss that solely relies
    on the next state :math:`s'`, independent of the next action or current
    policy.

    References
    ----------
    .. [1] Fujimoto, S., Chang, W.D., Smith, E., Gu, S., Precup, D., Meger, D.
       (2023). For SALE: State-Action Representation Learning for Deep
       Reinforcement Learning. In Advances in Neural Information Processing
       Systems 36, pp. 61573-61624. Available from
       https://proceedings.neurips.cc/paper_files/paper/2023/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html
    """
