from collections import OrderedDict, namedtuple
from functools import partial

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax import nnx
from numpy import typing as npt

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .td3 import double_q_deterministic_bootstrap_estimate


class LAP:
    r"""Replay buffer for loss-adjusted prioritized experience replay (LAP).

    For each quantity, we store all samples in NumPy array that will be
    preallocated once the size of the quantities is know, that is, when the
    first transition sample is added. This makes sampling faster than when
    we use a deque.

    Loss-adjusted prioritized experience replay (LAP) [1]_ is based on
    prioritized experience replay (PER) [2]_. PER is a sampling scheme for
    replay buffers, in which transitions are sampled in proportion to their
    temporal-difference (TD) error. The intuitive argument behind PER is that
    training on the highest error samples will result in the largest
    performance gain.

    PER changes the traditional uniformly sampled replay buffers. The
    probability of sampling a transition i is proportional to the absolute TD
    error :math:`|\delta_i|`, set to the power of a hyper-parameter
    :math:`\alpha` to smooth out extremes:

    .. math::

        p(i)
        =
        \frac{|\delta_i|^{\alpha} + \epsilon}
        {\sum_j |\delta_j|^{\alpha} + \epsilon},

    where a small constant :math:`\epsilon` is added to ensure each transition
    is sampled with non-zero probability. This is necessary as often the
    current TD error is approximated by the TD error when i was last sampled.

    LAP changes this to

    .. math::

        p(i)
        =
        \frac{\max(|\delta_i|^{\alpha}, 1)}
        {\sum_j \max(|\delta_j|^{\alpha}, 1)},

    which leads to uniform sampling of transitions with a TD error smaller than
    1 to avoid the bias introduced from using MSE and prioritization. A LAP
    replay buffer is supposed to be paired with a Huber loss with a threshold
    of 1 to switch between MSE and L1 loss.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'termination']. These names have to be used as key word arguments when
        adding a sample. When sampling a batch, the arrays will be returned in
        this order.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    .. [2] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """

    buffer: OrderedDict[str, npt.NDArray[float]]
    buffer_size: int
    current_len: int
    insert_idx: int
    max_priority: float
    priority: npt.NDArray[float]
    sampled_indices: npt.NDArray[int]

    def __init__(
        self,
        buffer_size: int,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "termination",
            ]
        if dtypes is None:
            dtypes = [
                float,
                int if discrete_actions else float,
                float,
                float,
                int,
            ]
        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0
        self.max_priority = 1.0
        self.priority = np.empty(self.buffer_size)
        self.sampled_indices = np.empty(0, dtype=int)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape,
                    dtype=self.buffer[k].dtype,
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v
        self.priority[self.insert_idx] = self.max_priority
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.
        """
        self.sampled_indices = rng.choice(
            np.arange(self.current_len, dtype=int),
            size=batch_size,
            replace=False,
            p=self.priority[: self.current_len]
            / sum(self.priority[: self.current_len]),
        )
        return [
            jnp.asarray(self.buffer[k][self.sampled_indices])
            for k in self.buffer
        ]

    def update_priority(self, priority):
        self.priority[self.sampled_indices] = priority
        self.max_priority = max(max(priority), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = max(self.priority[: self.current_len])

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len


def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """TODO"""
    return x / jnp.maximum(np.mean(jnp.abs(x), axis=-1), eps)


class SALE(nnx.Module):
    """SALE: state-action learned embedding.

    Although the embeddings are learned by considering the dynamics of the
    environment, their purpose is solely to improve the input to the value
    function and policy, and not to serve as a world model for planning or
    estimating rollouts.
    """

    state_embedding: nnx.Module
    state_action_embedding: nnx.Module

    def __init__(
        self, state_embedding: nnx.Module, state_action_embedding: nnx.Module
    ):
        self.state_embedding = state_embedding
        self.state_action_embedding = state_action_embedding

    def __call__(self, state, action):
        zs = self.state_embedding(state)
        zs_action = jnp.concatenate((zs, action), axis=-1)
        zsa = self.state_action_embedding(zs_action)
        return zsa, zs


class ActorSALE(nnx.Module):
    """TODO"""

    policy_net: nnx.Module
    l0: nnx.Linear

    def __init__(
        self,
        policy_net: nnx.Module,
        n_state_features: int,
        hidden_nodes: int,
        rngs: nnx.Rngs,
    ):
        self.policy_net = policy_net
        self.l0 = nnx.Linear(n_state_features, hidden_nodes, rngs=rngs)

    def __call__(self, state: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        """pi(state, zs)."""
        h = avg_l1_norm(self.l0(state))
        he = jnp.concatenate(
            (h, zs), axis=-1
        )  # hidden_nodes + n_embedding_dimensions
        return self.policy_net(he)


class CriticSALE(nnx.Module):
    """TODO"""

    q_net: nnx.Module
    q0: nnx.Linear

    def __init__(
        self,
        q_net: nnx.Module,
        n_state_features: int,
        n_action_features: int,
        hidden_nodes: int,
        rngs: nnx.Rngs,
    ):
        self.q_net = q_net
        self.q0 = nnx.Linear(
            n_state_features + n_action_features, hidden_nodes, rngs=rngs
        )

    def __call__(
        self, sa: jnp.ndarray, zsa: jnp.ndarray, zs: jnp.ndarray
    ) -> jnp.ndarray:
        """Q(s, a, zsa, zs)."""
        h = avg_l1_norm(self.q0(sa))
        embeddings = jnp.concatenate((zsa, zs), axis=-1)
        he = jnp.concatenate(
            (h, embeddings), axis=-1
        )  # hidden_nodes + 2 * n_embedding_dimensions
        return self.q_net(he)


def state_action_embedding_loss(
    embedding: SALE,
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
    zsa, _ = embedding(observation, action)
    zsp = jax.lax.stop_gradient(embedding.state_embedding(next_observation))
    return optax.squared_error(predictions=zsa, targets=zsp).mean()


def sample_actions(
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    action_scale: jnp.ndarray,
    exploration_noise: float,
    embedding: SALE,
    actor: ActorSALE,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """TODO"""
    action = actor(obs, embedding.state_embedding(obs))
    eps = (
        exploration_noise * action_scale * jax.random.normal(key, action.shape)
    )
    exploring_action = action + eps
    return jnp.clip(exploring_action, action_low, action_high)


def sample_target_actions(
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    action_scale: jnp.ndarray,
    exploration_noise: float,
    noise_clip: float,
    embedding: SALE,
    actor: ActorSALE,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """TODO"""
    action = actor(obs, embedding.state_embedding(obs))
    eps = (
        exploration_noise * action_scale * jax.random.normal(key, action.shape)
    )
    scaled_noise_clip = action_scale * noise_clip
    clipped_eps = jnp.clip(eps, -scaled_noise_clip, scaled_noise_clip)
    return jnp.clip(action + clipped_eps, action_low, action_high)


@nnx.jit
def td7_update_embedding(
    embedding: SALE,
    embedding_optimizer: nnx.Optimizer,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
) -> float:
    """TODO"""
    embedding_loss_value, grads = nnx.value_and_grad(
        state_action_embedding_loss, argnums=0
    )(embedding, observations, actions, next_observations)
    embedding_optimizer.update(grads)
    return embedding_loss_value


@nnx.jit
def td7_update_critic(
    embedding: SALE,
    critic: ContinuousClippedDoubleQNet,
    critic_target: ContinuousClippedDoubleQNet,
    critic_optimizer: nnx.Optimizer,
    gamma: float,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    next_observation: jnp.ndarray,
    next_action: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    min_priority: int = 1,
) -> tuple[float, jnp.ndarray]:
    """TODO

    TODO

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html
    """
    zsa, zs = embedding(observation, action)
    next_zsa, next_zs = embedding(next_observation, next_action)

    q_bootstrap = double_q_deterministic_bootstrap_estimate(
        reward,
        terminated,
        gamma,
        critic_target,
        next_observation,
        next_action,
        additional_args=dict(zsa=next_zsa, zs=next_zs),
    )

    def sum_of_qnet_losses(q: ContinuousClippedDoubleQNet):
        q1_pred = q.q1(jnp.concatenate((observation, action), axis=-1), zsa=zsa, zs=zs).squeeze()
        q2_pred = q.q2(jnp.concatenate((observation, action), axis=-1), zsa=zsa, zs=zs).squeeze()
        abs_max_td_error = jnp.maximum(
            jnp.abs(q1_pred - q_bootstrap),  # TODO we compute these differences twice...
            jnp.abs(q2_pred - q_bootstrap),
        )
        return optax.huber_loss(
            predictions=q1_pred, targets=q_bootstrap, delta=min_priority
        ).mean() + optax.huber_loss(
            predictions=q2_pred, targets=q_bootstrap, delta=min_priority
        ).mean(), abs_max_td_error

    (q_loss_value, abs_max_td_error), grads = nnx.value_and_grad(
        sum_of_qnet_losses, has_aux=True
    )(critic)
    critic_optimizer.update(grads)

    return q_loss_value, abs_max_td_error


def deterministic_policy_gradient_loss(
    embedding: SALE,
    critic: ContinuousClippedDoubleQNet,
    observation: jnp.ndarray,
    actor: ActorSALE,
) -> jnp.ndarray:
    r"""TODO"""
    zs = embedding.state_embedding(observation)
    action = actor(observation, zs)
    zsa = embedding.state_action_embedding(
        jnp.concatenate((zs, action), axis=-1)
    )
    obs_act = jnp.concatenate((observation, actor(observation, zs)), axis=-1)
    # - to perform gradient ascent with a minimizer
    return -critic(obs_act, zs=zs, zsa=zsa).mean()


@nnx.jit
def td7_update_actor(
    embedding: SALE,
    actor: ActorSALE,
    actor_optimizer: nnx.Optimizer,
    critic: ContinuousClippedDoubleQNet,
    observation: jnp.ndarray,
) -> float:
    """TODO"""
    actor_loss_value, grads = nnx.value_and_grad(
        deterministic_policy_gradient_loss, argnums=3
    )(embedding, critic, observation, actor)
    actor_optimizer.update(grads)
    return actor_loss_value


def create_td7_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    n_embedding_dimensions=256,
    state_embedding_hidden_nodes: list[int] | tuple[int] = (256,),
    state_action_embedding_hidden_nodes: list[int] | tuple[int] = (256,),
    embedding_activation: str = "elu",
    embedding_learning_rate: float = 1e-3,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "elu",
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
    embedding = SALE(state_embedding, state_action_embedding)
    embedding_optimizer = nnx.Optimizer(
        embedding, optax.adam(learning_rate=embedding_learning_rate)
    )

    n_linear_encoding_nodes = policy_hidden_nodes[0]
    policy_net = MLP(
        n_linear_encoding_nodes + n_embedding_dimensions,
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    actor = ActorSALE(
        policy,
        env.observation_space.shape[0],
        n_linear_encoding_nodes,
        rngs,
    )
    actor_optimizer = nnx.Optimizer(
        actor, optax.adam(learning_rate=policy_learning_rate)
    )

    n_linear_encoding_nodes = q_hidden_nodes[0]
    n_q_inputs = n_linear_encoding_nodes + 2 * n_embedding_dimensions
    q1 = MLP(
        n_q_inputs,
        1,
        q_hidden_nodes,
        q_activation,
        rngs,
    )
    critic1 = CriticSALE(
        q1,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        n_linear_encoding_nodes,
        rngs,
    )
    q2 = MLP(
        n_q_inputs,
        1,
        q_hidden_nodes,
        q_activation,
        rngs,
    )
    critic2 = CriticSALE(
        q2,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        n_linear_encoding_nodes,
        rngs,
    )
    critic = ContinuousClippedDoubleQNet(critic1, critic2)
    critic_optimizer = nnx.Optimizer(
        critic, optax.adam(learning_rate=q_learning_rate)
    )

    return namedtuple(
        "TD7State",
        [
            "embedding",
            "embedding_optimizer",
            "actor",
            "actor_optimizer",
            "critic",
            "critic_optimizer",
        ],
    )(
        embedding,
        embedding_optimizer,
        actor,
        actor_optimizer,
        critic,
        critic_optimizer,
    )


def train_td7(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    embedding: SALE,
    embedding_optimizer: nnx.Optimizer,
    actor: ActorSALE,
    actor_optimizer: nnx.Optimizer,
    critic: ContinuousClippedDoubleQNet,
    critic_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    target_delay: int = 250,
    policy_delay: int = 2,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.2,
    noise_clip: float = 0.5,
    lap_alpha: float = 0.4,
    lap_min_priority: float = 1.0,
    learning_starts: int = 25_000,
    actor_target: ActorSALE | None = None,
    critic_target: ContinuousClippedDoubleQNet | None = None,
    logger: LoggerBase | None = None,
) -> tuple[
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
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    env.observation_space.dtype = np.float32
    replay_buffer = LAP(buffer_size)

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

    if actor_target is None:
        actor_target = nnx.clone(actor)
    if critic_target is None:
        critic_target = nnx.clone(critic)

    for global_step in tqdm.trange(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(embedding, actor, jnp.asarray(obs), action_key)
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
                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminations,
                ) = replay_buffer.sample_batch(batch_size, rng)

                # policy smoothing: sample next actions from target policy
                key, sampling_key = jax.random.split(key, 2)
                next_actions = _sample_target_actions(
                    embedding, actor_target, next_observations, sampling_key
                )
                q_loss_value, abs_td_error = td7_update_critic(
                    embedding,
                    critic,
                    critic_target,
                    critic_optimizer,
                    gamma,
                    observations,
                    actions,
                    next_observations,
                    next_actions,
                    rewards,
                    terminations,
                )
                priority = jnp.maximum(abs_td_error, lap_min_priority) ** lap_alpha
                replay_buffer.update_priority(priority)

                if logger is not None:
                    logger.record_stat(
                        "q loss", q_loss_value, step=global_step + 1
                    )
                    logger.record_epoch("q", critic, step=global_step + 1)

                if global_step % policy_delay == 0:
                    actor_loss_value = td7_update_actor(
                        embedding, actor, actor_optimizer, critic, observations
                    )
                    if logger is not None:
                        logger.record_stat(
                            "policy loss",
                            actor_loss_value,
                            step=global_step + 1,
                        )
                        logger.record_epoch("policy", actor, step=global_step + 1)

                embedding_loss_value = td7_update_embedding(
                    embedding,
                    embedding_optimizer,
                    observations,
                    actions,
                    next_observations,
                )
                if logger is not None:
                    logger.record_stat(
                        "embedding loss",
                        embedding_loss_value,
                        step=global_step + 1,
                    )
                    logger.record_epoch(
                        "embedding", actor, step=global_step + 1
                    )

                if global_step % target_delay == 0:
                    hard_target_net_update(actor, actor_target)
                    hard_target_net_update(critic, critic_target)

                    if logger is not None:
                        logger.record_epoch(
                            "policy_target", actor_target, step=global_step + 1
                        )
                        logger.record_epoch(
                            "q_target", critic_target, step=global_step + 1
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
        embedding,
        embedding_optimizer,
        actor,
        actor_target,
        actor_optimizer,
        critic,
        critic_target,
        critic_optimizer,
    )
