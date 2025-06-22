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

from ..blox.checkpointing import CheckpointState, maybe_train_and_checkpoint
from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.embedding.sale import (
    SALE,
    ActorSALE,
    CriticSALE,
    DeterministicSALEPolicy,
    update_sale,
)
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..blox.replay_buffer import LAP
from ..blox.target_net import hard_target_net_update
from ..logging.logger import LoggerBase
from .ddpg import sample_actions
from .td3 import sample_target_actions


@dataclasses.dataclass
class ValueClippingState:
    min_value: float = 1e8
    max_value: float = -1e8
    min_target_value: float = 0.0
    max_target_value: float = 0.0


@partial(
    nnx.jit,
    static_argnames=[
        "gamma",
        "min_priority",
        "q_min",
        "q_max",
    ],
)
def td7_update_critic(
    fixed_embedding: SALE,
    fixed_embedding_target: SALE,
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
    min_priority: float,
    q_min: float,
    q_max: float,
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    r"""TD7 critic update.

    Uses ``critic_optimizer`` to update ``critic``. The target values
    :math:`y_i` are generated with

    .. math::

        y_i = r_i + (1 - t_i)
        \gamma
        \texttt{clip}(
        \min(
        Q_1(o_{i+1}, a_{i+1}, z^{sa_{i+1}}_{t-1}, z^{a_{i+1}}_{t-1}),
        Q_2(o_{i+1}, a_{i+1}, z^{sa_{i+1}}_{t-1}, z^{a_{i+1}}_{t-1})
        ),
        Q_{\min}, Q_{\max}),

    based on target network ``critic_target`` and next actions obtained with
    target policy smoothing. The values of the next state are clipped to the
    range defined by ``q_min`` and ``q_max``.

    For each Q network of the critic, we compute the Huber loss [1]_ between
    :math:`y_i` and :math:`Q(s, a, z^{sa}_t, z^s_t)`.

    Parameters
    ----------
    fixed_embedding : SALE
        Encoder with index t.

    fixed_embedding_target : SALE
        Encoder with index t-1.

    critic : CriticSALE
        Live critic.

    critic_target : CriticSALE
        Target critic.

    critic_optimizer : nnx.Optimizer
        Optimizer for critic.

    gamma : float
        Discount factor.

    observation : array
        Observations :math:`o_t`.

    action : array
        Actions :math:`a_t`.

    next_observation : array
        Next observations :math:`o_{t+1}`.

    next_action : array
        Sampled target actions :math:`a_{t+1}` obtained with target policy
        smoothing.

    reward : array
        Rewards :math:`r_{t+1}`.

    terminated : array
        Indicates if a terminal state was reached in this step.

    min_priority : float
        Value is used as delta for the Huber loss.

    q_min : float
        Minimum value.

    q_max : float
        Maximum value.

    Returns
    -------
    q_loss_value : float
        Loss value.

    max_abs_td_error : array
        Maximum of the absolute TD errors of the two Q networks.

    q_target : array
        Target values :math:`y_i` for the critic.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html
    """
    zsa, zs = fixed_embedding(observation, action)
    next_zsa, next_zs = fixed_embedding_target(next_observation, next_action)

    next_obs_act = jnp.concatenate((next_observation, next_action), axis=-1)
    q_next_target = critic_target(
        next_obs_act, zsa=next_zsa, zs=next_zs
    ).squeeze()
    # Extrapolation error is the tendency for deep value functions to
    # extrapolate to unrealistic values on state-action pairs which are rarely
    # seen in the dataset. Extrapolation error has a significant impact in
    # offline RL, where the RL agent learns from a given dataset rather than
    # collecting its own experience, as the lack of feedback on overestimated
    # values can result in divergence. Surprisingly, we observe a similar
    # phenomenon in online RL, when increasing the number of dimensions in the
    # state-action input to the value function. Our hypothesis is that the
    # state-action embedding zsa expands the action input and makes the value
    # function more likely to over-extrapolate on unknown actions. Fortunately,
    # extrapolation error can be comated in a straightforward manner in online
    # RL, where poor estimates are corrected by feedback from interacting with
    # the environment. Consequently, we only need to stabilize the value
    # estimate until the correction occurs. This can be achieved in SALE by
    # tracking the range of values in the dataset D (estimated over sampled
    # mini-batches during training), and then bounding the target as follows.
    q_next_target = jnp.clip(q_next_target, q_min, q_max)
    q_target = reward + (1 - terminated) * gamma * q_next_target

    def sum_of_qnet_losses(q: ContinuousClippedDoubleQNet):
        obs_act = jnp.concatenate((observation, action), axis=-1)
        q1_pred = q.q1(obs_act, zsa=zsa, zs=zs).squeeze()
        q2_pred = q.q2(obs_act, zsa=zsa, zs=zs).squeeze()
        max_abs_td_error = jnp.maximum(
            # TODO we compute these differences twice...
            jnp.abs(q1_pred - q_target),
            jnp.abs(q2_pred - q_target),
        )
        return (
            optax.huber_loss(
                predictions=q1_pred, targets=q_target, delta=min_priority
            ).mean()
            + optax.huber_loss(
                predictions=q2_pred, targets=q_target, delta=min_priority
            ).mean(),
            max_abs_td_error,
        )

    (q_loss_value, max_abs_td_error), grads = nnx.value_and_grad(
        sum_of_qnet_losses, has_aux=True
    )(critic)
    critic_optimizer.update(grads)

    return q_loss_value, max_abs_td_error, q_target


def deterministic_policy_gradient_loss_sale(
    embedding: SALE,
    critic: ContinuousClippedDoubleQNet,
    observation: jnp.ndarray,
    actor: ActorSALE,
) -> jnp.ndarray:
    r"""Deterministic policy gradient loss.

    In comparison to :func:`~.blox.losses.deterministic_policy_gradient_loss`,
    this function also takes a SALE encoder, and it computes the mean of the
    two critics.

    Parameters
    ----------
    embedding : SALE
        Encoder.

    critic : ContinuousClippedDoubleQNet
        Critic network.

    observation : array
        Batch of observations.

    actor : ActorSALE
        Actor that should be updated.

    Returns
    -------
    actor_loss : float
        Actor loss.
    """
    zs = embedding.state_embedding(observation)
    action = actor(observation, zs)
    zsa = embedding.state_action_embedding(
        jnp.concatenate((zs, action), axis=-1)
    )
    obs_act = jnp.concatenate((observation, actor(observation, zs)), axis=-1)
    # - to perform gradient ascent with a minimizer
    return -critic.mean(obs_act, zs=zs, zsa=zsa).mean()


@nnx.jit
def td7_update_actor(
    embedding: SALE,
    actor: ActorSALE,
    actor_optimizer: nnx.Optimizer,
    critic: ContinuousClippedDoubleQNet,
    observation: jnp.ndarray,
) -> float:
    """Update actor.

    Parameters
    ----------
    embedding : SALE
        Encoder.

    actor : ActorSALE
        Actor that should be updated.

    actor_optimizer : nnx.Optimizer
        Optimizer for actor.

    critic : ContinuousClippedDoubleQNet
        Critic network.

    observation : array
        Batch of observations.
    """
    actor_loss_value, grads = nnx.value_and_grad(
        deterministic_policy_gradient_loss_sale, argnums=3
    )(embedding, critic, observation, actor)
    actor_optimizer.update(grads)
    return actor_loss_value


def create_td7_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    n_embedding_dimensions=256,
    state_embedding_hidden_nodes: list[int] | tuple[int] = (256,),
    state_action_embedding_hidden_nodes: list[int] | tuple[int] = (256,),
    embedding_activation: str = "elu",
    embedding_learning_rate: float = 3e-4,
    policy_sa_encoding_nodes: int = 256,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_sa_encoding_nodes: int = 256,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "elu",
    q_learning_rate: float = 3e-4,
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

    policy_net = MLP(
        policy_sa_encoding_nodes + n_embedding_dimensions,
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        rngs,
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    actor = ActorSALE(
        policy,
        env.observation_space.shape[0],
        policy_sa_encoding_nodes,
        rngs,
    )
    actor_optimizer = nnx.Optimizer(
        actor, optax.adam(learning_rate=policy_learning_rate)
    )

    n_q_inputs = q_sa_encoding_nodes + 2 * n_embedding_dimensions
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
        q_sa_encoding_nodes,
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
        q_sa_encoding_nodes,
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
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    lap_alpha: float = 0.4,
    lap_min_priority: float = 1.0,
    use_checkpoints: bool = True,
    max_episodes_when_checkpointing: int = 20,
    steps_before_checkpointing: int = 750_000,
    reset_weight: float = 0.9,
    batch_size: int = 256,
    learning_starts: int = 25_000,
    replay_buffer: LAP | None = None,
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
    LAP,
]:
    r"""TD7.

    TD7 [1]_ is an extension of TD3 with the following tricks:

    * SALE: state-action learned embeddings, a method that learns embeddings
      jointly over both state and action by modeling the dynamics of the
      environment in latent space
    * checkpoints: similar to representation learning, early stopping and
      checkpoints are used to enhance the performance of a model
    * loss-adjusted prioritized experience replay

    The offline version of TD7 uses an additional behavior cloning loss. This
    is the reason why the algorithm is called TD7: TD3 + 4 additions.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    embedding : SALE
        SALE embedding network.

    embedding_optimizer : nnx.Optimizer
        Optimizer for embedding.

    actor : nnx.Module
        Maps state and zs to action.

    actor_optimizer : nnx.Optimizer
        Optimizer for actor.

    critic : ContinuousClippedDoubleQNet
        Maps state, action, and zsa to value.

    critic_optimizer: nnx.Optimizer
        Optimizer for critic.

    seed : int, optional
        Seed for random number generators in Jax and NumPy.

    total_timesteps : int, optional
        Number of steps to execute in the environment.

    buffer_size : int, optional
        Size of the replay buffer.

    gamma : float, optional
        Discount factor.

    target_delay : int, optional
        Delayed target net updates. The target nets are updated every
        ``target_delay`` steps.

    policy_delay : int, optional
        Delayed policy updates. The policy is updated every ``policy_delay``
        steps.

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

    use_checkpoints : bool, optional
        Checkpoint policy networks and delay training steps for policy
        assessment.

    max_episodes_when_checkpointing : int, optional
        Maximum number of assessment episodes when checkpointing.

    steps_before_checkpointing : int, optional
        Number of training epochs before checkpointing.

    reset_weight : float, optional
        Reset weight for best minimum return when checkpointing starts. When
        the performance varies a lot with the initial state, it might make
        sense to set the weight to a lower value in the range (0, 1].
        If the maximum return is always negative, it makes sense to set it to
        a value > 1.

    batch_size : int, optional
        Size of a batch during gradient computation.

    learning_starts : int, optional
        Learning starts after this number of random steps was taken in the
        environment.

    replay_buffer : LAP
        Replay buffer.

    actor_target : nnx.Module, optional
        Target actor. Only has to be set if we want to continue training
        from an old state.

    critic_target : ContinuousDoubleQNet, optional
        Target network. Only has to be set if we want to continue training
        from an old state.

    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    embedding : SALE
        State-action learned embedding or, if checkpointing is activated, last
        checkpoint.

    embedding_optimizer : nnx.Optimizer
        Optimizer for embedding.

    actor : nnx.Module
        Final actor or, if checkpointing is activated, last checkpoint.

    actor_target : nnx.Module
        Target actor.

    policy_optimizer : nnx.Optimizer
        Actor optimizer.

    critic : ContinuousClippedDoubleQNet
        Final critic.

    critic_target : ContinuousClippedDoubleQNet
        Target network.

    critic_optimizer : nnx.Optimizer
        Optimizer for critic.

    replay_buffer : LAP
        Replay buffer.

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

    A checkpoint is a snapshot of the parameters of a model, captured at a
    specific time during training. In RL, using the checkpoint of a policy that
    obtained a high reward during training, instead of the current policy,
    improves the stability of the performance at test time. For off-policy RL
    algorithms, the standard training paradigm is to train after each time step.
    However, this means that the policy changes throughout each episode, making
    it hard to evaluate the performance. Similar to many on-policy algorithms,
    TD7 keeps the policy fixed for several assessment episodes and then batches
    the training that would have occurred. In a similar manner to evolutionary
    approaches, we can use these assessment episodes to judge if the current
    policy outperforms the previous best policy and checkpoint accordingly.
    At evaluation time, the checkpoint policy is used, rather than the current
    policy. To preserve learning speed and sample efficiency, we use the minimum
    performance to assess a policy, which penalizes unstable policies.

    Loss-adjusted prioritized experience replay uses a prioritized replay buffer
    paired with the Huber loss for the value function.

    Implementation details:

    * ELU activation function is recommended for the critic.
    * The target networks are updated periodically with a hard update. This
      change in comparison to TD3 is necessary because of the fixed encoders.
    * To stabilize the value estimate we track the range of values in the
      dataset and then bound the target values. This is necessary because
      expanding the inputs to the value network with the embedding leads to
      extrapolation error.

    Logging

    * ``embedding loss`` - value of the loss function for ``embedding``
    * ``q loss`` - value of the loss function for ``q``
    * ``policy loss`` - value of the loss function for the actor
    * ``training steps`` - number of training steps in one batch update
    * ``min_value`` - minimum value of Q observed so far
    * ``min_target_value`` - less frequently updated target value
    * ``max_value`` - maximum value of Q observed so far
    * ``max_target_value`` - less frequently updated target value
    * ``episodes_since_udpate`` - number of assessment episodes since last
       actor update
    * ``timesteps_since_upate`` - number of environment steps since last actor
      update
    * ``max_episodes_before_update`` - maximum number of episodes allowed
      before next actor update
    * ``min_return`` - minimum return observed for current actor
    * ``best_min_return`` - best minimum return observed for any previous actor

    Checkpointing

    * ``actor_checkpoint`` - checkpoint of actor
    * ``fixed_embedding_checkpoint`` - checkpoint of corresponding embedding
    * ``embedding`` - SALE encoder
    * ``fixed_embedding`` - SALE encoder used with actor
    * ``q`` - clipped double Q network, critic
    * ``policy`` - target policy, actor
    * ``q_target`` - target network for the critic
    * ``policy_target`` - target network for the actor
    * ``fixed_embedding_target`` - target network for SALE encoder

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
    if replay_buffer is None:
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
            target_policy_noise,
            noise_clip,
        )
    )

    epoch = 0

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    if actor_target is None:
        actor_target = nnx.clone(actor)
    if critic_target is None:
        critic_target = nnx.clone(critic)

    fixed_embedding = nnx.clone(embedding)
    fixed_embedding_target = nnx.clone(embedding)

    policy = DeterministicSALEPolicy(fixed_embedding, actor)
    policy_target = DeterministicSALEPolicy(
        fixed_embedding_target, actor_target
    )

    if use_checkpoints:
        actor_checkpoint = nnx.clone(actor)
        fixed_embedding_checkpoint = nnx.clone(embedding)

    value_clipping_state = ValueClippingState()
    checkpoint_state = CheckpointState()

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
        accumulated_reward += reward

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        if use_checkpoints:
            # only train when not evaluating the checkpoint
            training_steps = 0
        else:
            training_steps = 1

        if (termination or truncated) and use_checkpoints:
            update_checkpoint, training_steps = maybe_train_and_checkpoint(
                checkpoint_state,
                steps_per_episode,
                accumulated_reward,
                epoch,
                reset_weight,
                max_episodes_when_checkpointing,
                steps_before_checkpointing,
            )
            if update_checkpoint:
                hard_target_net_update(actor, actor_checkpoint)
                hard_target_net_update(
                    fixed_embedding, fixed_embedding_checkpoint
                )
                epochs = {
                    "actor_checkpoint": actor_checkpoint,
                    "fixed_embedding_checkpoint": fixed_embedding_checkpoint,
                }
                if logger is not None:
                    for k, v in epochs.items():
                        logger.record_epoch(k, v, step=global_step + 1)
            if logger is not None:
                for k, v in checkpoint_state.__dict__.items():
                    logger.record_stat(k, v, step=global_step + 1)

        if global_step >= learning_starts:
            for delayed_train_step_idx in range(1, training_steps + 1):
                metrics = {}
                epochs = {}
                epoch += 1

                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminations,
                ) = replay_buffer.sample_batch(batch_size, rng)

                embedding_loss_value = update_sale(
                    embedding,
                    embedding_optimizer,
                    observations,
                    actions,
                    next_observations,
                )

                # policy smoothing: sample next actions from target policy
                key, sampling_key = jax.random.split(key, 2)
                next_actions = _sample_target_actions(
                    policy_target, next_observations, sampling_key
                )
                q_loss_value, max_abs_td_error, q_target = td7_update_critic(
                    fixed_embedding,
                    fixed_embedding_target,
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
                    lap_min_priority,
                    value_clipping_state.min_target_value,
                    value_clipping_state.max_target_value,
                )
                value_clipping_state.min_value = min(
                    value_clipping_state.min_value, float(q_target.min())
                )
                value_clipping_state.max_value = max(
                    value_clipping_state.max_value, float(q_target.max())
                )
                priority = (
                    jnp.maximum(max_abs_td_error, lap_min_priority) ** lap_alpha
                )
                replay_buffer.update_priority(priority)

                if epoch % policy_delay == 0:
                    actor_loss_value = td7_update_actor(
                        fixed_embedding,
                        actor,
                        actor_optimizer,
                        critic,
                        observations,
                    )
                    if logger is not None:
                        metrics["policy loss"] = actor_loss_value
                        epochs["policy"] = actor

                if epoch % target_delay == 0:
                    hard_target_net_update(actor, actor_target)
                    hard_target_net_update(critic, critic_target)
                    hard_target_net_update(
                        fixed_embedding, fixed_embedding_target
                    )
                    hard_target_net_update(embedding, fixed_embedding)

                    replay_buffer.reset_max_priority()
                    value_clipping_state.min_target_value = (
                        value_clipping_state.min_value
                    )
                    value_clipping_state.max_target_value = (
                        value_clipping_state.max_value
                    )

                    if logger is not None:
                        epochs["policy_target"] = actor_target
                        epochs["q_target"] = critic_target
                        epochs["fixed_embedding"] = fixed_embedding
                        epochs["fixed_embedding_target"] = (
                            fixed_embedding_target
                        )

                if logger is not None:
                    metrics["embedding loss"] = embedding_loss_value
                    metrics["q loss"] = q_loss_value
                    metrics.update(value_clipping_state.__dict__)
                    epochs["embedding"] = embedding
                    epochs["q"] = critic

                    log_step = (
                        global_step
                        + 1
                        - training_steps
                        + delayed_train_step_idx
                    )
                    for k, v in metrics.items():
                        logger.record_stat(k, v, step=log_step)
                    for k, v in epochs.items():
                        logger.record_epoch(k, v, step=log_step)
            if logger is not None and training_steps > 0:
                logger.record_stat(
                    "training steps", training_steps, step=global_step + 1
                )

        if termination or truncated:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=global_step + 1
                )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()

            steps_per_episode = 0
            accumulated_reward = 0.0
        else:
            obs = next_obs

    return namedtuple(
        "TD7Result",
        [
            "embedding",
            "embedding_optimizer",
            "actor",
            "actor_target",
            "actor_optimizer",
            "critic",
            "critic_target",
            "critic_optimizer",
            "replay_buffer",
        ],
    )(
        fixed_embedding_checkpoint if use_checkpoints else embedding,
        embedding_optimizer,
        actor_checkpoint if use_checkpoints else actor,
        actor_target,
        actor_optimizer,
        critic,
        critic_target,
        critic_optimizer,
        replay_buffer,
    )
