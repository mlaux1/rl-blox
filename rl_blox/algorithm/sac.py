from collections import namedtuple
from functools import partial

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm.rich import trange

from ..blox.double_qnet import ContinuousClippedDoubleQNet
from ..blox.function_approximator.gaussian_mlp import GaussianMLP
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import (
    GaussianTanhPolicy,
    StochasticPolicyBase,
)
from ..blox.losses import sac_loss
from ..blox.replay_buffer import ReplayBuffer
from ..blox.target_net import soft_target_net_update
from ..logging.logger import LoggerBase
from .dqn import train_step_with_loss


def sac_actor_loss(
    policy: StochasticPolicyBase,
    q: ContinuousClippedDoubleQNet,
    alpha: float,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    r"""Actor loss for Soft Actor-Critic with double Q learning.

    .. math::

        \mathcal{L}(\theta)
        =
        \frac{1}{N}
        \sum_{o \in \mathcal{D}, a \sim \pi_{\theta}(a|o)}
        \alpha \log \pi_{\theta}(a|o)
        -
        \min(Q_1(o, a), Q_2(o, a))

    Parameters
    ----------
    policy : StochasticPolicyBase
        Policy.

    q : ContinuousClippedDoubleQNet
        Action-value function represented by double Q network.

    alpha : float
        Entropy coefficient.

    action_key : array
        Random key for action generation.

    observations : array, (n_observations,) + observation_space.shape
        Batch of observations.

    Returns
    -------
    actor_loss : array, shape ()
        Loss value.
    """
    actions = policy.sample(observations, action_key)
    log_prob = policy.log_probability(observations, actions)
    obs_act = jnp.concatenate((observations, actions), axis=-1)
    q_value = q(obs_act).squeeze()
    actor_loss = (alpha * log_prob - q_value).mean()
    return actor_loss


class EntropyCoefficient(nnx.Module):
    """Entropy coefficient alpha, internally represented by log of alpha."""

    log_alpha: nnx.Param[jnp.ndarray]

    def __init__(self, log_alpha: jnp.ndarray):
        self.log_alpha = nnx.Param(log_alpha)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_alpha.value)


def sac_exploration_loss(
    policy: StochasticPolicyBase,
    target_entropy: float,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
    alpha: EntropyCoefficient,
) -> jnp.ndarray:
    r"""Exploration loss used to update entropy coefficient alpha.

    Parameters
    ----------
    policy : StochasticPolicyBase
        Policy.

    target_entropy : float
        Target value for entropy.

    action_key : array
        Key for random sampling.

    observations : array, shape (n_observations,) + observation_space.shape
        Observations.

    alpha : EntropyCoefficient
        Entropy coefficient, internally represented by log alpha.

    Returns
    -------
    loss : array, shape ()
        Loss value.
    """
    actions = policy.sample(observations, action_key)
    log_prob = policy.log_probability(observations, actions)
    return (-alpha() * (log_prob + target_entropy)).mean()


class EntropyControl:
    """Automatic entropy tuning."""

    autotune: bool
    target_entropy: float
    _alpha: EntropyCoefficient
    alpha_: jnp.ndarray
    optimizer: nnx.Optimizer | None

    def __init__(self, env, alpha, autotune, learning_rate):
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -float(
                jnp.prod(jnp.array(env.action_space.shape))
            )
            self._alpha = EntropyCoefficient(jnp.zeros(1))
            self.alpha_ = self._alpha()
            self.optimizer = nnx.Optimizer(
                self._alpha,
                optax.adam(learning_rate=learning_rate),
                wrt=nnx.Param,
            )
        else:
            self.target_entropy = alpha
            self.alpha_ = alpha
            self.optimizer = None

    def update(self, policy, observations, action_key):
        """Update entropy coefficient alpha."""
        if not self.autotune:
            return 0.0

        exploration_loss, self.alpha_ = _update_entropy_coefficient(
            self.optimizer,
            policy,
            self.target_entropy,
            action_key,
            observations,
            self._alpha,
        )
        return exploration_loss


@nnx.jit
def _update_entropy_coefficient(
    optimizer,
    policy,
    target_entropy,
    action_key,
    observations,
    log_alpha,
):
    exploration_loss, grad = nnx.value_and_grad(
        sac_exploration_loss, argnums=4
    )(
        policy,
        target_entropy,
        action_key,
        observations,
        log_alpha,
    )
    optimizer.update(log_alpha, grad)
    alpha = log_alpha()
    return exploration_loss, alpha


def create_sac_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_shared_head: bool = False,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "swish",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 1e-3,
    seed: int = 0,
) -> namedtuple:
    """Create components for SAC algorithm with default configuration."""
    env.action_space.seed(seed)

    policy_net = GaussianMLP(
        policy_shared_head,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        nnx.Rngs(seed),
    )
    policy = GaussianTanhPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate), wrt=nnx.Param
    )

    q1 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )

    q2 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed + 1),
    )
    q = ContinuousClippedDoubleQNet(q1, q2)
    q_optimizer = nnx.Optimizer(
        q, optax.adam(learning_rate=q_learning_rate), wrt=nnx.Param
    )

    return namedtuple(
        "SACState",
        [
            "policy",
            "policy_optimizer",
            "q",
            "q_optimizer",
        ],
    )(policy, policy_optimizer, q, q_optimizer)


def train_sac(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    q_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    total_episodes: int | None = None,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    learning_starts: float = 5_000,
    entropy_learning_rate: float = 1e-3,
    policy_delay: int = 2,
    target_network_delay: int = 1,
    alpha: float = 0.2,
    autotune: bool = True,
    replay_buffer: ReplayBuffer | None = None,
    q_target: ContinuousClippedDoubleQNet | None = None,
    entropy_control: EntropyControl | None = None,
    logger: LoggerBase | None = None,
    global_step: int = 0,
    progress_bar: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    EntropyControl,
    ReplayBuffer,
]:
    r"""Soft actor-critic (SAC).

    Soft actor-critic [1]_ [2]_ is a maximum entropy algorithm, i.e., it
    optimizes (for :math:`\gamma=1`)

    .. math::

        \pi^*
        =
        \arg\max_{\pi} \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}}
        \left[
        r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))
        \right],

    where :math:`\alpha` is the temperature parameter that determines the
    relative importance of the entropy term :math:`\mathcal{H}` against the
    reward.

    In addition, this implementation allows to automatically tune the
    temperature :math:`\alpha.`, uses a
    :class:`~.blox.double_qnet.ContinuousClippedDoubleQNet`, and uses target
    networks [3]_ for both Q networks.

    Parameters
    ----------
    env : gymnasium.Env
        Gymnasium environment.

    policy : StochasticPolicyBase
        Stochastic policy.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy.

    q : ContinuousClippedDoubleQNet
        Clipped double soft Q network.

    q_optimizer : nnx.Optimizer
        Optimizer for critic.

    seed : int
        Seed for random number generation.

    total_timesteps : int
        Total timesteps for training.

    total_episodes : int, optional
        Total episodes for training. This is an alternative termination
        criterion for training. Set it to None to use ``total_timesteps`` or
        set it to a positive integer to overwrite the step criterion.

    buffer_size : int
        The replay memory buffer size.

    gamma : float, optional (default: 0.99)
        Discount factor.

    tau : float, optional (default: 0.005)
        Target smoothing coefficient.

    batch_size : int
        The batch size of sample from the reply memory.

    learning_starts : int
        Timestep to start learning.

    entropy_learning_rate : float
        The learning rate of the Q network optimizer.

    policy_delay : int
        Delayed policy updates. The policy is updated every ``policy_delay``
        steps.

    target_network_delay : int
        The target networks are updated every ``target_network_delay`` steps.

    alpha : float
        Entropy regularization coefficient.

    autotune : bool
        Automatic tuning of the entropy coefficient.

    replay_buffer : ReplayBuffer
        Replay buffer.

    q_target : ContinuousClippedDoubleQNet
        Target network for q.

    entropy_control : EntropyControl
        State of entropy tuning.

    logger : LoggerBase, optional
        Experiment logger.

    global_step : int, optional
        Global step to start training from. If not set, will start from 0.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    policy
        Final policy.
    policy_optimizer
        Policy optimizer.
    q
        Clipped double soft Q network.
    q_target
        Target network of q.
    q_optimizer
        Optimizer of q.
    entropy_control
        State of entropy tuning.
    replay_buffer : ReplayBuffer
        Replay buffer.

    Notes
    -----

    Parameters

    * :math:`\pi(a|o)` with weights :math:`\theta^{\pi}` - stochastic ``policy``
    * :math:`Q(o, a)` with weights :math:`\theta^{Q}` - critic network
      ``q``, composed of two q networks :math:`Q_i(o, a)` with index i
      (see :class:`~.blox.double_qnet.ContinuousClippedDoubleQNet`)
    * :math:`Q'(o, a)` with weights :math:`\theta^{Q'}` - target network
      ``q_target``, initialized as a copy of ``q``
    * :math:`R` - ``replay_buffer``

    Algorithm

    * Randomly sample ``learning_starts`` actions and record transitions in
      :math:`R`
    * For each step :math:`t`

      * Sample :math:`a_t \sim \pi(a_t|o_t)`
      * Take a step in the environment ``env`` and observe result
      * Store transition :math:`(o_t, a_t, r_t, o_{t+1}, d_{t+1})` in :math:`R`,
        where :math:`d` indicates if a terminal state was reached
      * Sample mini-batch of ``batch_size`` transitions from :math:`R` to
        update the networks
      * Update critic networks with :func:`~.blox.losses.sac_loss`
      * If ``t % policy_delay == 0``

        * Update actor ``policy_delay`` times with :func:`sac_update_actor`
        * Update temperature with :class:`EntropyControl`
      * If ``t % target_network_delay == 0``

        * Update target network :math:`Q'` with
          :func:`~.blox.target_net.soft_target_net_update`

    Logging

    * ``q loss`` - value of the loss function for ``q``
    * ``q mean`` - mean Q value of batch used to update the critic
    * ``policy loss`` - value of the loss function for the actor

    Checkpointing

    * ``q`` - critic
    * ``policy`` - target policy
    * ``q_target`` - target network for the critic

    References
    ----------
    .. [1] Haarnoja, T., Zhou, A., Abbeel, P. & Levine, P. (2018).
       Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
       Learning with a Stochastic Actor. In Proceedings of the 35th
       International Conference on Machine Learning, PMLR 80:1861-1870.
       https://proceedings.mlr.press/v80/haarnoja18b

    .. [2] Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S.,
       Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P. & Levine, P. (2018).
       Soft Actor-Critic Algorithms and Applications. arXiv.
       http://arxiv.org/abs/1812.05905

    .. [3] Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015). Human-level
       control through deep reinforcement learning. Nature 518, 529–533.
       https://doi.org/10.1038/nature14236
    """
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    chex.assert_scalar_in(tau, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    if q_target is None:
        q_target = nnx.clone(q)

    if entropy_control is None:
        entropy_control = EntropyControl(
            env, alpha, autotune, entropy_learning_rate
        )

    @nnx.jit
    def _sample_action(policy, obs, action_key):
        return policy.sample(obs, action_key)

    env.observation_space.dtype = np.float32
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(buffer_size)

    train_step = partial(train_step_with_loss, sac_loss)
    train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

    episode_idx = 0
    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0
    accumulated_reward = 0.0

    for global_step in trange(
        global_step, total_timesteps, disable=not progress_bar
    ):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_action(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, termination, truncation, info = env.step(action)
        steps_per_episode += 1
        accumulated_reward += reward

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        if global_step >= learning_starts:
            batch = replay_buffer.sample_batch(batch_size, rng)

            key, action_key = jax.random.split(key, 2)
            q_loss_value, q_mean = train_step(
                q_optimizer,
                q,
                q_target,
                policy,
                action_key,
                entropy_control.alpha_,
                batch,
                gamma,
            )
            stats = {"q loss": q_loss_value, "q mean": q_mean}
            updated_modules = {"q": q}

            if global_step % policy_delay == 0:
                # compensate for delay by doing 'policy_frequency' updates
                for _ in range(policy_delay):
                    key, action_key = jax.random.split(key, 2)
                    policy_loss_value = sac_update_actor(
                        policy,
                        policy_optimizer,
                        q,
                        action_key,
                        batch.observation,
                        entropy_control.alpha_,
                    )
                    stats["policy loss"] = policy_loss_value
                    updated_modules["policy"] = policy

                    key, action_key = jax.random.split(key, 2)
                    exploration_loss_value = entropy_control.update(
                        policy, batch.observation, key
                    )
                    if autotune:
                        stats["alpha"] = float(
                            jnp.array(entropy_control.alpha_).squeeze()
                        )
                        stats["alpha loss"] = exploration_loss_value

            if global_step % target_network_delay == 0:
                soft_target_net_update(q, q_target, tau)
                updated_modules["q_target"] = q_target

            if logger is not None:
                for k, v in stats.items():
                    logger.record_stat(k, v, step=global_step + 1)
                for k, v in updated_modules.items():
                    logger.record_epoch(k, v, step=global_step + 1)

        if termination or truncation:
            if logger is not None:
                logger.record_stat(
                    "return", accumulated_reward, step=global_step + 1
                )
                logger.stop_episode(steps_per_episode)
            episode_idx += 1
            if total_episodes is not None and episode_idx >= total_episodes:
                break
            if logger is not None:
                logger.start_new_episode()
            obs, _ = env.reset()
            steps_per_episode = 0
            accumulated_reward = 0.0
        else:
            obs = next_obs

    return namedtuple(
        "SACResult",
        [
            "policy",
            "policy_optimizer",
            "q",
            "q_target",
            "q_optimizer",
            "entropy_control",
            "replay_buffer",
        ],
    )(
        policy,
        policy_optimizer,
        q,
        q_target,
        q_optimizer,
        entropy_control,
        replay_buffer,
    )


@nnx.jit
def sac_update_actor(
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: ContinuousClippedDoubleQNet,
    action_key: jnp.ndarray,
    observation: jnp.ndarray,
    alpha: jnp.ndarray,
) -> float:
    """SAC update of actor.

    Uses ``policy_optimizer`` to update ``policy`` with the
    :func:`sac_actor_loss`.

    Parameters
    ----------
    policy : StochasticPolicyBase
        Policy.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy.

    q : nnx.Module
        Double soft Q network.

    action_key : jnp.ndarray
        PRNG Key for action sampling.

    observation : jnp.ndarray
        Observations from mini-batch.

    alpha : jnp.ndarray
        Entropy coefficient.

    Returns
    -------
    loss : float
        Actor loss.

    See also
    --------
    sac_actor_loss
        The loss function used during the optimization step.
    """
    loss, grads = nnx.value_and_grad(sac_actor_loss, argnums=0)(
        policy, q, alpha, action_key, observation
    )
    policy_optimizer.update(policy, grads)
    return loss
