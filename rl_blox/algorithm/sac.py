from collections import namedtuple

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax import nnx
from jax.typing import ArrayLike

from ..logging.logger import LoggerBase
from .ddpg import MLP, ReplayBuffer, mse_action_value_loss, update_target


# TODO consolidate implementations
class GaussianMLP(nnx.Module):
    """Probabilistic neural network that predicts a Gaussian distribution.

    Parameters
    ----------
    shared_head
        All nodes of the last hidden layer are connected to mean AND log_std.

    n_features
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    rngs
        Random number generator.
    """

    shared_head: bool
    n_outputs: int
    hidden_layers: list[nnx.Linear]
    output_layers: list[nnx.Linear]

    def __init__(
        self,
        shared_head: bool,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.shared_head = shared_head
        self.n_outputs = n_outputs

        self.hidden_layers = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.output_layers = []
        if shared_head:
            self.output_layers.append(
                nnx.Linear(n_in, 2 * n_outputs, rngs=rngs)
            )
        else:
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for layer in self.hidden_layers:
            x = nnx.swish(layer(x))

        if self.shared_head:
            y = self.output_layers[0](x)
            mean, log_var = jnp.split(y, (self.n_outputs,), axis=-1)
        else:
            mean = self.output_layers[0](x)
            log_var = self.output_layers[1](x)

        return mean, log_var


# TODO merge with Gaussian policy from REINFORCE branch
class StochasticPolicyBase(nnx.Module):
    """Base class for probabilistic policies."""

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Compute action probabilities for given observation."""
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from policy given observation."""
        raise NotImplementedError("Subclasses must implement sample method.")

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        raise NotImplementedError(
            "Subclasses must implement log_probability method."
        )


# TODO merge with Gaussian policy from REINFORCE branch
class GaussianPolicy(StochasticPolicyBase):
    r"""Gaussian policy represented with a function approximator.

    The gaussian policy maps observations to mean and log variance of an
    action, hence, represents the distribution :math:`\pi(a|o)`.
    """

    net: nnx.Module
    """Underlying function approximator."""

    action_scale: nnx.Variable[jnp.ndarray]
    """Scales for each component of the action."""

    action_bias: nnx.Variable[jnp.ndarray]
    """Offset for each component of the action."""

    def __init__(self, policy_net: nnx.Module, action_space: gym.spaces.Box):
        self.net = policy_net
        self.action_scale = nnx.Variable(
            jnp.array((action_space.high - action_space.low) / 2.0)
        )
        self.action_bias = nnx.Variable(
            jnp.array((action_space.high + action_space.low) / 2.0)
        )

    def __call__(
        self, observation: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        y, log_var = self.net(observation)
        mean = nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)
        log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
        std = jnp.exp(log_std)
        return mean, std

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from Gaussian distribution."""
        mean, std = self(observation)
        return jax.random.normal(key, mean.shape) * std + mean

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        mean, std = self(observation)
        # same as
        # -jnp.log(std)
        # - 0.5 * jnp.log(2.0 * jnp.pi)
        # - 0.5 * ((action - mean) / std) ** 2
        return distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=std
        ).log_prob(action)


def sac_actor_loss(
    policy: StochasticPolicyBase,
    q1: nnx.Module,
    q2: nnx.Module,
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

    q1 : nnx.Module
        First action-value function.

    q2 : nnx.Module
        Second action-value function.

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
    qf1_pi = q1(obs_act).squeeze()
    qf2_pi = q2(obs_act).squeeze()
    min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
    actor_loss = (alpha * log_prob - min_qf_pi).mean()
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
                self._alpha, optax.adam(learning_rate=learning_rate)
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
    optimizer.update(grad)
    alpha = log_alpha()
    return exploration_loss, alpha


def create_sac_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_shared_head: bool = False,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
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
        nnx.Rngs(seed),
    )
    policy = GaussianPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate)
    )

    q1 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        nnx.Rngs(seed),
    )
    q1_optimizer = nnx.Optimizer(q1, optax.adam(learning_rate=q_learning_rate))

    q2 = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        nnx.Rngs(seed + 1),
    )
    q2_optimizer = nnx.Optimizer(q2, optax.adam(learning_rate=q_learning_rate))

    return namedtuple(
        "SACState",
        [
            "policy",
            "policy_optimizer",
            "q1",
            "q1_optimizer",
            "q2",
            "q2_optimizer",
        ],
    )(policy, policy_optimizer, q1, q1_optimizer, q2, q2_optimizer)


class SampleMeanVarStreamX:
    n: int
    mean: jnp.ndarray | None
    p: jnp.ndarray | None
    var: jnp.ndarray | None

    def __init__(self):
        self.n = 0
        self.mean = None
        self.p = None
        self.var = None

    def add_sample(self, x: ArrayLike, last_mean: jnp.ndarray | None = None):
        x = jnp.asarray(x)

        if self.n == 0:
            self.mean = jnp.zeros_like(x)
            self.p = jnp.zeros_like(x)

        self.n += 1
        if last_mean is None:
            last_mean = self.mean
        self.mean = self.mean + (x - self.mean) / self.n
        self.p = self.p + (x - last_mean) * (x - self.mean)
        if self.n >= 2:
            self.var = self.p / (self.n - 1)
        else:
            self.var = jnp.ones_like(x)

    def __repr__(self):
        return (
            f"SampleMeanVarStreamX("
            f"n={self.n}, "
            f"mean={self.mean}, "
            f"var={self.var})"
        )


class NormalizeObservationBase:
    def add_sample(self, x: ArrayLike):
        raise NotImplementedError(
            "Subclasses must implement add_sample method."
        )

    def transform(self, x: ArrayLike) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement transform method.")


class NormalizeObservationStreamX(NormalizeObservationBase):
    running_stats: SampleMeanVarStreamX

    def __init__(self):
        self.running_stats = SampleMeanVarStreamX()

    def add_sample(self, obs: ArrayLike):
        self.running_stats.add_sample(obs)

    def transform(self, obs: ArrayLike) -> jnp.ndarray:
        obs = jnp.asarray(obs)
        return (
            obs - jnp.broadcast_to(self.running_stats.mean, obs.shape)
        ) / jnp.broadcast_to(
            jnp.sqrt(self.running_stats.var + jnp.finfo(obs.dtype).eps),
            obs.shape,
        )

    def __repr__(self):
        return (
            f"NormalizeObservationStreamX(running_stats={self.running_stats})"
        )


class ScaleRewardBase:
    def add_sample(self, r: ArrayLike, terminal: bool):
        raise NotImplementedError(
            "Subclasses must implement add_sample method."
        )

    def transform(self, r: ArrayLike) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement transform method.")


class ScaleRewardStreamX(ScaleRewardBase):
    running_stats: SampleMeanVarStreamX
    u: jnp.ndarray | None
    gamma: float

    def __init__(self, gamma):
        self.running_stats = SampleMeanVarStreamX()
        self.u = None
        self.gamma = gamma

    def add_sample(self, r: ArrayLike, terminal: bool):
        if self.u is None:
            self.u = jnp.array(0.0)
        self.u = self.gamma * (1.0 - float(terminal)) * self.u + jnp.asarray(r)
        self.running_stats.add_sample(self.u, jnp.array(0.0))

    def transform(self, r: ArrayLike) -> jnp.ndarray:
        r = jnp.asarray(r)
        return r / jnp.sqrt(self.running_stats.var + jnp.finfo(r.dtype).eps)

    def __repr__(self):
        return f"ScaleRewardStreamX(running_stats={self.running_stats})"


def train_sac(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: StochasticPolicyBase,
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
    batch_size: int = 256,
    learning_starts: float = 5_000,
    entropy_learning_rate: float = 1e-3,
    policy_frequency: int = 2,
    target_network_frequency: int = 1,
    alpha: float = 0.2,
    autotune: bool = True,
    q1_target: nnx.Module | None = None,
    q2_target: nnx.Module | None = None,
    entropy_control: EntropyControl | None = None,
    logger: LoggerBase | None = None,
    observation_normalizer: NormalizeObservationBase | None = None,
    reward_scaler: ScaleRewardBase | None = None,
) -> tuple[
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    nnx.Module,
    nnx.Module,
    nnx.Optimizer,
    EntropyControl,
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
    relative importance of the optimal policy.

    In addition, this implementation allows to automatically tune the
    temperature :math:`\alpha.`, uses double Q learning [3]_, and uses target
    networks [4]_ for both Q networks. In total, there are four action-value
    networks q1, q1_target, q2, and q2_target as well as one policy network.

    Parameters
    ----------
    env
        Gymnasium environment.
    policy
        Stochastic policy.
    q1
        First soft Q network.
    q2
        Second soft Q network.
    seed : int
        Seed for random number generation.
    total_timesteps
        Total timesteps of the experiments.
    buffer_size
        The replay memory buffer size.
    gamma
        The discount factor gamma.
    tau : float, optional (default: 0.005)
        Target smoothing coefficient.
    batch_size
        The batch size of sample from the reply memory.
    learning_starts
        Timestep to start learning.
    policy_lr
        The learning rate of the policy network optimizer.
    entropy_learning_rate
        The learning rate of the Q network optimizer.
    policy_frequency
        Frequency of training policy (delayed).
    target_network_frequency
        The frequency of updates for the target networks.
    alpha
        Entropy regularization coefficient.
    autotune
        Automatic tuning of the entropy coefficient.
    q1_target
        Target network for q1.
    q2_target
        Target network for q2.
    entropy_control
        State of entropy tuning.
    logger : LoggerBase, optional
        Experiment logger.
    observation_normalizer
        Normalize observations with running statistics.
    reward_scaler
        Scale rewards with running statistics.

    Returns
    -------
    policy
        Final policy.
    policy_optimizer
        Policy optimizer.
    q1
        First soft Q network.
    q1_target
        Target network of q1.
    q1_optimizer
        Optimizer of q1.
    q2
        Second soft Q network.
    q2_target
        Target network of q2.
    q2_optimizer
        Optimizer of q2.
    entropy_control
        State of entropy tuning.

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

    .. [3] Hasselt, H. (2010). Double Q-learning. In Advances in Neural
       Information Processing Systems 23.
       https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html

    .. [4] Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015). Human-level
       control through deep reinforcement learning. Nature 518, 529–533.
       https://doi.org/10.1038/nature14236
    """
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    if q1_target is None:
        q1_target = nnx.clone(q1)
    if q2_target is None:
        q2_target = nnx.clone(q2)

    if entropy_control is None:
        entropy_control = EntropyControl(
            env, alpha, autotune, entropy_learning_rate
        )

    @nnx.jit
    def _sample_action(policy, obs, action_key):
        return policy.sample(obs, action_key)

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    if observation_normalizer is not None:
        observation_normalizer.add_sample(obs)
    steps_per_episode = 0

    for global_step in tqdm.trange(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_action(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, termination, truncation, info = env.step(action)
        if observation_normalizer is not None:
            observation_normalizer.add_sample(obs)
        if reward_scaler is not None:
            reward_scaler.add_sample(reward, termination)
        steps_per_episode += 1

        done = termination or truncation
        if done:
            if logger is not None:
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()
                if "episode" in info:
                    logger.record_stat("return", info["episode"]["r"])
            if observation_normalizer is not None:
                observation_normalizer.add_sample(next_obs)
            obs, _ = env.reset()
            steps_per_episode = 0

        rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        obs = next_obs

        if global_step > learning_starts:
            observations, actions, rewards, next_observations, terminations = (
                rb.sample_batch(batch_size, rng)
            )
            if observation_normalizer is not None:
                observations = observation_normalizer.transform(observations)
                next_observations = observation_normalizer.transform(
                    next_observations
                )
            if reward_scaler is not None:
                rewards = reward_scaler.transform(rewards)

            key, action_key = jax.random.split(key, 2)
            q1_loss_value, q2_loss_value = sac_update_critic(
                q1,
                q1_target,
                q1_optimizer,
                q2,
                q2_target,
                q2_optimizer,
                policy,
                gamma,
                observations,
                actions,
                rewards,
                next_observations,
                terminations,
                action_key,
                entropy_control.alpha_,
            )
            if logger is not None:
                logger.record_stat("q1 loss", q1_loss_value, step=global_step)
                logger.record_epoch("q1", q1)
                logger.record_stat("q2 loss", q2_loss_value, step=global_step)
                logger.record_epoch("q2", q2)

            if global_step % policy_frequency == 0:
                # compensate for delay by doing 'policy_frequency' updates
                for _ in range(policy_frequency):
                    key, action_key = jax.random.split(key, 2)
                    policy_loss_value = sac_update_actor(
                        policy,
                        policy_optimizer,
                        q1,
                        q2,
                        action_key,
                        observations,
                        entropy_control.alpha_,
                    )
                    key, action_key = jax.random.split(key, 2)
                    exploration_loss_value = entropy_control.update(
                        policy, observations, key
                    )
                    if logger is not None:
                        logger.record_stat(
                            "policy loss", policy_loss_value, step=global_step
                        )
                        logger.record_epoch("policy", policy, step=global_step)
                        logger.record_stat(
                            "alpha",
                            float(entropy_control.alpha_[0]),
                            step=global_step,
                        )
                        if autotune:
                            logger.record_stat(
                                "alpha loss",
                                exploration_loss_value,
                                step=global_step,
                            )
                            logger.record_epoch(
                                "alpha", alpha, step=global_step
                            )

            if global_step % target_network_frequency == 0:
                update_target(q1, q1_target, tau)
                logger.record_epoch("q1_target", q1_target, step=global_step)
                update_target(q2, q2_target, tau)
                logger.record_epoch("q2_target", q2_target, step=global_step)

    return (
        policy,
        policy_optimizer,
        q1,
        q1_target,
        q1_optimizer,
        q2,
        q2_target,
        q2_optimizer,
        entropy_control,
    )


@nnx.jit
def sac_update_actor(
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q1: nnx.Module,
    q2: nnx.Module,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
    alpha: jnp.ndarray,
) -> float:
    """SAC update of actor.

    See also
    --------
    sac_actor_loss
        The loss function used during the optimization step.
    """
    loss, grads = nnx.value_and_grad(sac_actor_loss, argnums=0)(
        policy, q1, q2, alpha, action_key, observations
    )
    policy_optimizer.update(grads)
    return loss


@nnx.jit
def sac_update_critic(
    q1: nnx.Module,
    q1_target: nnx.Module,
    q1_optimizer: nnx.Optimizer,
    q2: nnx.Module,
    q2_target: nnx.Module,
    q2_optimizer: nnx.Optimizer,
    policy: StochasticPolicyBase,
    gamma: float,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    terminations: jnp.ndarray,
    action_key: jnp.ndarray,
    alpha: jnp.ndarray,
) -> tuple[float, float]:
    r"""SAC update of critic.

    This function updates both q1 and q2.

    Uses the bootstrap estimate

    .. math::

        r_{t+1} + \gamma
        \left[\min(Q_1(o_{t+1}, a_{t+1}), Q_2(o_{t+1}, a_{t+1}))
        - \alpha \log \pi(a_{t+1}|o_{t+1})\right]

    based on the target networks of :math:`Q_1, Q_2` as a target value for the
    Q network update with a mean squared error loss.

    Parameters
    ----------
    q1 : nnx.Module
        First action-value network.

    q1_target : nnx.Module
        Target network of q1.

    q1_optimizer : nnx.Optimizer
        Optimizer of q1.

    q2 : nnx.Module
        Second action-value network.

    q2_target : nnx.Module
        Target network of q2.

    q2_optimizer : nnx.Optimizer
        Optimizer of q2.

    policy : StochasticPolicyBase
        Policy.

    gamma : float
        Discount factor of discounted infinite horizon return model.

    observations : array
        Observations :math:`o_t`.

    actions : array
        Actions :math:`a_t`.

    rewards : array
        Rewards :math:`r_{t+1}`.

    next_observations : array
        Next observations :math:`o_{t+1}`.

    terminations : array
        Indicates if a terminal state was reached in this step.

    action_key : array
        Random key for action sampling.

    alpha : float
        Entropy coefficient.

    Returns
    -------
    q1_loss_value : float
        Loss for q1.

    q2_loss_value : float
        Loss for q2.

    See also
    --------
    .ddpg.mse_action_value_loss
        The mean squared error loss.
    """
    next_actions = policy.sample(next_observations, action_key)
    next_log_pi = policy.log_probability(next_observations, next_actions)
    next_obs_act = jnp.concatenate((next_observations, next_actions), axis=-1)
    q1_next_target = q1_target(next_obs_act).squeeze()
    q2_next_target = q2_target(next_obs_act).squeeze()
    min_q_next_target = (
        jnp.minimum(q1_next_target, q2_next_target) - alpha * next_log_pi
    )
    q_target_value = rewards + (1 - terminations) * gamma * min_q_next_target

    q1_loss_value, q1_grads = nnx.value_and_grad(
        mse_action_value_loss, argnums=3
    )(observations, actions, q_target_value, q1)
    q1_optimizer.update(q1_grads)
    q2_loss_value, q2_grads = nnx.value_and_grad(
        mse_action_value_loss, argnums=3
    )(observations, actions, q_target_value, q2)
    q2_optimizer.update(q2_grads)

    return q1_loss_value, q2_loss_value
