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

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import DeterministicTanhPolicy
from ..logging.logger import LoggerBase


# TODO consolidate replay buffer implementations
class ReplayBuffer:
    buffer: OrderedDict[str, np.typing.NDArray[float]]

    def __init__(self, buffer_size: int, keys: list[str] | None = None):
        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "termination",
            ]
        self.buffer = OrderedDict()
        for k in keys:
            self.buffer[k] = np.empty(0, dtype=float)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

    def add_sample(self, **sample):
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape, dtype=float
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[jnp.ndarray]:
        indices = rng.integers(0, self.current_len, batch_size)
        return [jnp.asarray(self.buffer[k][indices]) for k in self.buffer]

    def __len__(self):
        return self.current_len


def mse_action_value_loss(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    q_target_values: jnp.ndarray,
    q: nnx.Module,
) -> jnp.ndarray:
    """Mean squared eror loss function for action-value function.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observation_features)
        Batch of observations.

    actions : array, shape (n_samples, n_action_dims)
        Batch of selected actions.

    q_target_values : array, shape (n_samples,)
        Actual action values that should be approximated.

    q : nnx.Module
        Q network.

    Returns
    -------
    loss : array, shape ()
        Mean squared distance between predicted and actual action values.
    """
    chex.assert_equal_shape_prefix((observations, actions), prefix_len=1)
    chex.assert_equal_shape_prefix(
        (observations, q_target_values), prefix_len=1
    )

    q_predicted = q(jnp.concatenate((observations, actions), axis=-1)).squeeze()
    chex.assert_equal_shape((q_predicted, q_target_values))

    return (
        2.0
        * optax.l2_loss(predictions=q_predicted, targets=q_target_values).mean()
    )


def deterministic_policy_value_loss(
    q: nnx.Module,
    observations: jnp.ndarray,
    policy: nnx.Module,
) -> jnp.ndarray:
    r"""Loss function for the deterministic policy of the actor.

    .. math::

        \mathcal{L}(\theta)
        =
        \frac{1}{N}
        \sum_{o \in \mathcal{D}}
        -Q_{\theta}(o, \pi(o))

    Parameters
    ----------
    q : nnx.Module
        Q network.

    observations : array, shape (n_samples, n_observation_features)
        Batch of observations.

    policy : nnx.Module
        Deterministic policy represented by neural network.

    Returns
    -------
    loss
        Negative value of the actions selected by the policy for the given
        observations.
    """
    return -q(
        jnp.concatenate((observations, policy(observations)), axis=-1)
    ).mean()


@nnx.jit
def ddpg_update_critic(
    policy_target: nnx.Module,
    q: nnx.Module,
    q_target: nnx.Module,
    q_optimizer: nnx.Optimizer,
    gamma: float,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    terminations: jnp.ndarray,
) -> float:
    r"""DDPG critic update.

    Uses the bootstrap estimate

    .. math::

        r_{t+1} + \gamma Q(o_{t+1}, \pi(o_{t+1}))

    based on the target network of :math:`Q,\pi` as a target value for the
    Q network update with a mean squared error loss.

    Parameters
    ----------
    policy_target : nnx.Module
        Target network of policy.

    q : nnx.Module
        Action-value function.

    q_target : nnx.Module
        Target network of q.

    q_optimizer : nnx.Optimizer
        Optimizer of q.

    gamma : float
        Discount factor of discounted infinite horizon return model.

    observations : array
        Observations :math:`o_t`.

    actions : array
        Actions :math:`a_t`.

    next_observations : array
        Next observations :math:`o_{t+1}`.

    rewards : array
        Rewards :math:`r_{t+1}`.

    terminations : array
        Indicates if a terminal state was reached in this step.

    Returns
    -------
    q_loss_value : float
        Loss value.

    See also
    --------
    mse_action_value_loss
        The mean squared error loss.
    """
    chex.assert_equal_shape_prefix((observations, actions), prefix_len=1)
    chex.assert_equal_shape((observations, next_observations))
    chex.assert_equal_shape_prefix((observations, rewards), prefix_len=1)
    chex.assert_equal_shape_prefix((observations, terminations), prefix_len=1)
    chex.assert_equal_shape((rewards, terminations))

    next_actions = policy_target(next_observations)
    q_target_next = q_target(
        jnp.concatenate((next_observations, next_actions), axis=-1)
    ).squeeze()
    q_bootstrap = rewards + (1 - terminations) * gamma * q_target_next

    q_loss_value, grads = nnx.value_and_grad(mse_action_value_loss, argnums=3)(
        observations, actions, q_bootstrap, q
    )
    q_optimizer.update(grads)

    return q_loss_value


@nnx.jit
def ddpg_update_actor(
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: nnx.Module,
    observations: jnp.ndarray,
) -> float:
    """DDPG actor update.

    See also
    --------
    deterministic_policy_value_loss
        The loss function used during the optimization step.
    """
    actor_loss_value, grads = nnx.value_and_grad(
        deterministic_policy_value_loss, argnums=2
    )(q, observations, policy)
    policy_optimizer.update(grads)
    return actor_loss_value


@nnx.jit
def update_target(net: nnx.Module, target_net: nnx.Module, tau: float) -> None:
    """Update target network inplace with Polyak averaging.

    Parameters
    ----------
    net : nnx.Module
        Live network.

    target_net : nnx.Module
        Target network.

    tau : float
        The step_size used to update the Polyak average, i.e., the coefficient
        with which the live network's parameters will be multiplied.
    """
    params = nnx.state(net)
    target_params = nnx.state(target_net)
    target_params = optax.incremental_update(params, target_params, tau)
    nnx.update(target_net, target_params)


def sample_actions(
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    action_scale: jnp.ndarray,
    exploration_noise: float,
    policy: DeterministicTanhPolicy,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Sample actions with deterministic policy and Gaussian action noise.

    Actions will be clipped to [action_low, action_high].
    """
    action = policy(obs)
    exploring_action = jax.random.multivariate_normal(
        key, action, jnp.diag(action_scale * exploration_noise)
    )
    return jnp.clip(exploring_action, action_low, action_high)


def create_ddpg_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_activation: str = "relu",
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_activation: str = "relu",
    q_learning_rate: float = 3e-4,
    seed: int = 0,
) -> namedtuple:
    """Create components for DDPG algorithm with default configuration."""
    env.action_space.seed(seed)

    policy_net = MLP(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        policy_hidden_nodes,
        policy_activation,
        nnx.Rngs(seed),
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate)
    )

    q = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
        q_activation,
        nnx.Rngs(seed),
    )
    q_optimizer = nnx.Optimizer(q, optax.adam(learning_rate=q_learning_rate))

    return namedtuple(
        "DDPGState",
        [
            "policy",
            "policy_optimizer",
            "q",
            "q_optimizer",
        ],
    )(policy, policy_optimizer, q, q_optimizer)


def train_ddpg(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy: nnx.Module,
    policy_optimizer: nnx.Optimizer,
    q: nnx.Module,
    q_optimizer: nnx.Optimizer,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    gradient_steps: int = 1,
    exploration_noise: float = 0.1,
    learning_starts: int = 25_000,
    policy_frequency: int = 2,
    policy_target: nnx.Optimizer | None = None,
    q_target: nnx.Optimizer | None = None,
    logger: LoggerBase | None = None,
) -> tuple[
    nnx.Module, nnx.Module, nnx.Optimizer, nnx.Module, nnx.Module, nnx.Optimizer
]:
    """Deep Deterministic Policy Gradients (DDPG).

    This is an off-policy actor-critic algorithm with a deterministic policy.
    The critic approximates the action-value function. The actor will maximize
    action values based on the approximation of the action-value function.

    DDPG [2]_ extends Deterministic Policy Gradients [1]_ to use neural
    networks as function approximators with target networks.

    Parameters
    ----------
    env: Vectorized Gymnasium environments.
    policy: Deterministic policy network.
    policy_optimizer: Optimizer for the policy network.
    q: Q network.
    q_optimizer: Optimizer for the Q network.
    seed: Seed for random number generators in Jax and NumPy.
    total_timesteps: Number of steps to execute in the environment.
    actor_learning_rate: Learning rate of the actor.
    q_learning_rate: Learning rate of the critic.
    buffer_size: Size of the replay buffer.
    gamma: Discount factor.
    tau
        Learning rate for polyak averaging of target policy and value function.
    batch_size
        Size of a batch during gradient computation.
    gradient_steps
        Number of gradient steps during one training phase.
    exploration_noise
        Exploration noise in action space. Will be scaled by half of the range
        of the action space.
    learning_starts
        Learning starts after this number of random steps was taken in the
        environment.
    policy_frequency
        The policy will only be updated after this number of steps. Target
        policy and value function will be updated with the same frequency. The
        value function will be updated after every step.
    policy_target
        Target policy. Only has to be set if we want to continue training
        from an old state.
    q_target
        Target network. Only has to be set if we want to continue training
        from an old state.
    logger : LoggerBase, optional
        Experiment logger.

    Returns
    -------
    policy
        Final policy.
    policy_target
        Target policy.
    policy_optimizer
        Policy optimizer.
    q
        Final state-action value function.
    q_target
        Target network.
    q_optimizer
        Optimizer for Q network.

    References
    ----------
    .. [1] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D. &
       Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms.
       In Proceedings of the 31st International Conference on Machine Learning,
       PMLR 32(1):387-395. https://proceedings.mlr.press/v32/silver14.html

    .. [2] Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T.,
       Tassa, Y., Silver, D. & Wierstra, D. (2016). Continuous control with
       deep reinforcement learning. In 4th International Conference on Learning
       Representations, {ICLR} 2016, San Juan, Puerto Rico, May 2-4, 2016,
       Conference Track Proceedings. http://arxiv.org/abs/1509.02971
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

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

    if logger is not None:
        logger.start_new_episode()
    obs, _ = env.reset(seed=seed)
    steps_per_episode = 0

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q_target is None:
        q_target = nnx.clone(q)

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

        rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=termination,
        )

        done = termination or truncated
        if done:
            if logger is not None:
                if "episode" in info:
                    logger.record_stat(
                        "return", info["episode"]["r"], step=global_step
                    )
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()

            obs, _ = env.reset()
            steps_per_episode = 0
            continue

        obs = next_obs

        if global_step > learning_starts:
            for _ in range(gradient_steps):
                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminations,
                ) = rb.sample_batch(batch_size, rng)

                q_loss_value = ddpg_update_critic(
                    policy_target,
                    q,
                    q_target,
                    q_optimizer,
                    gamma,
                    observations,
                    actions,
                    next_observations,
                    rewards,
                    terminations,
                )
                if logger is not None:
                    logger.record_stat("q loss", q_loss_value, step=global_step)
                    logger.record_epoch("q", q, step=global_step)

                if global_step % policy_frequency == 0:
                    actor_loss_value = ddpg_update_actor(
                        policy, policy_optimizer, q, observations
                    )

                    update_target(policy, policy_target, tau)
                    # TODO why is it updated less often than q?
                    update_target(q, q_target, tau)

                    if logger is not None:
                        logger.record_stat(
                            "policy loss", actor_loss_value, step=global_step
                        )
                        logger.record_epoch("policy", policy, step=global_step)
                        logger.record_epoch(
                            "policy_target", policy_target, step=global_step
                        )
                        logger.record_epoch(
                            "q_target", q_target, step=global_step
                        )

    return policy, policy_target, policy_optimizer, q, q_target, q_optimizer
