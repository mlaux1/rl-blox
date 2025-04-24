from collections import deque, namedtuple
from functools import partial

import chex
import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import nnx
from numpy.typing import ArrayLike


class ReplayBuffer:
    buffer: deque[tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

    def __init__(self, n_samples):
        self.buffer = deque(maxlen=n_samples)

    def add_sample(
        self, observation, action, reward, next_observation, terminated
    ):
        self.buffer.append(
            (observation, action, reward, next_observation, terminated)
        )

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in indices]
        observations = jnp.array([s[0] for s in samples])
        actions = jnp.array([s[1] for s in samples])
        rewards = jnp.array([s[2] for s in samples])
        next_observations = jnp.array([s[3] for s in samples])
        terminations = jnp.array([s[4] for s in samples])
        return observations, actions, rewards, next_observations, terminations


class MLP(nnx.Module):
    """Multilayer Perceptron.

    Parameters
    ----------
    n_features
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    rngs
        Random number generator.
    """

    n_outputs: int
    hidden_layers: list[nnx.Linear]
    output_layer: nnx.Linear

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs

        self.hidden_layers = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.output_layer = nnx.Linear(n_in, n_outputs, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = nnx.relu(
                layer(x)
            )  # TODO different activation in comparison to REINFORCE branch
        return self.output_layer(x)


class DeterministicPolicy(nnx.Module):
    r"""Deterministic policy represented with a function approximator.

    The deterministic policy directly maps observations to actions, hence,
    represents the function :math:`\pi(o) = a`.
    """

    policy_net: nnx.Module
    """Underlying function approximator."""

    action_scale: nnx.Variable[jnp.ndarray]
    """Scales for each component of the action."""

    action_bias: nnx.Variable[jnp.ndarray]
    """Offset for each component of the action."""

    def __init__(self, policy_net: nnx.Module, action_space: gym.spaces.Box):
        self.policy_net = policy_net
        self.action_scale = nnx.Variable(
            jnp.array((action_space.high - action_space.low) / 2.0)
        )
        self.action_bias = nnx.Variable(
            jnp.array((action_space.high + action_space.low) / 2.0)
        )

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        y = self.policy_net(observation)
        return nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)


def action_value_loss(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    q_target_values: jnp.ndarray,
    q: nnx.Module,
) -> jnp.ndarray:
    """Loss function for action-value function.

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
    loss
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
    """Loss function for the deterministic policy of the actor.

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
    """DDPG critic update."""
    chex.assert_equal_shape_prefix((observations, actions), prefix_len=1)
    chex.assert_equal_shape((observations, next_observations))
    chex.assert_equal_shape_prefix((observations, rewards), prefix_len=1)
    chex.assert_equal_shape_prefix((observations, terminations), prefix_len=1)
    chex.assert_equal_shape((rewards, terminations))

    # TODO why was it clipped to [-1, 1] before?
    next_actions = policy_target(next_observations)
    q_target_next = q_target(
        jnp.concatenate((next_observations, next_actions), axis=-1)
    ).squeeze()
    q_bootstrap = rewards + (1 - terminations) * gamma * q_target_next

    q_loss_value, grads = nnx.value_and_grad(action_value_loss, argnums=3)(
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
    """DDPG actor update."""
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
    policy: DeterministicPolicy,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Sample actions with deterministic policy and Gaussian action noise."""
    action = policy(obs)
    exploring_action = jax.random.multivariate_normal(
        key, action, jnp.diag(action_scale * exploration_noise)
    )
    return jnp.clip(exploring_action, action_low, action_high)


def create_ddpg_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_learning_rate: float = 3e-4,
    seed: int = 0,
):
    env.action_space.seed(seed)

    policy_net = MLP(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        policy_hidden_nodes,
        nnx.Rngs(seed),
    )
    policy = DeterministicPolicy(policy_net, env.action_space)
    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_learning_rate)
    )

    q = MLP(
        env.observation_space.shape[0] + env.action_space.shape[0],
        1,
        q_hidden_nodes,
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
    verbose: int = 0,
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
    verbose: Verbosity level.

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

    obs, _ = env.reset(seed=seed)

    if policy_target is None:
        policy_target = nnx.clone(policy)
    if q_target is None:
        q_target = nnx.clone(q)

    for t in range(total_timesteps):
        if t < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(
                _sample_actions(policy, jnp.asarray(obs), action_key)
            )

        next_obs, reward, terminated, truncated, info = env.step(action)

        rb.add_sample(obs, action, reward, next_obs, terminated)

        done = terminated or truncated
        if done:
            if verbose and "episode" in info:
                # TODO implement logging here
                print(
                    f"{t=}, length={info['episode']['l']}, "
                    f"return={info['episode']['r']}"
                )

            obs, _ = env.reset()
            continue

        obs = next_obs

        if t > learning_starts:
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
                if verbose >= 2:
                    print(f"{q_loss_value=}")
                    # TODO implement logging here
                    # TODO implement checkpointing here

                if t % policy_frequency == 0:
                    actor_loss_value = ddpg_update_actor(
                        policy, policy_optimizer, q, observations
                    )
                    if verbose >= 2:
                        print(f"{actor_loss_value=}")
                        # TODO implement logging here
                        # TODO implement checkpointing here

                    update_target(policy, policy_target, tau)
                    # TODO why is it updated less often than q?
                    update_target(q, q_target, tau)

    return policy, policy_target, policy_optimizer, q, q_target, q_optimizer
