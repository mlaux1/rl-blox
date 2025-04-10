from collections import namedtuple

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .ddpg import MLP, ReplayBuffer, action_value_loss, update_target


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

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        y, _ = self.net(observation)
        return nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from Gaussian distribution."""
        y, log_var = self.net(observation)
        mean = nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)
        return (
            jax.random.normal(key, mean.shape)
            # TODO compare to alternative approach from previous implementation
            * jnp.exp(jnp.clip(0.5 * log_var, -20.0, 2.0))
            + mean
        )

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        y, log_var = self.net(observation)
        mean = nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)
        # TODO compare to alternative approach from previous implementation
        log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
        std = jnp.exp(log_std)
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
    actions = policy.sample(observations, action_key)
    log_prob = policy.log_probability(observations, actions)
    obs_act = jnp.concatenate((observations, actions), axis=-1)
    qf1_pi = q1(obs_act).squeeze()
    qf2_pi = q2(obs_act).squeeze()
    min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
    actor_loss = (alpha * log_prob - min_qf_pi).mean()
    return actor_loss


def sac_exploration_loss(
    policy: StochasticPolicyBase,
    target_entropy: float,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
    log_alpha: dict,
) -> jnp.ndarray:
    actions = policy.sample(observations, action_key)
    log_prob = policy.log_probability(observations, actions)
    alpha_loss = (
        -jnp.exp(log_alpha["log_alpha"]) * (log_prob + target_entropy)
    ).mean()
    return alpha_loss


class EntropyControl:
    """Automatic entropy tuning."""

    alpha: jnp.ndarray
    autotune: bool
    target_entropy: jnp.ndarray
    log_alpha: dict[str, jnp.ndarray]
    optimizer: optax.adam
    optimizer_state: optax.OptState

    def __init__(self, env, alpha, autotune, learning_rate):
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -jnp.prod(jnp.array(env.action_space.shape))
            self.log_alpha = {"log_alpha": jnp.zeros(1)}
            self.alpha = jnp.exp(self.log_alpha["log_alpha"])
            self.optimizer = optax.adam(learning_rate=learning_rate)
            self.optimizer_state = self.optimizer.init(self.log_alpha)
        else:
            self.alpha = alpha

    def update(self, policy, observations, action_key):
        if not self.autotune:
            return 0.0

        exploration_loss, grad = jax.value_and_grad(
            sac_exploration_loss, argnums=4
        )(policy, self.target_entropy, action_key, observations, self.log_alpha)
        updates, self.optimizer_state = self.optimizer.update(
            grad, self.optimizer_state
        )
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        self.alpha = jnp.exp(self.log_alpha["log_alpha"])
        return exploration_loss


def create_sac_state(
    env: gym.Env[gym.spaces.Box, gym.spaces.Box],
    policy_shared_head: bool = False,
    policy_hidden_nodes: list[int] | tuple[int] = (256, 256),
    policy_learning_rate: float = 3e-4,
    q_hidden_nodes: list[int] | tuple[int] = (256, 256),
    q_learning_rate: float = 1e-3,
    seed: int = 0,
):
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
    verbose: int = 0,
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
    """Soft actor-critic.

    Parameters
    ----------
    env
        Gymnasium environment.
    policy
        Stochastic policy.
    q
        Soft Q network.
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
        State of entropy optimizer.
    verbose
        Verbosity level.

    Returns
    -------
    policy
    policy_optimizer
    q1
    q1_target
    q1_optimizer
    q2
    q2_target
    q2_optimizer
    entropy_control
    """
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    obs, _ = env.reset(seed=seed)

    if q1_target is None:
        q1_target = nnx.clone(q1)
    if q2_target is None:
        q2_target = nnx.clone(q2)

    if entropy_control is None:
        entropy_control = EntropyControl(
            env, alpha, autotune, entropy_learning_rate
        )

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

    obs, _ = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action = np.asarray(policy.sample(jnp.asarray(obs), action_key))

        next_obs, reward, termination, truncation, info = env.step(action)

        done = termination or truncation
        if done:
            if verbose and "episode" in info:
                # TODO implement logging here
                print(f"{global_step=}, episodic_return={info['episode']['r']}")

            obs, _ = env.reset()

        rb.add_sample(obs, action, reward, next_obs, termination)

        obs = next_obs

        if global_step > learning_starts:
            observations, actions, rewards, next_observations, dones = (
                rb.sample_batch(batch_size, rng)
            )

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
                dones,
                action_key,
                entropy_control.alpha,
            )

            if global_step % policy_frequency == 0:
                # compensate for delay by doing 'policy_frequency' updates
                for _ in range(policy_frequency):
                    key, action_key = jax.random.split(key, 2)
                    actor_loss_value = sac_update_actor(
                        policy,
                        policy_optimizer,
                        q1,
                        q2,
                        action_key,
                        observations,
                        entropy_control.alpha,
                    )
                    if autotune:
                        key, action_key = jax.random.split(key, 2)
                        exploration_loss_value = entropy_control.update(
                            policy, observations, key
                        )

            if global_step % target_network_frequency == 0:
                q1_target = update_target(q1, q1_target, tau)
                q2_target = update_target(q2, q2_target, tau)

            if verbose and global_step % 1_000 == 0:
                # TODO implement logging here
                print("losses/q1_loss", q1_loss_value, global_step)
                print("losses/q2_loss", q2_loss_value, global_step)
                print("losses/policy_loss", actor_loss_value, global_step)
                print("losses/alpha", entropy_control.alpha, global_step)
                if autotune:
                    print(
                        "losses/alpha_loss", exploration_loss_value, global_step
                    )

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
    dones: jnp.ndarray,
    action_key: jnp.ndarray,
    alpha: jnp.ndarray,
) -> tuple[float, float]:
    next_actions = policy.sample(next_observations, action_key)
    next_log_pi = policy.log_probability(next_observations, next_actions)
    next_obs_act = jnp.concatenate((next_observations, next_actions), axis=-1)
    q1_next_target = q1_target(next_obs_act).squeeze()
    q2_next_target = q2_target(next_obs_act).squeeze()
    min_q_next_target = (
        jnp.minimum(q1_next_target, q2_next_target) - alpha * next_log_pi
    )
    next_q_value = rewards + (1 - dones) * gamma * min_q_next_target

    q1_loss_value, q1_grads = nnx.value_and_grad(action_value_loss, argnums=3)(
        observations, actions, next_q_value, q1
    )
    q1_optimizer.update(q1_grads)
    q2_loss_value, q2_grads = nnx.value_and_grad(action_value_loss, argnums=3)(
        observations, actions, next_q_value, q2
    )
    q2_optimizer.update(q2_grads)

    return q1_loss_value, q2_loss_value
