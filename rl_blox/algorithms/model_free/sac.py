from functools import partial

import chex
import distrax
import flax
from flax import nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ddpg import ReplayBuffer, critic_loss


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


class GaussianPolicy(nnx.Module):
    r"""Gaussian policy represented with a function approximator.

    The gaussian policy maps observations to mean and log variance of an
    action, hence, represents the distribution :math:`\pi(a|o)`.
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


def sample_actions(
    policy: nnx.Module,
    obs: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mean, log_std = policy(obs)
    std = jnp.exp(log_std)
    normal = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
    # for reparameterization trick (mean + std * N(0,1))
    x_t = normal.sample(seed=key, sample_shape=())
    y_t = nnx.tanh(x_t)
    action = y_t * policy.action_scale + policy.action_bias
    log_prob = normal.log_prob(x_t)
    log_prob = log_prob.reshape(-1, 1)
    # Enforcing Action Bound
    log_prob -= jnp.log(policy.action_scale * (1 - y_t**2) + 1e-6)
    log_prob = log_prob.sum(1)
    return action, log_prob


def mean_action(policy: nnx.Module, obs: jnp.ndarray) -> jnp.ndarray:
    mean, _ = policy(obs)
    y_t = nnx.tanh(mean)
    action = y_t * policy.action_scale + policy.action_bias
    return action


def sac_actor_loss(
    policy: nnx.Module,
    q1: nnx.Module,
    q2: nnx.Module,
    alpha: float,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    action, log_prob = sample_actions(policy, observations, action_key)
    qf1_pi = q1(observations, action).squeeze()
    qf2_pi = q2(observations, action).squeeze()
    min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
    actor_loss = (alpha * log_prob - min_qf_pi).mean()
    return actor_loss


def sac_exploration_loss(
    policy: nnx.Module,
    target_entropy: float,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
    log_alpha: dict,
) -> jnp.ndarray:
    _, log_prob = sample_actions(policy, observations, action_key)
    alpha_loss = (
        -jnp.exp(log_alpha["log_alpha"]) * (log_prob + target_entropy)
    ).mean()
    return alpha_loss


class EntropyControl:
    """Automatic entropy tuning."""

    alpha: jnp.ndarray
    autotune: bool
    target_entropy: jnp.ndarray
    log_alpha: dict
    optimizer: optax.adam
    optimizer_state: optax.OptState

    def __init__(self, envs, alpha, autotune, learning_rate):
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -jnp.prod(
                jnp.array(envs.single_action_space.shape)
            )
            self.log_alpha = {"log_alpha": jnp.zeros(1)}
            self.alpha = jnp.exp(self.log_alpha["log_alpha"])
            self.optimizer = optax.adam(learning_rate=learning_rate)
            self.optimizer_state = self.optimizer.init(self.log_alpha)
        else:
            self.alpha = alpha

    def update(self, policy, policy_state, observations, action_key):
        if not self.autotune:
            return 0.0
        exploration_loss = partial(
            sac_exploration_loss,
            policy,
            self.target_entropy,
            policy_state,
            action_key,
            observations,
        )
        exploration_loss_value, alpha_grads = jax.value_and_grad(
            exploration_loss
        )(self.log_alpha)
        updates, self.optimizer_state = self.optimizer.update(
            alpha_grads, self.optimizer_state
        )
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        self.alpha = jnp.exp(self.log_alpha["log_alpha"])
        return exploration_loss_value


def train_sac(
    env,
    policy: nnx.Module,
    q: nnx.Module,
    seed: int = 1,
    total_timesteps: int = 1_000_000,
    buffer_size: int = 1_000_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    learning_starts: float = 5_000,
    policy_lr: float = 3e-4,
    q_lr: float = 1e-3,
    policy_frequency: int = 2,
    target_network_frequency: int = 1,
    alpha: float = 0.2,
    autotune: bool = True,
) -> tuple[
    nnx.Module,
    nnx.Module,
    nnx.Module,
]:
    """Soft actor-critic.

    Parameters
    ----------
    env: Vectorized Gymnasium environments.
    policy: Gaussian policy network.
    q: Soft Q network.
    seed: Seed for random number generation.
    total_timesteps: Total timesteps of the experiments
    buffer_size: The replay memory buffer size
    gamma: The discount factor gamma
    tau: Target smoothing coefficient (default: 0.005)
    batch_size: The batch size of sample from the reply memory
    learning_starts: Timestep to start learning
    policy_lr: The learning rate of the policy network optimizer
    q_lr: The learning rate of the Q network optimizer
    policy_frequency: Frequency of training policy (delayed)
    target_network_frequency: The frequency of updates for the target networks
    alpha: Entropy regularization coefficient.
    autotune: Automatic tuning of the entropy coefficient

    Returns
    -------
    policy
    policy parameters
    Q network
    Q1 parameters
    Q2 parameters
    """
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    assert isinstance(
        env.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    obs, _ = env.reset(seed=seed)

    policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate=policy_lr)
    )
    policy_target = nnx.clone(policy)
    q1 = q
    q1_optimizer = nnx.Optimizer(q1, optax.adam(learning_rate=q_lr))
    q2 = nnx.clone(q)
    q2_optimizer = nnx.Optimizer(q2, optax.adam(learning_rate=q_lr))

    entropy_control = EntropyControl(env, alpha, autotune, q_lr)

    env.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

    obs, _ = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            key, action_key = jax.random.split(key, 2)
            action, _ = sample_actions(policy, obs, action_key)
            action = np.asarray(action)

        next_obs, reward, termination, truncation, infos = env.step(action)

        if termination or truncation:
            print(f"{global_step=}, episodic_return={infos['episode']['r']}")

        rb.add_sample(obs, action, reward, next_obs, termination)

        obs = next_obs

        if global_step > learning_starts:
            observations, actions, rewards, next_observations, dones = (
                rb.sample_batch(batch_size, rng)
            )

            key, action_key = jax.random.split(key, 2)
            q1_loss_value, q1_state, q2_loss_value, q2_state = (
                sac_update_critic(
                    q1,
                    q2,
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
            )

            if (
                global_step % policy_frequency == 0
            ):  # TD 3 Delayed update support
                # compensate for the delay by doing 'policy_frequency' updates instead of 1
                for _ in range(policy_frequency):
                    key, action_key = jax.random.split(key, 2)
                    actor_loss_value, policy_state = sac_update_actor(
                        policy,
                        q1,
                        q2,
                        action_key,
                        observations,
                        entropy_control.alpha,
                    )
                    if autotune:
                        key, action_key = jax.random.split(key, 2)
                        exploration_loss_value = entropy_control.update(
                            policy, policy_state, observations, key
                        )

            # update the target networks
            if global_step % target_network_frequency == 0:
                q1_state = q1_state.replace(
                    target_params=optax.incremental_update(
                        q1_state.params, q1_state.target_params, tau
                    )
                )
                q2_state = q2_state.replace(
                    target_params=optax.incremental_update(
                        q2_state.params, q2_state.target_params, tau
                    )
                )

            if global_step % 1_000_000 == 0:
                print("losses/qf1_loss", q1_loss_value, global_step)
                print("losses/qf2_loss", q2_loss_value, global_step)
                print("losses/actor_loss", actor_loss_value, global_step)
                print("losses/alpha", entropy_control.alpha, global_step)
                if autotune:
                    print(
                        "losses/alpha_loss", exploration_loss_value, global_step
                    )

    # TODO return optimizers and state of entropy
    return policy, q1, q2


def sac_update_actor(
    policy: nnx.Module,
    q1: nnx.Module,
    q2: nnx.Module,
    action_key: jnp.ndarray,
    observations: jnp.ndarray,
    alpha: jnp.ndarray,
):
    actor_loss = partial(
        sac_actor_loss,
        policy,
        q1,
        q2,
        alpha,
        action_key,
        observations,
    )
    actor_loss_value, actor_grads = jax.value_and_grad(actor_loss)(
        policy_state.params
    )
    policy_state = policy_state.apply_gradients(grads=actor_grads)
    return actor_loss_value, policy_state


def sac_update_critic(
    q1,
    q2,
    policy,
    gamma,
    q1_state,
    q2_state,
    policy_state,
    observations,
    actions,
    rewards,
    next_observations,
    dones,
    action_key,
    alpha,
):
    next_state_actions, next_state_log_pi = sample_actions(
        policy, policy_state.params, next_observations, action_key
    )
    obs_act = jnp.concatenate((next_observations, next_state_actions), axis=-1)
    q1_next_target = q1(obs_act).squeeze()
    q2_next_target = q2(obs_act).squeeze()
    min_q_next_target = (
        jnp.minimum(q1_next_target, q2_next_target) - alpha * next_state_log_pi
    )
    next_q_value = rewards + (1 - dones) * gamma * min_q_next_target

    q_loss = partial(critic_loss, q, observations, actions, next_q_value)
    q1_loss_value, q1_grads = jax.value_and_grad(q_loss)(q1_state.params)
    q1_state = q1_state.apply_gradients(grads=q1_grads)
    q2_loss_value, q2_grads = jax.value_and_grad(q_loss)(q2_state.params)
    q2_state = q2_state.apply_gradients(grads=q2_grads)
    return q1_loss_value, q1_state, q2_loss_value, q2_state
