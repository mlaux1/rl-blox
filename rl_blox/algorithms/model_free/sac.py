from functools import partial
from typing import List, Tuple

import distrax
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ...policy.differentiable import GaussianMlpPolicyNetwork
from .ddpg import ReplayBuffer, critic_loss


class SoftMlpQNetwork(nn.Module):
    """Soft Q network represented by multilayer perceptron."""

    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    @nn.compact
    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray):
        x = jnp.concatenate([obs, act], -1)
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def sample_actions(
    policy: nn.Module,
    params: flax.core.FrozenDict,
    obs: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, log_std = policy.apply(params, obs)
    std = jnp.exp(log_std)
    normal = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
    x_t = normal.sample(
        seed=key, sample_shape=()
    )  # for reparameterization trick (mean + std * N(0,1))
    y_t = jnp.tanh(x_t)
    action = y_t * policy.action_scale + policy.action_bias
    log_prob = normal.log_prob(x_t)
    log_prob = log_prob.reshape(-1, 1)
    # Enforcing Action Bound
    log_prob -= jnp.log(policy.action_scale * (1 - y_t**2) + 1e-6)
    log_prob = log_prob.sum(1)
    return action, log_prob


def mean_actions(
    policy, params: flax.core.FrozenDict, obs: jnp.ndarray
) -> jnp.ndarray:
    mean, log_std = policy.apply(params, obs)
    std = jnp.exp(log_std)
    y_t = jnp.tanh(mean)
    action = y_t * policy.action_scale + policy.action_bias
    return action


class TargetTrainState(TrainState):
    """TrainState with additional target parameters.

    Target parameters are supposed to be more stable and will be updated by
    Polyak averaging.
    """

    target_params: flax.core.FrozenDict


def sac_actor_loss(
    policy: nn.Module,
    q: nn.Module,
    q1_state: TargetTrainState,
    q2_state: TargetTrainState,
    alpha: float,
    action_key: jax.random.PRNGKey,
    observations: jnp.ndarray,
    params: flax.core.FrozenDict,
) -> jnp.ndarray:
    action, log_prob = sample_actions(policy, params, observations, action_key)
    qf1_pi = q.apply(q1_state.params, observations, action).squeeze()
    qf2_pi = q.apply(q2_state.params, observations, action).squeeze()
    min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
    actor_loss = (alpha * log_prob - min_qf_pi).mean()
    return actor_loss


def sac_exploration_loss(
    policy: nn.Module,
    target_entropy: float,
    policy_state: TrainState,
    action_key: jax.random.PRNGKey,
    observations: jnp.ndarray,
    log_alpha: dict,
) -> jnp.ndarray:
    _, log_prob = sample_actions(
        policy, policy_state.params, observations, action_key
    )
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
    envs,
    policy: nn.Module,
    q: nn.Module,
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
) -> Tuple[
    GaussianMlpPolicyNetwork,
    flax.core.FrozenDict,
    nn.Module,
    flax.core.FrozenDict,
    flax.core.FrozenDict,
]:
    """Soft actor-critic.

    :param envs: Vectorized Gymnasium environments.
    :param policy: Gaussian policy network.
    :param q: Soft Q network.
    :param seed: Seed for random number generation.
    :param total_timesteps: Total timesteps of the experiments
    :param buffer_size: The replay memory buffer size
    :param gamma: The discount factor gamma
    :param tau: Target smoothing coefficient (default: 0.005)
    :param batch_size: The batch size of sample from the reply memory
    :param learning_starts: Timestep to start learning
    :param policy_lr: The learning rate of the policy network optimizer
    :param q_lr: The learning rate of the Q network optimizer
    :param policy_frequency: Frequency of training policy (delayed)
    :param target_network_frequency: The frequency of updates for the target networks
    :param alpha: Entropy regularization coefficient.
    :param autotune: Automatic tuning of the entropy coefficient
    :returns: A tuple of the policy, policy parameters, Q network,
              Q1 parameters, and Q2 parameters.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    action_space: gym.spaces.Box = envs.action_space

    obs, _ = envs.reset(seed=seed)
    action = action_space.sample()

    key, actor_key, q1_key, q2_key = jax.random.split(key, 4)
    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(actor_key, obs),
        tx=optax.adam(learning_rate=policy_lr),
    )
    q1_state = TargetTrainState.create(
        apply_fn=q.apply,
        params=q.init(q1_key, obs, action),
        target_params=q.init(q1_key, obs, action),
        tx=optax.adam(learning_rate=q_lr),
    )
    q2_state = TargetTrainState.create(
        apply_fn=q.apply,
        params=q.init(q2_key, obs, action),
        target_params=q.init(q2_key, obs, action),
        tx=optax.adam(learning_rate=q_lr),
    )
    policy.apply = jax.jit(policy.apply)
    q.apply = jax.jit(q.apply)

    update_critic = jax.jit(partial(sac_update_critic, q, policy, gamma))
    update_actor = jax.jit(partial(sac_update_actor, policy, q))

    entropy_control = EntropyControl(envs, alpha, autotune, q_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

    obs, _ = envs.reset(seed=seed)
    for global_step in range(total_timesteps):
        if global_step < learning_starts:
            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            key, action_key = jax.random.split(key, 2)
            actions, _ = sample_actions(
                policy, policy_state.params, obs, action_key
            )
            actions = jax.device_get(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if any(terminations) or any(truncations):
            print(f"{global_step=}, episodic_return={infos['episode']['r']}")

        rb.add_samples(obs, actions, rewards, next_obs, terminations)

        obs = next_obs

        if global_step > learning_starts:
            observations, actions, rewards, next_observations, dones = (
                rb.sample_batch(batch_size, rng)
            )

            key, action_key = jax.random.split(key, 2)
            q1_loss_value, q1_state, q2_loss_value, q2_state = update_critic(
                q1_state,
                q2_state,
                policy_state,
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                action_key,
                entropy_control.alpha,
            )

            if (
                global_step % policy_frequency == 0
            ):  # TD 3 Delayed update support
                # compensate for the delay by doing 'policy_frequency' updates instead of 1
                for _ in range(policy_frequency):
                    key, action_key = jax.random.split(key, 2)
                    actor_loss_value, policy_state = update_actor(
                        policy_state,
                        q1_state,
                        q2_state,
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

    return policy, policy_state.params, q, q1_state.params, q2_state.params


def sac_update_actor(
    policy, q, policy_state, q1_state, q2_state, action_key, observations, alpha
):
    actor_loss = partial(
        sac_actor_loss,
        policy,
        q,
        q1_state,
        q2_state,
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
    q,
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
    q1_next_target = q.apply(
        q1_state.target_params, next_observations, next_state_actions
    ).squeeze()
    q2_next_target = q.apply(
        q2_state.target_params, next_observations, next_state_actions
    ).squeeze()
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
