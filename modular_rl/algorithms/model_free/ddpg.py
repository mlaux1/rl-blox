from typing import Tuple, List
from collections import deque
from functools import partial

from numpy.typing import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import gymnasium as gym


class ReplayBuffer:
    buffer: deque[Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

    def __init__(self, n_samples):
        self.buffer = deque(maxlen=n_samples)

    def add_samples(self, observation, action, reward, next_observation, done):
        for i in range(len(done)):
            self.buffer.append((observation[i], action[i], reward[i], next_observation[i], done[i]))

    def sample_batch(
            self,
            batch_size: int,
            rng: np.random.Generator
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        observations = jnp.vstack([self.buffer[i][0] for i in indices])
        actions = jnp.stack([self.buffer[i][1] for i in indices])
        rewards = jnp.hstack([self.buffer[i][2] for i in indices])
        next_observations = jnp.vstack([self.buffer[i][3] for i in indices])
        dones = jnp.hstack([self.buffer[i][4] for i in indices])
        return observations, actions, rewards, next_observations, dones


class MlpQNetwork(nn.Module):
    r"""Q network represented by multilayer perceptron.

    A Q network maps observations and actions to their value, hence, represents
    the action-value function :math:`Q(o, a)`.
    """

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


class DeterministicMlpPolicyNetwork(nn.Module):
    r"""Deterministic policy represented by multilayer perceptron (MLP).

    The MLP directly maps observations to actions, hence, represents the
    function :math:`\pi(o) = a`.
    """

    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    action_dim: int
    """Dimension of the action space."""

    action_scale: jnp.ndarray
    """Scales for each component of the action."""

    action_bias: jnp.ndarray
    """Offset for each component of the action."""

    @nn.compact
    def __call__(self, x):
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x

    @staticmethod
    def create(actor_hidden_nodes: List[int], envs: gym.vector.SyncVectorEnv):
        return DeterministicMlpPolicyNetwork(
            hidden_nodes=actor_hidden_nodes,
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
            action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        )


class TargetTrainState(TrainState):
    """TrainState with additional target parameters.

    Target parameters are supposed to be more stable and will be updated by
    Polyak averaging.
    """
    target_params: flax.core.FrozenDict


def critic_loss(
        q: nn.Module,
        observations: np.ndarray,
        actions: np.ndarray,
        q_target: np.ndarray,
        q_params: flax.core.FrozenDict
) -> float:
    """Loss function for action-value function of the critic.

    :param q: Q network.
    :param observations: Batch of observations.
    :param actions: Batch of selected actions.
    :param q_target: Actual action values that should be approximated.
    :param q_params: Parameters of the Q network.
    :return: Mean squared distance between predicted and actual action values.
    """
    q_predicted = q.apply(q_params, observations, actions).squeeze()
    return 2.0 * optax.l2_loss(predictions=q_predicted, targets=q_target).mean()


def actor_loss(
        policy: nn.Module,
        q: nn.Module,
        q_state: TargetTrainState,
        observations: np.ndarray,
        policy_params: flax.core.FrozenDict
) -> float:
    """Loss function for the deterministic policy of the actor.

    :param policy: Deterministic policy represented by neural network.
    :param q: Q network.
    :param q_state: Training state of the Q network.
    :param observations: Batch of observations.
    :param policy_params: Parameters of the policy.
    :return: Negative value of the actions selected by the policy for the given
             observations.
    """
    return -q.apply(q_state.params, observations, policy.apply(policy_params, observations)).mean()


def ddpg_update_critic(
        policy: nn.Module,
        q: nn.Module,
        gamma: float,
        policy_state: TargetTrainState,
        q_state: TargetTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray
) -> Tuple[TargetTrainState, float]:
    """DDPG critic update."""
    # TODO why was it clipped to [-1, 1] before?
    next_actions = policy.apply(policy_state.target_params, next_observations)
    q_next_target = q.apply(q_state.target_params, next_observations, next_actions).reshape(-1)
    q_actual = (rewards + (1 - dones) * gamma * q_next_target).reshape(-1)

    loss = partial(critic_loss, q, observations, actions, q_actual)
    q_loss_value, grads = jax.value_and_grad(loss)(q_state.params)
    q_state = q_state.apply_gradients(grads=grads)

    return q_state, q_loss_value


def ddpg_update_actor(
        actor: nn.Module,
        q: nn.Module,
        actor_state: TargetTrainState,
        q_state: TargetTrainState,
        observations: np.ndarray
) -> Tuple[TargetTrainState, float]:
    """DDPG actor update."""
    loss = partial(actor_loss, actor, q, q_state, observations)
    actor_loss_value, grads = jax.value_and_grad(loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, actor_loss_value


def sample_actions(
        actor: nn.Module,
        actor_state: TargetTrainState,
        action_space: gym.spaces.Box,
        obs: np.ndarray,
        exploration_noise: float,
        rng: np.random.Generator
) -> np.ndarray:
    """Sample actions with deterministic policy and Gaussian action noise."""
    deterministic_actions = jax.device_get(
        actor.apply(actor_state.params, obs))
    noise = rng.normal(0.0, actor.action_scale * exploration_noise)
    exploring_actions = deterministic_actions + noise
    return exploring_actions.clip(action_space.low, action_space.high)


def train_ddpg(
        envs,
        policy: nn.Module,
        q: nn.Module,
        seed: int = 1,
        total_timesteps: int = 1_000_000,
        actor_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        gradient_steps: int = 1,
        exploration_noise: float = 0.1,
        learning_starts: int = 25_000,
        policy_frequency: int = 2,
        verbose: int = 0
) -> Tuple[nn.Module, flax.core.FrozenDict, nn.Module, flax.core.FrozenDict]:
    """Deep Deterministic Policy Gradients (DDPG).

    This is an off-policy actor-critic algorithm with a deterministic policy.
    The critic approximates the action-value function. The actor will maximize
    action values based on the approximation of the action-value function.

    :param envs: Vectorized Gymnasium environments.
    :param policy: Deterministic policy network.
    :param q: Q network.
    :param seed: Seed for random number generators in Jax and NumPy.
    :param total_timesteps: Number of steps to execute in the environment.
    :param actor_learning_rate: Learning rate of the actor.
    :param q_learning_rate: Learning rate of the critic.
    :param buffer_size: Size of the replay buffer.
    :param gamma: Discount factor.
    :param tau: Learning rate for polyak averaging of target policy and value
                function.
    :param batch_size: Size of a batch during gradient computation.
    :param gradient_steps: Number of gradient steps during one training phase.
    :param exploration_noise: Exploration noise in action space. Will be scaled
                              by half of the range of the action space.
    :param learning_starts: Learning starts after this number of random steps
                            was taken in the environment.
    :param policy_frequency: The policy will only be updated after this number
                             of steps. Target policy and value function will be
                             updated with the same frequency. The value
                             function will be updated after every step.
    :param verbose: Verbosity level.
    :returns: A tuple of the policy, policy parameters, Q network, and its
              parameters.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    action_space: gym.spaces.Box = envs.action_space

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(buffer_size)

    obs, _ = envs.reset(seed=seed)

    key, actor_key, q_key = jax.random.split(key, 3)
    policy_state = TargetTrainState.create(
        apply_fn=policy.apply,
        params=policy.init(actor_key, obs),
        target_params=policy.init(actor_key, obs),
        tx=optax.adam(learning_rate=actor_learning_rate),
    )
    q_state = TargetTrainState.create(
        apply_fn=q.apply,
        params=q.init(q_key, obs, action_space.sample()),
        target_params=q.init(q_key, obs, action_space.sample()),
        tx=optax.adam(learning_rate=q_learning_rate),
    )
    policy.apply = jax.jit(policy.apply)
    q.apply = jax.jit(q.apply)
    update_critic = jax.jit(partial(ddpg_update_critic, policy, q, gamma))
    update_actor = jax.jit(partial(ddpg_update_actor, policy, q))

    for t in range(total_timesteps):
        if t < learning_starts:
            actions = envs.action_space.sample()
        else:
            actions = sample_actions(policy, policy_state, envs.single_action_space, obs, exploration_noise, rng)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for final_info in infos.get("final_info", []):
            if verbose and "episode" in final_info:
                print(f"{t=}, return={final_info['episode']['r']}")
            break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add_samples(obs, actions, rewards, real_next_obs, terminations)

        obs = next_obs

        if t > learning_starts:
            for _ in range(gradient_steps):
                observations, actions, rewards, next_observations, dones = rb.sample_batch(batch_size, rng)

                q_state, q_loss_value = update_critic(
                    policy_state,
                    q_state,
                    observations,
                    actions,
                    next_observations,
                    rewards,
                    dones
                )
                if t % policy_frequency == 0:
                    policy_state, actor_loss_value = update_actor(
                        policy_state,
                        q_state,
                        observations
                    )
                    policy_state = policy_state.replace(
                        target_params=optax.incremental_update(policy_state.params, policy_state.target_params, tau)
                    )
                    q_state = q_state.replace(
                        target_params=optax.incremental_update(q_state.params, q_state.target_params, tau)
                    )

    return policy, policy_state.params, q, q_state.params
