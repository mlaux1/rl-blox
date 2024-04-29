from typing import List, Tuple
from functools import partial
from collections import deque
import gym
from numpy.typing import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
import optax
from modular_rl.policy.differentiable import NeuralNetwork, batched_nn_forward, DeterministicNNPolicy


class ReplayBuffer:
    buffer: deque[Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

    def __init__(self, n_samples):
        self.buffer = deque(maxlen=n_samples)

    def add_sample(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(
            self,
            batch_size: int,
            rng: np.random.Generator
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, List[bool]]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        states = jnp.vstack([self.buffer[i][0] for i in indices])
        actions = jnp.stack([self.buffer[i][1] for i in indices])
        rewards = jnp.hstack([self.buffer[i][2] for i in indices])
        next_states = jnp.vstack([self.buffer[i][3] for i in indices])
        dones = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, next_states, dones


class QNetwork(NeuralNetwork):
    r"""Approximation of the optimal action-value function Q*(s, a).

    .. math::

        Q^*(s, a) = \max_{\pi} \mathbb{E} \left[
        r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots
        | s_t = s, a_t = a, \pi \right]

    :param observation_space: observation space
    :param action_space: action space
    :param hidden_nodes: number of hidden nodes per hidden layer
    :param key: jax pseudo random number generator key
    :param learning_rate: learning rate for gradient descent
    :param n_train_iters_per_update: number of optimizer iterations per update
    """
    def __init__(
            self, observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            hidden_nodes: List[int], key: jax.random.PRNGKey,
            learning_rate: float = 1e-2, n_train_iters_per_update: int = 1):
        sizes = [observation_space.shape[0] + action_space.shape[0]] + hidden_nodes + [1]
        super(QNetwork, self).__init__(sizes, key)
        self.n_train_iters_per_update = n_train_iters_per_update
        self.solver = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.solver.init(self.theta)
        self.forward = jax.jit(batched_nn_forward)

    def update(
            self,
            states: jax.Array,
            actions: jax.Array,
            rewards: jax.Array,
            next_states: jax.Array,
            max_next_actions: jax.Array,
            gamma: float
    ):
        for _ in range(self.n_train_iters_per_update):
            theta_grad = jax.grad(
                partial(
                    q_network_loss,
                    states, actions, rewards, next_states, max_next_actions,
                    gamma
                )
            )(self.theta)
            updates, self.opt_state = self.solver.update(theta_grad, self.opt_state)
            self.theta = optax.apply_updates(self.theta, updates)

    def predict(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        x = jnp.hstack((states, actions))
        return self.forward(x, self.theta).squeeze()


@jax.jit
def q_network_loss(
        states: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_states: jax.Array,
        max_next_actions: jax.Array,
        gamma: float,
        theta: List[Tuple[jax.Array, jax.Array]]
) -> jnp.float32:
    """TODO"""
    next_states_max_next_actions = jnp.hstack((next_states, max_next_actions))
    next_state_action_values = batched_nn_forward(next_states_max_next_actions, theta).squeeze()
    target_values = rewards + gamma * next_state_action_values

    states_actions = jnp.hstack((states, actions))
    actual_values = batched_nn_forward(states_actions, theta).squeeze()

    return optax.l2_loss(predictions=actual_values, targets=target_values).mean()


class PolicyTrainer:
    """Contains the state of the policy optimizer."""
    def __init__(self, policy: NeuralNetwork,
                 optimizer=optax.adam,
                 learning_rate: float = 1e-2,
                 n_train_iters_per_update: int = 1):
        self.policy = policy
        self.n_train_iters_per_update = n_train_iters_per_update
        self.solver = optimizer(learning_rate=learning_rate)
        self.opt_state = self.solver.init(self.policy.theta)

    def update(
            self,
            policy_gradient_func,
            *args,
            **kwargs):
        for _ in range(self.n_train_iters_per_update):
            theta_grad = policy_gradient_func(
                self.policy, *args, **kwargs)
            updates, self.opt_state = self.solver.update(
                theta_grad, self.opt_state, self.policy.theta)
            self.policy.theta = optax.apply_updates(self.policy.theta, updates)


def dpg_policy_gradient(policy: DeterministicNNPolicy, states: jax.Array, q: QNetwork):
    return jax.grad(partial(dpg_policy_gradient_loss, states, q.theta))(policy.theta)


@jax.jit
def dpg_policy_gradient_loss(
        states: jax.Array,
        q_theta: List[Tuple[jax.Array, jax.Array]],
        policy_theta: List[Tuple[jax.Array, jax.Array]]
) -> jax.Array:
    max_actions = batched_nn_forward(states, policy_theta)
    x = jnp.hstack((states, max_actions))
    q = batched_nn_forward(x, q_theta).squeeze()
    return jnp.mean(q)


def train_ddpg(env: gym.Env, n_episodes, n_iters_before_update, n_updates, batch_size, noise_sigma, polyak, gamma, policy_learning_rate=1e-4):
    """Deep deterministic policy gradients.

    DDPG is an off-policy actor critic algorithm that combines a deterministic
    policy and deep Q networks.

    References
    ----------
    * DQN: Playing Atari with Deep Reinforcement Learning, https://arxiv.org/abs/1312.5602
    * DQN: Human-level control through deep reinforcement learning, https://www.nature.com/articles/nature14236
    * https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    """
    key = jax.random.PRNGKey(42)
    rng = np.random.default_rng(42)

    key, q_key = jax.random.split(key)
    q = QNetwork(env.observation_space, env.action_space, [64, 64], q_key, 1e-3)
    key, target_q_key = jax.random.split(key)
    target_q = QNetwork(env.observation_space, env.action_space, [64, 64], target_q_key, 1e-3)
    target_q.update_weights(q.theta, 0.0)

    key, policy_key = jax.random.split(key)
    policy = DeterministicNNPolicy(env.observation_space, env.action_space, [64, 64], policy_key, noise_sigma)
    key, target_policy_key = jax.random.split(key)
    target_policy = DeterministicNNPolicy(env.observation_space, env.action_space, [64, 64], target_policy_key, noise_sigma)
    policy_trainer = PolicyTrainer(policy, learning_rate=policy_learning_rate)

    buffer = ReplayBuffer(10000)

    t = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = np.asarray(policy.sample(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            buffer.add_sample(obs, action, reward, next_obs, done)
            t += 1

            obs = next_obs

            if t % n_iters_before_update == 0:
                for _ in range(n_updates):
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size, rng)
                    max_next_actions = target_policy.batch_predict(next_states)
                    q.update(states, actions, rewards, next_states, max_next_actions, gamma)
                    policy_trainer.update(dpg_policy_gradient, states, q)
                    target_q.update_weights(q.theta, polyak)
                    target_policy.update_weights(target_policy.theta, polyak)
