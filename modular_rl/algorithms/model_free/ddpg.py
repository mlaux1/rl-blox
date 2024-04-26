from typing import List, Tuple
from functools import partial
import gym
import jax
import jax.numpy as jnp
import optax
from modular_rl.policy.differentiable import NeuralNetwork, batched_nn_forward, DeterministicNNPolicy


class ReplayBuffer:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    # TODO


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

    def update(self, states: jax.Array,
               actions: jax.Array,  # TODO actions or max_actions?
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

    def update_weights(self, theta):
        raise NotImplementedError()
        # TODO copy weights to self.theta


@jax.jit
def q_network_loss(
        states: jax.Array,
        actions: jax.Array,  # TODO actions or max_actions?
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


def train_ddpg(env: gym.Env):
    """

    References
    ----------
    * DQN: Playing Atari with Deep Reinforcement Learning, https://arxiv.org/abs/1312.5602
    * DQN: Human-level control through deep reinforcement learning, https://www.nature.com/articles/nature14236
    """
    q = QNetwork(env.observation_space, env.action_space, [64, 64], 42, 1e-3)
    target_q = QNetwork(env.observation_space, env.action_space, [64, 64], 42, 1e-3)
    target_q.update_weights(q.theta)

    policy = DeterministicNNPolicy(env.observation_space, env.action_space, [64, 64], 42)

    done = False
    while not done:
        obs, _ = env.reset()

        # TODO
