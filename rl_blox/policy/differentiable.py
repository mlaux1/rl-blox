import math
from functools import partial
from typing import List, Tuple

import numpy as np
import distrax
import gymnasium as gym
import jax
from jax import numpy as jnp
import flax
import flax.linen as nn


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class GaussianMlpPolicyNetwork(nn.Module):
    """Gaussian policy represented by multilayer perceptron (MLP)."""

    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    action_dim: int
    """Dimension of the action space."""

    action_scale: jnp.ndarray
    """Scales for each component of the action."""

    action_bias: jnp.ndarray
    """Offset for each component of the action."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = nn.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std

    @staticmethod
    def create(
        actor_hidden_nodes: List[int], action_dim: int,
        action_scale: jnp.ndarray, action_bias: jnp.ndarray
    ) -> "GaussianMlpPolicyNetwork":
        return GaussianMlpPolicyNetwork(
            hidden_nodes=actor_hidden_nodes, action_dim=action_dim,
            action_scale=action_scale, action_bias=action_bias)


def sample_gaussian_actions(
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


class NeuralNetwork:
    """Base class of neural network policies.

    :param sizes: Numbers of neurons per layer, including input neurons and
                  output neurons.
    :param key: Jax pseudo random number generator key.
    """

    theta: List[Tuple[jax.Array, jax.Array]]

    def __init__(
        self,
        sizes: List[int],
        key: jax.random.PRNGKey,
        initializer=jax.nn.initializers.he_uniform,
    ):
        keys = jax.random.split(key, len(sizes))
        self.theta = [
            self._random_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    def _random_layer_params(
        self,
        m: int,
        n: int,
        key: jax.random.PRNGKey,
        initializer=jax.nn.initializers.he_uniform,
    ):
        w_key, b_key = jax.random.split(key)
        weight_initializer = initializer()
        bound = 1.0 / math.sqrt(m)
        return (
            weight_initializer(w_key, (n, m), jnp.float32),
            jax.random.uniform(b_key, (n,), jnp.float32, -bound, bound),
        )

    def update_weights(self, theta, polyak):
        for (sw, sb), (w, b) in zip(self.theta, theta):
            sw *= polyak
            sw += (1.0 - polyak) * w
            sb *= polyak
            sb *= (1.0 - polyak) * b


def nn_forward(
    x: jax.Array, theta: List[Tuple[jax.Array, jax.Array]]
) -> jax.Array:
    """Neural network forward pass.

    The neural network consists of fully connected layers with tanh activation
    functions in the hidden layers and no activation function in the last
    layer.

    :param x: 1D input vector.
    :param theta: Parameters (weights and biases) of the neural network.
    :returns: 1D output vector.
    """
    for W, b in theta[:-1]:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    W, b = theta[-1]
    y = jnp.dot(W, x) + b
    return y


batched_nn_forward = jax.vmap(nn_forward, in_axes=(0, None))


def nn_forward_tanh(
    x: jax.Array, theta: List[Tuple[jax.Array, jax.Array]]
) -> jax.Array:
    """Neural network forward pass with additional tanh at the output layer.

    The neural network consists of fully connected layers with tanh activation
    functions in the hidden layers and in the last layer.

    :param x: 1D input vector.
    :param theta: Parameters (weights and biases) of the neural network.
    :returns: 1D output vector.
    """
    for W, b in theta:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    return x


batched_nn_forward_tanh = jax.vmap(nn_forward_tanh, in_axes=(0, None))


class GaussianNNPolicy(NeuralNetwork):
    """Stochastic Gaussian policy for continuous action spaces with a neural network.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param hidden_nodes: Numbers of hidden nodes per hidden layer.
    :param key: Jax pseudo random number generator key for sampling network parameters and actions.
    """

    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    theta = jax.Array
    sampling_key: jax.random.PRNGKey

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        hidden_nodes: List[int],
        key: jax.random.PRNGKey,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.sampling_key, key = jax.random.split(key)

        sizes = (
            [self.observation_space.shape[0]]
            + hidden_nodes
            + [2 * self.action_space.shape[0]]
        )
        super(GaussianNNPolicy, self).__init__(sizes, key)
        self.sample_gaussian_nn = jax.jit(
            partial(
                sample_gaussian_nn, n_action_dims=self.action_space.shape[0]
            )
        )

    def sample(self, state: jax.Array) -> jax.Array:
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return self.sample_gaussian_nn(state, self.theta, key)


def sample_gaussian_nn(
    x: jax.Array,
    theta: List[Tuple[jax.Array, jax.Array]],
    key: jax.random.PRNGKey,
    n_action_dims: int,
) -> jax.Array:
    y = nn_forward(x, theta).squeeze()
    mu, log_sigma = jnp.split(y, [n_action_dims])
    log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
    sigma = jnp.exp(log_sigma)
    return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma).sample(
        seed=key, sample_shape=()
    )
    # return jax.random.normal(key, shape=mu.shape) * sigma + mu


def gaussian_log_probability(
    mu: jax.Array,
    log_sigma: jax.Array,
    action: jax.Array
) -> float:
    log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
    sigma = jnp.exp(log_sigma)
    return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma).log_prob(
        action
    )
    # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
    # return -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.square((action - mu) / sigma)


gaussian_log_probabilities = jax.vmap(
    gaussian_log_probability, in_axes=(0, 0, 0)
)


class SoftmaxMlpPolicy(nn.Module):
    r"""Stochastic softmax policy for discrete action spaces with a neural network.

    The neural network representing the policy :math:`\pi_{\theta}(a|s)` has
    :math:`|\mathcal{A}|` outputs, of which each represents the probability
    that the corresponding action is selected.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param hidden_nodes: Numbers of hidden nodes per hidden layer.
    :param key: Jax pseudo random number generator key for sampling network parameters and actions.
    """

    observation_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    theta = jax.Array
    sampling_key: jax.random.PRNGKey

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        hidden_nodes: List[int],
        key: jax.random.PRNGKey,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.actions = jnp.arange(
            self.action_space.start,
            self.action_space.start + self.action_space.n,
        )

        self.sampling_key, key = jax.random.split(key)

        sizes = (
            [self.observation_space.shape[0]]
            + hidden_nodes
            + [self.action_space.n]
        )
        super(SoftmaxNNPolicy, self).__init__(sizes, key)
        self.sample_softmax_nn = jax.jit(sample_softmax_nn)

    def sample(self, state: jax.Array):
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return (
            self.sample_softmax_nn(state, self.theta, key)
            + self.action_space.start
        )


def sample_softmax_nn(
    x: jax.Array,
    theta: List[Tuple[jax.Array, jax.Array]],
    key: jax.random.PRNGKey,
) -> jax.Array:
    logits = nn_forward(x, theta).squeeze()
    # return jax.random.categorical(key, logits)
    return distrax.Categorical(logits=logits).sample(seed=key, sample_shape=())


def softmax_log_probability(
    logits: jax.Array,
    action: jax.Array,
) -> float:
    return distrax.Categorical(logits=logits).log_prob(action)
    # return logits[action] - jax.scipy.special.logsumexp(logits)


softmax_log_probabilities = jax.vmap(
    softmax_log_probability, in_axes=(0, 0)
)
