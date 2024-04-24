import math
from functools import partial
from typing import List, Tuple

import distrax
import gymnasium as gym
import jax
from jax import numpy as jnp


class NeuralNetwork:
    """Base class of neural network policies.

    :param sizes: Numbers of neurons per layer, including input neurons and
                  output neurons.
    :param key: Jax pseudo random number generator key.
    """
    theta: List[Tuple[jax.Array, jax.Array]]

    def __init__(self, sizes: List[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(sizes))
        self.theta = [self._random_layer_params(m, n, k)
                      for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def _random_layer_params(self, m: int, n: int, key: jax.random.PRNGKey):
        w_key, b_key = jax.random.split(key)
        weight_initializer = jax.nn.initializers.he_uniform()
        bound = 1.0 / math.sqrt(m)
        return (
            weight_initializer(w_key, (n, m), jnp.float32),
            jax.random.uniform(b_key, (n,), jnp.float32, -bound, bound)
        )


def nn_forward(x: jax.Array, theta: List[Tuple[jax.Array, jax.Array]]) -> jax.Array:
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
            key: jax.random.PRNGKey):
        self.observation_space = observation_space
        self.action_space = action_space

        self.sampling_key, key = jax.random.split(key)

        sizes = [self.observation_space.shape[0]] + hidden_nodes + [2 * self.action_space.shape[0]]
        super(GaussianNNPolicy, self).__init__(sizes, key)
        self.sample_gaussian_nn = jax.jit(
            partial(sample_gaussian_nn, n_action_dims=self.action_space.shape[0]))

    def sample(self, state: jax.Array) -> jax.Array:
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return self.sample_gaussian_nn(state, self.theta, key)


def sample_gaussian_nn(
        x: jax.Array, theta: List[Tuple[jax.Array, jax.Array]],
        key: jax.random.PRNGKey, n_action_dims: int) -> jax.Array:
    y = nn_forward(x, theta).squeeze()
    mu, log_sigma = jnp.split(y, [n_action_dims])
    log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
    sigma = jnp.exp(log_sigma)
    return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma).sample(seed=key, sample_shape=())
    #return jax.random.normal(key, shape=mu.shape) * sigma + mu


def gaussian_log_probability(state: jax.Array, action: jax.Array, theta: List[Tuple[jax.Array, jax.Array]]) -> jnp.float32:
    y = nn_forward(state, theta).squeeze()
    mu, log_sigma = jnp.split(y, [action.shape[0]])
    log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
    sigma = jnp.exp(log_sigma)
    return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma).log_prob(action)
    # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
    #return -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.square((action - mu) / sigma)


batched_gaussian_log_probability = jax.vmap(gaussian_log_probability, in_axes=(0, 0, None))


class SoftmaxNNPolicy(NeuralNetwork):
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
            key: jax.random.PRNGKey):
        self.observation_space = observation_space
        self.action_space = action_space

        self.actions = jnp.arange(
            self.action_space.start, self.action_space.start + self.action_space.n)

        self.sampling_key, key = jax.random.split(key)

        sizes = [self.observation_space.shape[0]] + hidden_nodes + [self.action_space.n]
        super(SoftmaxNNPolicy, self).__init__(sizes, key)
        self.sample_softmax_nn = jax.jit(sample_softmax_nn)

    def sample(self, state: jax.Array):
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return self.sample_softmax_nn(state, self.theta, key) + self.action_space.start


def sample_softmax_nn(
        x: jax.Array, theta: List[Tuple[jax.Array, jax.Array]],
        key: jax.random.PRNGKey) -> jax.Array:
    logits = nn_forward(x, theta).squeeze()
    #return jax.random.categorical(key, logits)
    return distrax.Categorical(logits=logits).sample(seed=key, sample_shape=())


def softmax_log_probability(
        state: jax.Array, action: jax.Array,
        theta: List[Tuple[jax.Array, jax.Array]]) -> jnp.float32:
    logits = nn_forward(state, theta).squeeze()
    return distrax.Categorical(logits=logits).log_prob(action)
    #return logits[action] - jax.scipy.special.logsumexp(logits)


batched_softmax_log_probability = jax.vmap(softmax_log_probability, in_axes=(0, 0, None))
