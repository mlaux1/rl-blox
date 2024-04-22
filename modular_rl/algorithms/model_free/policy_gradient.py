from typing import List, Tuple, Optional
import math
from functools import partial
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import optax
import distrax
import chex
import gymnasium as gym


class EpisodeDataset:
    """Collects state-action-reward samples batched in episodes."""
    episodes: List[List[Tuple[jax.Array, jax.Array, jax.Array, float]]]

    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self.episodes.append([])

    def add_sample(self, state: jax.Array, action: jax.Array,
                   next_state: jax.Array, reward: float):
        assert len(self.episodes) > 0
        self.episodes[-1].append((state, action, next_state, reward))

    def dataset(self) -> Tuple[List[npt.ArrayLike], List[npt.ArrayLike], List[npt.ArrayLike], List[List[float]]]:
        states = []
        actions = []
        next_states = []
        rewards = []
        for episode in self.episodes:
            states.extend([s for s, _, _, _ in episode])
            actions.extend([a for _, a, _, _ in episode])
            rewards.append([r for _, _, _, r in episode])
            next_states.append([s for _, _, s, _ in episode])
        return states, actions, next_states, rewards

    def __len__(self) -> int:
        return sum(map(len, self.episodes))

    def average_return(self) -> float:
        return sum([sum([r for _, _, _, r in episode])
                    for episode in self.episodes]) / len(self.episodes)


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


class ValueFunctionApproximation(NeuralNetwork):
    def __init__(
            self, observation_state: gym.spaces.Space, hidden_nodes: List[int],
            key: jax.random.PRNGKey, learning_rate: float = 1e-2,
            n_train_iters_per_update: int = 1):
        sizes = [observation_state.shape[0]] + hidden_nodes + [1]
        super(ValueFunctionApproximation, self).__init__(sizes, key)
        self.n_train_iters_per_update = n_train_iters_per_update
        self.solver = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.solver.init(self.theta)
        self.forward = jax.jit(batched_nn_forward)

    def update(self, states: jax.Array, returns: jax.Array):
        for _ in range(self.n_train_iters_per_update):
            theta_grad = jax.grad(partial(value_loss, states, returns))(self.theta)
            updates, self.opt_state = self.solver.update(theta_grad, self.opt_state)
            self.theta = optax.apply_updates(self.theta, updates)

    def predict(self, states: jax.Array) -> jax.Array:
        return self.forward(states, self.theta).squeeze()


@jax.jit
def value_loss(states: jax.Array, returns: jax.Array,
               theta: List[Tuple[jax.Array, jax.Array]]) -> jnp.float32:
    """Value error as loss for the value function network."""
    values = batched_nn_forward(states, theta).squeeze()  # squeeze Nx1-D -> N-D
    chex.assert_equal_shape((values, returns))
    return optax.l2_loss(predictions=values, targets=returns).mean()


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
    mu, log_sigma = jnp.split(y, [action.shape[0]])  # TODO check dimensions
    log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
    sigma = jnp.exp(log_sigma)
    return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma).log_prob(action)
    # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
    #return -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.square((action - mu) / sigma)


batched_gaussian_log_probability = jax.vmap(gaussian_log_probability, in_axes=(0, 0, None))


@jax.jit
def gaussian_policy_gradient_pseudo_loss(
        states: jax.Array, actions: jax.Array, weights: jax.Array,
        theta: List[Tuple[jax.Array, jax.Array]]) -> jnp.float32:
    logp = batched_gaussian_log_probability(states, actions, theta)
    return -jnp.mean(weights * logp)  # - to perform gradient ascent with a minimizer


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


@jax.jit
def softmax_policy_gradient_pseudo_loss(
        states: jax.Array, actions: jax.Array, weights: jax.Array,
        theta: List[Tuple[jax.Array, jax.Array]]) -> jnp.float32:
    logp = batched_softmax_log_probability(states, actions, theta)
    return -jnp.mean(weights * logp)  # - to perform gradient ascent with a minimizer


class PolicyTrainer:
    def __init__(self, policy: NeuralNetwork,
                 optimizer = optax.adam,
                 learning_rate: float = 1e-2,
                 n_train_iters_per_update: int = 1):
        self.policy = policy
        self.n_train_iters_per_update = n_train_iters_per_update
        self.solver = optimizer(learning_rate=learning_rate)
        self.opt_state = self.solver.init(self.policy.theta)

    def update(
            self,
            policy_gradient_func,
            value_function: Optional[ValueFunctionApproximation],
            *args,
            **kwargs):
        for _ in range(self.n_train_iters_per_update):
            theta_grad = policy_gradient_func(
                self.policy, value_function, *args, **kwargs)
            updates, self.opt_state = self.solver.update(
                theta_grad, self.opt_state, self.policy.theta)
            self.policy.theta = optax.apply_updates(self.policy.theta, updates)


def reinforce_gradient(
        policy: NeuralNetwork,
        value_function: Optional[ValueFunctionApproximation],
        states: jax.Array, actions: jax.Array, returns: jax.Array,
        gamma_discount: Optional[jax.Array] = None
) -> jax.Array:
    r"""REINFORCE policy gradient update.

    REINFORCE is an abbreviation for *Reward Increment = Non-negative Factor x
    Offset Reinforcement x Characteristic Eligibility*. It is a policy gradient
    algorithm that directly optimizes parameters of a stochastic policy.

    We treat the episodic case, in which we define the performance measure as
    the value of the start state of the episode

    .. math::

        J(\theta) = v_{\pi_{\theta}}(s_0),

    where :math:`v_{\pi_{\theta}}` is the true value function for
    :math:`\pi_{\theta}`, the policy determined by :math:`\theta`.

    We use the policy gradient theorem to compute the policy gradient, which
    is the derivative of J with respect to the parameters of the policy.

    Policy Gradient Theorem
    -----------------------
    .. math::

        \nabla_{\theta}J(\theta)
        \propto \sum_s \mu(s) \sum_a Q_{\pi_{\theta}} (s, a) \nabla_{\theta} \pi_{\theta}(a|s)
        = \mathbb{E}_{s \sim \mu(s)}\left[ \sum_a Q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s) \right],

    where

    * :math:`\mu(s)` is the state distribution under policy :math:`\pi_{\theta}`
    * :math:`Q_{\pi_{\theta}}` is the state-action value function

    In practice, we have to estimate the policy gradient from samples
    accumulated by using the policy :math:`\pi_{\theta}`.

    .. math::

        \nabla_{\theta}J(\theta)
        \propto \mathbb{E}_{s \sim \mu(s)}\left[ \sum_a q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s) \right]
        = \mathbb{E}_{s \sim \mu(s)}\left[ \sum_a \textcolor{darkgreen}{\pi_{\theta} (a|s)} q_{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\textcolor{darkgreen}{\pi_{\theta} (a|s)}} \right]
        = \mathbb{E}_{s \sim \mu(s), \textcolor{darkgreen}{a \sim \pi_{\theta}}}\left[ q_{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\pi_{\theta} (a|s)} \right]
        = \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}\left[ \textcolor{darkgreen}{R} \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\pi_{\theta} (a|s)} \right]
        = \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}\left[ \underline{R} \textcolor{darkgreen}{\nabla_{\theta} \ln \pi_{\theta} (\underline{a}|\underline{s})} \right]
        \approx \textcolor{darkgreen}{\frac{1}{N}\sum_{(s, a, R)}}\underline{R} \nabla_{\theta} \ln \pi_{\theta} (\underline{a}|\underline{s})

    So we can estimate the policy gradient with N sampled states, actions, and
    returns.

    REINFORCE With Baseline
    -----------------------
    For any function b which only depends on the state,

    .. math::

        \mathbb{E}_{a_t \sim \pi_{\theta}} \left[\nabla_{\theta} \log \pi_{\theta} (a_t | s_t) b(s_t) \right] = 0

    This allows us to add or subtract any number of terms from the policy
    gradient without changing it in expectation. Any function b used in this
    way is called a baseline. The most common choice of baseline is the
    on-policy value function. This will reduce the variance of the estimate of
    the policy gradient, which makes learning faster and more stable. This
    encodes the intuition that if an agent gets what it expects, it should not
    change the parameters of the policy.

    References
    ----------
    [1] Williams, R.J. (1992). Simple statistical gradient-following algorithms
        for connectionist reinforcement learning. Mach Learn 8, 229–256.
        https://doi.org/10.1007/BF00992696
    [2] Sutton, R.S., McAllester, D., Singh, S., Mansour, Y. (1999). Policy
        Gradient Methods for Reinforcement Learning with Function Approximation.
        In Advances in Neural Information Processing Systems 12 (NIPS 1999).
        https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html

    Further resources:

    * https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
    * https://github.com/openai/spinningup/tree/master/spinup/examples/pytorch/pg_math
    * https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    * https://github.com/NadeemWard/pytorch_simple_policy_gradients/blob/master/reinforce/REINFORCE_discrete.py
    * https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    * http://incompleteideas.net/book/RLbook2020.pdf, page 326
    * https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    * https://link.springer.com/chapter/10.1007/978-3-642-27645-3_7, page 26
    * https://www.quora.com/What-is-log-probability-in-policy-gradient-reinforcement-learning
    * https://avandekleut.github.io/reinforce/

    :param policy: Policy that we want to update and has been used for exploration.
    :param value_function: Estimated value function.
    :param states: Samples that were collected with the policy.
    :param actions: Samples that were collected with the policy.
    :param returns: Samples that were collected with the policy.
    :param gamma_discount: Discounting for individual steps of the episode.
    :returns: REINFORCE policy gradient.
    """
    if value_function is not None:
        # state-value function as baseline, weights are advantages
        baseline = value_function.predict(states)
    else:
        # no baseline, weights are MC returns
        baseline = jnp.zeros_like(returns)
    weights = returns - baseline
    if gamma_discount is not None:
        weights *= gamma_discount

    if isinstance(policy, GaussianNNPolicy):  # TODO find another way without if-else
        return jax.grad(
            partial(gaussian_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)
    else:
        return jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)


def ac_policy_gradient(
        policy: NeuralNetwork,
        value_function: ValueFunctionApproximation,
        states: jax.Array, actions: jax.Array, next_states: jax.Array,
        rewards: jax.Array, gamma_discount: jax.Array, gamma: float
) -> jax.Array:
    V = value_function.predict(states)
    V_next = value_function.predict(next_states)
    TD_bootstrap_estimate = rewards + gamma * V_next - V
    weights = gamma_discount * TD_bootstrap_estimate

    if isinstance(policy, GaussianNNPolicy):  # TODO find another way without if-else
        return jax.grad(
            partial(gaussian_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)
    else:
        return jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, weights)
        )(policy.theta)


def discounted_reward_to_go(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


def train_reinforce_epoch(train_env, policy, policy_trainer, render_env, value_function, batch_size, gamma, train_after_episode=False):
    dataset = EpisodeDataset()
    if render_env is not None:
        env = render_env
    else:
        env = train_env

    dataset.start_episode()
    observation, _ = env.reset()
    while True:
        action = policy.sample(jnp.array(observation))
        next_observation, reward, terminated, truncated, _ = env.step(np.asarray(action))

        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if train_after_episode or len(dataset) >= batch_size:
                break

            env = train_env
            observation, _ = env.reset()
            dataset.start_episode()

    print(f"{dataset.average_return()=}")

    states, actions, _, returns, gamma_discount = prepare_policy_gradient_dataset(
        dataset, env.action_space, gamma)

    policy_trainer.update(
        reinforce_gradient, value_function, states, actions, returns,
        gamma_discount)

    if value_function is not None:
        value_function.update(states, returns)


def train_ac_epoch(train_env, policy, policy_trainer, render_env, value_function, batch_size, gamma, train_after_episode=False):
    dataset = EpisodeDataset()
    if render_env is not None:
        env = render_env
    else:
        env = train_env

    dataset.start_episode()
    observation, _ = env.reset()
    while True:
        action = policy.sample(jnp.array(observation))
        next_observation, reward, terminated, truncated, _ = env.step(np.asarray(action))

        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if train_after_episode or len(dataset) >= batch_size:
                break

            env = train_env
            observation, _ = env.reset()
            dataset.start_episode()

    print(f"{dataset.average_return()=}")

    states, actions, next_states, returns, gamma_discount = prepare_policy_gradient_dataset(
        dataset, env.action_space, gamma)
    rewards = jnp.hstack([jnp.hstack([r for _, _, _, r in episode])
                          for episode in dataset.episodes])

    policy_trainer.update(
        ac_policy_gradient, value_function, states, actions, next_states, rewards,
        gamma_discount, gamma)

    if value_function is not None:
        value_function.update(states, returns)


def prepare_policy_gradient_dataset(
        dataset: EpisodeDataset, action_space: gym.spaces.Space, gamma: float
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    states, actions, next_states, rewards = dataset.dataset()
    states = jnp.vstack(states)
    actions = jnp.stack(actions)
    next_states = jnp.vstack(next_states)
    if isinstance(action_space, gym.spaces.Discrete):
        actions -= action_space.start
    returns = jnp.hstack([discounted_reward_to_go(R, gamma) for R in rewards])
    gamma_discount = jnp.hstack([gamma ** jnp.arange(len(R)) for R in rewards])
    return states, actions, next_states, returns, gamma_discount
