from typing import List, Tuple
import math
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
import gymnasium as gym


class EpisodeDataset:
    episodes: List[List[Tuple[jax.Array, jax.Array, float]]]

    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self.episodes.append([])

    def add_sample(self, state: jax.Array, action: jax.Array, reward: float):
        assert len(self.episodes) > 0
        self.episodes[-1].append((state, action, reward))

    def dataset(self):
        states = []
        actions = []
        rewards = []
        for episode in self.episodes:
            states.extend([s for s, _, _ in episode])
            actions.extend([a for _, a, _ in episode])
            rewards.append([r for _, _, r in episode])
        return states, actions, rewards

    def __len__(self):
        return sum(map(len, self.episodes))

    def average_return(self):
        return sum([sum([r for _, _, r in episode])
                    for episode in self.episodes]) / len(self.episodes)


class NNPolicy:
    theta = jax.Array

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


class GaussianNNPolicy(NNPolicy):
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

    def sample(self, state: jax.Array):
        y = nn_forward(state, self.theta)
        mu, sigma = jnp.split(y, [self.action_space.shape[0]])
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.normal(key, shape=mu.shape) * sigma + mu


class SoftmaxNNPolicy(NNPolicy):
    r"""Stochastic softmax policy for discrete action spaces with a neural network.

    The neural network representing the policy :math:`\pi_{\theta}(a|s)` has
    :math:`|\mathcal{A}|` outputs, of which each represents the probability
    that the corresponding action is selected.

    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_nodes: Numbers of hidden nodes per hidden layer
    :param key: Jax random key for sampling network parameters and actions
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

    def sample(self, state: jax.Array):
        logits = nn_forward(state, self.theta)
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.categorical(key, logits) + self.action_space.start


@jax.jit
def nn_forward(x, theta):
    for W, b in theta[:-1]:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    W, b = theta[-1]
    y = jnp.dot(W, x) + b
    return y


@jax.jit
def gaussian_log_probability(state, action, theta):
    # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
    y = nn_forward(state, theta)
    mu, sigma = jnp.split(y, [action.shape[0]])
    return -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.square((action - mu) / sigma)


batched_gaussian_log_probability = jax.vmap(gaussian_log_probability, in_axes=(0, 0, None))


@jax.jit
def gaussian_policy_gradient_pseudo_loss(states, actions, returns, theta):
    logp = batched_gaussian_log_probability(states, actions, theta)
    return -jnp.dot(returns, logp)[0] / len(returns)  # - to perform gradient ascent with a minimizer
    # TODO why? TypeError: Gradient only defined for scalar-output functions. Output had shape: (1,).


@jax.jit
def softmax_log_probability(state, action, theta):
    logits = nn_forward(state, theta)
    return logits[action] - jax.scipy.special.logsumexp(logits)


batched_softmax_log_probability = jax.vmap(softmax_log_probability, in_axes=(0, 0, None))


@jax.jit
def softmax_policy_gradient_pseudo_loss(states, actions, returns, theta):
    logp = batched_softmax_log_probability(states, actions, theta)
    return -jnp.dot(returns, logp) / len(returns)  # - to perform gradient ascent with a minimizer


def reinforce_update(policy: NNPolicy, dataset: EpisodeDataset, gamma: float):
    r"""REINFORCE policy gradient update.

    REINFORCE is an abbreviation for *Reward Increment = Non-negative Factor x
    Offset Reinforcement x Characteristic Eligibility*. It is a policy gradient
    algorithm that directly optimizes parameters of a stochastic policy. It
    uses a Monte Carlo estimate of :math:`Q^{\pi}`.

    We treat the episodic case, in which we define the performance measure as
    the value of the start state of the episode

    .. math::

        J(\theta) = v_{\pi_{\theta}}(s_0),

    where :math:`v_{\pi_{\theta}}` is the true value function for
    :math:`\pi_{\theta}`, the policy determined by :math:`\theta`.

    TODO derive log derivative trick

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

    [1] Williams, R.J. (1992). Simple statistical gradient-following algorithms
        for connectionist reinforcement learning. Mach Learn 8, 229–256.
        https://doi.org/10.1007/BF00992696
    [2] Sutton, R.S., McAllester, D., Singh, S., Mansour, Y. (1999). Policy
        Gradient Methods for Reinforcement Learning with Function Approximation.
        In Advances in Neural Information Processing Systems 12 (NIPS 1999).
        https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html

    Further resources:
    * https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    * http://incompleteideas.net/book/RLbook2020.pdf, page 326
    * https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    * https://link.springer.com/chapter/10.1007/978-3-642-27645-3_7, page 26
    * https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
    * https://github.com/openai/spinningup/tree/master/spinup/examples/pytorch/pg_math
    * https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    * https://www.quora.com/What-is-log-probability-in-policy-gradient-reinforcement-learning
    * https://avandekleut.github.io/reinforce/

    :param policy: TODO
    :param dataset: TODO
    :param gamma: Reward discount factor.
    """
    states, actions, rewards = dataset.dataset()

    states = jnp.vstack(states)
    actions = jnp.stack(actions)

    # reward to go
    #returns = [discounted_reward_to_go(R, gamma) for R in rewards]
    returns = [reward_to_go(R) for R in rewards]
    returns = jnp.hstack(returns)

    # TODO include baseline

    if isinstance(policy, GaussianNNPolicy):
        return jax.grad(
            partial(gaussian_policy_gradient_pseudo_loss, states, actions, returns)
        )(policy.theta)
    else:
        actions -= policy.action_space.start
        return jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, returns)
        )(policy.theta)


#def reward_to_go(rewards):
#    return np.flip(np.cumsum(np.flip(rewards)))


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def discounted_reward_to_go(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


def train_one_epoch(train_env, render_env, policy, solver, opt_state, batch_size, gamma):
    dataset = EpisodeDataset()
    env = render_env

    dataset.start_episode()
    state, _ = env.reset()
    R = 0.0
    t = 0
    i = 1
    while True:
        action = policy.sample(jnp.array(state))
        next_state, reward, terminated, truncated, _ = env.step(np.asarray(action))

        done = terminated or truncated
        R += reward
        t += 1

        dataset.add_sample(state, action, reward)

        state = next_state

        if done:
            n_samples = len(dataset)
            #print(f"{i=}, {t=}, {R=}, {n_samples=}")
            dataset.start_episode()
            env = train_env
            state, _ = env.reset()
            R = 0.0
            t = 0
            i += 1

            if n_samples >= batch_size:
                break

    print(f"{dataset.average_return()=}")

    # gradient ascent
    theta_grad = reinforce_update(policy, dataset, gamma)
    updates, opt_state = solver.update(theta_grad, opt_state)
    policy.theta = optax.apply_updates(policy.theta, updates)
    return opt_state


if __name__ == "__main__":
    # note on MountainCar-v0: never reaches the goal -> never learns
    env_name = "CartPole-v1"
    #env_name = "MountainCar-v0"
    #env_name = "Pendulum-v1"
    #env_name = "HalfCheetah-v4"
    train_env = gym.make(env_name)
    train_env.reset(seed=42)
    render_env = gym.make(env_name, render_mode="human")
    render_env.reset(seed=42)

    observation_space = render_env.observation_space
    action_space = render_env.action_space
    policy = SoftmaxNNPolicy(observation_space, action_space, [32], jax.random.PRNGKey(42))
    #policy = GaussianNNPolicy(observation_space, action_space, [50], jax.random.PRNGKey(42))

    solver = optax.adam(learning_rate=1e-2)
    opt_state = solver.init(policy.theta)

    n_epochs = 50
    for _ in range(n_epochs):
        opt_state = train_one_epoch(train_env, train_env, policy, solver, opt_state, batch_size=5000, gamma=1.0)
