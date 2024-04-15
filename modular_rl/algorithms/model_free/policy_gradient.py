from typing import List, Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym


class EpisodeDataset:
    samples: List[List[Tuple[jax.Array, jax.Array, float]]]

    def __init__(self, episode_buffer_size=5):
        self.episode_buffer_size = episode_buffer_size
        self.samples = []

    def start_episode(self):
        if len(self.samples) >= self.episode_buffer_size:
            self.samples = self.samples[1:]
        self.samples.append([])

    def add_sample(self, state: jax.Array, action: jax.Array, reward: float):
        assert len(self.samples) > 0
        self.samples[-1].append((state, action, reward))

    def dataset(self):  # TODO return to go, discount factor
        states = []
        actions = []
        returns = []
        for episode in self.samples:
            states.extend([s for s, _, _ in episode])
            actions.extend([a for _, a, _ in episode])
            rewards = [r for _, _, r in episode]
            returns.append(rewards)
        return states, actions, returns


class NNPolicy:
    theta = jax.Array

    def __init__(self, sizes: List[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(sizes))
        self.theta = [self._random_layer_params(m, n, k)
                      for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def _random_layer_params(
            self, m: int, n: int, key: jax.random.PRNGKey,
            scale: float = 1e-1):
        w_key, b_key = jax.random.split(key)
        return (
            scale * jax.random.normal(w_key, (n, m)),
            scale * jax.random.normal(b_key, (n,))
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

    def _action_probabilities(self, state: jax.Array):
        return jax.nn.softmax(nn_forward(state, self.theta))

    def sample(self, state: jax.Array):
        probs = self._action_probabilities(state)
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.choice(key, self.actions, p=probs)


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
    return -jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * np.square((action - mu) / sigma)


batched_gaussian_log_probability = jax.vmap(gaussian_log_probability, in_axes=(0, 0, None, None))


@jax.jit
def gaussian_policy_gradient_pseudo_loss(states, actions, returns, theta):
    logp = batched_gaussian_log_probability(states, actions, theta)
    return jnp.dot(logp, returns)


@jax.jit
def softmax_log_probability(state, action, theta):
    logits = nn_forward(state, theta)
    return logits[action] - jax.scipy.special.logsumexp(logits)


batched_softmax_log_probability = jax.vmap(softmax_log_probability, in_axes=(0, 0, None, None))


@jax.jit
def softmax_policy_gradient_pseudo_loss(states, actions, returns, theta):
    logp = batched_softmax_log_probability(states, actions, theta)
    return jnp.dot(logp, returns)


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
    actions = jnp.stack(actions) - policy.action_space.start

    # reward to go
    returns = [discounted_reward_to_go(R, gamma) for R in rewards]
    returns = jnp.hstack(returns)

    # TODO baseline

    if isinstance(policy, GaussianNNPolicy):
        return jax.grad(
            partial(gaussian_policy_gradient_pseudo_loss, states, actions, returns)
        )(policy.theta)
    else:
        return jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, returns)
        )(policy.theta)


def reward_to_go(rewards):
    return np.flip(np.cumsum(np.flip(rewards)))


def discounted_reward_to_go(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


if __name__ == "__main__":
    # note on MountainCar-v0: never reaches the goal -> never learns
    #env = gym.make("CartPole-v1", render_mode="human")
    #env = gym.make("MountainCar-v0", render_mode="human")
    env = gym.make("Pendulum-v1", render_mode="human")

    observation_space = env.observation_space
    action_space = env.action_space
    #policy = SoftmaxNNPolicy(observation_space, action_space, [50], jax.random.PRNGKey(42))
    policy = GaussianNNPolicy(observation_space, action_space, [50], jax.random.PRNGKey(42))

    n_episodes = 1000
    gamma = 1.0
    learning_rate = 0.0001  # TODO use Adam
    random_state = np.random.RandomState(42)

    dataset = EpisodeDataset(episode_buffer_size=3)
    for i in range(n_episodes):
        print(f"{i=}")
        dataset.start_episode()
        state, _ = env.reset()
        done = False
        R = jnp.array(0.0)
        t = 0
        while not done:
            action = policy.sample(jnp.array(state))
            state, reward, terminated, truncated, _ = env.step(np.asarray(action))

            R += reward
            t += 1

            done = terminated or truncated
            dataset.add_sample(state, action, reward)
        print(f"Return {R}")

        # RL algorithm
        if (i + 1) % 1 == 0 and len(dataset.samples) > 5:
            # gradient ascent
            theta_grad = reinforce_update(policy, dataset, gamma)
            policy.theta = [(w + learning_rate * dw, b + learning_rate * db)
                            for (w, b), (dw, db) in zip(policy.theta, theta_grad)]
            #print(theta_grad)
