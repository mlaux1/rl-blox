from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym


class EpisodeDataset:
    samples : List[List[Tuple[jax.Array, jax.Array, float]]]

    def __init__(self):
        self.samples = []

    def start_episode(self):
        self.samples.append([])

    def add_sample(self, state: jax.Array, action: jax.Array, reward: float):
        assert len(self.samples) > 0
        self.samples[-1].append((state, action, reward))

    def episode_returns(self):
        episode_returns = []
        for episode in self.samples:
            rewards = [r for _, _, r in episode]
            episode_return = jnp.sum(jnp.array(rewards))
            episode_returns.append(episode_return)
        return episode_returns


class SoftmaxLinearPolicy:
    state_space: gym.spaces.Space
    action_space: gym.spaces.Space
    theta = jax.Array

    def __init__(
            self,
            state_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            key: jax.random.PRNGKey):
        self.state_space = state_space
        self.action_space = action_space

        self.actions = jnp.arange(self.action_space.start, self.action_space.start + self.action_space.n)

        self.theta = jax.random.normal(
            key, (self.action_space.n + self.state_space.shape[0] + 1,))
        key, self.sampling_key = jax.random.split(key)  # TODO necessary?

    def _feature_projection(self, state, action):
        one_hot_action = jnp.zeros(self.action_space.n)
        one_hot_action.at[action - self.action_space.start].set(1.0)
        return jnp.hstack((state, one_hot_action, jnp.array([1.0])))

    def _h(self, state, action, theta):
        return theta @ self._feature_projection(state, action)

    def action_probability(self, state, action, theta):
        H = jnp.exp(jnp.array([self._h(state, b, theta) for b in self.actions]))
        return jnp.exp(self._h(state, action, theta)) / jnp.sum(H)  # TODO numerically robust, logsumexp?

    def log_action_probability(self, state, action, theta):
        H = jnp.array([self._h(state, b, theta) for b in self.actions])
        h = self._h(state, action, theta)
        return h - jax.scipy.special.logsumexp(H)  # TODO check

    def sample(self, state):
        probs = []
        for a in self.actions:  # TODO vectorize
            probs.append(self.action_probability(state, a, self.theta))
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.choice(key, self.actions, p=jnp.array(probs))


def reinforce(policy: SoftmaxLinearPolicy, dataset: EpisodeDataset):  # TODO can we use a pseudo-objective for autodiff?
    """REINFORCE policy gradient.

    References

    https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    http://incompleteideas.net/book/RLbook2020.pdf, page 326
    https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    """
    n_episodes = len(dataset.samples)
    n_params = policy.theta.shape[0]

    episode_returns = dataset.episode_returns()

    # sum i=1 to n_episodes
    #     (sum t=0 to T-1 grad theta log pi(a_t | s_t)) ** 2
    #     * R[i]
    # /
    # sum i=1 to len(dataset.samples)
    #     (sum t=0 to T-1 grad theta log pi(a_t | s_t)) ** 2
    denoms = jnp.array([jnp.sum(jnp.array([jax.grad(lambda theta: policy.log_action_probability(st, at, theta))(policy.theta)
                                           for st, at, _ in dataset.samples[i]])) ** 2
                        for i in range(n_episodes)])
    noms = denoms * jnp.hstack(episode_returns)
    baseline = jnp.sum(noms / denoms, axis=0)

    # grad =
    # 1/n_episodes
    # sum i=1 to N
    #     sum t=0 to T-1 grad theta log pi (a_t | s_t) (R[i] - b)
    # return grad
    grad = jnp.zeros(n_params)
    for i in range(n_episodes):
        grad = grad + jnp.sum(jnp.array([
            jax.grad(lambda theta: policy.log_action_probability(st, at, theta))(policy.theta) * (episode_returns[i] - baseline)
            for st, at, _ in dataset.samples[i]]))
    grad = grad / n_episodes
    return grad


if __name__ == "__main__":
    state_space = gym.spaces.Box(low=np.array([-10.0]), high=np.array([10.0]))
    action_space = gym.spaces.Discrete(2)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    policy = SoftmaxLinearPolicy(state_space, action_space, subkey)


    n_episodes = 100
    n_steps = 100
    learning_rate = 0.0005
    random_state = np.random.RandomState(42)

    dataset = EpisodeDataset()
    for i in range(n_episodes):
        dataset.start_episode()
        state = jnp.array(np.array([10.0]) * np.sign(random_state.randn()))
        for t in range(n_steps):
            action = policy.sample(state)

            # environment
            if action == 0:  # transition dynamics
                next_state = state - 1.0
            else:
                next_state = state + 1.0
            reward = -next_state ** 2  # reward function

            dataset.add_sample(state, action, reward)

            state = next_state

        # RL algorithm
        if (i + 1) % 5 == 0:
            print(dataset.episode_returns())

            theta_grad = reinforce(policy, dataset)
            policy.theta += learning_rate * theta_grad

            dataset = EpisodeDataset()
