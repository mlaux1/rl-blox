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

        self.theta = jax.random.normal(
            key, (self.action_space.n + self.state_space.shape[0] + 1,))

    def _feature_projection(self, state, action):
        one_hot_action = jnp.zeros(self.action_space.n)
        one_hot_action[action - self.action_space.start] = 1.0
        return jnp.hstack((state, one_hot_action, jnp.array([1.0])))

    def _h(self, state, action):
        return self.theta @ self._feature_projection(state, action)

    def action_probability(self, state, action):
        H = jnp.exp([self._h(state, b)
                     for b in range(self.action_space.start,
                                    self.action_space.start + self.action_space.n)])
        return jnp.exp(self._h(state, action)) / jnp.sum(H)  # TODO numerically robust, logsumexp?

    def log_action_probability(self, state, action):
        H = [self._h(state, b) for b in range(self.action_space.start,
                                              self.action_space.start + self.action_space.n)]
        h = self._h(state, action)
        return h - jax.special.logsumexp(H)  # TODO check


def reinforce(policy: SoftmaxLinearPolicy, dataset: EpisodeDataset):  # TODO can we use a pseudo-objective for autodiff?
    """REINFORCE policy gradient.

    References

    https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    http://incompleteideas.net/book/RLbook2020.pdf, page 326
    https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    """
    episode_returns = []
    for episode in dataset.samples:
        rewards = [r for _, _, r in episode]
        episode_return = jnp.sum(jnp.array(rewards))
        episode_returns.append(episode_return)
    episode_returns = jnp.hstack(episode_returns)
    n_episodes = len(dataset.samples)

    baseline = jnp.zeros(len(policy.theta))
    for h in range(len(policy.theta)):  # can we vectorize this?
        # sum i=1 to len(dataset.samples)
        #     (sum t=0 to T-1 grad log pi(a_t | s_t)) ** 2
        #     * R[i]
        # /
        # sum i=1 to len(dataset.samples)
        #     (sum t=0 to T-1 grad log pi(a_t | s_t)) ** 2
        denoms = [jnp.sum(jnp.array([policy.log_action_probability(st, at)  # TODO gradient with respect to theta[h]
                                     for st, at, _ in range(len(dataset.samples[i]))])) ** 2
                  for i in range(n_episodes)]
        noms = [denoms[i] * episode_returns[i] for i in range(n_episodes)]
        baseline[h] = 0

    # grad =
    # 1/len(dataset.samples)
    # sum i=1 to N
    #     sum t=0 to T-1 grad log pi (a_t | s_t) (R[i] * b)
    # return grad


if __name__ == "__main__":
    state_space = gym.spaces.Box(low=np.array([-10.0]), high=np.array([10.0]))
    action_space = gym.spaces.Discrete(2)
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    policy = SoftmaxLinearPolicy(state_space, action_space, subkey)
