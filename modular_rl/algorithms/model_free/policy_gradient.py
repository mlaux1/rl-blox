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


class Policy:  # I think this must be a stochastic policy
    state_space: gym.spaces.Space
    action_space: gym.spaces.Space
    theta = jax.Array

    def __init__(self, state_space: gym.spaces.Space, action_space: gym.spaces.Space, key: jax.random.PRNGKey):
        self.state_space = state_space
        self.action_space = action_space

        self.theta = jax.random.normal(
            key, (self.action_space.shape[0], self.state_space.shape[0] + 1))

    def _project(self, state):
        return jnp.append(state, jnp.array([1.0]))

    def generate(self, state):
        return self.theta @ self._project(state)

    def density(self, state, action):
        raise NotImplementedError()


def reinforce(policy: Policy, dataset: EpisodeDataset):  # TODO can we use a pseudo-objective for autodiff?
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

    baseline = jnp.zeros(len(policy.theta))
    for h in range(len(policy.theta)):  # can we vectorize this?
        # sum i=1 to len(dataset.samples)
        #     (sum t=0 to T-1 grad log pi(a_t | s_t)) ** 2
        #     * R[i]
        # /
        # sum i=1 to len(dataset.samples)
        #     (sum t=0 to T-1 grad log pi(a_t | s_t)) ** 2
        baseline[h] = 0

    # grad =
    # 1/len(dataset.samples)
    # sum i=1 to N
    #     sum t=0 to T-1 grad log pi (a_t | s_t) (R[i] * b)
    # return grad


if __name__ == "__main__":
    state_space = gym.spaces.Box(low=np.array([-10.0]), high=np.array([10.0]))
    action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    policy = Policy(state_space, action_space, subkey)
