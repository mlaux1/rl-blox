from typing import List, Tuple
from functools import partial
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

    def dataset(self):
        states = []
        actions = []
        returns = []
        for episode in self.samples:
            states.extend([s for s, _, _ in episode])
            actions.extend([a for _, a, _ in episode])
            rewards = [r for _, _, r in episode]
            R = jnp.sum(jnp.array(rewards))  # TODO for other versions
            returns.extend([R] * len(episode))
        return states, actions, returns


class SoftmaxPolicy:
    state_space: gym.spaces.Space
    action_space: gym.spaces.Space
    theta = jax.Array

    def __init__(
            self,
            state_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            hidden_nodes: List[int],
            key: jax.random.PRNGKey):
        self.state_space = state_space
        self.action_space = action_space

        self.actions = jnp.arange(
            self.action_space.start, self.action_space.start + self.action_space.n)

        self.sampling_key, key = jax.random.split(key)

        sizes = [self.state_space.shape[0]] + hidden_nodes + [self.action_space.n]
        keys = jax.random.split(key, len(sizes))
        self.theta = [self._random_layer_params(m, n, k)
                      for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def _random_layer_params(self, m, n, key, scale=1e-1):
        w_key, b_key = jax.random.split(key)
        return (
            scale * jax.random.normal(w_key, (n, m)),
            scale * jax.random.normal(b_key, (n,))
        )

    def zeros_like_theta(self):
        return [
            (jnp.zeros_like(W), jnp.zeros_like(b))
            for W, b in self.theta
        ]

    def _h(self, state, theta):
        return nn_logits(state, theta)

    def action_probabilities(self, state, theta):
        return jax.nn.softmax(self._h(state, theta))

    def sample(self, state):
        probs = self.action_probabilities(state, self.theta)
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.choice(key, self.actions, p=probs)


def nn_logits(x, theta):
    for W, b in theta[:-1]:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    W, b = theta[-1]
    y = jnp.dot(W, x) + b
    return y


def log_probability(state, action, theta, action_start_index):
    y = nn_logits(state, theta)
    action_index = action - action_start_index
    return y[action_index] - jax.scipy.special.logsumexp(y)


def policy_gradient_pseudo_loss(states, actions, returns, log_action_probability, action_start_index, theta):
    loss = jnp.array(0.0)
    for s, a, R in zip(states, actions, returns):  # TODO parallelize
        logp = log_action_probability(s, a, theta, action_start_index)
        loss = loss + logp * R
    return loss


def policy_gradient_update(policy: SoftmaxPolicy, dataset: EpisodeDataset):  # TODO can we use a pseudo-objective for autodiff?
    """REINFORCE policy gradient.

    References

    https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    http://incompleteideas.net/book/RLbook2020.pdf, page 326
    https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
    https://github.com/openai/spinningup/tree/master/spinup/examples/pytorch/pg_math
    """
    states, actions, returns = dataset.dataset()
    return jax.grad(partial(policy_gradient_pseudo_loss, states, actions, returns, log_probability, policy.action_space.start))(policy.theta)


class OptimalPolicy:
    def sample(self, state):
        return 1 if state[0] < 0 else 0


if __name__ == "__main__":
    state_space = gym.spaces.Box(low=np.array([-10.0]), high=np.array([10.0]))
    action_space = gym.spaces.Discrete(2)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    policy = SoftmaxPolicy(state_space, action_space, [100], subkey)
    #policy = OptimalPolicy()

    n_episodes = 100
    n_steps = 100
    learning_rate = 0.0001  # TODO use Adam
    random_state = np.random.RandomState(42)

    dataset = EpisodeDataset()
    for i in range(n_episodes):
        dataset.start_episode()
        state = jnp.array(np.array([10.0]) * np.sign(random_state.randn()))
        R = jnp.array(0.0)
        #print("")
        for t in range(n_steps):
            action = policy.sample(state)

            # environment
            if action == 0:  # transition dynamics
                next_state = state - 0.1
            else:
                next_state = state + 0.1
            next_state = jnp.clip(next_state, -10, 10.0)  # TODO hard coded
            #print(f"\r{next_state[0]:2.3f}")
            reward = -jnp.abs(next_state[0])  # reward function
            R += reward

            dataset.add_sample(state, action, reward)

            state = next_state
        print(f"State {state}")
        print(f"Return {R}")

        # RL algorithm
        if (i + 1) % 5 == 0:
            theta_grad = policy_gradient_update(policy, dataset)
            #print(theta_grad)
            # gradient ascent
            policy.theta = [(w + learning_rate * dw, b + learning_rate * db)
                            for (w, b), (dw, db) in zip(policy.theta, theta_grad)]

            dataset = EpisodeDataset()
