from typing import List, Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym


class EpisodeDataset:
    samples: List[List[Tuple[jax.Array, jax.Array, float]]]

    def __init__(self):
        self.samples = []

    def start_episode(self):
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
            R = jnp.sum(jnp.array(rewards))  # TODO for other versions
            returns.extend([R] * len(episode))
        return states, actions, returns


class SoftmaxNNPolicy:
    r"""A stochastic softmax policy for discrete action spaces with a neural network.

    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_nodes: Numbers of hidden nodes per hidden layer
    :param key: Jax random key for sampling network parameters and actions

    The neural network representing the policy :math:`\pi_{\theta}(a|s)` has
    :math:`|\mathcal{A}|` outputs, of which each represents the probability
    that the corresponding action is selected.
    """
    state_space: gym.spaces.Space
    action_space: gym.spaces.Space
    theta = jax.Array

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            hidden_nodes: List[int],
            key: jax.random.PRNGKey):
        self.observation_space = observation_space
        self.action_space = action_space

        self.actions = jnp.arange(
            self.action_space.start, self.action_space.start + self.action_space.n)

        self.sampling_key, key = jax.random.split(key)

        sizes = [self.observation_space.shape[0]] + hidden_nodes + [self.action_space.n]
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

    def zeros_like_theta(self):  # TODO?
        return [
            (jnp.zeros_like(W), jnp.zeros_like(b))
            for W, b in self.theta
        ]

    def _action_probabilities(self, state: jax.Array):
        return jax.nn.softmax(nn_logits(state, self.theta))

    def sample(self, state: jax.Array):
        probs = self._action_probabilities(state)
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.choice(key, self.actions, p=probs)


@jax.jit
def nn_logits(x, theta):
    for W, b in theta[:-1]:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    W, b = theta[-1]
    y = jnp.dot(W, x) + b
    return y


@jax.jit
def log_probability(state, action, theta, action_start_index):
    y = nn_logits(state, theta)
    action_index = action - action_start_index
    return y[action_index] - jax.scipy.special.logsumexp(y)


batched_log_probability = jax.vmap(log_probability, in_axes=(0, 0, None, None))


@jax.jit
def policy_gradient_pseudo_loss(states, actions, returns, action_start_index, theta):
    logp = batched_log_probability(states, actions, theta, action_start_index)
    return jnp.dot(logp, returns)


def policy_gradient_update(policy: SoftmaxNNPolicy, dataset: EpisodeDataset):
    """REINFORCE policy gradient.

    References

    https://www.deisenroth.cc/pdf/fnt_corrected_2014-08-26.pdf, page 29
    http://incompleteideas.net/book/RLbook2020.pdf, page 326
    https://media.suub.uni-bremen.de/handle/elib/4585, page 52
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
    https://github.com/openai/spinningup/tree/master/spinup/examples/pytorch/pg_math
    https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    """
    states, actions, returns = dataset.dataset()
    states = jnp.vstack(states)
    actions = jnp.hstack(actions)
    returns = jnp.hstack(returns)
    return jax.grad(partial(
        policy_gradient_pseudo_loss, states, actions, returns,
        policy.action_space.start))(policy.theta)


if __name__ == "__main__":
    # note on MountainCar-v0: never reaches the goal -> never learns
    env = gym.make("CartPole-v1", render_mode="human")

    observation_space = env.observation_space
    action_space = env.action_space
    policy = SoftmaxNNPolicy(observation_space, action_space, [30], jax.random.PRNGKey(42))

    n_episodes = 1000
    learning_rate = 0.0001  # TODO use Adam
    random_state = np.random.RandomState(42)

    dataset = EpisodeDataset()
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
        if (i + 1) % 10 == 0:
            theta_grad = policy_gradient_update(policy, dataset)
            #print(theta_grad)
            # gradient ascent
            policy.theta = [(w + learning_rate * dw, b + learning_rate * db)
                            for (w, b), (dw, db) in zip(policy.theta, theta_grad)]

            dataset = EpisodeDataset()
