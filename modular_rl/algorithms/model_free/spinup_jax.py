from typing import List
import math
from functools import partial
import optax
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym


class NNPolicy:
    theta = jax.Array

    def __init__(self, sizes: List[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(sizes))
        self.theta = [self._random_layer_params(m, n, k)
                      for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    """
    def _random_layer_params(
            self, m: int, n: int, key: jax.random.PRNGKey,
            scale: float = 1e-1):
        w_key, b_key = jax.random.split(key)
        return (
            scale * jax.random.normal(w_key, (n, m)),
            scale * jax.random.normal(b_key, (n,))
        )
    """

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

        self.actions = jnp.arange(self.action_space.n)

        self.sampling_key, key = jax.random.split(key)

        sizes = [self.observation_space.shape[0]] + hidden_nodes + [self.action_space.n]
        super(SoftmaxNNPolicy, self).__init__(sizes, key)

    def sample(self, state: jax.Array):
        logits = nn_forward(state, self.theta)
        self.sampling_key, key = jax.random.split(self.sampling_key)
        return jax.random.categorical(key, logits)


@jax.jit
def nn_forward(x, theta):
    for W, b in theta[:-1]:
        a = jnp.dot(W, x) + b
        x = jnp.tanh(a)
    W, b = theta[-1]
    y = jnp.dot(W, x) + b
    return y


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


@jax.jit
def softmax_log_probability(state, action, theta):
    logits = nn_forward(state, theta)
    return logits[action] - jax.scipy.special.logsumexp(logits)


batched_softmax_log_probability = jax.vmap(softmax_log_probability, in_axes=(0, 0, None))


@jax.jit
def softmax_policy_gradient_pseudo_loss(states, actions, returns, theta):
    logp = batched_softmax_log_probability(states, actions, theta)
    return -jnp.dot(returns, logp) / len(returns)  # - to perform gradient ascent with a minimizer


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    env.reset(seed=42)

    # make core of policy network
    policy = SoftmaxNNPolicy(env.observation_space, env.action_space, hidden_sizes, jax.random.PRNGKey(42))

    # make optimizer
    solver = optax.adam(learning_rate=lr)
    opt_state = solver.init(policy.theta)

    # for training policy
    def train_one_epoch(opt_state):
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(np.copy(obs))

            # act in the environment
            act = np.asarray(policy.sample(jnp.array(obs)))
            obs, rew, terminated, truncated, _ = env.step(act)

            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        states = jnp.vstack(batch_obs)
        actions = jnp.stack(batch_acts)
        returns = jnp.hstack(batch_weights)
        batch_loss = softmax_policy_gradient_pseudo_loss(
            states=states,
            actions=actions,
            returns=returns,
            theta=policy.theta
        )
        batch_grad = jax.grad(
            partial(softmax_policy_gradient_pseudo_loss, states, actions, returns)
        )(policy.theta)
        updates, opt_state = solver.update(batch_grad, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)
        return batch_loss, batch_rets, batch_lens, opt_state

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens, opt_state = train_one_epoch(opt_state)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
