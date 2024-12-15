from functools import partial
from typing import List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import optax

from ...policy.differentiable import (
    GaussianNNPolicy,
    NeuralNetwork,
    batched_gaussian_log_probability,
    batched_nn_forward,
    batched_softmax_log_probability,
)


class EpisodeDataset:
    """Collects samples batched in episodes."""

    episodes: List[List[Tuple[jax.Array, jax.Array, jax.Array, float]]]

    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self.episodes.append([])

    def add_sample(
        self,
        state: jax.Array,
        action: jax.Array,
        next_state: jax.Array,
        reward: float,
    ):
        assert len(self.episodes) > 0
        self.episodes[-1].append((state, action, next_state, reward))

    def _indices(self) -> List[int]:
        indices = []
        for episode in self.episodes:
            indices.extend([t for t in range(len(episode))])
        return indices

    def _states(self) -> np.ndarray:
        states = []
        for episode in self.episodes:
            states.extend([s for s, _, _, _ in episode])
        return np.vstack(states)

    def _actions(self) -> np.ndarray:
        actions = []
        for episode in self.episodes:
            actions.extend([a for _, a, _, _ in episode])
        return np.stack(actions)

    def _next_states(self) -> np.ndarray:
        next_states = []
        for episode in self.episodes:
            next_states.extend([s for _, _, s, _ in episode])
        return np.vstack(next_states)

    def _rewards(self) -> List[List[float]]:
        rewards = []
        for episode in self.episodes:
            rewards.append([r for _, _, _, r in episode])
        return rewards

    def __len__(self) -> int:
        return sum(map(len, self.episodes))

    def average_return(self) -> float:
        return sum(
            [sum([r for _, _, _, r in episode]) for episode in self.episodes]
        ) / len(self.episodes)

    def prepare_policy_gradient_dataset(
        self, action_space: gym.spaces.Space, gamma: float
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        states = jnp.array(self._states())
        actions = jnp.array(self._actions())
        next_states = jnp.array(self._next_states())
        if isinstance(action_space, gym.spaces.Discrete):
            actions -= action_space.start
        returns = jnp.hstack(
            [discounted_reward_to_go(R, gamma) for R in self._rewards()]
        )
        gamma_discount = gamma ** jnp.hstack(self._indices())
        return states, actions, next_states, returns, gamma_discount


def discounted_reward_to_go(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


class ValueFunctionApproximation(nn.Module):
    """Approximation of the state-value function V(s).

    Note that a value function is usually specific for a policy because the
    policy influences action selection. Hence, reusing an old estimation of
    the value function for a new policy is an approximation. In addition,
    we use a function approximator that is trained on a finite number of
    samples, which also is an approximation.

    :param observation_space: observation space
    :param hidden_nodes: number of hidden nodes per hidden layer
    """
    hidden_nodes: List[int]
    """Numbers of hidden nodes of the MLP."""

    @nn.compact
    def __call__(self, x):
        for n_nodes in self.hidden_nodes:
            x = nn.Dense(n_nodes)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

    @staticmethod
    def create(hidden_nodes: List[int]):
        return ValueFunctionApproximation(hidden_nodes=hidden_nodes)


def value_loss(
        v: nn.Module,
        observations: jax.Array,
        returns: jax.Array,
        v_params: flax.core.FrozenDict
) -> float:
    """Value error as loss for the value function network.

    :param v: Value function network.
    :param observations: Batch of observations.
    :param returns: Episode returns.
    :param v_params: Parameters of the Q network.
    :return: Mean squared distance between predicted and actual action values.
    """
    values = v.apply(v_params, observations).squeeze()
    return 2.0 * optax.l2_loss(predictions=values, targets=returns).mean()


def update_value_function(
        v: nn.Module,
        v_state: TrainState,
        observations: jax.Array,
        returns: jax.Array
) -> Tuple[TrainState, float]:
    """Value function update."""
    v_loss_value, grads = jax.value_and_grad(
        partial(value_loss, v, observations, returns)
    )(v_state.params)
    v_state = v_state.apply_gradients(grads=grads)
    return v_state, v_loss_value


@jax.jit
def gaussian_policy_gradient_pseudo_loss(
    states: jax.Array,
    actions: jax.Array,
    weights: jax.Array,
    theta: List[Tuple[jax.Array, jax.Array]],
) -> jnp.float32:
    logp = batched_gaussian_log_probability(states, actions, theta)
    return -jnp.mean(
        weights * logp
    )  # - to perform gradient ascent with a minimizer


@jax.jit
def softmax_policy_gradient_pseudo_loss(
    states: jax.Array,
    actions: jax.Array,
    weights: jax.Array,
    theta: List[Tuple[jax.Array, jax.Array]],
) -> jnp.float32:
    logp = batched_softmax_log_probability(states, actions, theta)
    return -jnp.mean(
        weights * logp
    )  # - to perform gradient ascent with a minimizer


class PolicyTrainer:
    """Contains the state of the policy optimizer."""

    def __init__(
        self,
        policy: NeuralNetwork,
        optimizer=optax.adam,
        learning_rate: float = 1e-2,
        n_train_iters_per_update: int = 1,
    ):
        self.policy = policy
        self.n_train_iters_per_update = n_train_iters_per_update
        self.solver = optimizer(learning_rate=learning_rate)
        self.opt_state = self.solver.init(self.policy.theta)

    def update(
        self,
        policy_gradient_func,
        value_function: Optional[ValueFunctionApproximation],
        value_function_params: Optional[flax.core.FrozenDict],
        *args,
        **kwargs,
    ):
        for _ in range(self.n_train_iters_per_update):
            theta_grad = policy_gradient_func(
                self.policy, value_function, value_function_params,
                *args, **kwargs
            )
            updates, self.opt_state = self.solver.update(
                theta_grad, self.opt_state, self.policy.theta
            )
            self.policy.theta = optax.apply_updates(self.policy.theta, updates)


def reinforce_gradient(
    policy: NeuralNetwork,
    value_function: Optional[ValueFunctionApproximation],
    value_function_params: Optional[flax.core.FrozenDict],
    states: jax.Array,
    actions: jax.Array,
    returns: jax.Array,
    gamma_discount: Optional[jax.Array] = None,
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
    * https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

    :param policy: Policy that we want to update and has been used for exploration.
    :param value_function: Estimated value function.
    :param value_function_params: Parameters of the value function network.
    :param states: Samples that were collected with the policy.
    :param actions: Samples that were collected with the policy.
    :param returns: Samples that were collected with the policy.
    :param gamma_discount: Discounting for individual steps of the episode.
    :returns: REINFORCE policy gradient.
    """
    if value_function is not None:
        # state-value function as baseline, weights are advantages
        baseline = value_function.apply(value_function_params, states)
    else:
        # no baseline, weights are MC returns
        baseline = jnp.zeros_like(returns)
    weights = returns - baseline
    if gamma_discount is not None:
        weights *= gamma_discount

    if isinstance(
        policy, GaussianNNPolicy
    ):  # TODO find another way without if-else
        return jax.grad(
            partial(
                gaussian_policy_gradient_pseudo_loss, states, actions, weights
            )
        )(policy.theta)
    else:
        return jax.grad(
            partial(
                softmax_policy_gradient_pseudo_loss, states, actions, weights
            )
        )(policy.theta)


def train_reinforce_epoch(
    train_env : gym.Env,
    policy,
    policy_trainer,
    render_env: Optional[gym.Env],
    value_function: nn.Module,
    value_function_state: TrainState,
    batch_size: int,
    gamma: float,
    train_after_episode: bool = False,
    n_train_iters_per_update: int = 1
):
    dataset = EpisodeDataset()
    if render_env is not None:
        env = render_env
    else:
        env = train_env

    dataset.start_episode()
    observation, _ = env.reset()
    while True:
        action = policy.sample(jnp.array(observation))
        next_observation, reward, terminated, truncated, _ = env.step(
            np.asarray(action)
        )

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

    states, actions, _, returns, gamma_discount = (
        dataset.prepare_policy_gradient_dataset(env.action_space, gamma)
    )

    policy_trainer.update(
        reinforce_gradient,
        value_function,
        value_function_state.params,
        states,
        actions,
        returns,
        gamma_discount,
    )

    if value_function is not None:
        update = jax.jit(partial(update_value_function, v=value_function))
        for i in range(n_train_iters_per_update):
            value_function_state, v_loss_value = update(
                v_state=value_function_state, observations=states,
                returns=returns)
