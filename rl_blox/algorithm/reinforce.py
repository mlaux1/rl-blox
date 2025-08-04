import contextlib
from collections import namedtuple
from collections.abc import Callable
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.gaussian_mlp import GaussianMLP
from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import (
    GaussianPolicy,
    SoftmaxPolicy,
    StochasticPolicyBase,
)
from ..blox.losses import mse_value_loss, stochastic_policy_gradient_pseudo_loss
from ..logging.logger import LoggerBase


class EpisodeDataset:
    """Collects samples batched in episodes."""

    episodes: list[list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]]]

    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self.episodes.append([])

    def add_sample(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        reward: float,
    ):
        assert len(self.episodes) > 0
        self.episodes[-1].append(
            (observation, action, next_observation, reward)
        )

    def _indices(self) -> list[int]:
        indices = []
        for episode in self.episodes:
            indices.extend([t for t in range(len(episode))])
        return indices

    def _observations(self) -> list:
        observations = []
        for episode in self.episodes:
            observations.extend([o for o, _, _, _ in episode])
        return observations

    def _actions(self) -> list:
        actions = []
        for episode in self.episodes:
            actions.extend([a for _, a, _, _ in episode])
        return actions

    def _nest_observations(self) -> list:
        next_observations = []
        for episode in self.episodes:
            next_observations.extend([s for _, _, s, _ in episode])
        return next_observations

    def _rewards(self) -> list[list[float]]:
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
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        observations = jnp.array(self._observations())
        actions = jnp.array(self._actions())
        next_observations = jnp.array(self._nest_observations())
        if isinstance(action_space, gym.spaces.Discrete):
            actions -= action_space.start
        returns = jnp.hstack(
            [discounted_reward_to_go(R, gamma) for R in self._rewards()]
        )
        gamma_discount = gamma ** jnp.hstack(self._indices())
        return observations, actions, next_observations, returns, gamma_discount


def discounted_reward_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """Computes the discounted return for each step.

    Parameters
    ----------
    rewards : list
        Rewards of one episode.

    gamma : float
        Discount factor.

    Returns
    -------
    discounted_returns : array
        Discounted return until the end of the episode.
    """
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


def reinforce_gradient(
    policy: StochasticPolicyBase,
    value_function: nnx.Module | None,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    returns: jnp.ndarray,
    gamma_discount: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""REINFORCE policy gradient.

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

    **Policy Gradient Theorem**

    .. math::

        \nabla_{\theta}J(\theta)
        & \propto
        \sum_s \mu(s) \sum_a Q_{\pi_{\theta}} (s, a)
        \nabla_{\theta} \pi_{\theta}(a|s)\\
        &=
        \mathbb{E}_{s \sim \mu(s)}
        \left[
        \sum_a Q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s)
        \right],

    where

    * :math:`\mu(s)` is the state distribution under policy :math:`\pi_{\theta}`
    * :math:`Q_{\pi_{\theta}}` is the state-action value function

    In practice, we have to estimate the policy gradient from samples
    accumulated by using the policy :math:`\pi_{\theta}`.

    .. math::

        \nabla_{\theta}J(\theta)
        &\propto
        \mathbb{E}_{s \sim \mu(s)}
        \left[
        \sum_a q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s)
        \right]\\
        &=
        \mathbb{E}_{s \sim \mu(s)}
        \left[
        \sum_a \textcolor{darkgreen}{\pi_{\theta} (a|s)} q_{\pi_{\theta}}(s, a)
        \frac{\nabla_{\theta} \pi_{\theta} (a|s)}
        {\textcolor{darkgreen}{\pi_{\theta} (a|s)}}
        \right]\\
        &=
        \mathbb{E}_{s \sim \mu(s), \textcolor{darkgreen}{a \sim \pi_{\theta}}}
        \left[
        q_{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta} (a|s)}
        {\pi_{\theta} (a|s)}
        \right]\\
        &=
        \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}
        \left[
        \textcolor{darkgreen}{R} \frac{\nabla_{\theta} \pi_{\theta} (a|s)}
        {\pi_{\theta} (a|s)}
        \right]\\
        &=
        \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}
        \left[
        \underline{R} \textcolor{darkgreen}{\nabla_{\theta}
        \ln \pi_{\theta} (\underline{a}|\underline{s})}
        \right]\\
        &\approx
        \textcolor{darkgreen}{\frac{1}{N}\sum_{(s, a, R)}}\underline{R}
        \nabla_{\theta} \ln \pi_{\theta} (\underline{a}|\underline{s})

    So we can estimate the policy gradient with N sampled states, actions, and
    returns.

    **REINFORCE With Baseline**

    For any function b which only depends on the state,

    .. math::

        \mathbb{E}_{a_t \sim \pi_{\theta}}
        \left[
        \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) b(s_t)
        \right]
        = 0

    This allows us to add or subtract any number of terms from the policy
    gradient without changing it in expectation. Any function b used in this
    way is called a baseline. The most common choice of baseline is the
    on-policy value function. This will reduce the variance of the estimate of
    the policy gradient, which makes learning faster and more stable. This
    encodes the intuition that if an agent gets what it expects, it should not
    change the parameters of the policy.

    References
    ----------
    .. [1] Williams, R.J. (1992). Simple statistical gradient-following
       algorithms for connectionist reinforcement learning. Mach Learn 8,
       229–256. https://doi.org/10.1007/BF00992696
    .. [2] Sutton, R.S., McAllester, D., Singh, S., Mansour, Y. (1999). Policy
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

    Parameters
    ----------
    policy
        Probabilistic policy that we want to update and has been used for
        exploration.
    value_function
        Estimated value function that should be used as a baseline. Set it to
        None if you do not want to use a baseline.
    observations
        Samples that were collected with the policy.
    actions
        Samples that were collected with the policy.
    returns
        Samples that were collected with the policy.
    gamma_discount
        Discounting for individual steps of the episode.

    Returns
    -------
    loss
        REINFORCE pseudo loss.
    grad
        REINFORCE policy gradient.

    See Also
    --------
    .blox.losses.stochastic_policy_gradient_pseudo_loss
        The pseudo loss that is used to compute the REINFORCE gradient. As
        weights for the pseudo loss we use R(o), the Monte Carlo return for the
        observation o. If a value function is provided, we use the difference
        between the Monte Carlo return and the value function as weights. If
        gamma_discount is provided, we multiply the weights with the
        discounting factor for each step of the episode.
    """
    if value_function is not None:
        # state-value function as baseline, weights are advantages
        baseline = value_function(observations).squeeze()
    else:
        # no baseline, weights are MC returns
        baseline = jnp.zeros_like(returns)
    weights = returns - baseline
    if gamma_discount is not None:
        weights *= gamma_discount

    return nnx.value_and_grad(
        stochastic_policy_gradient_pseudo_loss, argnums=3
    )(observations, actions, weights, policy)


def create_policy_gradient_continuous_state(
    env: gym.Env,
    policy_shared_head: bool = True,
    policy_hidden_nodes: list[int] | tuple[int] = (32,),
    policy_activation: str = "swish",
    policy_learning_rate: float = 1e-4,
    policy_optimizer: Callable = optax.adamw,
    value_network_hidden_nodes: list[int] | tuple[int] = (50, 50),
    value_network_learning_rate: float = 1e-2,
    value_network_optimizer: Callable = optax.adamw,
    seed: int = 0,
):
    observation_space: gym.spaces.Box = env.observation_space
    if len(observation_space.shape) > 1:
        raise ValueError("Only flat observation spaces are supported.")
    action_space: gym.spaces.Box = env.action_space
    if len(action_space.shape) > 1:
        raise ValueError("Only flat action spaces are supported.")

    policy_net = GaussianMLP(
        shared_head=policy_shared_head,
        n_features=observation_space.shape[0],
        n_outputs=action_space.shape[0],
        hidden_nodes=list(policy_hidden_nodes),
        activation=policy_activation,
        rngs=nnx.Rngs(seed),
    )
    policy = GaussianPolicy(policy_net)
    policy_optimizer = nnx.Optimizer(
        policy, policy_optimizer(policy_learning_rate), wrt=nnx.Param
    )

    value_function = MLP(
        n_features=observation_space.shape[0],
        n_outputs=1,
        hidden_nodes=list(value_network_hidden_nodes),
        activation="swish",
        rngs=nnx.Rngs(seed),
    )
    value_function_optimizer = nnx.Optimizer(
        value_function,
        value_network_optimizer(value_network_learning_rate),
        wrt=nnx.Param,
    )

    return namedtuple(
        "PolicyGradientState",
        [
            "policy",
            "policy_optimizer",
            "value_function",
            "value_function_optimizer",
        ],
    )(policy, policy_optimizer, value_function, value_function_optimizer)


def create_policy_gradient_discrete_state(
    env: gym.Env,
    policy_hidden_nodes: list[int] | tuple[int] = (32,),
    policy_learning_rate: float = 1e-4,
    policy_optimizer: Callable = optax.adam,
    value_network_hidden_nodes: list[int] | tuple[int] = (50, 50),
    value_network_learning_rate: float = 1e-2,
    value_network_optimizer: Callable = optax.adamw,
    seed: int = 0,
):
    observation_space: gym.spaces.Box = env.observation_space
    if len(observation_space.shape) > 1:
        raise ValueError("Only flat observation spaces are supported.")
    action_space: gym.spaces.Discrete = env.action_space
    if action_space.start != 0:
        raise ValueError("We assume that the minimum action is 0!")

    policy_net = MLP(
        n_features=observation_space.shape[0],
        n_outputs=int(action_space.n),
        hidden_nodes=list(policy_hidden_nodes),
        activation="swish",
        rngs=nnx.Rngs(seed),
    )
    policy = SoftmaxPolicy(policy_net)
    policy_optimizer = nnx.Optimizer(
        policy, policy_optimizer(policy_learning_rate), wrt=nnx.Param
    )

    value_function = MLP(
        n_features=observation_space.shape[0],
        n_outputs=1,
        hidden_nodes=list(value_network_hidden_nodes),
        activation="swish",
        rngs=nnx.Rngs(seed),
    )
    value_function_optimizer = nnx.Optimizer(
        value_function,
        value_network_optimizer(value_network_learning_rate),
        wrt=nnx.Param,
    )

    return namedtuple(
        "PolicyGradientState",
        [
            "policy",
            "policy_optimizer",
            "value_function",
            "value_function_optimizer",
        ],
    )(policy, policy_optimizer, value_function, value_function_optimizer)


def train_reinforce(
    env: gym.Env,
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP | None = None,
    value_function_optimizer: nnx.Optimizer | None = None,
    seed: int = 0,
    policy_gradient_steps: int = 1,
    value_gradient_steps: int = 1,
    total_timesteps: int = 1_000_000,
    gamma: float = 1.0,
    steps_per_update: int = 1_000,
    train_after_episode: bool = False,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Optimizer, nnx.Module, nnx.Optimizer]:
    """Train with REINFORCE.

    Parameters
    ----------
    env : gym.Env
        Environment.

    policy : nnx.Module
        Probabilistic policy network. Maps observations to probability
        distribution over actions.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy network.

    value_function : nnx.Module or None, optional
        Policy network. Maps observations to expected returns.

    value_function_optimizer : nnx.Optimizer or None, optional
        Optimizer for value function network.

    seed : int, optional
        Seed for random number generation.

    policy_gradient_steps : int, optional
        Number of gradient descent steps for the policy network.

    value_gradient_steps : int, optional
        Number of gradient descent steps for the value network.

    total_timesteps
        Total timesteps of the experiments.

    gamma : float, optional
        Discount factor for rewards.

    steps_per_update
        Number of samples to collect before updating the policy. Alternatively
        you can train after each episode.

    train_after_episode : bool, optional
        Train after each episode. Alternatively you can train after collecting
        a certain number of samples.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    policy : StochasticPolicyBase
        Final policy.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy network.

    value_function : nnx.Module
        Value function.

    value_function_optimizer : nnx.Optimizer
        Optimizer for value function.
    """
    key = jax.random.key(seed)
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    step = 0
    while step < total_timesteps:
        key, skey = jax.random.split(key, 2)
        dataset = sample_trajectories(
            env, policy, skey, logger, train_after_episode, steps_per_update
        )
        step += len(dataset)
        progress.update(len(dataset))

        observations, actions, _, returns, gamma_discount = (
            dataset.prepare_policy_gradient_dataset(env.action_space, gamma)
        )

        p_loss = train_policy_reinforce(
            policy,
            policy_optimizer,
            policy_gradient_steps,
            value_function,
            observations,
            actions,
            returns,
            gamma_discount,
        )
        if logger is not None:
            logger.record_stat(
                "policy loss", p_loss, episode=logger.n_episodes - 1
            )
            logger.record_epoch("policy", policy)

        if value_function is not None:
            assert value_function_optimizer is not None
            v_loss = train_value_function(
                value_function,
                value_function_optimizer,
                value_gradient_steps,
                observations,
                returns,
            )
            if logger is not None:
                logger.record_stat(
                    "value function loss", v_loss, episode=logger.n_episodes - 1
                )
                logger.record_epoch("value_function", value_function)
    progress.close()

    return namedtuple(
        "REINFORCEResult",
        [
            "policy",
            "policy_optimizer",
            "value_function",
            "value_function_optimizer",
        ],
    )(policy, policy_optimizer, value_function, value_function_optimizer)


def sample_trajectories(
    env: gym.Env,
    policy: StochasticPolicyBase,
    key: jnp.ndarray,
    logger: LoggerBase,
    train_after_episode: bool,
    total_steps: int,
) -> EpisodeDataset:
    """Sample trajectories with stochastic policy.

    Parameters
    ----------
    env : gym.Env
        Environment in which we collect samples.

    policy : StochasticPolicyBase
        Policy from which we sample actions.

    key : array
        Pseudo random number generator key for action sampling.

    logger : Logger
        Logs average return.

    train_after_episode : bool
        Collect exactly one episode of samples.

    total_steps : int
        Collect a minimum of total_steps, but continues to the end of the
        episode.

    Returns
    -------
    dataset : EpisodeDataset
        Collected samples organized in episodes.
    """
    if key is None:
        key = jax.random.key(0)

    dataset = EpisodeDataset()
    dataset.start_episode()

    if logger is not None:
        logger.start_new_episode()

    @nnx.jit
    def sample(policy, observation, subkey):
        return policy.sample(observation, subkey)

    steps_per_episode = 0
    observation, _ = env.reset()
    while True:
        key, subkey = jax.random.split(key)
        action = np.asarray(sample(policy, jnp.array(observation), subkey))

        next_observation, reward, terminated, truncated, _ = env.step(action)

        steps_per_episode += 1
        done = terminated or truncated

        dataset.add_sample(observation, action, next_observation, reward)

        observation = next_observation

        if done:
            if logger is not None:
                logger.stop_episode(steps_per_episode)
                logger.start_new_episode()
            steps_per_episode = 0

            if train_after_episode or len(dataset) >= total_steps:
                break

            observation, _ = env.reset()
            dataset.start_episode()
    if logger is not None:
        logger.record_stat(
            "average return",
            dataset.average_return(),
            episode=logger.n_episodes - 1,
        )
    return dataset


# DEPRECATED: for backward compatibility
collect_samples = sample_trajectories
with contextlib.suppress(ImportError):
    from warnings import deprecated

    collect_samples = deprecated("collect_samples is deprecated!")(
        collect_samples
    )


@partial(nnx.jit, static_argnames=["value_gradient_steps"])
def train_value_function(
    value_function,
    value_function_optimizer,
    value_gradient_steps,
    observations,
    returns,
):
    v_loss = 0.0
    for _ in range(value_gradient_steps):
        v_loss, v_grad = nnx.value_and_grad(mse_value_loss, argnums=2)(
            observations, returns, value_function
        )
        value_function_optimizer.update(value_function, v_grad)
    return v_loss


@partial(nnx.jit, static_argnames=["policy_gradient_steps"])
def train_policy_reinforce(
    policy,
    policy_optimizer,
    policy_gradient_steps,
    value_function,
    observations,
    actions,
    returns,
    gamma_discount,
):
    p_loss = 0.0
    for _ in range(policy_gradient_steps):
        p_loss, p_grad = reinforce_gradient(
            policy,
            value_function,
            observations,
            actions,
            returns,
            gamma_discount,
        )
        policy_optimizer.update(policy, p_grad)
    return p_loss
