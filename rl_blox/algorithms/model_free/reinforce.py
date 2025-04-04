from collections import namedtuple

import chex
import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


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

    def _observations(self) -> np.ndarray:
        observations = []
        for episode in self.episodes:
            observations.extend([o for o, _, _, _ in episode])
        return np.vstack(observations)

    def _actions(self) -> np.ndarray:
        actions = []
        for episode in self.episodes:
            actions.extend([a for _, a, _, _ in episode])
        return np.stack(actions)

    def _nest_observations(self) -> np.ndarray:
        next_observations = []
        for episode in self.episodes:
            next_observations.extend([s for _, _, s, _ in episode])
        return np.vstack(next_observations)

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


def discounted_reward_to_go(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


class MLP(nnx.Module):
    """Multilayer Perceptron.

    Parameters
    ----------
    n_features
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    rngs
        Random number generator.
    """

    n_outputs: int
    hidden_layers: list[nnx.Linear]
    output_layer: nnx.Linear
    rngs: nnx.Rngs

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs

        self.hidden_layers = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.output_layer = nnx.Linear(n_in, n_outputs, rngs=rngs)

        self.rngs = rngs

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = nnx.swish(layer(x))
        return self.output_layer(x)


class GaussianMLP(nnx.Module):
    """Probabilistic neural network that predicts a Gaussian distribution.

    Parameters
    ----------
    shared_head
        All nodes of the last hidden layer are connected to mean AND log_std.

    n_features
        Number of features.

    n_outputs
        Number of output components.

    hidden_nodes
        Numbers of hidden nodes of the MLP.

    rngs
        Random number generator.
    """

    shared_head: bool
    n_outputs: int
    hidden_layers: list[nnx.Linear]
    output_layers: list[nnx.Linear]
    rngs: nnx.Rngs

    def __init__(
        self,
        shared_head: bool,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.shared_head = shared_head
        self.n_outputs = n_outputs

        self.hidden_layers = []
        n_in = n_features
        for n_out in hidden_nodes:
            self.hidden_layers.append(nnx.Linear(n_in, n_out, rngs=rngs))
            n_in = n_out

        self.output_layers = []
        if shared_head:
            self.output_layers.append(
                nnx.Linear(n_in, 2 * n_outputs, rngs=rngs)
            )
        else:
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))
            self.output_layers.append(nnx.Linear(n_in, n_outputs, rngs=rngs))

        self.rngs = rngs

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for layer in self.hidden_layers:
            x = nnx.swish(layer(x))

        if self.shared_head:
            y = self.output_layers[0](x)
            mean, log_var = jnp.split(y, (self.n_outputs,), axis=-1)
        else:
            mean = self.output_layers[0](x)
            log_var = self.output_layers[1](x)

        return mean, log_var


class ProbabilisticPolicyBase(nnx.Module):
    """Base class for probabilistic policies."""

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Compute action probabilities for given observation."""
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def sample(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Sample action from policy given observation."""
        raise NotImplementedError("Subclasses must implement sample method.")

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        raise NotImplementedError(
            "Subclasses must implement log_probability method."
        )


class GaussianPolicy(ProbabilisticPolicyBase):
    """Gaussian policy.

    Wraps a Gaussian neural network that maps observations to a Gaussian
    distribution over actions, i.e., mean vector and log variance vector.

    Parameters
    ----------
    net : nnx.Module
        Gaussian neural network.

    rngs
        Random number generator.
    """

    net: nnx.Module
    rngs: nnx.Rngs

    def __init__(
        self,
        net: nnx.Module,
        rngs: nnx.Rngs,
    ):
        self.net = net
        self.rngs = rngs

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.net(observation)[0]

    def sample(self, observation):
        """Sample action from Gaussian distribution."""
        mean, log_var = self.net(observation)
        return (
            jax.random.normal(self.rngs.params(), mean.shape)
            * jnp.exp(jnp.clip(0.5 * log_var, -20.0, 2.0))
            + mean
        )

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        mean, log_var = self.net(observation)
        log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
        std = jnp.exp(log_std)
        return distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=std
        ).log_prob(action)
        # return -jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * ((action - mean) / std) ** 2


class SoftmaxPolicy(ProbabilisticPolicyBase):
    r"""Softmax policy.

    Wraps a softmax neural network that maps observations to the logits of each
    action.

    Parameters
    ----------
    net : nnx.Module
        Gaussian neural network.

    rngs
        Random number generator.
    """

    net: nnx.Module
    rngs: nnx.Rngs

    def __init__(
        self,
        net: nnx.Module,
        rngs: nnx.Rngs,
    ):
        self.net = net
        self.rngs = rngs

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return nnx.softmax(self.logits(observation))

    def logits(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.net(observation)

    def sample(self, observation: jnp.ndarray) -> jnp.ndarray:
        return distrax.Categorical(logits=self.logits(observation)).sample(
            seed=self.rngs.params(),
            sample_shape=(),
        )

    def log_probability(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        return distrax.Categorical(logits=self.logits(observation)).log_prob(
            action
        )


@nnx.jit
def value_loss(
    observations: jnp.ndarray,
    returns: jnp.ndarray,
    value_function: nnx.Module,
) -> jnp.ndarray:
    """Value error as loss for the value function network.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observation_features)
        Observations.

    returns : array, shape (n_samples,)
        Target values, obtained, e.g., through Monte Carlo sampling.

    value_function : nnx.Module
        Value function that maps observations to expected returns.

    Returns
    -------
    loss : float
        Value function loss.
    """
    values = value_function(observations).squeeze()  # squeeze Nx1-D -> N-D
    chex.assert_equal_shape((values, returns))
    return optax.l2_loss(predictions=values, targets=returns).mean()


def policy_gradient_pseudo_loss(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    weights: jnp.ndarray,
    policy: nnx.Module,
) -> jnp.ndarray:
    """Pseudo loss for the policy gradient.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observation_features)
        Observations.

    actions : array, shape (n_samples, n_action_features)
        Actions.

    weights : array, shape (n_samples,)
        Weights for the policy gradient.

    policy : nnx.Module
        Policy.

    Returns
    -------
    loss : float
        Pseudo loss for the policy gradient.
    """
    logp = policy.log_probability(observations, actions)
    return -jnp.mean(
        weights * logp
    )  # - to perform gradient ascent with a minimizer


@nnx.jit
def reinforce_gradient(
    policy: ProbabilisticPolicyBase,
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
        \propto \sum_s \mu(s) \sum_a Q_{\pi_{\theta}} (s, a) \nabla_{\theta} \pi_{\theta}(a|s)
        = \mathbb{E}_{s \sim \mu(s)}\left[ \sum_a Q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s) \right],

    where

    * :math:`\mu(s)` is the state distribution under policy :math:`\pi_{\theta}`
    * :math:`Q_{\pi_{\theta}}` is the state-action value function

    In practice, we have to estimate the policy gradient from samples
    accumulated by using the policy :math:`\pi_{\theta}`.

    .. math::

        \nabla_{\theta}J(\theta)
        \propto
        \mathbb{E}_{s \sim \mu(s)}
        \left[
        \sum_a q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta} (a|s)
        \right]
        =
        \mathbb{E}_{s \sim \mu(s)}
        \left[
        \sum_a \textcolor{darkgreen}{\pi_{\theta} (a|s)} q_{\pi_{\theta}}(s, a)
        \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\textcolor{darkgreen}{\pi_{\theta} (a|s)}}
        \right]
        =
        \mathbb{E}_{s \sim \mu(s), \textcolor{darkgreen}{a \sim \pi_{\theta}}}
        \left[
        q_{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\pi_{\theta} (a|s)}
        \right]
        =
        \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}
        \left[
        \textcolor{darkgreen}{R} \frac{\nabla_{\theta} \pi_{\theta} (a|s)}{\pi_{\theta} (a|s)}
        \right]
        =
        \mathbb{E}_{s \sim \mu(s), a \sim \pi_{\theta}}
        \left[
        \underline{R} \textcolor{darkgreen}{\nabla_{\theta} \ln \pi_{\theta} (\underline{a}|\underline{s})}
        \right]
        \approx
        \textcolor{darkgreen}{\frac{1}{N}\sum_{(s, a, R)}}\underline{R} \nabla_{\theta} \ln \pi_{\theta} (\underline{a}|\underline{s})

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
    .. [1] Williams, R.J. (1992). Simple statistical gradient-following algorithms
       for connectionist reinforcement learning. Mach Learn 8, 229–256.
       https://doi.org/10.1007/BF00992696
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
        Estimated value function.
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
    """
    if value_function is not None:
        # state-value function as baseline, weights are advantages
        baseline = value_function(observations)
    else:
        # no baseline, weights are MC returns
        baseline = jnp.zeros_like(returns)
    weights = returns - baseline
    if gamma_discount is not None:
        weights *= gamma_discount

    return nnx.value_and_grad(policy_gradient_pseudo_loss, argnums=3)(
        observations, actions, weights, policy
    )


def create_policy_gradient_continuous_state(
    env: gym.Env,
    policy_shared_head: bool = True,
    policy_hidden_nodes: list[int] | tuple[int] = (32,),
    policy_learning_rate: float = 1e-4,
    value_network_hidden_nodes: list[int] | tuple[int] = (50, 50),
    value_network_learning_rate: float = 1e-2,
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
        rngs=nnx.Rngs(seed),
    )
    policy = GaussianPolicy(policy_net, rngs=policy_net.rngs)
    policy_optimizer = nnx.Optimizer(policy, optax.adamw(policy_learning_rate))

    value_function = MLP(
        n_features=observation_space.shape[0],
        n_outputs=1,
        hidden_nodes=list(value_network_hidden_nodes),
        rngs=nnx.Rngs(seed),
    )
    value_function_optimizer = nnx.Optimizer(
        value_function, optax.adamw(value_network_learning_rate)
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
    value_network_hidden_nodes: list[int] | tuple[int] = (50, 50),
    value_network_learning_rate: float = 1e-2,
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
        rngs=nnx.Rngs(seed),
    )
    policy = SoftmaxPolicy(policy_net, rngs=policy_net.rngs)
    policy_optimizer = nnx.Optimizer(policy, optax.adamw(policy_learning_rate))

    value_function = MLP(
        n_features=observation_space.shape[0],
        n_outputs=1,
        hidden_nodes=list(value_network_hidden_nodes),
        rngs=nnx.Rngs(seed),
    )
    value_function_optimizer = nnx.Optimizer(
        value_function, optax.adamw(value_network_learning_rate)
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


def train_reinforce_epoch(
    env: gym.Env,
    policy: ProbabilisticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP | None = None,
    value_function_optimizer: nnx.Optimizer | None = None,
    policy_gradient_steps: int = 1,
    value_gradient_steps: int = 1,
    total_steps: int = 1000,
    gamma: float = 1.0,
    train_after_episode: bool = False,
    verbose: int = 0,
):
    """Train with REINFORCE for one epoch.

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

    policy_gradient_steps : int, optional
        Number of gradient descent steps for the policy network.

    value_gradient_steps : int, optional
        Number of gradient descent steps for the value network.

    total_steps : int, optional
        Number of samples to collect before updating the policy. Alternatively
        you can train after each episode.

    gamma : float, optional
        Discount factor for rewards.

    train_after_episode : bool, optional
        Train after each episode. Alternatively you can train after collecting
        a certain number of samples.

    verbose : int, optional
        Verbosity level.
    """
    dataset = EpisodeDataset()

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
            if train_after_episode or len(dataset) >= total_steps:
                break

            observation, _ = env.reset()
            dataset.start_episode()

    if verbose:
        print(
            f"[REINFORCE] Average return in sampled "
            f"dataset: {dataset.average_return():.3f}"
        )

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
    if verbose >= 2:
        print(f"[REINFORCE] Policy loss: {p_loss:.3f}")

    if value_function is not None:
        assert value_function_optimizer is not None
        v_loss = train_value_function(
            value_function,
            value_function_optimizer,
            value_gradient_steps,
            observations,
            returns,
        )
        if verbose >= 2:
            print(f"[REINFORCE] Value function loss: {v_loss:.3f}")


def train_value_function(
    value_function,
    value_function_optimizer,
    value_gradient_steps,
    observations,
    returns,
):
    v_loss = 0.0
    for _ in range(value_gradient_steps):
        v_loss, v_grad = nnx.value_and_grad(value_loss, argnums=2)(
            observations, returns, value_function
        )
        value_function_optimizer.update(v_grad)
    return v_loss


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
        policy_optimizer.update(p_grad)
    return p_loss
