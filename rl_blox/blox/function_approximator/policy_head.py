import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx


class DeterministicTanhPolicy(nnx.Module):
    r"""Policy with tanh scaling for continuous action spaces.

    The deterministic policy directly maps observations to actions, hence,
    represents the function :math:`\pi(o) = a`.

    The output of the underlying function approximator is scaled with tanh
    to [-1, 1] and then scaled to [action_space.low, action_space.high].

    Parameters
    ----------
    net : nnx.Module
        Neural network that maps observations to unscaled actions.
    """

    policy_net: nnx.Module
    """Underlying function approximator."""

    action_scale: nnx.Variable[jnp.ndarray]
    """Scales for each component of the action."""

    action_bias: nnx.Variable[jnp.ndarray]
    """Offset for each component of the action."""

    def __init__(self, policy_net: nnx.Module, action_space: gym.spaces.Box):
        self.policy_net = policy_net
        self.action_scale = nnx.Variable(
            jnp.array((action_space.high - action_space.low) / 2.0)
        )
        self.action_bias = nnx.Variable(
            jnp.array((action_space.high + action_space.low) / 2.0)
        )

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        y = self.policy_net(observation)
        return self.scale_output(y)

    def scale_output(self, y: jnp.ndarray) -> jnp.ndarray:
        return nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)


class StochasticPolicyBase(nnx.Module):
    """Base class for probabilistic policies.

    A subclass must define the functions

    * :func:`~StochasticPolicyBase.__call__`
    * :func:`~StochasticPolicyBase.sample`
    * :func:`~StochasticPolicyBase.log_probability`
    """

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Compute action probabilities for given observation."""
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from policy given observation.

        Parameters
        ----------
        observation : array
            Observation.

        key : array
            Pseudo random number generator key for sampling.

        Returns
        -------
        action : array
            Sampled action.
        """
        raise NotImplementedError("Subclasses must implement sample method.")

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation.

        Parameters
        ----------
        observation : array
            Observation.

        action : array
            Action.

        Returns
        -------
        log_prob : array
            Log probability of action given observation.
        """
        raise NotImplementedError(
            "Subclasses must implement log_probability method."
        )


class GaussianTanhPolicy(StochasticPolicyBase):
    r"""Gaussian policy with tanh scaling for continuous action spaces.

    The gaussian policy maps observations to mean and log variance of an
    action, hence, represents the distribution :math:`\pi(a|o)`.

    The output of the underlying function approximator is scaled with tanh
    to [-1, 1] and then scaled to [action_space.low, action_space.high].

    Parameters
    ----------
    net : nnx.Module
        Gaussian neural network that maps observations to unscaled mean and log
        variance.
    """

    net: nnx.Module
    """Underlying function approximator."""

    action_scale: nnx.Variable[jnp.ndarray]
    """Scales for each component of the action."""

    action_bias: nnx.Variable[jnp.ndarray]
    """Offset for each component of the action."""

    def __init__(self, policy_net: nnx.Module, action_space: gym.spaces.Box):
        self.net = policy_net
        self.action_scale = nnx.Variable(
            jnp.array((action_space.high - action_space.low) / 2.0)
        )
        self.action_bias = nnx.Variable(
            jnp.array((action_space.high + action_space.low) / 2.0)
        )

    def __call__(
        self, observation: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        y, log_var = self.net(observation)
        mean = nnx.tanh(y) * jnp.broadcast_to(
            self.action_scale.value, y.shape
        ) + jnp.broadcast_to(self.action_bias.value, y.shape)
        log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
        std = jnp.exp(log_std)
        return mean, std

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from Gaussian distribution."""
        mean, std = self(observation)
        return jax.random.normal(key, mean.shape) * std + mean

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation."""
        mean, std = self(observation)
        # same as
        # -jnp.log(std)
        # - 0.5 * jnp.log(2.0 * jnp.pi)
        # - 0.5 * ((action - mean) / std) ** 2
        return dist.MultivariateNormalDiag(loc=mean, scale_diag=std).log_prob(
            action
        )


class GaussianPolicy(StochasticPolicyBase):
    """Gaussian policy for continuous action spaces.

    Wraps a Gaussian neural network that maps observations to a Gaussian
    distribution over actions, i.e., mean vector and log variance vector.

    Parameters
    ----------
    net : nnx.Module
        Gaussian neural network that maps observations to mean and log variance.
    """

    net: nnx.Module

    def __init__(self, net: nnx.Module):
        self.net = net

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.net(observation)[0]

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from Gaussian distribution."""
        mean, log_var = self.net(observation)
        log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
        std = jnp.exp(log_std)
        # same as
        # jax.random.normal(key, mean.shape)
        # * jnp.exp(jnp.clip(0.5 * log_var, -20.0, 2.0))
        # + mean
        return dist.MultivariateNormalDiag(loc=mean, scale_diag=std).sample(
            seed=key,
            sample_shape=(),
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
        # same as
        # -jnp.log(std)
        # - 0.5 * jnp.log(2.0 * jnp.pi)
        # - 0.5 * ((action - mean) / std) ** 2
        return dist.MultivariateNormalDiag(loc=mean, scale_diag=std).log_prob(
            action
        )


class SoftmaxPolicy(StochasticPolicyBase):
    r"""Softmax policy for discrete action spaces.

    Wraps a softmax neural network that maps observations to the logits of each
    action.

    Parameters
    ----------
    net : nnx.Module
        Neural network that maps observations to logits.
    """

    net: nnx.Module

    def __init__(self, net: nnx.Module):
        self.net = net

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        return nnx.softmax(self.logits(observation))

    def logits(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.net(observation)

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        return dist.Categorical(logits=self.logits(observation)).sample(
            seed=key,
            sample_shape=(),
        )

    def log_probability(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        return dist.Categorical(logits=self.logits(observation)).log_prob(
            action
        )
