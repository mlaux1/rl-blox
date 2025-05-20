import chex
import jax.numpy as jnp
import optax
from flax import nnx

from .function_approximator.policy_head import StochasticPolicyBase


def mse_continuous_action_value_loss(
    observation: jnp.ndarray,
    action: jnp.ndarray,
    q_target_values: jnp.ndarray,
    q: nnx.Module,
) -> jnp.ndarray:
    r"""Mean squared error loss for continuous action-value function.

    For a given action-value function :math:`q(o, a)` and target values
    :math:`R(o, a)`, the loss is defined as

    .. math::

        \mathcal{L}(q)
        = \frac{1}{2 N} \sum_{i=1}^{N} (q(o_i, a_i) - R(o_i, a_i))^2.

    :math:`R(o, a)` could be the Monte Carlo return.

    Parameters
    ----------
    observation : array, shape (n_samples, n_observation_features)
        Batch of observations.

    action : array, shape (n_samples, n_action_dims)
        Batch of selected actions.

    q_target_values : array, shape (n_samples,)
        Actual action values that should be approximated.

    q : nnx.Module
        Q network that maps a pair of observation and action to the action
        value. These networks are used for continuous action spaces.

    Returns
    -------
    loss : array, shape ()
        Mean squared error between predicted and actual action values.
    """
    chex.assert_equal_shape_prefix((observation, action), prefix_len=1)
    chex.assert_equal_shape_prefix((observation, q_target_values), prefix_len=1)

    q_predicted = q(jnp.concatenate((observation, action), axis=-1)).squeeze()
    chex.assert_equal_shape((q_predicted, q_target_values))

    return optax.squared_error(
        predictions=q_predicted, targets=q_target_values
    ).mean()


def mse_discrete_action_value_loss(
    observation: jnp.ndarray,
    action: jnp.ndarray,
    q_target_values: jnp.ndarray,
    q: nnx.Module,
) -> jnp.ndarray:
    r"""Mean squared error loss for discrete action-value function.

    For a given action-value function :math:`q(o, a)` and target values
    :math:`R(o, a)`, the loss is defined as

    .. math::

        \mathcal{L}(q)
        = \frac{1}{2 N} \sum_{i=1}^{N} (q(o_i, a_i) - R(o_i, a_i))^2.

    :math:`R(o, a)` could be the Monte Carlo return.

    Parameters
    ----------
    observation : array, shape (n_samples, n_observation_features)
        Batch of observations.

    action : array, shape (n_samples,)
        Batch of selected actions.

    q_target_values : array, shape (n_samples,)
        Actual action values that should be approximated.

    q : nnx.Module
        Q network that maps observation to the action-values of each action of
        the discrete action space.

    Returns
    -------
    loss : array, shape ()
        Mean squared error between predicted and actual action values.
    """
    chex.assert_equal_shape_prefix((observation, action), prefix_len=1)
    chex.assert_equal_shape_prefix((observation, q_target_values), prefix_len=1)

    q_predicted = q(observation)[
        jnp.arange(len(observation), dtype=int), action.astype(int)
    ]
    chex.assert_equal_shape((q_predicted, q_target_values))

    return optax.squared_error(
        predictions=q_predicted, targets=q_target_values
    ).mean()


def mse_value_loss(
    observations: jnp.ndarray,
    v_target_values: jnp.ndarray,
    v: nnx.Module,
) -> jnp.ndarray:
    r"""Mean squared error as loss for a value function network.

    For a given value function :math:`v(o)` and target values :math:`R(o)`, the
    loss is defined as

    .. math::

        \mathcal{L}(v) = \frac{1}{2 N} \sum_{i=1}^{N} (v(o_i) - R(o_i))^2.

    :math:`R(o)` could be the Monte Carlo return.

    Parameters
    ----------
    observations : array, shape (n_samples, n_observation_features)
        Observations.

    v_target_values : array, shape (n_samples,)
        Target values, obtained, e.g., through Monte Carlo sampling.

    v : nnx.Module
        Value function that maps observations to expected returns.

    Returns
    -------
    loss : float
        Value function loss.
    """
    values = v(observations).squeeze()  # squeeze Nx1-D -> N-D
    chex.assert_equal_shape((values, v_target_values))
    return optax.l2_loss(predictions=values, targets=v_target_values).mean()


def stochastic_policy_gradient_pseudo_loss(
    observation: jnp.ndarray,
    action: jnp.ndarray,
    weight: jnp.ndarray,
    policy: StochasticPolicyBase,
) -> jnp.ndarray:
    r"""Pseudo loss for the stochastic policy gradient.

    For a given stochastic policy :math:`\pi_{\theta}(a|o)`, observations
    :math:`o_i`, actions :math:`a_i`, and corresponding weights :math:`w_i`,
    the pseudo loss is defined as

    .. math::

        \mathcal{L}(\theta)
        = -\frac{1}{N} \sum_{i=1}^{N} w_i \ln \pi_{\theta}(a_i|o_i)
        \approx -\mathbb{E} \left[ w \ln \pi_{\theta}(a|o) \right]

    where :math:`w` depends on the algorithm:

    * REINFORCE: :math:`w = \gamma^t R_0` or :math:`w = \gamma^t R_t`
      (causality trick, less variance) or
      :math:`w = \gamma^t (R_t - \hat{v}(o_t))` (with baseline, even less
      variance)
    * Actor-Critic:
      :math:`w = \gamma^t (R_t + \gamma \hat{v}(o_{t+1}) - \hat{v}(o_t))`

    We take the negative value of the pseudo loss, because we want to perform
    gradient ascent with the policy gradient, but we use a gradient descent
    optimizer.

    Parameters
    ----------
    observation : array, shape (n_samples, n_observation_features)
        Observations.

    action : array, shape (n_samples, n_action_features)
        Actions.

    weight : array, shape (n_samples,)
        Weights for the policy gradient.

    policy : nnx.Module
        Policy :math:`\pi(a|o)`. We have to be able to compute
        :math:`\log \pi(a|o)` with
        `policy.log_probability(observations, actions)`.

    Returns
    -------
    loss : float
        Pseudo loss for the policy gradient.

    See Also
    --------
    .algorithm.reinforce.reinforce_gradient
        Uses this function to calculate the REINFORCE policy gradient.

    .algorithm.actor_critic.actor_critic_policy_gradient
        Uses this function to calculate the actor-critic policy gradient.
    """
    logp = policy.log_probability(observation, action)
    chex.assert_equal_shape((weight, logp))
    # - to perform gradient ascent with a minimizer
    return -jnp.mean(weight * logp)


def deterministic_policy_gradient_loss(
    q: nnx.Module,
    observation: jnp.ndarray,
    policy: nnx.Module,
) -> jnp.ndarray:
    r"""Loss function for the deterministic policy gradient.

    .. math::

        \mathcal{L}(\theta)
        =
        \frac{1}{N}
        \sum_{o \in \mathcal{D}}
        -Q_{\theta}(o, \pi(o))

    Parameters
    ----------
    q : nnx.Module
        Q network.

    observation : array, shape (n_samples, n_observation_features)
        Batch of observations.

    policy : nnx.Module
        Deterministic policy :math:`\pi(o) = a` represented by neural network.

    Returns
    -------
    loss : float
        Negative value of the actions selected by the policy for the given
        observations.
    """
    obs_act = jnp.concatenate((observation, policy(observation)), axis=-1)
    # - to perform gradient ascent with a minimizer
    return -q(obs_act).mean()
