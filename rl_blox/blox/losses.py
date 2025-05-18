import chex
import jax.numpy as jnp
import optax
from flax import nnx

from .function_approximator.policy_head import StochasticPolicyBase


def mse_value_loss(
    observations: jnp.ndarray,
    v_target: jnp.ndarray,
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

    v_target : array, shape (n_samples,)
        Target values, obtained, e.g., through Monte Carlo sampling.

    v : nnx.Module
        Value function that maps observations to expected returns.

    Returns
    -------
    loss : float
        Value function loss.
    """
    values = v(observations).squeeze()  # squeeze Nx1-D -> N-D
    chex.assert_equal_shape((values, v_target))
    return optax.l2_loss(predictions=values, targets=v_target).mean()


def stochastic_policy_gradient_pseudo_loss(
    observation: jnp.ndarray,
    action: jnp.ndarray,
    weight: jnp.ndarray,
    policy: StochasticPolicyBase,
) -> jnp.ndarray:
    r"""Pseudo loss for the policy gradient.

    For a given stochastic policy :math:`\pi(a|o)`, observations :math:`o_i`,
    actions :math:`a_i`, and corresponding weights :math:`w_i`, the pseudo loss
    is defined as

    .. math::

        \mathcal{L}(\pi)
        =
        -\frac{1}{N} \sum_{i=1}^{N} w_i \log \pi(a_i|o_i).

    The calculation of weights depends on the specific algorithm. We take the
    negative value of the pseudo loss, because we want to perform gradient
    ascent with the policy gradient, but we use a gradient descent optimizer.

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
    return -jnp.mean(
        weight * logp
    )  # - to perform gradient ascent with a minimizer
