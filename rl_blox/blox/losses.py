import chex
import jax
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
    :math:`y_i`, the loss is defined as

    .. math::

        \mathcal{L}(q)
        = \frac{1}{N} \sum_{i=1}^{N} (q(o_i, a_i) - y_i)^2.

    :math:`y_i` could be the Monte Carlo return.

    Parameters
    ----------
    observation : array, shape (n_samples, n_observation_features)
        Batch of observations.

    action : array, shape (n_samples, n_action_dims)
        Batch of selected actions.

    q_target_values : array, shape (n_samples,)
        Target action values :math:`y_i` that should be approximated.

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
) -> tuple[float, float]:
    r"""Mean squared error loss for discrete action-value function.

    For a given action-value function :math:`q(o, a)` and target values
    :math:`y_i`, the loss is defined as

    .. math::

        \mathcal{L}(q)
        = \frac{1}{N} \sum_{i=1}^{N} (q(o_i, a_i) - y_i)^2.

    :math:`y_i` could be the Monte Carlo return.

    Parameters
    ----------
    observation : array, shape (n_samples, n_observation_features)
        Batch of observations.

    action : array, shape (n_samples,)
        Batch of selected actions.

    q_target_values : array, shape (n_samples,)
        Target action values :math:`y_i` that should be approximated.

    q : nnx.Module
        Q network that maps observation to the action-values of each action of
        the discrete action space.

    Returns
    -------
    loss : float
        Mean squared error between predicted and actual action values.

    q_mean : float
        Mean of the predicted action values.
    """
    chex.assert_equal_shape_prefix((observation, action), prefix_len=1)
    chex.assert_equal_shape_prefix((observation, q_target_values), prefix_len=1)

    q_predicted = q(observation)[
        jnp.arange(len(observation), dtype=int), action.astype(int)
    ]
    chex.assert_equal_shape((q_predicted, q_target_values))

    return (
        optax.squared_error(
            predictions=q_predicted, targets=q_target_values
        ).mean(),
        q_predicted.mean(),
    )


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


def dqn_loss(
    q: nnx.Module,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float = 0.99,
) -> tuple[float, float]:
    r"""Deep Q-network (DQN) loss.

    This loss requires a continuous state space and a discrete action space.

    For a mini-batch, we calculate target values of :math:`Q`

    .. math::

        y_i = r_i + (1 - t_i) \gamma \max_{a'} Q(o_{i+1}, a'),

    where :math:`r_i` is the immediate reward obtained in the transition,
    :math:`t_i` indicates if a terminal state was reached in this transition,
    :math:`\gamma` (``gamma``) is the discount factor, and :math:`o_{i+1}` is
    the observation after the transition.

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : nnx.Module
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float, default=0.99
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given mini-batch.

    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I.,
       Wierstra, D., Riedmiller, M. (2013). Playing Atari with Deep
       Reinforcement Learning. https://arxiv.org/abs/1312.5602
    """
    obs, action, reward, next_obs, terminated = batch

    next_q = jax.lax.stop_gradient(q(next_obs))
    max_next_q = jnp.max(next_q, axis=1)

    q_target_values = jnp.array(reward) + (1 - terminated) * gamma * max_next_q

    return mse_discrete_action_value_loss(obs, action, q_target_values, q)


@nnx.jit
def nature_dqn_loss(
    q: nnx.Module,
    q_target: nnx.Module,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float = 0.99,
) -> tuple[float, float]:
    r"""Deep Q-network (DQN) loss with target network.

    This loss requires a continuous state space and a discrete action space.

    For a mini-batch, we calculate target values of :math:`Q`

    .. math::

        y_i = r_i + (1 - t_i) \gamma \max_{a'} Q'(o_{i+1}, a'),

    where :math:`r_i` is the immediate reward obtained in the transition,
    :math:`t_i` indicates if a terminal state was reached in this transition,
    :math:`\gamma` (``gamma``) is the discount factor, :math:`o_{i+1}` is
    the observation after the transition, and :math:`Q'` is the target network
    (``q_target``).

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : nnx.Module
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target : nnx.Module
        Target network for ``q``.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float, default=0.99
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given minibatch.

    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control
       through deep reinforcement learning. Nature 518, 529–533 (2015).
       https://doi.org/10.1038/nature14236
    """
    obs, action, reward, next_obs, terminated = batch

    next_q = jax.lax.stop_gradient(q_target(next_obs))
    max_next_q = jnp.max(next_q, axis=1)

    target = jnp.array(reward) + (1 - terminated) * gamma * max_next_q

    return mse_discrete_action_value_loss(obs, action, target, q)


@nnx.jit
def ddqn_loss(
    q: nnx.Module,
    q_target: nnx.Module,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float = 0.99,
) -> tuple[float, float]:
    r"""Deep double Q-network (DDQN) loss.

    This loss requires a continuous state space and a discrete action space.

    For a mini-batch, we calculate target values of :math:`Q`

    .. math::

        y_i = r_i + (1 - t_i) \gamma Q'(o_{i+1}, \arg\max_{a'} Q(o_{i+1}, a')),

    where :math:`r_i` is the immediate reward obtained in the transition,
    :math:`t_i` indicates if a terminal state was reached in this transition,
    :math:`\gamma` (``gamma``) is the discount factor, :math:`o_{i+1}` is
    the observation after the transition, and :math:`Q'` is the target network
    (``q_target``).

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : nnx.Module
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target : nnx.Module
        Target network for ``q``.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float, default=0.99
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given minibatch.

    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement
       Learning with Double Q-Learning. Proceedings of the AAAI Conference on
       Artificial Intelligence, 30(1). https://doi.org/10.1609/aaai.v30i1.10295
    """
    obs, action, reward, next_obs, terminated = batch

    next_q = jax.lax.stop_gradient(q(next_obs))
    indices = jnp.argmax(next_q, axis=1).reshape(-1, 1)
    next_q_t = jax.lax.stop_gradient(q_target(next_obs))
    next_vals = jnp.take_along_axis(next_q_t, indices, axis=1).squeeze()

    target = jnp.array(reward) + (1 - terminated) * gamma * next_vals

    pred = q(obs)
    pred = pred[jnp.arange(len(pred)), action]

    return optax.squared_error(pred, target).mean(), pred.mean()
