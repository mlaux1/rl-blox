import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .double_qnet import ContinuousClippedDoubleQNet
from .function_approximator.policy_head import StochasticPolicyBase


def masked_mse_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray
) -> float:
    """Masked mean squared error loss.

    Parameters
    ----------
    predictions : array, shape (n_samples, n_features)
        Predicted values.

    targets : array, shape (n_samples, n_features)
        Target values.

    mask : array, shape (n_samples,)
        Mask indicating which values to include in the loss calculation with 1.

    Returns
    -------
    loss : float
        Masked mean squared error loss.
    """
    return jnp.mean(
        optax.squared_error(predictions=predictions, targets=targets)
        * mask[:, jnp.newaxis]
    )


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


def mse_continuous_action_value_loss(
    observation: jnp.ndarray,
    action: jnp.ndarray,
    q_target_values: jnp.ndarray,
    q: nnx.Module,
) -> tuple[float, float]:
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
    loss : float
        Mean squared error between predicted and actual action values.

    q_mean : float
        Mean of the predicted action values.
    """
    chex.assert_equal_shape_prefix((observation, action), prefix_len=1)
    chex.assert_equal_shape_prefix((observation, q_target_values), prefix_len=1)

    q_predicted = q(jnp.concatenate((observation, action), axis=-1)).squeeze()
    chex.assert_equal_shape((q_predicted, q_target_values))

    return (
        optax.squared_error(
            predictions=q_predicted, targets=q_target_values
        ).mean(),
        q_predicted.mean(),
    )


def ddpg_loss(
    q: nnx.Module,
    q_target_value: nnx.Module,
    policy_target: nnx.Module,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float,
) -> tuple[float, float]:
    r"""Critic loss of DDPG.

    This loss requires continuous state and action spaces.

    For a mini-batch, we calculate target values of :math:`Q'`

    .. math::

        y_i = r_i + (1 - t_i) \gamma Q'(o_{i+1}, \pi'(o_{i+1})),

    where :math:`r_i` (``reward``) is the immediate reward obtained in the
    transition, :math:`o_{i+1}` (``next_observation``) is the observation
    after the transition, :math:`\pi` is the deterministic policy network,
    :math:`\gamma` (``gamma``) is the discount factor, and :math:`t_i`
    (``terminated``) indicates if a terminal state was reached in this
    transition.

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : nnx.Module
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target_value : nnx.Module
        Target network for ``q``.
    policy_target : nnx.Module
        Deterministic target policy :math:`\pi'(o) = a`.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given mini-batch.

    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T.,
       Tassa, Y., Silver, D. & Wierstra, D. (2016). Continuous control with
       deep reinforcement learning. In 4th International Conference on Learning
       Representations, {ICLR} 2016, San Juan, Puerto Rico, May 2-4, 2016,
       Conference Track Proceedings. http://arxiv.org/abs/1509.02971
    """
    observation, action, reward, next_observation, terminated = batch
    next_actions = jax.lax.stop_gradient(policy_target(next_observation))
    next_obs_act = jnp.concatenate((next_observation, next_actions), axis=-1)
    q_next = jax.lax.stop_gradient(q_target_value(next_obs_act).squeeze())
    q_target_value = reward + (1 - terminated) * gamma * q_next
    return mse_continuous_action_value_loss(
        observation, action, q_target_value, q
    )


def td3_loss(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    next_action: jnp.ndarray,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float,
) -> tuple[float, float]:
    r"""Critic loss of TD3.

    This loss requires continuous state and action spaces.

    For a mini-batch, we calculate target values of :math:`Q'`

    .. math::

        y_i = r_i + (1 - t_i)
        \gamma \min(Q_1(o_{i+1}, a_{i+1}), Q_2(o_{i+1}, a_{i+1})),

    where :math:`r_i` (``reward``) is the immediate reward obtained in the
    transition, :math:`o_{i+1}` (``next_observation``) is the observation
    after the transition, :math:`a_{i+1}` (``next_action``) is the next action,
    :math:`\gamma` (``gamma``) is the discount factor, and :math:`t_i`
    (``terminated``) indicates if a terminal state was reached in this
    transition.

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : ContinuousClippedDoubleQNet
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target : ContinuousClippedDoubleQNet
        Target network for ``q``.
    next_action : jnp.ndarray
        Sampled target actions :math:`a_{t+1}`.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given mini-batch.
    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T.,
       Tassa, Y., Silver, D. & Wierstra, D. (2016). Continuous control with
       deep reinforcement learning. In 4th International Conference on Learning
       Representations, {ICLR} 2016, San Juan, Puerto Rico, May 2-4, 2016,
       Conference Track Proceedings. http://arxiv.org/abs/1509.02971
    """
    observation, action, reward, next_observation, terminated = batch
    next_obs_act = jnp.concatenate((next_observation, next_action), axis=-1)
    q_next = jax.lax.stop_gradient(q_target(next_obs_act).squeeze())
    q_target_value = reward + (1 - terminated) * gamma * q_next
    return _mse_clipped_double_q_loss(q_target_value, q, action, observation)


def _mse_clipped_double_q_loss(q_target_value, q, action, observation):
    obs_act = jnp.concatenate((observation, action), axis=-1)
    q1_predicted = q.q1(obs_act).squeeze()
    q1_loss = optax.squared_error(
        predictions=q1_predicted, targets=q_target_value
    ).mean()
    q2_predicted = q.q2(obs_act).squeeze()
    q2_loss = optax.squared_error(
        predictions=q2_predicted, targets=q_target_value
    ).mean()
    return q1_loss + q2_loss, jnp.minimum(q1_predicted, q2_predicted).mean()


def td3_lap_loss(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    next_action: jnp.ndarray,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float,
    min_priority: float,
) -> tuple[float, tuple[float, jnp.ndarray]]:
    r"""Critic loss of TD3 with LAP.

    This loss requires continuous state and action spaces.

    For a mini-batch, we calculate target values of :math:`Q'`

    .. math::

        y_i = r_i + (1 - t_i)
        \gamma \min(Q_1(o_{i+1}, a_{i+1}), Q_2(o_{i+1}, a_{i+1})),

    where :math:`r_i` (``reward``) is the immediate reward obtained in the
    transition, :math:`o_{i+1}` (``next_observation``) is the observation
    after the transition, :math:`a_{i+1}` (``next_action``) is the next action,
    :math:`\gamma` (``gamma``) is the discount factor, and :math:`t_i`
    (``terminated``) indicates if a terminal state was reached in this
    transition.

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} Huber(y_i - Q(o_i, a_i)).

    Parameters
    ----------
    q : ContinuousClippedDoubleQNet
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target : ContinuousClippedDoubleQNet
        Target network for ``q``.
    next_action : jnp.ndarray
        Sampled target actions :math:`a_{t+1}`.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float
        Discount factor :math:`\gamma`.
    min_priority : float
        Minimum priority, delta for the Huber loss.

    Returns
    -------
    loss : float
        The computed loss for the given mini-batch.
    auxiliary : tuple
        Auxiliary information about the loss.
        (1) q_mean (float): Mean of the predicted action values.
        (2) max_abs_td_error (jnp.ndarray): Maximum over two Q networks of
        absolute temporal difference (TD) error for each sample in the batch.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html
    """
    observation, action, reward, next_observation, terminated = batch
    next_obs_act = jnp.concatenate((next_observation, next_action), axis=-1)
    q_next = jax.lax.stop_gradient(q_target(next_obs_act).squeeze())
    q_target_value = reward + (1 - terminated) * gamma * q_next
    obs_act = jnp.concatenate((observation, action), axis=-1)
    q1_predicted = q.q1(obs_act).squeeze()
    q2_predicted = q.q2(obs_act).squeeze()
    td_error1 = jnp.abs(q1_predicted - q_target_value)
    td_error2 = jnp.abs(q2_predicted - q_target_value)
    return (
        huber_loss(td_error1, min_priority).mean()
        + huber_loss(td_error2, min_priority).mean(),
        (
            jnp.minimum(q1_predicted, q2_predicted).mean(),
            jnp.maximum(td_error1, td_error2),
        ),
    )


def huber_loss(abs_errors: jnp.ndarray, delta: float) -> jnp.ndarray:
    # 0.5 * err^2                  if |err| <= d
    # 0.5 * d^2 + d * (|err| - d)  if |err| > d
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def sac_loss(
    q: ContinuousClippedDoubleQNet,
    q_target: ContinuousClippedDoubleQNet,
    policy: StochasticPolicyBase,
    action_key: jnp.ndarray,
    alpha: float,
    batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
    gamma: float,
) -> tuple[float, float]:
    r"""Soft Actor-Critic (SAC) loss.

    This loss requires continuous state and action spaces.

    The target values will be generated to estimate the soft Q function as
    described in [1]_ with a target network. In addition, we use a clipped
    double Q network.

    For a mini-batch, we calculate target values :math:`y_i` for the critic

    .. math::

        y_i = r_i + (1 - t_i) \gamma
        \left[\min(Q_1(o_{i+1}, a_{i+1}), Q_2(o_{i+1}, a_{i+1}))
        - \alpha \log \pi(a_{i+1}|o_{i+1})\right]

    where :math:`r_i` (``reward``) is the immediate reward obtained in the
    transition, :math:`o_{i+1}` (``next_observation``) is the observation
    after the transition, :math:`a_{i+1} \sim \pi(a_{i+1}|o_{i+1})` is the next
    action sampled from the ``policy``, :math:`\gamma` (``gamma``) is the
    discount factor, :math:`\alpha` is the entropy coefficient, and :math:`t_i`
    (``terminated``) indicates if a terminal state was reached in this
    transition.

    Based on these target values, the loss is defined as

    .. math::

        \mathcal{L}(Q) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(o_i, a_i))^2.

    Parameters
    ----------
    q : ContinuousClippedDoubleQNet
        Deep Q-network :math:`Q(o, a)`. For a given observation, the neural
        network predicts the value of each action from the discrete action
        space.
    q_target : ContinuousClippedDoubleQNet
        Target network for ``q``.
    policy : StochasticPolicyBase
        Policy.
    action_key : array
        Random key for action sampling.
    alpha : float
        Entropy coefficient.
    batch : tuple
        Mini-batch of transitions. Contains in this order: observations
        :math:`o_i`, actions :math:`a_i`, rewards :math:`r_i`, next
        observations :math:`o_{i+1}`, termination flags :math:`t_i`.
    gamma : float
        Discount factor :math:`\gamma`.

    Returns
    -------
    loss : float
        The computed loss for the given mini-batch.
    q_mean : float
        Mean of the predicted action values.

    References
    ----------
    .. [1] Haarnoja, T., Tang, H., Abbeel, P., Levine, S. (2017). Reinforcement
       Learning with Deep Energy-Based Policies. In Proceedings of the 34th
       International Conference on Machine Learning, PMLR 70:1352-1361, 2017.
       https://proceedings.mlr.press/v70/haarnoja17a.html
    """
    observation, action, reward, next_observation, terminated = batch

    next_actions = jax.lax.stop_gradient(
        policy.sample(next_observation, action_key)
    )
    next_log_pi = jax.lax.stop_gradient(
        policy.log_probability(next_observation, next_actions)
    )
    next_obs_act = jnp.concatenate((next_observation, next_actions), axis=-1)
    q_next_target = jax.lax.stop_gradient(
        q_target(next_obs_act).squeeze() - alpha * next_log_pi
    )
    q_target_value = reward + (1 - terminated) * gamma * q_next_target

    return _mse_clipped_double_q_loss(q_target_value, q, action, observation)


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
