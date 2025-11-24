from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.losses import stochastic_policy_gradient_pseudo_loss
from ..logging.logger import LoggerBase
from .reinforce import sample_trajectories, train_value_function


def a2c_policy_gradient(
    policy: StochasticPolicyBase,
    value_function: nnx.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    gamma_discount: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    r"""Actor-critic policy gradient.

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
    next_observations
        Samples that were collected with the policy.
    rewards
        Samples that were collected with the policy.
    gamma_discount
        Discounting for individual steps of the episode.
    gamma
        Discount factor.

    Returns
    -------
    loss
        Actor-critic policy gradient pseudo loss.
    grad
        Actor-critic policy gradient.

    See Also
    --------
    .blox.losses.stochastic_policy_gradient_pseudo_loss
        The pseudo loss that is used to compute the policy gradient. As
        weights for the pseudo loss we use the TD error
        :math:`\delta_t = r_t + \gamma v(o_{t+1}) - v(o_t)` multiplied by the
        discounting factor for the step of the episode.
    """
    v = value_function(observations).squeeze()
    v_next = value_function(next_observations).squeeze()
    td_bootstrap_estimate = rewards + gamma * v_next - v
    weights = gamma_discount * td_bootstrap_estimate

    return nnx.value_and_grad(
        stochastic_policy_gradient_pseudo_loss, argnums=3
    )(observations, actions, weights, policy)


def train_a2c(
    env: gym.vector.SyncVectorEnv,
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP,
    value_function_optimizer: nnx.Optimizer,
    seed: int = 0,
    num_envs: int = 1,
    policy_gradient_steps: int = 1,
    value_gradient_steps: int = 1,
    total_timesteps: int = 1_000_000,
    gamma: float = 1.0,
    steps_per_update: int = 1_000,
    train_after_episode: bool = False,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Optimizer, nnx.Module, nnx.Optimizer]:
    """Train with actor-critic.

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

    steps_per_update : int, optional
        Number of samples to collect before updating the policy. Alternatively
        you can train after each episode.

    train_after_episode : bool, optional
        Train after each episode. Alternatively you can train after collecting
        a certain number of samples.

    key : jnp.ndarray, optional
        Pseudo random number generator key for action sampling.

    logger : logger.LoggerBase, optional
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
    obs, _ = env.reset(seed=seed)
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    step = 0

    while step < total_timesteps:

        rollout_data = []

        for _ in range(steps_per_update):

            key, action_key = jax.random.split(key)

            mean_and_log_std = policy(obs)
            action_mean = mean_and_log_std[..., 0]
            log_std_param = mean_and_log_std[..., 1]

            action_std = jnp.exp(log_std_param)

            action_distribution = tfd.Normal(loc=action_mean, scale=action_std)

            actions = action_distribution.sample(seed=action_key)
            log_probs = action_distribution.log_prob(actions)

            values = value_function(obs)

            actions_for_env = jnp.expand_dims(actions, axis=-1)
            next_obs, rewards, terminations, truncations, infos = env.step(
                actions_for_env
            )

            rollout_data.append(
                (
                    obs,
                    actions,
                    rewards,
                    values.squeeze(),
                    log_probs,
                    terminations,
                )
            )

            obs = next_obs

        # Update progress bar and total step count
        total_steps_collected = num_envs * steps_per_update
        step += total_steps_collected
        progress.update(total_steps_collected)
        break  # For testing purposes

        observations, actions, next_observations, returns, gamma_discount = (
            dataset.prepare_policy_gradient_dataset(env.action_space, gamma)
        )
        rewards = jnp.hstack(
            [
                jnp.hstack([r for _, _, _, r in episode])
                for episode in dataset.episodes
            ]
        )

        p_loss = train_policy_a2c(
            policy,
            policy_optimizer,
            policy_gradient_steps,
            value_function,
            observations,
            actions,
            next_observations,
            rewards,
            gamma_discount,
            gamma,
        )
        if logger is not None:
            logger.record_stat(
                "policy loss", p_loss, episode=logger.n_episodes - 1
            )
            logger.record_epoch("policy", policy)

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

    # return namedtuple(
    #     "ActorCriticResult",
    #     [
    #         "policy",
    #         "policy_optimizer",
    #         "value_function",
    #         "value_function_optimizer",
    #     ],
    # )(policy, policy_optimizer, value_function, value_function_optimizer)
    return rollout_data  # For testing purposes


@partial(nnx.jit, static_argnames=["policy_gradient_steps", "gamma"])
def train_policy_a2c(
    policy,
    policy_optimizer,
    policy_gradient_steps,
    value_function,
    observations,
    actions,
    next_observations,
    rewards,
    gamma_discount,
    gamma,
):
    p_loss = 0.0
    for _ in range(policy_gradient_steps):
        p_loss, p_grad = a2c_policy_gradient(
            policy,
            value_function,
            observations,
            actions,
            next_observations,
            rewards,
            gamma_discount,
            gamma,
        )
        policy_optimizer.update(policy, p_grad)
    return p_loss
