from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.gae import compute_gae
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


def collect_rollout(
    env: gym.vector.SyncVectorEnv,
    policy: StochasticPolicyBase,
    value_function: nnx.Module,
    key: jnp.ndarray,
    steps_per_update: int,
    start_obs: jnp.ndarray,
) -> tuple[list, jnp.ndarray, jnp.ndarray]:
    """Collects a rollout of experience using the current policy."""

    rollout_data = []
    obs = start_obs

    for _ in range(steps_per_update):
        key, action_key = jax.random.split(key)

        mean_and_log_std = policy(obs)
        action_mean = mean_and_log_std[..., 0]
        log_std_param = mean_and_log_std[..., 1]
        action_std = jnp.exp(log_std_param)

        action_distribution = tfd.Normal(loc=action_mean, scale=action_std)
        actions = action_distribution.sample(seed=action_key)
        log_probs = action_distribution.log_prob(actions)

        values = value_function(obs).squeeze()

        actions_for_env = jnp.expand_dims(actions, axis=-1)
        next_obs, rewards, terminations, truncations, infos = env.step(
            actions_for_env
        )

        rollout_data.append(
            (
                obs,
                actions,
                rewards,
                values,
                log_probs,
                terminations,
            )
        )

        obs = next_obs

    last_values = value_function(obs).squeeze()

    return rollout_data, obs, last_values


def train_a2c(
    env: gym.vector.SyncVectorEnv,
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP,
    value_function_optimizer: nnx.Optimizer,
    seed: int = 0,
    num_envs: int = 1,
    total_timesteps: int = 1_000_000,
    steps_per_update: int = 1_000,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
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

        key, rollout_key = jax.random.split(key)
        rollout_data, obs, last_values = collect_rollout(
            env, policy, value_function, rollout_key, steps_per_update, obs
        )

        rollout_obs = [d[0] for d in rollout_data]
        rollout_actions = [d[1] for d in rollout_data]
        rollout_rewards = [d[2] for d in rollout_data]
        rollout_values = [d[3] for d in rollout_data]
        rollout_log_probs = [d[4] for d in rollout_data]
        rollout_terminations = [d[5] for d in rollout_data]

        advantages, returns = compute_gae(
            rewards=jnp.array(rollout_rewards),
            values=jnp.array(rollout_values),
            next_values=last_values,
            terminateds=jnp.array(rollout_terminations),
            gamma=gamma,
            lmbda=lmbda,
        )

        all_obs = jnp.concatenate(rollout_obs)
        all_actions = jnp.concatenate(rollout_actions)
        all_log_probs = jnp.concatenate(rollout_log_probs)
        all_advantages = advantages.flatten()
        all_returns = returns.flatten()

        print(f"all_obs: {all_obs}")
        print(f"all_actions: {all_actions}")
        print(f"all_log_probs: {all_log_probs}")
        print(f"Shape of all_returns: {all_returns.shape}")
        print(f"Shape of all_advantages: {all_advantages.shape}")

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
