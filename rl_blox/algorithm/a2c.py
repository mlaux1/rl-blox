from collections import namedtuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.gae import compute_gae
from ..logging.logger import LoggerBase


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
                infos,
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

    env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    step = 0

    while step < total_timesteps:

        total_steps_collected = num_envs * steps_per_update

        key, rollout_key = jax.random.split(key)
        rollout_data, obs, last_values = collect_rollout(
            env, policy, value_function, rollout_key, steps_per_update, obs
        )

        for data_step in rollout_data:
            infos = data_step[6]

            if "episode" in infos:
                finished_envs_mask = infos["_episode"]

                finished_returns = infos["episode"]["r"][finished_envs_mask]
                finished_lengths = infos["episode"]["l"][finished_envs_mask]

                for i in range(len(finished_returns)):
                    ep_return = finished_returns[i]
                    ep_length = finished_lengths[i]
                    current_log_step = step + total_steps_collected
                    print(
                        f"Step {current_log_step}: "
                        f"Ep. Return={ep_return:.2f}, "
                        f"Length={ep_length}"
                    )
                    if logger is not None:
                        logger.record_stat(
                            "episodic_return", ep_return, current_log_step
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

        all_advantages = (all_advantages - all_advantages.mean()) / (
            all_advantages.std() + 1e-8
        )

        # print(f"all_obs: {all_obs}")
        # print(f"all_actions: {all_actions}")
        # print(f"all_log_probs: {all_log_probs}")
        # print(f"Shape of all_returns: {all_returns.shape}")
        # print(f"Shape of all_advantages: {all_advantages.shape}")

        def actor_loss_fn(p: StochasticPolicyBase):
            mean_and_log_std = p(all_obs)
            action_mean, log_std_param = (
                mean_and_log_std[..., 0],
                mean_and_log_std[..., 1],
            )
            action_std = jnp.exp(log_std_param)
            dist = tfd.Normal(loc=action_mean, scale=action_std)
            log_probs_new = dist.log_prob(all_actions)
            entropy = dist.entropy().mean()
            policy_gradient_loss = -(log_probs_new * all_advantages).mean()
            return policy_gradient_loss - entropy_coef * entropy

        def critic_loss_fn(vf: nnx.Module):
            values_pred = vf(all_obs).squeeze()
            return jnp.mean((all_returns - values_pred) ** 2)

        p_loss, p_grad = nnx.value_and_grad(actor_loss_fn)(policy)
        v_loss, v_grad = nnx.value_and_grad(critic_loss_fn)(value_function)

        policy_optimizer.update(policy, p_grad)
        value_function_optimizer.update(value_function, v_grad)

        step += total_steps_collected
        progress.update(total_steps_collected)

    progress.close()
    return namedtuple(
        "A2CResult",
        [
            "policy",
            "policy_optimizer",
            "value_function",
            "value_function_optimizer",
        ],
    )(policy, policy_optimizer, value_function, value_function_optimizer)
