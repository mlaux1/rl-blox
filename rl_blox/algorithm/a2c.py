from collections import deque, namedtuple
from functools import partial
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.gae import compute_gae
from ..blox.losses import stochastic_policy_gradient_pseudo_loss
from ..logging.logger import LoggerBase
from .reinforce import train_value_function

BatchDataset = namedtuple(
    "BatchDataset",
    ["obs", "actions", "rewards", "terminations", "truncations"],
)


def collect_trajectories(
    envs: gym.vector.VectorEnv,
    policy: StochasticPolicyBase,
    key: jnp.ndarray,
    last_observation: jnp.ndarray,
    steps_per_update: int,
    logger: LoggerBase | None = None,
    global_step: int = 0,
) -> tuple[BatchDataset, jnp.ndarray, int, list[float]]:
    """
    Collects a batch of trajectories from vectorized environments.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        Vectorized environment.
    policy : StochasticPolicyBase
        Policy used for sampling actions.
    key : jnp.ndarray
        JAX random key for action sampling.
    last_observation : jnp.ndarray
        The observation from the previous step (or reset) to start rolling out from.
        Shape: (Num_Envs, Obs_Dim).
    steps_per_update : int
        Number of steps to run per environment. Total samples collected will be
        steps_per_update * num_envs.
    logger : LoggerBase | None, optional
        Logger to record episodic returns.
    global_step : int, optional
        Current global step count for logging alignment.

    Returns
    -------
    dataset : BatchDataset
        The collected batch of data. Contains:
        - obs: Observations (Time, Num_Envs, *Obs_Shape)
        - actions: Actions (Time, Num_Envs, *Action_Shape)
        - rewards: Rewards (Time, Num_Envs)
        - terminations: Termination flags (Time, Num_Envs)
        - truncations: Truncation flags (Time, Num_Envs)
    last_observation : jnp.ndarray
        The final observation after the rollout, used for bootstrapping value estimates.
    global_step : int
        The updated global step count.
    episodic_returns : list[float]
        A list of returns for all episodes that finished during this collection window.
    """
    obs_list, act_list, rew_list, term_list, trunc_list = [], [], [], [], []
    episodic_returns = []

    obs = last_observation
    num_envs = envs.num_envs

    rng = key

    for _ in range(steps_per_update):
        rng, subkey = jax.random.split(rng)

        action = np.array(policy.sample(obs, subkey))

        next_obs, reward, terminated, truncated, infos = envs.step(action)

        if "episode" in infos and "_episode" in infos:
            finished_mask = infos["_episode"]
            if np.any(finished_mask):
                new_returns = infos["episode"]["r"][finished_mask]
                episodic_returns.extend(new_returns)

                if logger is not None:
                    for ret in new_returns:
                        logger.record_stat("return", ret, step=global_step)
                        logger.start_new_episode()

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        term_list.append(terminated)
        trunc_list.append(truncated)

        obs = next_obs
        global_step += num_envs

    dataset = BatchDataset(
        obs=jnp.array(np.stack(obs_list)),
        actions=jnp.array(np.stack(act_list)),
        rewards=jnp.array(np.stack(rew_list)),
        terminations=jnp.array(np.stack(term_list)),
        truncations=jnp.array(np.stack(trunc_list)),
    )

    return dataset, jnp.array(obs), global_step, episodic_returns


def train_a2c(
    envs: gym.vector.VectorEnv,
    policy: StochasticPolicyBase,
    policy_optimizer: nnx.Optimizer,
    value_function: MLP,
    value_function_optimizer: nnx.Optimizer,
    seed: int = 0,
    policy_gradient_steps: int = 1,
    value_gradient_steps: int = 1,
    total_timesteps: int = 1_000_000,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    steps_per_update: int = 5,
    log_frequency: int = 20_000,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Optimizer, nnx.Module, nnx.Optimizer]:
    """
    Train Advantage Actor-Critic (A2C) with Vectorized Environments.

    Parameters
    ----------
    env : gym.vector.VectorEnv
        The vectorized environment.

    policy : nnx.Module
        Probabilistic policy network. Maps observations to probability
        distribution over actions.

    policy_optimizer : nnx.Optimizer
        Optimizer for policy network.

    value_function : nnx.Module
        Value network. Maps observations to expected returns.

    value_function_optimizer : nnx.Optimizer
        Optimizer for value function network.

    seed : int, optional
        Seed for random number generation.

    policy_gradient_steps : int, optional
        Number of gradient descent steps per data batch for the policy.

    value_gradient_steps : int, optional
        Number of gradient descent steps per data batch for the value function.

    total_timesteps : int
        Total environment steps to train for (across all environments).

    gamma : float, optional
        Discount factor for rewards.

    gae_lambda : float, optional
        Smoothing factor for GAE (bias-variance trade-off).

    steps_per_update : int, optional
        Number of steps to collect per environment before performing an update
        (n-step lookahead).

    log_frequency : int, optional
        Frequency (in timesteps) to print stats to stdout.

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

    last_obs, _ = envs.reset(seed=seed)
    last_obs = jnp.array(last_obs)

    global_step = 0
    last_log_step = 0
    return_buffer = deque(maxlen=100)

    p_loss, v_loss = 0.0, 0.0

    progress = tqdm(total=total_timesteps, disable=not progress_bar)

    while global_step < total_timesteps:
        key, col_key = jax.random.split(key)

        dataset, last_obs, global_step, episodic_returns = collect_trajectories(
            envs,
            policy,
            col_key,
            last_obs,
            steps_per_update,
            logger,
            global_step,
        )

        return_buffer.extend(episodic_returns)
        steps_collected = steps_per_update * envs.num_envs
        progress.update(steps_collected)

        observations, actions, advantages, returns = prepare_a2c_batch(
            dataset,
            value_function,
            last_obs,
            envs.single_action_space,
            gamma,
            gae_lambda,
        )

        p_loss = train_policy_a2c(
            policy,
            policy_optimizer,
            policy_gradient_steps,
            observations,
            actions,
            advantages,
        )

        v_loss = train_value_function(
            value_function,
            value_function_optimizer,
            value_gradient_steps,
            observations,
            returns,
        )

        if logger is not None:
            logger.record_stat("policy_loss", p_loss, step=global_step)
            logger.record_stat("value_loss", v_loss, step=global_step)
            if global_step % (steps_collected * 10) == 0:
                logger.record_epoch("policy", policy)
                logger.record_epoch("value_function", value_function)

        if (
            log_frequency is not None
            and (global_step - last_log_step) >= log_frequency
        ):
            avg_return = (
                np.mean(return_buffer) if len(return_buffer) > 0 else 0.0
            )
            tqdm.write(
                f"Step: {global_step} | "
                f"Avg Return (last 100): {avg_return:.2f} | "
                f"P_Loss: {p_loss:.3f} | "
                f"V_Loss: {v_loss:.3f}"
            )
            last_log_step = global_step

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


def prepare_a2c_batch(
    dataset: BatchDataset,
    value_function: nnx.Module,
    last_observation: jnp.ndarray,
    action_space: gym.spaces.Space,
    gamma: float,
    lmbda: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes GAE for vectorized data and flattens dimensions.

    Parameters
    ----------
    dataset : BatchDataset
        The batch of experience collected from `collect_trajectories`.
    value_function : nnx.Module
        The critic network to estimate values.
    last_observation : jnp.ndarray
        The observation at the end of the batch used for bootstrapping.
    action_space : gym.spaces.Space
        The action space.
    gamma : float
        Discount factor.
    lmbda : float
        GAE smoothing parameter.

    Returns
    -------
    observations : jnp.ndarray
        Flattened observations (Time * Num_Envs, Features).
    actions : jnp.ndarray
        Flattened actions (Time * Num_Envs, Action_Dim).
    advantages : jnp.ndarray
        Flattened advantage estimates (Time * Num_Envs,).
    returns : jnp.ndarray
        Flattened return targets (Time * Num_Envs,).
    """
    T, N = dataset.obs.shape[:2]
    flat_obs = dataset.obs.reshape(-1, *dataset.obs.shape[2:])

    values = value_function(flat_obs).squeeze()
    values = values.reshape(T, N)

    next_values_bootstrap = value_function(last_observation).squeeze()
    bootstrap_expanded = jnp.expand_dims(next_values_bootstrap, 0)

    all_next_values = jnp.concatenate([values[1:], bootstrap_expanded], axis=0)

    def get_gae_for_env(rewards, vals, next_val, terms):
        return compute_gae(rewards, vals, next_val, terms, gamma, lmbda)

    gae_result = jax.vmap(get_gae_for_env, in_axes=(1, 1, 1, 1))(
        dataset.rewards, values, all_next_values, dataset.terminations
    )

    advantages = gae_result.advantages.T
    returns = gae_result.returns.T

    flat_observations = flat_obs
    flat_actions = dataset.actions.reshape(-1, *dataset.actions.shape[2:])
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)

    if isinstance(action_space, gym.spaces.Discrete):
        flat_actions -= action_space.start

    return flat_observations, flat_actions, flat_advantages, flat_returns


def a2c_policy_gradient(
    policy: StochasticPolicyBase,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    A2C policy gradient loss.

    Parameters
    ----------
    policy
        Probabilistic policy network.
    observations
        Batch of observations.
    actions
        Batch of actions taken.
    advantages
        Advantage estimates.

    Returns
    -------
    loss
        A2C policy gradient pseudo loss.
    grad
        Gradients for the policy parameters.
    """
    return nnx.value_and_grad(
        stochastic_policy_gradient_pseudo_loss, argnums=3
    )(observations, actions, advantages, policy)


@partial(nnx.jit, static_argnames=["policy_gradient_steps"])
def train_policy_a2c(
    policy,
    policy_optimizer,
    policy_gradient_steps,
    observations,
    actions,
    advantages,
):
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    normalized_advantages = (advantages - adv_mean) / adv_std

    p_loss = 0.0
    for _ in range(policy_gradient_steps):
        p_loss, p_grad = a2c_policy_gradient(
            policy,
            observations,
            actions,
            normalized_advantages,
        )
        policy_optimizer.update(policy, p_grad)
    return p_loss
