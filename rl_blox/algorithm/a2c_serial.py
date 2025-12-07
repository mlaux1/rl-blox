from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm.rich import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.gae import compute_gae
from ..blox.losses import stochastic_policy_gradient_pseudo_loss
from ..logging.logger import LoggerBase
from .reinforce import EpisodeDataset, sample_trajectories, train_value_function


def a2c_policy_gradient(
    policy: StochasticPolicyBase,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""A2C policy gradient.

    Parameters
    ----------
    policy
        Probabilistic policy that we want to update and has been used for
        exploration.
    observations
        Samples that were collected with the policy.
    actions
        Samples that were collected with the policy.
    advantages
        Pre-computed advantage estimates (via GAE).

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


def train_a2c(
    env: gym.Env,
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
    steps_per_update: int = 1_000,
    train_after_episode: bool = False,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Optimizer, nnx.Module, nnx.Optimizer]:
    """Train with Advantage Actor-Critic (A2C).

    This implementation uses a single environment (synchronous) and calculates
    advantages using Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    env : gym.Env
        Environment.

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
        Number of gradient descent steps per update for the policy.

    value_gradient_steps : int, optional
        Number of gradient descent steps per update for the value function.

    total_timesteps : int
        Total timesteps of the experiment.

    gamma : float, optional
        Discount factor for rewards.

    gae_lambda : float, optional
        Smoothing factor for GAE (bias-variance trade-off).

    steps_per_update : int, optional
        Number of samples to collect before updating.

    train_after_episode : bool, optional
        Train after each episode.

    key : jnp.ndarray, optional
        Pseudo random number generator key for action sampling.

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
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    step = 0
    total_episodes_finished = 0

    while step < total_timesteps:
        key, skey = jax.random.split(key, 2)
        dataset = sample_trajectories(
            env, policy, skey, logger, train_after_episode, steps_per_update
        )
        step += len(dataset)
        progress.update(len(dataset))

        num_episodes_in_batch = len(dataset.episodes)
        batch_average_return = dataset.average_return()
        print(
            f"Step ~{step}: "
            f"Finished {num_episodes_in_batch} episodes. "
            f"Average Return in batch = {batch_average_return:.2f}"
        )
        total_episodes_finished += num_episodes_in_batch

        observations, actions, advantages, returns = prepare_a2c_dataset(
            dataset, value_function, env.action_space, gamma, gae_lambda
        )

        p_loss = train_policy_a2c(
            policy,
            policy_optimizer,
            policy_gradient_steps,
            observations,
            actions,
            advantages,
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

    return namedtuple(
        "A2CResult",
        [
            "policy",
            "policy_optimizer",
            "value_function",
            "value_function_optimizer",
        ],
    )(policy, policy_optimizer, value_function, value_function_optimizer)


def prepare_a2c_dataset(
    dataset: EpisodeDataset,
    value_function: nnx.Module,
    action_space: gym.spaces.Space,
    gamma: float,
    lmbda: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes GAE advantages and targets for the collected episodes.
    """
    all_observations = []
    all_actions = []
    all_advantages = []
    all_returns = []

    for episode in dataset.episodes:
        obs, acts, next_obs, rewards = zip(*episode)

        obs_arr = jnp.array(obs)
        rewards_arr = jnp.array(rewards)

        values = value_function(obs_arr).squeeze()

        if values.ndim == 0:
            values = jnp.expand_dims(values, 0)

        values_exp = jnp.expand_dims(values, -1)
        rewards_exp = jnp.expand_dims(rewards_arr, -1)

        next_value_exp = jnp.array([0.0])

        terminateds_exp = jnp.zeros_like(rewards_exp, dtype=jnp.float32)
        terminateds_exp = terminateds_exp.at[-1].set(1.0)

        gae_result = compute_gae(
            rewards_exp,
            values_exp,
            next_value_exp,
            terminateds_exp,
            gamma=gamma,
            lmbda=lmbda,
        )

        all_observations.extend(obs)
        all_actions.extend(acts)

        all_advantages.append(gae_result.advantages.flatten())
        all_returns.append(gae_result.returns.flatten())

    observations = jnp.array(all_observations)
    actions = jnp.array(all_actions)

    if isinstance(action_space, gym.spaces.Discrete):
        actions -= action_space.start

    advantages = jnp.concatenate(all_advantages)
    returns = jnp.concatenate(all_returns)

    return observations, actions, advantages, returns


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
