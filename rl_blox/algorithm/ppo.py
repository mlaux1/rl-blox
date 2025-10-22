from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.policy_head import (
    GaussianPolicy,
    GaussianTanhPolicy,
    SoftmaxPolicy,
    StochasticPolicyBase,
)
from ..blox.gae import compute_gae
from ..logging.logger import LoggerBase


def collect_trajectories(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    key: jnp.ndarray,
    batch_size: int = 64,
    logger: LoggerBase | None = None,
    last_observation=None,
    global_step: int = 0,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Any,
    int,
]:
    """
    Run and collect trajectories until at least `batch_size` steps are gathered.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        The vectorized environment to interact with.
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    key : jnp.ndarray
        Random key.
    batch_size : int, optional
        Minimum number of steps to collect.
    logger : LoggerBase, optional
        Experiment Logger.
    last_observation : Any, optional
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    global_step : int, optional
        Global step count

    Returns
    -------
    - observation : jnp.ndarray
        Array of observations.
    - action : jnp.ndarray
        Actions taken per step.
    - reward : jnp.ndarray
        Array of rewards per step.
    - terminated : jnp.ndarray
        Flags indicating episode termination per step.
    - next_value : jnp.ndarray
        Array of predicted values for next steps per step.
    last_observation
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    global_step : int, optional
        Global step count
    """

    @nnx.jit
    def sample(policy, observation, subkey):
        return policy.sample(observation, subkey)

    @nnx.jit
    def value(value_fn, observation):
        return value_fn(observation).flatten()

    def add_to_batch(batch, value):
        return (
            jnp.array(value[None, ...])
            if batch == None
            else jnp.concat([batch, value[None, ...]], axis=0)
        )

    observations, actions, rewards, terminated_arr, next_values = (
        None,
        None,
        None,
        None,
        None,
    )
    obs = envs.reset()[0] if last_observation is None else last_observation

    for _ in range(batch_size):
        key, subkey = jax.random.split(key)
        action = sample(actor, obs, subkey)
        next_obs, reward, terminated, truncated, info = envs.step(
            np.asarray(action)
        )

        observations = add_to_batch(observations, obs)
        actions = add_to_batch(actions, action)
        rewards = add_to_batch(rewards, reward)

        obs = jnp.copy(next_obs)
        if "episode" in info.keys():
            if logger is not None:
                finished_reward_len_obs = [
                    (r, l, o)
                    for r, l, o, f in zip(
                        info["episode"]["r"],
                        info["episode"]["l"],
                        info["final_obs"],
                        info["_episode"],
                    )
                    if f
                ]
                for i, (r, l, o) in enumerate(finished_reward_len_obs):
                    global_step += int(l)
                    logger.record_stat("return", float(r), step=global_step)
                    logger.start_new_episode()
                    obs = obs.at[i].set(o)

        next_value = value(critic, obs)
        terminated_arr = add_to_batch(terminated_arr, terminated)
        next_values = add_to_batch(next_values, next_value)
        obs = next_obs

    def reshape_batch(batch):
        return jnp.permute_dims(batch, (1, 0)).flatten()

    def reshape_obs_batch(observations):
        return jnp.permute_dims(observations, (1, 0, 2)).reshape(
            -1, envs.observation_space.shape[1]
        )

    return namedtuple(
        "PPO_Trajectory",
        [
            "observation",
            "action",
            "reward",
            "terminated",
            "next_value",
            "last_observation",
            "global_step",
        ],
    )(
        reshape_obs_batch(observations),
        reshape_batch(actions),
        reshape_batch(rewards),
        reshape_batch(terminated_arr),
        reshape_batch(next_values),
        obs,
        global_step,
    )


def entropy(
    actor: StochasticPolicyBase, observations: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the entropy for PPO loss.

    Parameters
    ----------
    actor : StochasticPolicyBase
        The actor network.
    observations : jnp.ndarray
        Batch of observations.

    Returns
    -------
    entropy : jnp.ndarray
        The computed entropy.
    """
    if type(actor) == SoftmaxPolicy:
        logits = actor.logits(observations)
        entropy = dist.Categorical(logits=logits).entropy()
    else:
        if type(actor) == GaussianTanhPolicy:
            mean, std = actor(observations)
        elif type(actor) == GaussianPolicy:
            mean, log_var = actor(observations)
            log_std = jnp.clip(0.5 * log_var, -20.0, 2.0)
            std = jnp.exp(log_std)
        entropy = dist.Normal(loc=mean, scale=std).entropy()
    return jnp.mean(entropy)


def ppo_loss(
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    old_logps: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip: float = 0.2,
) -> jnp.ndarray:
    """
    Calculate the PPO loss.

    Parameters
    ----------
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    old_logps : jnp.ndarray
        Log probabilities of actions calculated during rollout.
    observations : jnp.ndarray
        Batch of observations.
    actions : jnp.ndarray
        Actions taken in each observation.
    advantages : jnp.ndarray
        Estimated advantages for each action.
    returns : jnp.ndarray
        Computed returns.
    clip : float, optional
        Clipping range for the PPO objective.

    Returns
    -------
    loss : jnp.ndarray
        The computed PPO loss for the batch.
    """
    logps = actor.log_probability(observations, actions)
    ratios = jnp.exp(logps - old_logps)
    surrogate1 = ratios * advantages
    surrogate2 = jnp.clip(ratios, 1 - clip, 1 + clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = critic(observations)
    value_loss = jnp.mean((returns - values) ** 2)

    return policy_loss + 0.5 * value_loss - 0.01 * entropy(actor, observations)


def update_ppo(
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    next_value: jnp.ndarray,
    epochs: int = 1,
) -> jnp.ndarray:
    """
    Updates the PPO agent

    Args:
        actor : StochasticPolicyBase
            The actor network
        critic : nnx.Module
            The critic network
        observation : jnp.ndarray
            Array of observations.
        action : jnp.ndarray
            Actions taken per step.
        reward : jnp.ndarray
            Array of rewards per step.
        terminated : jnp.ndarray
            Flags indicating episode termination per step.
        next_value : jnp.ndarray
            Array of predicted next_values per step.
        epochs : int, optional
            Number of training epochs.

    Returns:
    - loss_val : jnp.ndarray
        Calculated loss.
    """
    advs, returns = compute_gae(
        reward, critic(observation).flatten(), next_value, terminated
    )
    logp = actor.log_probability(observation, action)
    loss_grad_fn = nnx.value_and_grad(ppo_loss, argnums=(0, 1))

    for _ in range(epochs):
        (loss_val), (grad_actor, grad_critic) = loss_grad_fn(
            actor, critic, logp, observation, action, advs, returns
        )
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

    return loss_val


def train_ppo(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    iterations: int = 3000,
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    """
    Train a PPO agent.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        The vectorized training environment.
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    optimizer_actor : nnx.Optimizer
        Optimizer for the actor network.
    optimizer_critic : nnx.Optimizer
        Optimizer for the critic network.
    iterations : int, optional
        Number of training iterations.
    epochs : int, optional
        Number of training epochs per iteration.
    batch_size : int, optional
        Batch size per update.
    seed : int, optional
        Random seed for reproducibility.
    logger : LoggerBase, optional
        Experiment Logger.
    progress_bar : bool, optional
        Display a progress bar during training.

    Returns
    -------
    - actor : StochasticPolicyBase
        Trained actor network.
    - critic : nnx.Module
        Trained critic network.
    - optimizer_actor : nnx.Optimizer
        Updated actor optimizer.
    - optimizer_critic : nnx.Optimizer
        Updated critic optimizer.
    """
    key = jax.random.key(seed)
    last_observation, _ = envs.reset(seed=seed)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert (
        envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    ), "Vectorized Env has to be instantiated with the SAME_STEP autoreset mode."

    if logger is not None:
        logger.start_new_episode()

    update_ppo_jitted = nnx.jit(update_ppo, static_argnames="epochs")

    global_step = 0
    for iteration in trange(iterations, disable=not progress_bar):
        key, subkey = jax.random.split(key)
        (
            observation,
            action,
            reward,
            terminated,
            next_value,
            last_observation,
            global_step,
        ) = collect_trajectories(
            envs,
            actor,
            critic,
            subkey,
            batch_size,
            logger,
            last_observation,
            global_step,
        )

        loss_val = update_ppo_jitted(
            actor,
            critic,
            optimizer_actor,
            optimizer_critic,
            observation,
            action,
            reward,
            terminated,
            next_value,
            epochs,
        )

        if logger is not None:
            logger.record_stat("loss", loss_val, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
