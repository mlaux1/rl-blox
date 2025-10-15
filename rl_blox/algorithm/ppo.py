import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
from tqdm.rich import trange
from collections import namedtuple
from typing import Any

from ..blox.gae import compute_gae

@nnx.jit
def select_action_deterministic(
    actor: nnx.Module,
    obs: jnp.ndarray,
    key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select an action using the actor's policy in a deterministic way.

    Parameters
    ----------
    actor : nnx.Module
        The actor network.
    obs : jnp.ndarray
        Last observation.
    key : jnp.ndarray
        Random key. Used for action sampling.

    Returns
    -------
    action : jnp.ndarray
        Selected action.
    logp : jnp.ndarray
        Log-probability of the selected action.
    """
    logits = actor(obs)
    probs = jax.nn.softmax(logits)
    action = jax.random.categorical(key, logits)
    logp = jnp.log(probs[action])
    return namedtuple('SelectedAction', ['action', 'logp'])(action, logp)


def ppo_loss(
    actor: nnx.Module,
    critic: nnx.Module,
    old_logps: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip: float=0.2
) -> jnp.ndarray:
    """
    Calculate the PPO loss.

    Parameters
    ----------
    actor : nnx.Module
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
    logits = actor(observations)
    probs = jax.nn.softmax(logits)
    logps = jnp.log(jnp.take_along_axis(probs, actions[:, None], axis=1).squeeze())

    ratios = jnp.exp(logps - old_logps)
    surrogate1 = ratios * advantages
    surrogate2 = jnp.clip(ratios, 1-clip, 1+clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = critic(observations)
    value_loss = jnp.mean((returns - values) ** 2)

    entropy = -jnp.mean(jnp.sum(probs * (logits - jax.scipy.special.logsumexp(logits)), axis=-1))
    return policy_loss + 0.5 * value_loss - 0.01 * entropy


def collect_trajectories(
    env: gym.Env,
    actor: nnx.Module,
    critic: nnx.Module,
    key: jnp.ndarray,
    batch_size: int=64,
    last_observation=None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
    """
    Run and collect trajectories until at least `batch_size` steps are gathered.

    Parameters
    ----------
    env : gym.Env
        The environment to interact with.
    actor : nnx.Module
        The actor network.
    critic : nnx.Module
        The critic network.
    key : jnp.ndarray
        Random key.
    batch_size : int, optional
        Minimum number of steps to collect.
    last_observation
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.

    Returns
    -------
    - observations : jnp.ndarray
        Array of observations.
    - actions : jnp.ndarray
        Actions taken per step.
    - logps : jnp.ndarray
        Log probabilities of selected actions.
    - rewards : jnp.ndarray
        Array of rewards per step.
    - terminateds : jnp.ndarray
        Flags indicating episode termination per step.
    - next_values : jnp.ndarray
        Array of predicted values for next steps per step.
    last_observation
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    """
    actions, logps, observations, rewards, terminateds, next_values = [], [], [], [], [], []
    
    obs, _ = env.reset() if last_observation is None else last_observation, None
    for _ in range(batch_size):
        key, subkey = jax.random.split(key)
        action, logp = select_action_deterministic(actor, obs, subkey)
        next_obs, reward, terminated, truncated, _ = env.step(int(action))
        next_value = critic(obs)

        actions.append(action)
        logps.append(logp)
        observations.append(obs)
        rewards.append(reward)
        terminateds.append(terminated)
        next_values.append(next_value)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            
    return namedtuple('PPO_Trajectory',
                      ['observations', 'actions', 'logps', 'rewards', 'terminateds', 'next_values', 'last_observation'])(
        jnp.stack(observations),
        jnp.stack(actions),
        jnp.stack(logps),
        jnp.array(rewards).flatten(),
        jnp.array(terminateds, dtype=jnp.float32).flatten(),
        jnp.stack(next_values).flatten(),
        obs)


@nnx.jit
def calculate_ppo_grads(
    actor: nnx.Module,
    critic: nnx.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    old_logps: jnp.ndarray,
    rewards: jnp.ndarray,
    terminateds: jnp.ndarray,
    next_values: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate gradients for PPO update

    Args:
        actor : nnx.Module
            The actor network
        critic : nnx.Module
            The critic network
        observations : jnp.ndarray
            Array of observations.
        actions : jnp.ndarray
            Actions taken per step.
        old_logps : jnp.ndarray
            Log probabilities of selected actions.
        rewards : jnp.ndarray
            Array of rewards per step.
        terminateds : jnp.ndarray
            Flags indicating episode termination per step.
        next_values : jnp.ndarray
            Array of predicted next_values per step.

    Returns:
    - loss_val : jnp.ndarray
        Calculated loss.
    - grad_actor : jnp.ndarray
        Gradients for actor update.
    - grad_critic : jnp.ndarray
        Gradients for critic update.
    """
    advs, returns = compute_gae(rewards, critic(observations).flatten(), next_values, terminateds)
    loss_grad_fn = nnx.value_and_grad(ppo_loss, argnums=(0,1))
    (loss_val), (grad_actor, grad_critic) = loss_grad_fn(
            actor, critic, old_logps, observations, actions, advs, returns
    )   
    return loss_val, grad_actor, grad_critic


def train_ppo(
        env: gym.Env,
        actor: nnx.Module,
        critic: nnx.Module,
        optimizer_actor: nnx.Optimizer,
        optimizer_critic: nnx.Optimizer,
        episodes: int=1000,
        batch_size: int=64,
        seed: int=1,
        progress_bar: bool=True
    ) -> tuple[nnx.Module, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    """
    Train a PPO agent.

    Parameters
    ----------
    env : gym.Env
        The training environment.
    actor : nnx.Module
        The actor network.
    critic : nnx.Module
        The critic network.
    optimizer_actor : nnx.Optimizer
        Optimizer for the actor network.
    optimizer_critic : nnx.Optimizer
        Optimizer for the critic network.
    episodes : int, optional
        Number of training episodes.
    batch_size : int, optional
        Batch size per update.
    seed : int, optional
        Random seed for reproducibility.
    progress_bar : bool, optional
        Display a progress bar during training.

    Returns
    -------
    - actor : nnx.Module
        Trained actor network.
    - critic : nnx.Module
        Trained critic network.
    - optimizer_actor : nnx.Optimizer
        Updated actor optimizer.
    - optimizer_critic : nnx.Optimizer
        Updated critic optimizer.
    """
    key = jax.random.key(seed)
    last_observation, _ = env.reset(seed=seed)

    for episode in trange(episodes, disable=not progress_bar):
        key, subkey = jax.random.split(key)
        observations, actions, logps, rewards, terminateds, next_values, last_observation = collect_trajectories(
            env, actor, critic, subkey, batch_size, last_observation)

        loss_val, grad_actor, grad_critic = calculate_ppo_grads(
            actor, critic, observations, actions, logps, rewards, terminateds, next_values)
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

        if episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss_val:.3f}, Return: {sum(rewards)}")
            
    return actor, critic, optimizer_actor, optimizer_critic
