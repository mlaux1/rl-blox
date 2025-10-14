import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
from tqdm.rich import trange
from collections import namedtuple

from rl_blox.blox.function_approximator.mlp import MLP

@nnx.jit
def select_action_deterministic(
    actor: MLP,
    obs: jax.Array,
    key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Select an action using the actor's policy in a deterministic way

    Args:
        actor (MLP): Actor MLP
        obs (jax.Array): Observation
        key (jax.Array): Key. Used for sampling the action

    Returns:
        tuple[jax.Array, jax.Array]: Selected action and it's probability
    """
    logits = actor(obs)
    probs = jax.nn.softmax(logits)
    action = jax.random.categorical(key, logits)
    logp = jnp.log(probs[action])
    return namedtuple('SelectedAction', ['action', 'logp'])(action, logp)

@jax.jit
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float=0.99,
    lam: float=0.95
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation

    Args:
        rewards (jax.Array): Rewards
        values (jax.Array): Values
        dones (jax.Array): Dones
        gamma (float, optional): Gamma. Defaults to 0.99.
        lam (float, optional): Lambda. Defaults to 0.95.

    Returns:
        tuple[jax.Array, jax.Array]: Advantages and returns per step
    """
    def step(carry, inputs):
        gae, next_value = carry
        reward, value, done = inputs
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        step,
        (0.0, 0.0),
        (rewards[::-1], values[::-1], dones[::-1])
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return namedtuple('GAE', ['advantages', 'returns'])(advantages, returns)

@nnx.jit
def ppo_loss(
    actor: MLP,
    critic: MLP,
    old_logps: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip: float=0.2
) -> jax.Array:
    """Calculates the PPO loss

    Args:
        actor (MLP): Actor
        critic (MLP): Critic
        old_logps (jax.Array): Log probabilites of the actions calculated during collecting trajectories
        states (jax.Array): Observations
        actions (jax.Array): Actions
        advs (jax.Array): Advantages
        returns (jax.Array): Returns
        clip (float, optional): Clip values. Defaults to 0.2.

    Returns:
        jax.Array: PPO loss for batch
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

    entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1))
    return policy_loss + 0.5 * value_loss - 0.01 * entropy

def collect_trajectories(
    env: gym.Env,
    actor: nnx.Module,
    critic: nnx.Module,
    key: jax.Array,
    batch_size: int=64
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Runs and collects trajectories until at least batch_size amount
    of steps are given.

    Args:
        env (gym.Env): Environment
        actor (nnx.Module): Actor
        critic (nnx.Module): Critic
        key (jax.Array): Random key
        batch_size (int, optional): Minimum amount of steps to run. Defaults to 64.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]: 
        actions, logps, observations, rewards, dones, values
    """
    actions, logps, observations, rewards, dones, values = [], [], [], [], [], []
    
    obs, _ = env.reset()
    while True:
        value = critic(obs)
        key, subkey = jax.random.split(key)
        action, logp = select_action_deterministic(actor, obs, subkey)
        next_obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

        actions.append(action)
        logps.append(logp)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        obs = next_obs
        if done:
            obs, _ = env.reset()
            if len(observations) > batch_size:
                break
            
    return namedtuple('PPO_Trajectory', ['observations', 'actions', 'logps', 'rewards', 'dones', 'values'])(
        jnp.stack(observations),
        jnp.stack(actions),
        jnp.stack(logps),
        jnp.array(rewards).flatten(),
        jnp.array(dones, dtype=jnp.float32).flatten(),
        jnp.stack(values).flatten())

def train(
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
    """Train a PPO agent

    Args:
        env (gym.Env): Environment
        actor (nnx.Module): Actor
        critic (nnx.Module): Critic
        optimizer_actor (nnx.Optimizer): Actor optimizer
        optimizer_critic (nnx.Optimizer): Critic optimizer
        episodes (int, optional): Episode count. Defaults to 1000.
        batch_size (int, optional): Batch size. Defaults to 64.
        seed (int, optional): Seed. Defaults to 1.
        progress_bar (bool, optional): Display progress bar. Defaults to True.

    Returns:
        tuple[nnx.Module, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
        Trained actor, critic, actor optimizer and critic optimizer
    """
    key = jax.random.key(seed)
    env.reset(seed=seed)

    for episode in trange(episodes, disable=not progress_bar):
        key, subkey = jax.random.split(key)
        states, actions, old_logps, rewards, dones, values = collect_trajectories(
            env, actor, critic, subkey, batch_size)

        advs, returns = compute_gae(rewards, values, dones)
        key, subkey = jax.random.split(key)
        samples = jax.random.choice(subkey, advs.shape[0], shape=(batch_size,), replace=False)

        loss_grad_fn = nnx.value_and_grad(ppo_loss, argnums=(0,1))
        (loss_val), (grad_actor, grad_critic) = loss_grad_fn(
            actor, critic, old_logps[samples], states[samples],
            actions[samples], advs[samples], returns[samples])
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

        if episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss_val:.3f}, Return: {sum(rewards)}")
            
    return actor, critic, optimizer_actor, optimizer_critic
