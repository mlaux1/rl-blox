import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
from tqdm.rich import trange
from collections import namedtuple


@nnx.jit
def select_action_deterministic(
    actor: nnx.Module,
    obs: jax.Array,
    key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Select an action using the actor's policy in a deterministic way.

    Parameters
    ----------
    actor : MLP
        The actor network.
    obs : jax.Array
        Last observation.
    key : jax.Array
        Random key. Used for action sampling.

    Returns
    -------
    action : jax.Array
        Selected action.
    logp : jax.Array
        Log-probability of the selected action.
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
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : jax.Array
        Array of rewards per step.
    values : jax.Array
        Array of predicted values per step.
    dones : jax.Array
        Flags indicating episode termination per step.
    gamma : float, optional
        Discount factor for rewards.
    lam : float, optional
        Smoothing factor for bias-variance trade-off.

    Returns
    -------
    - advantages : jax.Array
        Advantage estimates per step.
    - returns : jax.Array
        Computed returns per step.
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
    actor: nnx.Module,
    critic: nnx.Module,
    old_logps: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip: float=0.2
) -> jax.Array:
    """
    Calculate the PPO loss.

    Parameters
    ----------
    actor : nnx.Module
        The actor network.
    critic : nnx.Module
        The critic network.
    old_logps : jax.Array
        Log probabilities of actions calculated during rollout.
    observations : jax.Array
        Batch of observations.
    actions : jax.Array
        Actions taken in each observation.
    advantages : jax.Array
        Estimated advantages for each action.
    returns : jax.Array
        Computed returns.
    clip : float, optional
        Clipping range for the PPO objective.

    Returns
    -------
    loss : jax.Array
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
    key: jax.Array,
    batch_size: int=64
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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
    key : jax.Array
        Random key.
    batch_size : int, optional
        Minimum number of steps to collect.

    Returns
    -------
    - observations : jax.Array
        Array of observations.
    - actions : jax.Array
        Actions taken per step.
    - logps : jax.Array
        Log probabilities of selected actions.
    - rewards : jax.Array
        Array of rewards per step.
    - dones : jax.Array
        Flags indicating episode termination per step.
    - values : jax.Array
        Array of predicted values per step.
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
