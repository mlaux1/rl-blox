import gymnasium as gym
import jax
import jax.numpy as jnp


def generate_rollout(
    env: gym.Env,
    policy,
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate one episode with a policy.

    Parameters
    ----------
    env : gym.Env
        Environment.

    policy : callable
        Policy, called as ``policy(observation=observation, key=key)``.

    seed : int, optional
        Seed for the environment reset and the policy's random keys.

    Returns
    -------
    observations : array, shape (n_steps + 1,) + observation shape
        Observations, including the last one.

    actions : array, shape (n_steps,) + action shape
        Actions.

    rewards : array, shape (n_steps,)
        Rewards.
    """
    key = jax.random.key(seed)

    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False

    obs = []
    actions = []
    rewards = []

    obs.append(observation)

    while not (terminated or truncated):
        key, subkey = jax.random.split(key)

        action = policy(observation=observation, key=subkey)
        observation, reward, terminated, truncated, info = env.step(int(action))

        obs.append(observation)
        actions.append(action)
        rewards.append(reward)

    return jnp.array(obs), jnp.array(actions), jnp.array(rewards)
