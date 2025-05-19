import gymnasium as gym
import jax
import jax.numpy as jnp


def generate_rollout(
    env: gym.Env,
    policy,
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.key(seed)

    observation, _ = env.reset()
    terminated = False
    truncated = False

    obs = []
    actions = []
    rewards = []

    obs.append(observation)

    while not terminated or truncated:
        key, subkey = jax.random.split(key)

        action = policy(observation=observation, key=subkey)
        observation, reward, terminated, truncated, info = env.step(int(action))

        obs.append(observation)
        actions.append(action)
        rewards.append(reward)

    return jnp.array(obs), jnp.array(actions), jnp.array(rewards)
