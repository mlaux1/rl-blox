import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithms.model_free.dqn import MLP, train_dqn

# Set up environment
env_name = "CliffWalking-v0"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)


# Train
q = train_dqn(
    env,
    epsilon=0.05,
    seed=seed,
    total_timesteps=1_000,
)
env.close()

eval_env = gym.make(env_name, render_mode="human")

obs, _ = eval_env.reset()


while True:
    action = int(jnp.argmax(q([obs])))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs
