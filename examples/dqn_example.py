import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.spaces.utils import flatdim

from rl_blox.algorithms.model_free.dqn import MLP, train_dqn

# Set up environment
env_name = "Taxi-v3"
env = gym.make(env_name, render_mode="human")
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

nnx_rngs = nnx.Rngs(seed)
q_net = MLP(1, 16, env.action_space.n, nnx_rngs)

# Train
q = train_dqn(
    q_net,
    env,
    epsilon=0.5,
    learning_rate=0.1,
    seed=seed,
    total_timesteps=10_000,
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
