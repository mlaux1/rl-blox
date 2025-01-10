import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.spaces.utils import flatdim

from rl_blox.algorithms.model_free.dqn import MLP, train_dqn

# Set up environment
env_name = "CliffWalking-v0"
env = gym.make(env_name)
seed = 42
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

nnx_rngs = nnx.Rngs(seed)
q_net = MLP(1, 10, env.action_space.n, nnx_rngs)

# Train
q = train_dqn(
    q_net,
    env,
    epsilon=1.00,
    decay=0.97,
    learning_rate=0.001,
    seed=seed,
    total_timesteps=100_000,
)
env.close()

eval_env = gym.make(env_name, render_mode="human")

obs, _ = eval_env.reset()


while True:
    action = int(jnp.argmax(q([obs])))
    print(f"q_vals: {q([obs])} -> action {action}")
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs
