import gymnasium as gym
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
