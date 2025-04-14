import gymnasium as gym
from rl_blox.algorithms.model_free.ext.crossq import train_crossq


env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
train_crossq(env)
