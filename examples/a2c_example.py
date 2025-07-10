import gymnasium as gym

from rl_blox.algorithm.a2c import train_a2c

env_name = "CartPole-v1"
train_envs = gym.make_vec(env_name, 2)
seed = 42


train_a2c(train_envs, None, None)
