import gymnasium as gym

from rl_blox.blox.value_policy import get_greedy_action, make_q_table
from rl_blox.algorithm.dynaq import train_dynaq

env = gym.make("CliffWalking-v0")

q_table = make_q_table(env)

train_dynaq(
    env,
    q_table,
    total_timesteps=1_000,
)
env.close()
