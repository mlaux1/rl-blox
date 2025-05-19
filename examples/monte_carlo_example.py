from functools import partial

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.monte_carlo import train_monte_carlo
from rl_blox.blox.value_policy import get_greedy_action, make_q_table
from rl_blox.util.experiment_helper import generate_rollout

ENV_NAME = "CliffWalking-v0"
NUM_STEPS = 50_000


train_env = gym.make(ENV_NAME)
train_env = RecordEpisodeStatistics(train_env)

q_table = make_q_table(train_env)

q_table, _ = train_monte_carlo(train_env, q_table, total_timesteps=NUM_STEPS)

train_env.close()

print(q_table)

policy = partial(get_greedy_action, key=42, q_table=q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
