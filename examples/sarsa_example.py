from functools import partial

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from jax.random import PRNGKey

from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import get_epsilon_greedy_action, make_q_table
from rl_blox.util.experiment_helper import generate_rollout

NUM_TIMESTEPS = 200_00
LEARNING_RATE = 0.1
EPSILON = 0.2
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "CliffWalking-v0"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=2000)

q_table = make_q_table(env)

q_table = train_sarsa(
    KEY,
    env,
    q_table,
    alpha=LEARNING_RATE,
    epsilon=EPSILON,
    total_timesteps=NUM_TIMESTEPS,
)

env.close()

# create and run the final policy
policy = partial(
    get_epsilon_greedy_action, key=KEY, q_table=q_table, epsilon=0.2
)

print(q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
