from functools import partial

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.double_q_learning import train_double_q_learning
from rl_blox.blox.value_policy import get_greedy_action, make_q_table
from rl_blox.util.experiment_helper import generate_rollout

NUM_STEPS = 50_000
LEARNING_RATE = 0.1
EPSILON = 0.1
ENV_NAME = "CliffWalking-v0"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=2000)

q_table1 = make_q_table(env)
q_table2 = make_q_table(env)

q_table1, q_table2 = train_double_q_learning(
    env,
    q_table1,
    q_table2,
    learning_rate=LEARNING_RATE,
    epsilon=EPSILON,
    total_timesteps=NUM_STEPS,
    seed=42,
)

env.close()


# create and run the final policy
q_table = q_table1 + q_table2
policy = partial(get_greedy_action, key=42, q_table=q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
