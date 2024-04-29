from functools import partial

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from jax.random import PRNGKey

from modular_rl.algorithms.model_free.sarsa import sarsa
from modular_rl.helper.experiment_helper import generate_rollout
from modular_rl.policy.value_policy import get_greedy_action, make_q_table

NUM_EPISODES = 2000
LEARNING_RATE = 0.05
EPSILON = 0.05
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "CliffWalking-v0"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, deque_size=NUM_EPISODES)

q_table = make_q_table(env)

ep_rewards = sarsa(
    KEY, env, q_table,
    alpha=LEARNING_RATE, epsilon=EPSILON, num_episodes=NUM_EPISODES)

env.close()

# create and run the final policy
policy = partial(get_greedy_action, key=KEY, q_table=q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
