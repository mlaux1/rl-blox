from functools import partial

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from jax.random import PRNGKey

from rl_blox.algorithm.q_learning import q_learning
from rl_blox.algorithm.sarsa import sarsa
from rl_blox.blox.value_policy import get_greedy_action, make_q_table
from rl_blox.util.experiment_helper import generate_rollout

NUM_EPISODES = 2000
LEARNING_RATE = 0.05
EPSILON = 0.05
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "Blackjack-v1"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=NUM_EPISODES)

# create the q table
q_table = make_q_table(env)

# train using Q-Learning
q_table, ep_rewards = q_learning(
    KEY,
    env,
    q_table,
    alpha=LEARNING_RATE,
    epsilon=EPSILON,
    num_episodes=NUM_EPISODES,
)

env.close()

# create and run the final policy
policy = partial(get_greedy_action, key=KEY, q_table=q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=NUM_EPISODES)

sarsa_q_table = make_q_table(env)

sarsa_q_table, ep_rewards = sarsa(
    KEY,
    env,
    sarsa_q_table,
    alpha=LEARNING_RATE,
    epsilon=EPSILON,
    num_episodes=NUM_EPISODES,
)

env.close()

# create and run the final policy
sarsa_policy = partial(get_greedy_action, key=KEY, q_table=sarsa_q_table)

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, sarsa_policy)
test_env.close()
