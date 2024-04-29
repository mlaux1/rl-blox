import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from jax.random import PRNGKey

from modular_rl.algorithms.model_free.q_learning import q_learning
from modular_rl.algorithms.model_free.sarsa import sarsa
from modular_rl.policy.value_policy import make_q_table, get_greedy_action

NUM_EPISODES = 1000
LEARNING_RATE = 0.1
EPSILON = 0.05
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "Taxi-v3"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, deque_size=NUM_EPISODES)

# create the q table
q_table = make_q_table(env)

# train using Q-Learning
ep_rewards = q_learning(
    KEY, env, q_table,
    alpha=LEARNING_RATE, epsilon=EPSILON, num_episodes=NUM_EPISODES)

env.close()

# create and run the final policy

policy = get_greedy_action(KEY, )

# test_env = gym.make(ENV_NAME, render_mode="human")
# generate_rollout(test_env, policy)
# test_env.close()

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, deque_size=NUM_EPISODES)

sarsa_q_table = make_q_table(env)

ep_rewards = sarsa(
    KEY, env, q_table,
    alpha=LEARNING_RATE, epsilon=EPSILON, num_episodes=NUM_EPISODES)

env.close()

# show final policy rollout
# test_env = gym.make(ENV_NAME, render_mode="human")
# generate_rollout(test_env, policy)
# test_env.close()