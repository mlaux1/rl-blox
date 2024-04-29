import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from jax.random import PRNGKey

from modular_rl.algorithms.model_free.q_learning_jax import q_learning
from modular_rl.policy.value_policy import make_q_table

NUM_EPISODES = 1000
LEARNING_RATE = 0.1
EPSILON = 0.05
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "CliffWalking-v0"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, deque_size=NUM_EPISODES)

q_table = make_q_table(env)

ep_rewards = q_learning(
    KEY, env, q_table,
    alpha=LEARNING_RATE, epsilon=EPSILON, num_episodes=NUM_EPISODES)

env.close()
