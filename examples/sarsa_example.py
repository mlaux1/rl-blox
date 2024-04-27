import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from modular_rl.algorithms.model_free.sarsa import sarsa
from modular_rl.helper.experiment_helper import generate_rollout
from modular_rl.policy.base_policy import EpsilonGreedyPolicy

NUM_EPISODES = 2000
LEARNING_RATE = 0.1
EPSILON = 0.1
KEY = 42
WINDOW_SIZE = 10
ENV_NAME = "Taxi-v3"

train_env = gym.make(ENV_NAME)

sarsa_env = RecordEpisodeStatistics(train_env, deque_size=NUM_EPISODES)
policy = EpsilonGreedyPolicy(
    train_env.observation_space,
    train_env.action_space,
    epsilon=EPSILON
)
sarsa = sarsa(
    sarsa_env, policy, alpha=LEARNING_RATE, key=KEY, num_episodes=NUM_EPISODES)
sarsa_env.close()

test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
