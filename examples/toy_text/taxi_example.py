import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from rl_experiments.evaluation.plotting import plot_training_stats

from modular_rl.algorithms.model_free.q_learning import QLearning
from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.helper.experiment_helper import generate_rollout
from modular_rl.policy.base_policy import EpsilonGreedyPolicy

NUM_EPISODES = 2000
LEARNING_RATE = 0.1
EPSILON = 0.1
KEY = 42
WINDOW_SIZE = 10

train_env = gym.make("Taxi-v3")

sarsa_env = RecordEpisodeStatistics(train_env, deque_size=NUM_EPISODES)
policy = EpsilonGreedyPolicy(
    train_env.observation_space,
    train_env.action_space,
    epsilon=EPSILON
)
sarsa = Sarsa(sarsa_env, policy, alpha=LEARNING_RATE, key=KEY)
sarsa.train(NUM_EPISODES)
sarsa_env.close()

test_env = gym.make("Taxi-v3", render_mode="human")
generate_rollout(test_env, sarsa.policy)
test_env.close()

q_learning_env = RecordEpisodeStatistics(train_env, deque_size=NUM_EPISODES)
q_policy = EpsilonGreedyPolicy(
    train_env.observation_space,
    train_env.action_space,
    epsilon=EPSILON
)
q_learning = QLearning(
    q_learning_env,
    q_policy,
    alpha=LEARNING_RATE,
    key=KEY)
q_learning.train(NUM_EPISODES)
q_learning_env.close()

test_env = gym.make("Taxi-v3", render_mode="human")
generate_rollout(test_env, q_learning.policy)
test_env.close()

plot_training_stats(
    np.array(sarsa_env.return_queue),
    np.array(sarsa_env.length_queue),
    rolling_length=WINDOW_SIZE,
    label="SARSA",
)

plot_training_stats(
    np.array(q_learning_env.return_queue),
    np.array(q_learning_env.length_queue),
    rolling_length=WINDOW_SIZE,
    label="Q_Learning",
)

