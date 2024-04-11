import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.algorithms.model_free.q_learning import QLearning
from modular_rl.policy.base_policy import EpsilonGreedyPolicy
from modular_rl.helper.experiment_helper import generate_rollout

from rl_experiments.evaluation.plotting import plot_training_stats


num_episodes = 2000
learning_rate = 0.1
epsilon = 0.1

train_env = gym.make("Taxi-v3")

sarsa_env = RecordEpisodeStatistics(train_env, deque_size=num_episodes)
policy = EpsilonGreedyPolicy(
    train_env.observation_space, train_env.action_space, epsilon=epsilon
)
sarsa = Sarsa(sarsa_env, policy, alpha=learning_rate, key=0)
sarsa.train(num_episodes)
sarsa_env.close()

test_env = gym.make("Taxi-v3", render_mode="human")
generate_rollout(test_env, sarsa.target_policy)
test_env.close()

q_learning_env = RecordEpisodeStatistics(train_env, deque_size=num_episodes)
q_learning = QLearning(q_learning_env, alpha=learning_rate, epsilon=epsilon)
q_learning.train(num_episodes)
q_learning_env.close()

test_env = gym.make("Taxi-v3", render_mode="human")
generate_rollout(test_env, q_learning.target_policy)
test_env.close()

plot_training_stats(
    np.array(sarsa_env.return_queue),
    np.array(sarsa_env.length_queue),
    rolling_length=100,
    label="SARSA",
)

plot_training_stats(
    np.array(q_learning_env.return_queue),
    np.array(q_learning_env.length_queue),
    rolling_length=100,
    label="Q_Learning",
)

