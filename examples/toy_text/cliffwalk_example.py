import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.algorithms.model_free.q_learning import QLearning
from modular_rl.policy.base_policy import EpsilonGreedyPolicy
from modular_rl.helper.experiment_helper import generate_rollout
from rl_experiments.evaluation.plotting import plot_training_stats

num_episodes = 1000
learning_rate = 0.1
epsilon = 0.1

train_env = gym.make("CliffWalking-v0")

sarsa_env = gym.wrappers.RecordEpisodeStatistics(
    train_env, deque_size=num_episodes)
policy = EpsilonGreedyPolicy(
    train_env.observation_space, train_env.action_space, epsilon=epsilon
)
sarsa = Sarsa(sarsa_env, policy, alpha=learning_rate, key=0)
sarsa.train(num_episodes)

test_env = gym.make("CliffWalking-v0", render_mode="human")
generate_rollout(test_env, sarsa.target_policy)

#q_learning_env = gym.wrappers.RecordEpisodeStatistics(
#    train_env, deque_size=num_episodes
#)
#q_learning = QLearning(q_learning_env, alpha=learning_rate, epsilon=epsilon)
#q_learning.train(num_episodes)

# generate_rollout(test_env, q_learning.target_policy)

# train_env.close()
