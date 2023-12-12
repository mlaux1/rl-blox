import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.algorithms.model_free.q_learning import QLearning
from modular_rl.policy.base_policy import EpsilonGreedyPolicy
from modular_rl.helper.experiment_helper import generate_rollout, moving_average

num_episodes = 1000
learning_rate = 0.1
epsilon = 0.1

train_env = gym.make("CliffWalking-v0")

sarsa_env = gym.wrappers.RecordEpisodeStatistics(train_env, deque_size=num_episodes)
policy = EpsilonGreedyPolicy(
    train_env.observation_space, train_env.action_space, epsilon=epsilon
)
sarsa = Sarsa(sarsa_env, policy, alpha=learning_rate)
sarsa.train(num_episodes)

test_env = gym.make("CliffWalking-v0", render_mode="human")
generate_rollout(test_env, sarsa.target_policy)

q_learning_env = gym.wrappers.RecordEpisodeStatistics(
    train_env, deque_size=num_episodes
)
q_learning = QLearning(q_learning_env, alpha=learning_rate, epsilon=epsilon)
q_learning.train(num_episodes)

generate_rollout(test_env, q_learning.target_policy)

# train_env.close()

rolling_length = 100
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))


axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
sarsa_reward_moving_average = moving_average(
    np.array(sarsa_env.return_queue), rolling_length
)
q_learning_reward_moving_average = moving_average(
    np.array(q_learning_env.return_queue), rolling_length
)
axs[0].plot(
    range(len(sarsa_reward_moving_average)), sarsa_reward_moving_average, label="SARSA"
)
axs[0].plot(
    range(len(q_learning_reward_moving_average)),
    q_learning_reward_moving_average,
    label="Q-Learning",
)

axs[1].set_title("Episode lengths")
sarsa_length_moving_average = moving_average(
    np.array(sarsa_env.length_queue), rolling_length
)
q_learning_length_moving_average = moving_average(
    np.array(q_learning_env.length_queue), rolling_length
)
axs[1].plot(
    range(len(sarsa_length_moving_average)), sarsa_length_moving_average, label="SARSA"
)
axs[1].plot(
    range(len(q_learning_length_moving_average)),
    q_learning_length_moving_average,
    label="Q-Learning",
)
plt.tight_layout()
plt.legend()
plt.show()
