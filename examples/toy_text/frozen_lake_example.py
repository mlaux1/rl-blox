import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modular_rl.algorithms.model_free.sarsa import Sarsa
from modular_rl.policy.base_policy import EpsilonGreedyPolicy
from modular_rl.helper.experiment_helper import generate_rollout


train_env = gym.make("FrozenLake-v1")
train_env = gym.wrappers.RecordEpisodeStatistics(train_env, deque_size=100000)


policy = EpsilonGreedyPolicy(
    train_env.observation_space, train_env.action_space, epsilon=0.01
)
alg = Sarsa(train_env, policy, alpha=0.2)

train_returns = alg.train(100000)

train_env.close()

test_env = gym.make("FrozenLake-v1", render_mode="human")

generate_rollout(test_env, alg.target_policy)

train_env.close()


rolling_length = 100
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(train_env.return_queue).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(train_env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
plt.tight_layout()
plt.show()
