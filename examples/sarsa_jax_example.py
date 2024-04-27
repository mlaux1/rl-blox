import gymnasium as gym
import jax.numpy as jnp
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.spaces.utils import flatdim

from modular_rl.algorithms.model_free.sarsa_jax import sarsa
from jax.random import PRNGKey
from modular_rl.helper.experiment_helper import generate_rollout

NUM_EPISODES = 10000
LEARNING_RATE = 0.1
EPSILON = 0.1
KEY = PRNGKey(42)
WINDOW_SIZE = 10
ENV_NAME = "CliffWalking-v0"

train_env = gym.make(ENV_NAME)

sarsa_env = RecordEpisodeStatistics(train_env, deque_size=NUM_EPISODES)

q_table = jnp.zeros(
            shape=(flatdim(train_env.observation_space), flatdim(train_env.action_space)),
            dtype=jnp.float32,
        )

sarsa = sarsa(
    KEY,
    sarsa_env,
    q_table,
    alpha=LEARNING_RATE,
    epsilon=EPSILON,
    num_episodes=NUM_EPISODES)

sarsa_env.close()
