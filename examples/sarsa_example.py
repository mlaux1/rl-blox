from functools import partial

import gymnasium as gym
import jax
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import get_epsilon_greedy_action, make_q_table
from rl_blox.logging.logger import AIMLogger
from rl_blox.util.experiment_helper import generate_rollout

NUM_TIMESTEPS = 200_00
LEARNING_RATE = 0.1
EPSILON = 0.2
ENV_NAME = "CliffWalking-v0"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=2000)

q_table = make_q_table(env)

logger = AIMLogger()
logger.define_experiment(env_name=ENV_NAME, algorithm_name="SARSA")

q_table = train_sarsa(
    env,
    q_table,
    learning_rate=LEARNING_RATE,
    epsilon=EPSILON,
    total_timesteps=NUM_TIMESTEPS,
    seed=42,
    logger=logger,
)

env.close()

logger.run.close()

# create and run the final policy
policy = partial(
    get_epsilon_greedy_action,
    q_table=q_table,
    epsilon=0.2,
    key=jax.random.key(42),
)


test_env = gym.make(ENV_NAME, render_mode="human")
generate_rollout(test_env, policy)
test_env.close()
