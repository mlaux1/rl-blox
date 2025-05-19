import gymnasium as gym
import jax
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import epsilon_greedy_policy, make_q_table
from rl_blox.logging.logger import AIMLogger

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

# Show the final policy
eval_env = gym.make(ENV_NAME, render_mode="human")
obs, _ = eval_env.reset()

key = jax.random.key(42)

while True:
    action = int(epsilon_greedy_policy(q_table, obs, EPSILON, key))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs

logger.run.close()
