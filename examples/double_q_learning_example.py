import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.double_q_learning import train_double_q_learning
from rl_blox.blox.value_policy import greedy_policy, make_q_table

NUM_STEPS = 50_000
LEARNING_RATE = 0.1
EPSILON = 0.1
ENV_NAME = "CliffWalking-v1"

env = gym.make(ENV_NAME)
env = RecordEpisodeStatistics(env, buffer_length=2000)

q_table1 = make_q_table(env)
q_table2 = make_q_table(env)

q_table1, q_table2 = train_double_q_learning(
    env,
    q_table1,
    q_table2,
    learning_rate=LEARNING_RATE,
    epsilon=EPSILON,
    total_timesteps=NUM_STEPS,
    seed=42,
)

env.close()


# Show the final policy
q_table = q_table1 + q_table2

eval_env = gym.make(ENV_NAME, render_mode="human")
obs, _ = eval_env.reset()

while True:
    action = int(greedy_policy(q_table, obs))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs
