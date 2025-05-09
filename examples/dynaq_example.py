import gymnasium as gym

from rl_blox.algorithm.dynaq import train_dynaq
from rl_blox.blox.value_policy import get_greedy_action, make_q_table
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "CliffWalking-v0"
env = gym.make(env_name)

q_table = make_q_table(env)

hparams = dict(
    gamma=0.99,
    learning_rate=0.05,
    epsilon=0.1,
    n_planning_steps=10,
    buffer_size=100,
    seed=1,
)

logger = LoggerList([AIMLogger(), StandardLogger(verbose=2)])
logger.define_experiment(env_name, algorithm_name="Dyna-Q")

q_table = train_dynaq(
    env,
    q_table,
    **hparams,
    total_timesteps=5_000,
    logger=logger,
)
env.close()

env = gym.make(env_name, render_mode="human")
for _ in range(5):
    obs, _ = env.reset()
    done = False
    accumulated_reward = 0.0
    while not done:
        act = int(get_greedy_action(None, q_table, obs))
        obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        accumulated_reward += reward
    print(f"return={accumulated_reward}")
env.close()
