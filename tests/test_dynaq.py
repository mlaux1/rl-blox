import gymnasium as gym

from rl_blox.algorithm.dynaq import train_dynaq
from rl_blox.blox.value_policy import make_q_table


def test_dynaq():
    env = gym.make("CliffWalking-v0")
    q_table = make_q_table(env)

    _ = train_dynaq(
        env,
        q_table,
        learning_rate=0.05,
        epsilon=0.05,
        total_timesteps=10,
    )

    env.close()
