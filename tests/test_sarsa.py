import gymnasium as gym

from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import make_q_table


def test_sarsa():
    env = gym.make("CliffWalking-v0")
    q_table = make_q_table(env)

    _ = train_sarsa(
        env,
        q_table,
        total_timesteps=10,
        seed=0,
    )

    env.close()
