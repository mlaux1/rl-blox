import gymnasium as gym

from rl_blox.algorithm.q_learning import train_q_learning
from rl_blox.blox.value_policy import make_q_table


def test_q_learning():
    env = gym.make("CliffWalking-v0")
    q_table = make_q_table(env)

    train_q_learning(
        env,
        q_table,
        total_timesteps=10,
    )

    env.close()
