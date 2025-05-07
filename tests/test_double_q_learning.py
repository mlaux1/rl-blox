import gymnasium as gym

from rl_blox.algorithm.double_q_learning import train_double_q_learning
from rl_blox.blox.value_policy import make_q_table


def test_double_q_learning():
    env = gym.make("CliffWalking-v0")
    q_table1 = make_q_table(env)
    q_table2 = make_q_table(env)

    train_double_q_learning(
        env,
        q_table1,
        q_table2,
        total_timesteps=10,
        seed=42,
    )

    env.close()
