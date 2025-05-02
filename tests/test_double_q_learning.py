import gymnasium as gym
import jax

from rl_blox.algorithm.double_q_learning import double_q_learning
from rl_blox.blox.value_policy import make_q_table


def test_double_q_learning():
    env = gym.make("CliffWalking-v0")
    q_table1 = make_q_table(env)
    q_table2 = make_q_table(env)

    double_q_learning(
        jax.random.key(0),
        env,
        q_table1,
        q_table2,
        alpha=0.05,
        epsilon=0.05,
        num_episodes=1,
    )

    env.close()
