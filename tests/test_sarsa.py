import gymnasium as gym
import jax

from rl_blox.algorithm.sarsa import sarsa
from rl_blox.blox.value_policy import make_q_table


def test_sarsa():
    env = gym.make("CliffWalking-v0")
    q_table = make_q_table(env)

    _ = sarsa(
        jax.random.key(0),
        env,
        q_table,
        alpha=0.05,
        epsilon=0.05,
        num_episodes=1,
    )

    env.close()
