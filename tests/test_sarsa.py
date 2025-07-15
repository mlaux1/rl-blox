from rl_blox.algorithm.sarsa import train_sarsa
from rl_blox.blox.value_policy import make_q_table


def test_sarsa(tabular_test_env):
    q_table = make_q_table(tabular_test_env)

    _ = train_sarsa(
        tabular_test_env,
        q_table,
        total_timesteps=10,
        seed=0,
    )
