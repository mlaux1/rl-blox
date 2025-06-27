from rl_blox.algorithm.monte_carlo import train_monte_carlo
from rl_blox.blox.value_policy import make_q_table


def test_monte_carlo(tabular_test_env):
    q_table = make_q_table(tabular_test_env)
    _ = train_monte_carlo(tabular_test_env, q_table, total_timesteps=10)
