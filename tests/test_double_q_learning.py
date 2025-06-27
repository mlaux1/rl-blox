from rl_blox.algorithm.double_q_learning import train_double_q_learning
from rl_blox.blox.value_policy import make_q_table


def test_double_q_learning(tabular_test_env):
    q_table1 = make_q_table(tabular_test_env)
    q_table2 = make_q_table(tabular_test_env)

    train_double_q_learning(
        tabular_test_env,
        q_table1,
        q_table2,
        total_timesteps=10,
        seed=42,
    )
