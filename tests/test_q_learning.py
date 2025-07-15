from rl_blox.algorithm.q_learning import train_q_learning
from rl_blox.blox.value_policy import make_q_table


def test_q_learning(tabular_test_env):
    q_table = make_q_table(tabular_test_env)

    train_q_learning(
        tabular_test_env,
        q_table,
        total_timesteps=10,
    )
