from rl_blox.algorithm.dynaq import train_dynaq
from rl_blox.blox.value_policy import make_q_table


def test_dynaq(tabular_test_env):
    q_table = make_q_table(tabular_test_env)

    _ = train_dynaq(
        tabular_test_env,
        q_table,
        learning_rate=0.05,
        epsilon=0.05,
        total_timesteps=10,
    )

    tabular_test_env.close()
