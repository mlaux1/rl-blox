from rl_blox.algorithm.td3 import create_td3_state, train_td3


def test_td3(inverted_pendulum_env):
    td3_state = create_td3_state(
        inverted_pendulum_env,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        q_hidden_nodes=[128, 128],
        q_learning_rate=1e-2,
        seed=42,
    )

    train_td3(
        inverted_pendulum_env,
        td3_state.policy,
        td3_state.policy_optimizer,
        td3_state.q,
        td3_state.q_optimizer,
        total_timesteps=10,
    )
