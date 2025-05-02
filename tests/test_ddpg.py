import gymnasium as gym

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg


def test_ddpg():
    env = gym.make("InvertedPendulum-v5")

    ddpg_state = create_ddpg_state(
        env,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        q_hidden_nodes=[128, 128],
        q_learning_rate=1e-2,
        seed=42,
    )

    train_ddpg(
        env,
        ddpg_state.policy,
        ddpg_state.policy_optimizer,
        ddpg_state.q,
        ddpg_state.q_optimizer,
        total_timesteps=10,
    )

    env.close()
