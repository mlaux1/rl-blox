import gymnasium as gym

from rl_blox.algorithm.sac import create_sac_state, train_sac


def test_sac():
    env = gym.make("Pendulum-v1")

    sac_state = create_sac_state(
        env,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        q_hidden_nodes=[128, 128],
        q_learning_rate=1e-2,
        seed=42,
    )

    train_sac(
        env,
        sac_state.policy,
        sac_state.policy_optimizer,
        sac_state.q1,
        sac_state.q1_optimizer,
        sac_state.q2,
        sac_state.q2_optimizer,
        total_timesteps=10,
    )

    env.close()
