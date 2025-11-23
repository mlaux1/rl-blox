import gymnasium as gym

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state


def test_a2c(inverted_pendulum_env):
    num_envs_for_test = 2
    env_name = "InvertedPendulum-v5"
    test_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_name) for _ in range(num_envs_for_test)]
    )

    ac_state = create_policy_gradient_continuous_state(
        test_env,
        policy_shared_head=True,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[128, 128],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_a2c(
        inverted_pendulum_env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        seed=42,
        total_timesteps=10,
        num_envs=num_envs_for_test,
    )

    test_env.close()
