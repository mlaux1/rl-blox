import gymnasium as gym

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state


def test_a2c(inverted_pendulum_env):
    env_id = inverted_pendulum_env.spec.id
    num_envs = 2

    def make_env():
        return gym.make(env_id)

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    ac_state = create_policy_gradient_continuous_state(
        inverted_pendulum_env,
        policy_shared_head=True,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[128, 128],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_a2c(
        envs,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        seed=42,
        total_timesteps=50,
        steps_per_update=5,
    )

    envs.close()
