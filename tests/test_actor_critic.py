from rl_blox.algorithm.actor_critic import train_ac
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state


def test_actor_critic(inverted_pendulum_env):
    ac_state = create_policy_gradient_continuous_state(
        inverted_pendulum_env,
        policy_shared_head=True,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[128, 128],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_ac(
        inverted_pendulum_env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        seed=42,
        total_timesteps=10,
    )
