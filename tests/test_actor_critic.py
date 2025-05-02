import gymnasium as gym

from rl_blox.algorithm.actor_critic import train_ac
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state


def test_actor_critic():
    env = gym.make("InvertedPendulum-v5")

    ac_state = create_policy_gradient_continuous_state(
        env,
        policy_shared_head=True,
        policy_hidden_nodes=[32, 32],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[128, 128],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_ac(
        env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        key=ac_state.key,
        total_timesteps=10,
    )

    env.close()
