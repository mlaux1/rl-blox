import gymnasium as gym

from rl_blox.algorithm.a2c import train_a2c
from rl_blox.algorithm.reinforce import create_policy_gradient_continuous_state


def test_a2c(inverted_pendulum_env):
    num_envs_for_test = 3
    steps_per_update_for_test = 5
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

    rollout_data = train_a2c(
        test_env,
        ac_state.policy,
        ac_state.policy_optimizer,
        ac_state.value_function,
        ac_state.value_function_optimizer,
        seed=42,
        total_timesteps=20,
        steps_per_update=steps_per_update_for_test,
        num_envs=num_envs_for_test,
    )

    test_env.close()

    # was the right amount of data collected?
    assert (
        len(rollout_data) == steps_per_update_for_test
    ), f"Expected {steps_per_update_for_test} steps, but got {len(rollout_data)}"

    first_step_data = rollout_data[0]
    obs, actions, rewards, values, log_probs, terminations = first_step_data

    # Are the shapes correct?
    single_obs_shape = test_env.single_observation_space.shape
    single_action_shape = test_env.single_action_space.shape

    assert obs.shape == (
        num_envs_for_test,
        *single_obs_shape,
    ), f"Observations shape is wrong. Expected {(num_envs_for_test, *single_obs_shape)}, got {obs.shape}"

    assert actions.shape == (
        num_envs_for_test,
    ), f"Stored actions shape is wrong. Expected {(num_envs_for_test,)}, got {actions.shape}"

    assert rewards.shape == (
        num_envs_for_test,
    ), f"Rewards shape is wrong. Expected {(num_envs_for_test,)}, got {rewards.shape}"

    assert values.shape == (
        num_envs_for_test,
    ), f"Values shape is wrong. Expected {(num_envs_for_test,)}, got {values.shape}"

    assert log_probs.shape == (
        num_envs_for_test,
    ), f"Log_probs shape is wrong. Expected {(num_envs_for_test,)}, got {log_probs.shape}"

    print("\n--- Data from the first step of the rollout ---")
    print(f"Observations (first env):\n{obs[0]}")
    print(f"Actions taken:\n{actions}")
    print(f"Rewards received:\n{rewards}")
    print(f"Critic's value predictions:\n{values}")
    print("---------------------------------------------")
