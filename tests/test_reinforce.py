import gymnasium as gym
import jax
import pytest
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.algorithm.reinforce import (
    GaussianMLP,
    GaussianPolicy,
    SoftmaxPolicy,
    create_policy_gradient_continuous_state,
    discounted_reward_to_go,
    sample_trajectories,
    train_reinforce,
)


def test_reinforce():
    env = gym.make("InvertedPendulum-v5")
    reinforce_state = create_policy_gradient_continuous_state(
        env,
        policy_shared_head=True,
        policy_hidden_nodes=[64, 64],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[256, 256],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_reinforce(
        env,
        reinforce_state.policy,
        reinforce_state.policy_optimizer,
        reinforce_state.value_function,
        reinforce_state.value_function_optimizer,
        key=reinforce_state.key,
        total_timesteps=10,
    )


def test_data_collection_discrete():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset(seed=42)
    key = jax.random.key(42)
    policy = SoftmaxPolicy(
        MLP(
            env.observation_space.shape[0],
            int(env.action_space.n),
            [32, 32],
            "swish",
            nnx.Rngs(42),
        )
    )
    env.close()
    total_steps = 100
    dataset = sample_trajectories(env, policy, key, None, False, total_steps)
    assert len(dataset) >= total_steps
    # regression test:
    assert dataset.average_return() == 20.8


def test_data_collection_continuous():
    env_name = "InvertedPendulum-v5"
    env = gym.make(env_name)
    env.reset(seed=42)
    key = jax.random.key(42)
    policy = GaussianPolicy(
        GaussianMLP(
            True,
            env.observation_space.shape[0],
            env.action_space.shape[0],
            [32, 32],
            nnx.Rngs(42),
        )
    )
    env.close()
    total_steps = 100
    dataset = sample_trajectories(env, policy, key, None, False, total_steps)
    assert len(dataset) >= total_steps
    # regression test:
    assert dataset.average_return() == 5.8


def test_discounted_reward_to_go():
    assert_array_equal(
        discounted_reward_to_go([1.0, 2.0, 3.0], 0.9), [5.23, 4.7, 3.0]
    )
