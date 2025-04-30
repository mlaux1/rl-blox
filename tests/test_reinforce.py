import gymnasium as gym
import jax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithms.model_free.reinforce import (
    MLP,
    GaussianMLP,
    GaussianPolicy,
    SoftmaxPolicy,
    sample_trajectories,
    discounted_reward_to_go,
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
