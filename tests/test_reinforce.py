import jax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithm.reinforce import (
    create_policy_gradient_continuous_state,
    discounted_reward_to_go,
    sample_trajectories,
    train_reinforce,
)
from rl_blox.blox.function_approximator.gaussian_mlp import GaussianMLP
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    GaussianPolicy,
    SoftmaxPolicy,
)


def test_reinforce(inverted_pendulum_env):
    reinforce_state = create_policy_gradient_continuous_state(
        inverted_pendulum_env,
        policy_shared_head=True,
        policy_hidden_nodes=[64, 64],
        policy_learning_rate=3e-4,
        value_network_hidden_nodes=[256, 256],
        value_network_learning_rate=1e-2,
        seed=42,
    )

    train_reinforce(
        inverted_pendulum_env,
        reinforce_state.policy,
        reinforce_state.policy_optimizer,
        reinforce_state.value_function,
        reinforce_state.value_function_optimizer,
        seed=42,
        total_timesteps=10,
    )


def test_data_collection_discrete(cart_pole_env):
    seed = 42
    cart_pole_env.reset(seed=seed)
    key = jax.random.key(seed)
    policy = SoftmaxPolicy(
        MLP(
            cart_pole_env.observation_space.shape[0],
            int(cart_pole_env.action_space.n),
            [32, 32],
            "swish",
            nnx.Rngs(seed),
        )
    )
    total_steps = 100
    dataset = sample_trajectories(
        cart_pole_env, policy, key, None, False, total_steps
    )
    assert len(dataset) >= total_steps
    # regression test:
    assert dataset.average_return() == 20.8


def test_data_collection_continuous(inverted_pendulum_env):
    seed = 42
    inverted_pendulum_env.reset(seed=seed)
    key = jax.random.key(seed)
    policy = GaussianPolicy(
        GaussianMLP(
            True,
            inverted_pendulum_env.observation_space.shape[0],
            inverted_pendulum_env.action_space.shape[0],
            [32, 32],
            "swish",
            nnx.Rngs(seed),
        )
    )
    total_steps = 100
    dataset = sample_trajectories(
        inverted_pendulum_env, policy, key, None, False, total_steps
    )
    assert len(dataset) >= total_steps
    # regression test:
    assert dataset.average_return() == 5.8


def test_discounted_reward_to_go():
    assert_array_equal(
        discounted_reward_to_go([1.0, 2.0, 3.0], 0.9), [5.23, 4.7, 3.0]
    )
