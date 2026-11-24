import inspect

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from numpy.testing import assert_array_almost_equal, assert_array_equal

from rl_blox.algorithm.ddqn_per import train_ddqn_per
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.losses import ddqn_per_loss
from rl_blox.blox.replay_buffer import PrioritizedReplayBuffer


def test_ddqn_per_same_seed_gives_same_result(cart_pole_env):
    def train():
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_ddqn_per(
            q_net,
            cart_pole_env,
            PrioritizedReplayBuffer(100, discrete_actions=True),
            optimizer,
            batch_size=8,
            total_timesteps=100,
            seed=1,
            progress_bar=False,
        )
        return jax.tree.leaves(nnx.state(q_net, nnx.Param))

    for a, b in zip(train(), train(), strict=True):
        assert_array_equal(a, b)


def test_ddqn_per_waits_for_learning_starts(cart_pole_env):
    def train(total_timesteps):
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        params = jax.tree.map(
            jnp.copy, jax.tree.leaves(nnx.state(q_net, nnx.Param))
        )
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_ddqn_per(
            q_net,
            cart_pole_env,
            PrioritizedReplayBuffer(100, discrete_actions=True),
            optimizer,
            batch_size=8,
            learning_starts=40,
            total_timesteps=total_timesteps,
            progress_bar=False,
        )
        new_params = jax.tree.leaves(nnx.state(q_net, nnx.Param))
        return any(
            not jnp.array_equal(a, b)
            for a, b in zip(params, new_params, strict=True)
        )

    assert not train(total_timesteps=40)
    assert train(total_timesteps=60)


def test_ddqn_per_default_total_timesteps_is_int():
    default = (
        inspect.signature(train_ddqn_per).parameters["total_timesteps"].default
    )
    assert isinstance(default, int)


def test_ddqn_per_loss_returns_td_error_per_sample():
    q = MLP(4, 2, [10], "relu", nnx.Rngs(0))
    q_target = MLP(4, 2, [10], "relu", nnx.Rngs(1))
    key1, key2 = jax.random.split(jax.random.key(0))
    obs = jax.random.normal(key1, (8, 4))
    next_obs = jax.random.normal(key2, (8, 4))
    action = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])
    reward = jnp.arange(8, dtype=float)
    terminated = jnp.array([0, 0, 0, 1, 0, 0, 0, 1])

    _, (_, abs_td_error) = ddqn_per_loss(
        q, q_target, (obs, action, reward, next_obs, terminated), 0.9
    )

    next_action = jnp.argmax(q(next_obs), axis=1)
    next_value = q_target(next_obs)[jnp.arange(8), next_action]
    target = reward + 0.9 * (1 - terminated) * next_value
    expected = jnp.abs(q(obs)[jnp.arange(8), action] - target)
    assert_array_almost_equal(abs_td_error, expected, decimal=5)


def test_ddqn_per_updates_priority_per_sample(cart_pole_env):
    buffer = PrioritizedReplayBuffer(100, discrete_actions=True)
    priorities = []
    update_priority = buffer.update_priority

    def record_update_priority(priority):
        priorities.append(jnp.asarray(priority))
        update_priority(priority)

    buffer.update_priority = record_update_priority
    q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)

    train_ddqn_per(
        q_net,
        cart_pole_env,
        buffer,
        optimizer,
        batch_size=8,
        total_timesteps=40,
        progress_bar=False,
    )

    assert priorities
    for priority in priorities:
        assert priority.shape == (8,)
        assert len(jnp.unique(priority)) > 1


def test_ddqn_per_trains_with_float_actions(cart_pole_env):
    q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
    train_ddqn_per(
        q_net,
        cart_pole_env,
        PrioritizedReplayBuffer(100),
        optimizer,
        batch_size=8,
        total_timesteps=20,
        progress_bar=False,
    )
