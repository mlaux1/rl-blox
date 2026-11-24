import inspect

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithm.ddqn_per import train_ddqn_per
from rl_blox.blox.function_approximator.mlp import MLP
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
