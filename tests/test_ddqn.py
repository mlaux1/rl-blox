import inspect

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer


def test_ddqn(cart_pole_env):
    seed = 42

    rb = ReplayBuffer(100)

    q_net = MLP(
        cart_pole_env.observation_space.shape[0],
        int(cart_pole_env.action_space.n),
        [10],
        "relu",
        nnx.Rngs(seed),
    )

    optimizer = nnx.Optimizer(q_net, optax.rprop(0.0003), wrt=nnx.Param)

    train_ddqn(
        q_net,
        cart_pole_env,
        rb,
        optimizer,
        seed=seed,
        total_timesteps=10,
    )


def test_ddqn_same_seed_gives_same_result(cart_pole_env):
    def train():
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_ddqn(
            q_net,
            cart_pole_env,
            ReplayBuffer(100, discrete_actions=True),
            optimizer,
            batch_size=8,
            total_timesteps=100,
            seed=1,
            progress_bar=False,
        )
        return jax.tree.leaves(nnx.state(q_net, nnx.Param))

    for a, b in zip(train(), train(), strict=True):
        assert_array_equal(a, b)


def test_ddqn_waits_for_learning_starts(cart_pole_env):
    def train(total_timesteps):
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        params = jax.tree.map(
            jnp.copy, jax.tree.leaves(nnx.state(q_net, nnx.Param))
        )
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_ddqn(
            q_net,
            cart_pole_env,
            ReplayBuffer(100, discrete_actions=True),
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


def test_ddqn_default_total_timesteps_is_int():
    default = (
        inspect.signature(train_ddqn).parameters["total_timesteps"].default
    )
    assert isinstance(default, int)


def test_ddqn_trains_with_float_actions(cart_pole_env):
    q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
    train_ddqn(
        q_net,
        cart_pole_env,
        ReplayBuffer(100),
        optimizer,
        batch_size=8,
        total_timesteps=20,
        progress_bar=False,
    )
