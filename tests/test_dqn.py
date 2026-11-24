import inspect
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import optax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithm.dqn import train_dqn
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import ReplayBuffer


def test_dqn(cart_pole_env):
    seed = 42

    rb = ReplayBuffer(100, discrete_actions=True)

    q_net = MLP(
        cart_pole_env.observation_space.shape[0],
        int(cart_pole_env.action_space.n),
        [10],
        "relu",
        nnx.Rngs(seed),
    )

    optimizer = nnx.Optimizer(q_net, optax.rprop(0.0003), wrt=nnx.Param)

    train_dqn(
        q_net,
        cart_pole_env,
        rb,
        optimizer,
        seed=seed,
        total_timesteps=10,
    )


def test_dqn_same_seed_gives_same_result(cart_pole_env):
    def train():
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_dqn(
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


def make_q_net_and_optimizer(n_inputs=4):
    q_net = MLP(n_inputs, 2, [10], "relu", nnx.Rngs(0))
    return q_net, nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)


def test_dqn_stops_after_total_episodes(cart_pole_env):
    env = gym.wrappers.RecordEpisodeStatistics(cart_pole_env, buffer_length=10)
    q_net, optimizer = make_q_net_and_optimizer()

    result = train_dqn(
        q_net,
        env,
        ReplayBuffer(1_000, discrete_actions=True),
        optimizer,
        batch_size=8,
        total_timesteps=10_000,
        total_episodes=3,
        progress_bar=False,
    )

    assert len(env.length_queue) == 3
    assert result.global_step == sum(env.length_queue)


def test_dqn_returns_number_of_steps(cart_pole_env):
    q_net, optimizer = make_q_net_and_optimizer()

    result = train_dqn(
        q_net,
        cart_pole_env,
        ReplayBuffer(100, discrete_actions=True),
        optimizer,
        batch_size=8,
        total_timesteps=50,
        global_step=20,
        progress_bar=False,
    )

    assert result.global_step == 50


def test_dqn_with_task_scheduler():
    def set_force(env, context):
        env.unwrapped.force_mag = context[0]

    task_set = DiscreteTaskSet(
        gym.make("CartPole-v1"),
        set_force,
        np.array([[5.0], [10.0]]),
        context_aware=True,
    )
    q_net, optimizer = make_q_net_and_optimizer(n_inputs=5)
    train_st = partial(
        train_dqn,
        q_net=q_net,
        replay_buffer=ReplayBuffer(1_000, discrete_actions=True),
        optimizer=optimizer,
        batch_size=8,
    )

    result = train_uts(
        task_set, train_st, total_timesteps=300, progress_bar=False
    )

    assert np.sum(result.training_steps) == 300


def test_dqn_default_total_timesteps_is_int():
    default = inspect.signature(train_dqn).parameters["total_timesteps"].default
    assert isinstance(default, int)
