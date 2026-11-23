from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.space import (
    space_grow,
    space_instance_set,
    train_space,
)
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet, SchedulerResult
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.blox.state_value import greedy_value
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 10


class TenStepEnv(gym.Env):
    """Episodes of 10 steps without reward."""

    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        observation = np.zeros(1, dtype=np.float32)
        return observation, 0.0, False, self.t >= EPISODE_LENGTH, {}


def make_task_set(n_tasks=3):
    return DiscreteTaskSet(
        TenStepEnv(),
        lambda env, context: None,
        np.zeros((n_tasks, 1)),
        context_aware=False,
    )


def random_actions_st(
    env, total_timesteps, total_episodes, global_step, seed, **kwargs
):
    """Backbone stub that only acts randomly for the scheduled episodes."""
    env.action_space.seed(seed)
    step = global_step
    for episode in range(total_episodes):
        env.reset(seed=seed + episode)
        done = False
        while not done and step < total_timesteps:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            step += 1
    return step


class ScriptedValues:
    """Value function whose start-state values follow a script.

    ``script(sweep, task_id)`` is the value of a task in the ``sweep``-th
    computation of all start-state values (0: before the first iteration).
    It knows the task because SPaCE informs its task selectables.
    """

    def __init__(self, script, n_tasks):
        self.script = script
        self.n_tasks = n_tasks
        self.task_id = 0
        self.n_calls = 0

    def select_task(self, task_id):
        self.task_id = task_id

    def __call__(self, observation):
        sweep = self.n_calls // self.n_tasks
        self.n_calls += 1
        return jnp.full(len(observation), self.script(sweep, self.task_id))


def run_space(script, n_tasks=3, **kwargs):
    values = ScriptedValues(script, n_tasks)
    logger = MemoryLogger()
    result = train_space(
        make_task_set(n_tasks),
        random_actions_st,
        values,
        task_selectables=[values],
        logger=logger,
        progress_bar=False,
        **kwargs,
    )
    return result, logger


def test_space_grow_within_relative_band():
    assert space_grow(1.05, 1.0, eta=0.05)
    assert space_grow(0.95, 1.0, eta=0.05)
    assert not space_grow(1.06, 1.0, eta=0.05)
    assert not space_grow(0.94, 1.0, eta=0.05)


def test_space_grow_never_after_zero_mean_value():
    # V_0 = 0, so the first iteration never grows the instance set
    assert not space_grow(0.1, 0.0, eta=0.5)


def test_space_instance_set_prefers_highest_improvement():
    instance_set = space_instance_set(
        np.array([1.0, 5.0, 3.0, 2.0]), np.array([0.0, 4.5, 0.0, 2.0]), 2
    )

    # PIC: 1.0, 0.5, 3.0, 0.0
    assert instance_set.tolist() == [2, 0]


def test_space_instance_set_breaks_ties_by_index():
    instance_set = space_instance_set(np.ones(4), np.zeros(4), 3)

    assert instance_set.tolist() == [0, 1, 2]


def test_train_space_grows_once_values_stagnate():
    # Constant values: the first iteration compares with V_0 = 0 and does
    # not grow, every later one grows until all tasks are included. All PICs
    # are 0, so ties put the tasks in index order.
    result, logger = run_space(
        lambda sweep, task_id: 1.0, total_timesteps=10 * EPISODE_LENGTH
    )

    assert isinstance(result, SchedulerResult)
    assert result.training_steps.sum() == 10 * EPISODE_LENGTH
    assert result.eval_steps == 0
    assert logger.stats["instance_set_size"][:4] == [1, 2, 3, 3]
    task_ids = logger.stats["task_id"]
    assert task_ids[1:] == [0, 0, 1, 0, 1, 2, 0, 1, 2]


def test_train_space_starts_with_random_task():
    first_tasks = {
        run_space(
            lambda sweep, task_id: 1.0,
            n_tasks=5,
            total_timesteps=EPISODE_LENGTH,
            seed=seed,
        )[1].stats["task_id"][0]
        for seed in range(10)
    }

    assert len(first_tasks) > 1


def test_train_space_does_not_grow_while_values_change():
    # The values grow by 10% per iteration, more than eta = 5%
    _, logger = run_space(
        lambda sweep, task_id: 1.1**sweep, total_timesteps=10 * EPISODE_LENGTH
    )

    assert set(logger.stats["instance_set_size"]) == {1}


def test_train_space_grows_by_kappa():
    _, logger = run_space(
        lambda sweep, task_id: 1.0,
        n_tasks=5,
        total_timesteps=10 * EPISODE_LENGTH,
        kappa=3,
    )

    assert logger.stats["instance_set_size"][:3] == [1, 4, 5]


def test_train_space_selects_tasks_with_highest_improvement():
    # Task 2's value rises fastest, so it is selected after each iteration
    def script(sweep, task_id):
        return sweep * (task_id + 1.0)

    _, logger = run_space(script, total_timesteps=5 * EPISODE_LENGTH)

    assert set(logger.stats["task_id"][1:]) == {2}
    _, pic = logger.get_stat("task_pic_2", x_key="step")
    assert np.all(pic == 3.0)


def test_train_space_warmup_trains_tasks_round_robin():
    _, logger = run_space(
        lambda sweep, task_id: 1.0,
        total_timesteps=4 * EPISODE_LENGTH,
        warmup_episodes=1,
    )

    assert logger.stats["task_id"][:3] == [0, 1, 2]


def test_train_space_passes_views_of_multitask_replay_buffer():
    received_buffers = []

    def record_buffer_st(replay_buffer, **kwargs):
        received_buffers.append(replay_buffer)
        return random_actions_st(**kwargs)

    values = ScriptedValues(lambda sweep, task_id: 1.0, 2)
    result = train_space(
        make_task_set(2),
        record_buffer_st,
        values,
        total_timesteps=4 * EPISODE_LENGTH,
        replay_buffer=MultiTaskReplayBuffer(ReplayBuffer(buffer_size=100), 2),
        task_selectables=[values],
        progress_bar=False,
    )

    assert result.training_steps.sum() == 4 * EPISODE_LENGTH
    # A random first task, then [0] and [0, 1] (constant values: ties)
    assert [buffer.task_id for buffer in received_buffers[1:]] == [0, 0, 1]


def test_train_space_with_ddqn():
    def set_force(env, context):
        env.unwrapped.force_mag = context[0]

    task_set = DiscreteTaskSet(
        gym.make("CartPole-v1"),
        set_force,
        np.array([[5.0], [10.0], [15.0]]),
        context_aware=True,
    )
    q_net = MLP(5, 2, [16], "relu", rngs=nnx.Rngs(0))
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        optimizer=nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param),
        q_target_net=nnx.clone(q_net),
        batch_size=32,
    )
    logger = MemoryLogger()

    result = train_space(
        task_set,
        train_st,
        partial(greedy_value, q_net),
        total_timesteps=500,
        replay_buffer=MultiTaskReplayBuffer(
            ReplayBuffer(buffer_size=1_000, discrete_actions=True), 3
        ),
        logger=logger,
        progress_bar=False,
    )

    assert np.sum(result.training_steps) == 500
    assert len(logger.stats["instance_set_size"]) > 0
    assert np.all(np.isfinite(logger.stats["mean_value"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_timesteps": 0},
        {"episodes_per_task": 0},
        {"eta": 0.0},
        {"eta": 1.0},
        {"kappa": 0},
    ],
)
def test_train_space_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        train_space(
            make_task_set(),
            None,
            None,
            **({"total_timesteps": 100} | kwargs),
            progress_bar=False,
        )
