from functools import partial

import gymnasium as gym
from flax import nnx

from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.blox.multitask import (
    TaskSelectionMixin,
    n_warmup_selections,
    train_task,
)
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 200  # fixed length of Pendulum episodes


def make_train_st(env):
    state = create_sac_state(
        env, policy_hidden_nodes=[8], q_hidden_nodes=[8], seed=0
    )
    # Random actions without updates keep the tests fast
    return partial(
        train_sac,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_target=nnx.clone(state.q),
        q_optimizer=state.q_optimizer,
        entropy_control=EntropyControl(env, 0.2, False, 1e-3),
        replay_buffer=ReplayBuffer(buffer_size=1_000),
        learning_starts=10_000,
    )


def test_train_task_counts_steps_of_all_episodes():
    env = gym.make("Pendulum-v1")

    _, steps, returns = train_task(
        make_train_st(env),
        env,
        task_id=0,
        n_episodes=2,
        total_timesteps=10_000,
        global_step=100,
        seed=0,
    )

    assert steps == 2 * EPISODE_LENGTH
    assert returns.shape == (2,)


def test_train_task_stops_when_budget_runs_out():
    env = gym.make("Pendulum-v1")

    _, steps, returns = train_task(
        make_train_st(env),
        env,
        task_id=0,
        n_episodes=2,
        total_timesteps=1_300,
        global_step=1_000,
        seed=0,
    )

    assert steps == 300
    assert returns.shape == (1,)


def test_train_task_informs_task_selectables_and_logger():
    env = gym.make("Pendulum-v1")
    selectable = TaskSelectionMixin()
    logger = MemoryLogger()

    train_task(
        make_train_st(env),
        env,
        task_id=3,
        n_episodes=1,
        total_timesteps=10_000,
        global_step=0,
        seed=0,
        task_selectables=[selectable],
        logger=logger,
    )

    assert selectable.task_id == 3
    assert logger.stats["task_id"] == [3]


def test_n_warmup_selections_covers_whole_rounds():
    assert n_warmup_selections(10, 0, 1) == 0
    assert n_warmup_selections(10, 2, 1) == 20
    # At least 3 episodes per task with 2 episodes per selection
    assert n_warmup_selections(10, 3, 2) == 20
