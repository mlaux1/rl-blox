from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from flax import nnx

from rl_blox.algorithm.active_mt import train_active_mt
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.blox.multitask import (
    DiscreteTaskSet,
    SchedulerResult,
    TaskSelector,
)
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 200  # fixed length of Pendulum episodes


def make_task_set():
    def set_gravity(env, context):
        env.unwrapped.g = context[0]

    contexts = np.array([[5.0], [10.0], [15.0]])
    return DiscreteTaskSet(
        gym.make("Pendulum-v1"), set_gravity, contexts, context_aware=True
    )


def make_train_st(task_set):
    env = task_set.get_task(0)
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
        learning_starts=10_000,
    )


class LastTaskSelector(TaskSelector):
    """Always selects the last task and records the feedback."""

    def __init__(self, tasks):
        super().__init__(tasks)
        self.rewards = []

    def select(self) -> int:
        super().select()
        return self.tasks[-1]

    def feedback(self, reward: float):
        super().feedback(reward)
        self.rewards.append(reward)


def test_train_active_mt():
    task_set = make_task_set()
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=1_000), 3)
    logger = MemoryLogger()

    result = train_active_mt(
        task_set,
        make_train_st(task_set),
        r_max=1.0,
        total_timesteps=4 * EPISODE_LENGTH,
        task_selector="Round Robin",
        replay_buffer=replay_buffer,
        logger=logger,
        progress_bar=False,
    )

    assert isinstance(result, SchedulerResult)
    assert result.training_steps.tolist() == [
        2 * EPISODE_LENGTH,
        EPISODE_LENGTH,
        EPISODE_LENGTH,
    ]
    assert result.eval_steps == 0
    assert logger.stats["task_id"] == [0, 1, 2, 0]
    assert [
        len(buffer) for buffer in replay_buffer.buffers
    ] == result.training_steps.tolist()


def test_train_active_mt_gives_feedback_after_warmup():
    task_set = make_task_set()
    selector = LastTaskSelector(np.arange(3))
    logger = MemoryLogger()

    train_active_mt(
        task_set,
        make_train_st(task_set),
        r_max=1.0,
        total_timesteps=5 * EPISODE_LENGTH + 50,
        task_selector=selector,
        warmup_episodes=1,
        replay_buffer=MultiTaskReplayBuffer(ReplayBuffer(buffer_size=500), 3),
        logger=logger,
        progress_bar=False,
    )

    assert logger.stats["task_id"] == [0, 1, 2, 2, 2, 2]
    # No feedback for the warm-up and the selection that ran out of budget
    assert len(selector.rewards) == 2
    assert all(reward < 0.0 for reward in selector.rewards)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_selector": "Random"},
        {"total_timesteps": 0},
        {"episodes_per_task": 0},
    ],
)
def test_train_active_mt_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        train_active_mt(
            make_task_set(),
            None,
            r_max=1.0,
            **({"total_timesteps": 100} | kwargs),
            progress_bar=False,
        )
