import copy
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike
from tqdm.rich import tqdm

from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


class ContextualMultiTaskDefinition(metaclass=ABCMeta):
    """Defines a multi-task environment."""

    def __init__(self, contexts: ArrayLike):
        self.contexts = contexts

    def get_task_context(self, task_id: int) -> ArrayLike:
        assert task_id < len(self.contexts)
        return self.contexts[task_id]

    @abstractmethod
    def get_task(self, task_id: int) -> gym.Env:
        """Returns the task environment for the given task ID."""

    @abstractmethod
    def get_solved_threshold(self, task_id: int) -> float:
        """Performance threshold for a task to be considered solved."""

    @abstractmethod
    def get_unsolvable_threshold(self, task_id: int) -> float:
        """Performance threshold for a task to be considered unsolvable."""

    def __len__(self) -> int:
        return len(self.contexts)


def train_smt(
    mt_def: ContextualMultiTaskDefinition,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    b1: int = 17_000_000,
    b2: int = 3_000_000,
    scheduling_interval: int = 1_000,
    kappa: float = 0.8,
    K: int = 3,
    n_average: int = 3,
    learning_starts: int = 5_000,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple:
    r"""Scheduled Multi-Task (SMT) training.

    Multi-task RL faces the challenge of varying task difficulties, often
    leading to negative transfer when simpler tasks overshadow the learning of
    more complex ones. To overcome this challenge, SMT strategically prioritizes
    more challenging tasks, thereby enhancing overall learning efficiency. SMT
    uses a dynamic task prioritization strategy, underpinned by an effective
    metric for assessing task difficulty. This metric ensures an efficient and
    targeted allocation of training resources.

    Parameters
    ----------
    mt_def
        The multi-task environment definition.

    train_st : callable
        The single-task training algorithm.

    replay_buffer : MultiTaskReplayBuffer
        Replay buffer.

    b1 : int
        Total number of timesteps to train the agent in first phase.
        Corresponds to :math:`B_1` in the paper [1]_.

    b2 : int
        Total number of timesteps to train the agent in second phase.
        Corresponds to :math:`B_2` in the paper [1]_.

    scheduling_interval : int
        Number of steps after which the task scheduling is performed.

    kappa : float
        Budget for each task is initially set to :math:`\kappa B_{total}`
        with :math:`B_{total} = B_1 + B_2`.

    K : int
        Number of tasks to train in each iteration.

    n_average : int
        Number of tasks to average the performance over when checking if a task
        is solved or unsolvable.

    learning_starts : int
        Number of steps to wait before starting training per task.

    learning_starts : int
        Timestep to start learning.

    seed : int
        Seed for random number generation.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    result
        The training result. Same as the result of the `train_st` function.

    training_steps : np.ndarray
        Number of training steps for each task.

    References
    ----------
    .. [1] Cho, M., Park, J., Lee, S., Sung, Y. (2024). Hard tasks first:
       multi-task reinforcement learning through task scheduling. In
       Proceedings of the 41st International Conference on Machine Learning,
       Vol. 235. JMLR.org, Article 340, 8556–8577.
       https://icml.cc/virtual/2024/poster/33388
    """
    rng = np.random.default_rng(seed)

    n_tasks = len(mt_def)
    training_pool = set(rng.choice(n_tasks, size=K, replace=False))
    main_pool = set(range(n_tasks)) - training_pool
    solved_pool = set()
    unsolvable_pool = set()

    b_total = b1 + b2
    global_step = 0
    training_steps = np.zeros(n_tasks, dtype=int)
    avg_training_performances = np.full(n_tasks, -np.finfo(float).max)
    training_performances = [deque(maxlen=n_average) for _ in range(n_tasks)]
    task_budgets = np.full(n_tasks, kappa * b_total)

    progress = tqdm(total=b_total, disable=not progress_bar)

    while global_step < b1:
        updated_training_pool = copy.deepcopy(training_pool)
        for task_id in training_pool:
            env = mt_def.get_task(task_id)
            env_with_stats = gym.wrappers.RecordEpisodeStatistics(
                env, buffer_length=n_average
            )
            replay_buffer.select_task(task_id)

            total_timesteps = global_step + scheduling_interval
            result_st = train_st(
                env=env_with_stats,
                learning_starts=learning_starts,
                total_timesteps=total_timesteps,
                replay_buffer=replay_buffer,
                seed=seed + global_step,
                logger=logger,
                global_step=global_step,
                progress_bar=False,
            )
            training_steps[task_id] += scheduling_interval
            global_step = total_timesteps

            progress.update(scheduling_interval)

            training_performances[task_id].extend(env_with_stats.return_queue)
            avg_training_performances[task_id] = np.mean(
                training_performances[task_id]
            )

            if logger is not None:
                logger.record_stat("task_id", task_id, global_step + 1)
                logger.record_stat(
                    "task_performance",
                    avg_training_performances[task_id],
                    global_step + 1,
                )

            M = mt_def.get_solved_threshold(task_id)
            if avg_training_performances[task_id] > M:
                solved_pool.add(task_id)
                updated_training_pool.remove(task_id)
            elif training_steps[task_id] >= task_budgets[task_id]:
                m = mt_def.get_unsolvable_threshold(task_id)
                if avg_training_performances[task_id] < m:
                    unsolvable_pool.add(task_id)
                    updated_training_pool.remove(task_id)
                else:
                    main_pool.add(task_id)
                    updated_training_pool.remove(task_id)

        # The original SMT algorithm would evaluate the performance on each task
        # and reset the networks here, i.e., randomly initialize them.

        early_stop_stage = False
        while len(updated_training_pool) < K:
            if len(main_pool) == 0:
                # All tasks outside the training pool are solved or unsolvable.
                if len(updated_training_pool) == 0:  # No tasks left to train.
                    early_stop_stage = True
                # else: Continue until training pool is solved or unsolvable.
                break

            # Select task with the lowest performance (from main pool)
            main_pool_indices = list(main_pool)
            worst_index = np.argmin(
                avg_training_performances[main_pool_indices]
            )
            worst = main_pool_indices[worst_index]
            # Move it to training pool
            updated_training_pool.add(worst)
            main_pool.remove(worst)
            # Set its budget to kappa * B (B: remaining total budget)
            task_budgets[worst] = kappa * (b_total - global_step)

            if logger is not None:
                logger.record_stat("worst task", worst, global_step + 1)
                logger.record_stat(
                    "worst performance",
                    avg_training_performances[worst],
                    global_step + 1,
                )

        if early_stop_stage:
            break

        training_pool = updated_training_pool

    if len(unsolvable_pool) > 0:
        result_st = smt_stage2(
            mt_def,
            train_st,
            replay_buffer,
            unsolvable_pool,
            training_steps,
            global_step,
            scheduling_interval,
            b_total,
            learning_starts,
            logger,
            seed,
            progress,
        )
    progress.close()

    return result_st, training_steps, avg_training_performances


def smt_stage2(
    mt_def,
    train_st,
    replay_buffer,
    unsolvable_pool,
    training_steps,
    global_step,
    scheduling_interval,
    b_total,
    learning_starts,
    logger,
    seed,
    progress,
):
    """SMT will iterate over the unsolvable tasks in stage 2."""
    while global_step < b_total:
        for task_id in unsolvable_pool:
            env = mt_def.get_task(task_id)
            replay_buffer.select_task(task_id)
            total_timesteps = global_step + scheduling_interval
            result_st = train_st(
                env=env,
                learning_starts=learning_starts,
                total_timesteps=total_timesteps,
                replay_buffer=replay_buffer,
                seed=seed + global_step,
                logger=logger,
                global_step=global_step,
                progress_bar=False,
            )
            training_steps[task_id] += scheduling_interval
            global_step = total_timesteps

            progress.update(scheduling_interval)

            if global_step + 1 >= b_total:
                break

    return result_st
