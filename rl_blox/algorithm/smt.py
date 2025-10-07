import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike
from tqdm.rich import tqdm

from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


class ContextInObservationWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wrapper to add context to the observation of the environment."""

    def __init__(self, env: gym.Env, context: ArrayLike):
        gym.utils.RecordConstructorArgs.__init__(self, context=context)
        gym.Wrapper.__init__(self, env)
        self.context = context
        self._observation_space = gym.spaces.Box(
            low=np.concatenate(
                (self.context, env.observation_space.low), axis=0
            ),
            high=np.concatenate(
                (self.context, env.observation_space.high), axis=0
            ),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate((self.context, obs), axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            np.concatenate((self.context, obs), axis=0),
            reward,
            terminated,
            truncated,
            info,
        )


class ContextualMultiTaskDefinition(metaclass=ABCMeta):
    """Defines a multi-task environment."""

    def __init__(self, contexts: ArrayLike, context_in_observation: bool):
        self.contexts = contexts
        self.context_in_observation = context_in_observation

    def get_task(self, task_id: int) -> gym.Env:
        """Returns the task environment for the given task ID."""
        assert 0 <= task_id < len(self.contexts)
        context = self.contexts[task_id]
        st_env = self._get_env(context)
        if self.context_in_observation:
            return ContextInObservationWrapper(st_env, context=context)
        else:
            return st_env

    @abstractmethod
    def _get_env(self, context: ArrayLike) -> gym.Env:
        """Returns the base environment without context."""

    @abstractmethod
    def get_solved_threshold(self, task_id: int) -> float:
        """Performance threshold for a task to be considered solved (>=)."""

    @abstractmethod
    def get_unsolvable_threshold(self, task_id: int) -> float:
        """Performance threshold for a task to be considered unsolvable (<=)."""

    def __len__(self) -> int:
        return len(self.contexts)


def train_smt(
    mt_def: ContextualMultiTaskDefinition,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    b1: int = 17_000_000,
    b2: int = 3_000_000,
    scheduling_interval: int = 1,
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
    more challenging tasks, thereby enhancing overall learning efficiency.

    Parameters
    ----------
    mt_def
        The multi-task environment definition.

    train_st : callable
        The single-task training algorithm. The training function should accept
        the following parameters:

        - ``env``: The environment to train on.
        - ``learning_starts``: Number of steps to wait before starting training.
        - ``total_timesteps``: Total number of timesteps to train the agent.
        - ``replay_buffer``: The replay buffer to use for training.
        - ``seed``: Seed for random number generation.
        - ``logger``: Logger to record training statistics.
        - ``global_step``: Current global step in the training process.
        - ``progress_bar``: Flag to enable/disable the tqdm progress bar.

        No other parameters will be passed to the training function by SMT.
        Hence, for any other parameters, the training function should be
        prepared with ``functools.partial`` or similar.

    replay_buffer : MultiTaskReplayBuffer
        Replay buffer.

    b1 : int
        Total number of timesteps to train the agent in first stage.
        Corresponds to :math:`B_1` in the paper [1]_.

    b2 : int
        Total number of timesteps to train the agent in second stage.
        Corresponds to :math:`B_2` in the paper [1]_.

    scheduling_interval : int
        Number of episodes after which the task scheduling is performed.

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

    seed : int
        Seed for random number generation.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    result
        The training result. Same as the result of the ``train_st`` function.

    training_steps : np.ndarray
        Number of training steps for each task.

    training_performances : np.ndarray
        Average training performance for each task from stage 1.

    Notes
    -----
    In comparison to the original SMT algorithm [1]_, this implementation
    does not reset networks. Furthermore, it allows any underlying RL
    algorithm to be used as a backbone (not just SAC) and instead of estimating
    the task encoding with a recurrent neural network, it can use a context
    vector provided by the multi-task environment definition to distinguish
    tasks.

    References
    ----------
    .. [1] Cho, M., Park, J., Lee, S., Sung, Y. (2024). Hard tasks first:
       multi-task reinforcement learning through task scheduling. In
       Proceedings of the 41st International Conference on Machine Learning,
       Vol. 235. JMLR.org, Article 340, 8556–8577.
       https://icml.cc/virtual/2024/poster/33388
    """
    rng = np.random.default_rng(seed)

    b_total = b1 + b2
    global_step = 0
    training_steps = np.zeros(len(mt_def), dtype=int)
    progress = tqdm(total=b_total, disable=not progress_bar)

    avg_training_performances, global_step, result_st, unsolvable_pool = (
        smt_stage1(
            mt_def,
            train_st,
            replay_buffer,
            training_steps,
            global_step,
            scheduling_interval,
            b1,
            b_total,
            n_average,
            K,
            kappa,
            learning_starts,
            logger,
            seed,
            rng,
            progress,
        )
    )

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
    else:
        warnings.warn(
            "Unsolvable pool is empty. There is no second SMT stage.",
            stacklevel=2,
        )
    progress.close()

    return result_st, training_steps, avg_training_performances


def smt_stage1(
    mt_def,
    train_st,
    replay_buffer,
    training_steps,
    global_step,
    scheduling_interval,
    b1,
    b_total,
    n_average,
    K,
    kappa,
    learning_starts,
    logger,
    seed,
    rng,
    progress,
):
    n_tasks = len(mt_def)
    training_pool = set(rng.choice(n_tasks, size=K, replace=False))
    main_pool = set(range(n_tasks)) - training_pool
    solved_pool = set()
    unsolvable_pool = set()
    avg_training_performances = np.full(n_tasks, -np.finfo(float).max)
    training_performances = [deque(maxlen=n_average) for _ in range(n_tasks)]
    task_budgets = np.full(n_tasks, kappa * b_total)
    while global_step < b1:
        updated_training_pool = copy.deepcopy(training_pool)
        for task_id in training_pool:
            env = mt_def.get_task(task_id)
            env_with_stats = gym.wrappers.RecordEpisodeStatistics(
                env, buffer_length=scheduling_interval
            )
            replay_buffer.select_task(task_id)

            result_st = train_st(
                env=env_with_stats,
                learning_starts=learning_starts,
                total_timesteps=b1,
                total_episodes=scheduling_interval,
                replay_buffer=replay_buffer,
                seed=seed + global_step,
                logger=logger,
                global_step=global_step,
                progress_bar=False,
            )

            steps = sum(env_with_stats.length_queue)
            training_steps[task_id] += steps
            global_step += steps
            progress.update(steps)

            if len(env_with_stats.return_queue) != scheduling_interval:
                # early termination because we reached step limit
                unlogged_steps = b1 - global_step
                global_step = b1
                training_steps[task_id] += unlogged_steps
                progress.update(unlogged_steps)

            training_performances[task_id].extend(env_with_stats.return_queue)
            avg_training_performances[task_id] = np.mean(
                training_performances[task_id]
            )

            if logger is not None:
                logger.record_stat("task_id", task_id, step=global_step)
                logger.record_stat(
                    "task_performance",
                    avg_training_performances[task_id],
                    step=global_step,
                )
                logger.record_stat(
                    "pool_size_main", len(main_pool), step=global_step
                )
                logger.record_stat(
                    "pool_size_solved", len(solved_pool), step=global_step
                )
                logger.record_stat(
                    "pool_size_unsolvable",
                    len(unsolvable_pool),
                    step=global_step,
                )

            M = mt_def.get_solved_threshold(task_id)
            if avg_training_performances[task_id] >= M:
                solved_pool.add(task_id)
                updated_training_pool.remove(task_id)
            elif training_steps[task_id] >= task_budgets[task_id]:
                m = mt_def.get_unsolvable_threshold(task_id)
                if avg_training_performances[task_id] <= m:
                    unsolvable_pool.add(task_id)
                    updated_training_pool.remove(task_id)
                else:
                    main_pool.add(task_id)
                    updated_training_pool.remove(task_id)

            if global_step >= b1:
                print("break2")
                break

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
                logger.record_stat("worst task", worst, step=global_step)
                logger.record_stat(
                    "worst performance",
                    avg_training_performances[worst],
                    step=global_step,
                )

        if early_stop_stage:
            break

        training_pool = updated_training_pool

    if len(unsolvable_pool) == 0 and len(main_pool) > 0:
        # If there are no unsolvable tasks, iterate over the main pool instead.
        # This is different from the original implementation.
        unsolvable_pool = main_pool

    return avg_training_performances, global_step, result_st, unsolvable_pool


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

            env_with_stats = gym.wrappers.RecordEpisodeStatistics(
                env, buffer_length=scheduling_interval
            )

            result_st = train_st(
                env=env_with_stats,
                learning_starts=learning_starts,
                total_timesteps=b_total,
                total_episodes=scheduling_interval,
                replay_buffer=replay_buffer,
                seed=seed + global_step,
                logger=logger,
                global_step=global_step,
                progress_bar=False,
            )

            steps = sum(env_with_stats.length_queue)
            training_steps[task_id] += steps
            global_step += steps
            progress.update(steps)

            if len(env_with_stats.return_queue) != scheduling_interval:
                # early termination because we reached step limit
                unlogged_steps = b_total - global_step
                global_step = b_total
                training_steps[task_id] += unlogged_steps
                progress.update(unlogged_steps)

            if logger is not None:
                logger.record_stat("task_id", task_id, step=global_step)
                logger.record_stat(
                    "pool_size_unsolvable",
                    len(unsolvable_pool),
                    step=global_step,
                )

            if global_step >= b_total:
                break

    return result_st
