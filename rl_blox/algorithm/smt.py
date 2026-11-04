import copy
import warnings
from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet, TaskSelectionMixin
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


def train_smt(
    task_set: DiscreteTaskSet | VectorEnv,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    b1: int = 17_000_000,
    b2: int = 3_000_000,
    solved_threshold: float = -100.0,
    unsolvable_threshold: float = -1000.0,
    scheduling_interval: int = 1,
    kappa: float = 0.8,
    K: int = 3,
    n_average: int = 5,
    learning_starts: int = 5_000,
    seed: int = 0,
    task_selectables: list[TaskSelectionMixin] | None = None,
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
    task_set : DiscreteTaskSet or VectorEnv
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

    solved_threshold : float
        A task is solved once its training performance (see ``n_average``)
        is greater than this value. Corresponds to :math:`M` in the paper [1]_.

    unsolvable_threshold : float
        When a task has used up its budget, it is unsolvable if its training
        performance is less than this value. Otherwise, it returns to the main
        pool. Corresponds to :math:`m` in the paper [1]_.

    scheduling_interval : int
        Number of episodes after which the task scheduling is performed.

    kappa : float
        Share of the remaining budget of the first stage that the tasks in
        the training pool may use. When a task enters the training pool, its
        budget is set to :math:`\kappa (B_1 - B_{used}) / K`, where
        :math:`B_{used}` is the number of steps used so far. This follows
        Appendix G.1 of [1]_. Algorithm 1 of [1]_ uses :math:`\kappa B`
        instead, with :math:`B` the remaining total budget, but with this
        budget, tasks would hardly ever use up their budget in the first
        stage for the values of :math:`\kappa` and :math:`K` from the paper.

    K : int
        Number of tasks to train in each iteration.

    n_average : int
        Number of recent training episodes of a task whose mean return is the
        task's training performance. It is used to check if a task is solved
        or unsolvable.

    learning_starts : int
        Number of steps at the start of SMT in which the backbone only takes
        random actions and does not train. ``train_st`` compares it with the
        global step, so it counts steps over all tasks, not per task.

    seed : int
        Seed for random number generation.

    task_selectables : list, optional
        When a task is selected, these objects will be informed about the task
        index.

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
        Average training performance for each task.

    Warns
    -----
    UserWarning
        If ``K`` is not smaller than the number of tasks, because then there
        is no task scheduling.

    UserWarning
        If all tasks are solved in the first stage, so that the second stage
        trains all tasks round robin.

    Notes
    -----
    In comparison to the original SMT algorithm [1]_, this implementation
    does not reset networks. Furthermore, it allows any underlying RL
    algorithm to be used as a backbone (not just SAC) and instead of estimating
    the task encoding with a recurrent neural network, it can use a context
    vector provided by the multi-task environment definition to distinguish
    tasks. If no task is unsolvable after the first stage, the second stage
    trains all tasks that are not solved instead, including those that are
    still in the training pool. The second stage trains these tasks only
    until they are solved. Afterwards, or if all tasks are already solved in
    the first stage, it trains all tasks round robin until the budget is used
    up, so that SMT always uses the full budget :math:`B_1 + B_2`.

    References
    ----------
    .. [1] Cho, M., Park, J., Lee, S., Sung, Y. (2024). Hard tasks first:
       multi-task reinforcement learning through task scheduling. In
       Proceedings of the 41st International Conference on Machine Learning,
       Vol. 235. JMLR.org, Article 340, 8556–8577.
       https://icml.cc/virtual/2024/poster/33388
    """
    rng = np.random.default_rng(seed)
    if isinstance(task_set, VectorEnv):
        n_tasks = task_set.num_envs
    else:
        n_tasks = len(task_set)

    if K >= n_tasks:
        warnings.warn(
            f"K={K} is not smaller than the number of tasks ({n_tasks}). All "
            "tasks are trained at the same time, so there is no task "
            "scheduling.",
            stacklevel=2,
        )

    b_total = b1 + b2
    global_step = 0
    training_steps = np.zeros(n_tasks, dtype=int)
    training_performances = [deque(maxlen=n_average) for _ in range(n_tasks)]
    avg_training_performances = np.full(n_tasks, -np.finfo(float).max)
    progress = tqdm(total=b_total, disable=not progress_bar)

    global_step, result_st, unsolvable_pool = smt_stage1(
        task_set,
        train_st,
        replay_buffer,
        task_selectables,
        training_steps,
        training_performances,
        avg_training_performances,
        solved_threshold,
        unsolvable_threshold,
        global_step,
        scheduling_interval,
        b1,
        K,
        kappa,
        learning_starts,
        logger,
        seed,
        rng,
        progress,
    )

    if len(unsolvable_pool) == 0:
        warnings.warn(
            "All tasks are solved after the first SMT stage. The second stage "
            "trains all tasks round robin.",
            stacklevel=2,
        )

    result_st = smt_stage2(
        task_set,
        train_st,
        replay_buffer,
        task_selectables,
        unsolvable_pool,
        training_steps,
        training_performances,
        avg_training_performances,
        solved_threshold,
        global_step,
        scheduling_interval,
        b_total,
        learning_starts,
        logger,
        seed,
        progress,
        result_st,
    )
    progress.close()

    return result_st, training_steps, avg_training_performances


def smt_stage1(
    task_set,
    train_st,
    replay_buffer,
    task_selectables,
    training_steps,
    training_performances,
    avg_training_performances,
    solved_threshold,
    unsolvable_threshold,
    global_step,
    scheduling_interval,
    b1,
    K,
    kappa,
    learning_starts,
    logger,
    seed,
    rng,
    progress,
):
    if isinstance(task_set, VectorEnv):
        n_tasks = task_set.num_envs
    else:
        n_tasks = len(task_set)
    training_pool = set(rng.choice(n_tasks, size=K, replace=False))
    main_pool = set(range(n_tasks)) - training_pool
    solved_pool = set()
    unsolvable_pool = set()
    task_budgets = np.full(n_tasks, kappa * b1 / K)
    # A task's budget is spent from the moment it enters the training pool.
    steps_in_training_pool = np.zeros(n_tasks, dtype=int)

    while global_step < b1:
        updated_training_pool = copy.deepcopy(training_pool)
        for task_id in training_pool:
            if isinstance(task_set, VectorEnv):
                env = task_set.envs[task_id]
            else:
                env = task_set.get_task(task_id)
            env_with_stats = gym.wrappers.RecordEpisodeStatistics(
                env, buffer_length=scheduling_interval
            )
            replay_buffer.select_task(task_id)
            if task_selectables is not None:
                for ts in task_selectables:
                    ts.select_task(task_id)

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
            steps_in_training_pool[task_id] += steps
            global_step += steps
            progress.update(steps)

            if len(env_with_stats.return_queue) != scheduling_interval:
                # early termination because we reached step limit
                unlogged_steps = b1 - global_step
                global_step = b1
                training_steps[task_id] += unlogged_steps
                steps_in_training_pool[task_id] += unlogged_steps
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

            if avg_training_performances[task_id] > solved_threshold:
                solved_pool.add(task_id)
                updated_training_pool.remove(task_id)
            elif steps_in_training_pool[task_id] >= task_budgets[task_id]:
                if avg_training_performances[task_id] < unsolvable_threshold:
                    unsolvable_pool.add(task_id)
                    updated_training_pool.remove(task_id)
                else:
                    main_pool.add(task_id)
                    updated_training_pool.remove(task_id)

            if global_step >= b1:
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

            # Select task with the lowest performance (from main pool). Ties,
            # e.g., between tasks that were never trained, are broken at
            # random, so that untrained tasks are selected in random order.
            main_pool_indices = np.array(sorted(main_pool))
            performances = avg_training_performances[main_pool_indices]
            worst = int(
                rng.choice(
                    main_pool_indices[performances == performances.min()]
                )
            )
            # Move it to training pool
            updated_training_pool.add(worst)
            main_pool.remove(worst)
            # Set its budget to kappa * B / K (B: remaining budget of stage 1)
            task_budgets[worst] = kappa * (b1 - global_step) / K
            steps_in_training_pool[worst] = 0

            if logger is not None:
                logger.record_stat("worst task", worst, step=global_step)
                logger.record_stat(
                    "worst performance",
                    avg_training_performances[worst],
                    step=global_step,
                )

        training_pool = updated_training_pool

        if early_stop_stage:
            break

    unsolved_pool = main_pool | training_pool
    if len(unsolvable_pool) == 0 and len(unsolved_pool) > 0:
        # If there are no unsolvable tasks, iterate over all unsolved tasks
        # instead. This is different from the original implementation.
        unsolvable_pool = unsolved_pool

    return global_step, result_st, unsolvable_pool


def smt_stage2(
    task_set,
    train_st,
    replay_buffer,
    task_selectables,
    unsolvable_pool,
    training_steps,
    training_performances,
    avg_training_performances,
    solved_threshold,
    global_step,
    scheduling_interval,
    b_total,
    learning_starts,
    logger,
    seed,
    progress,
    result_st,
):
    """SMT trains the unsolvable tasks in stage 2 until they are solved.

    Once all of them are solved, all tasks are trained round robin.
    """
    n_tasks = len(training_steps)
    unsolved_pool = set(unsolvable_pool)
    while global_step < b_total:
        if len(unsolved_pool) > 0:
            scheduled_tasks = sorted(unsolved_pool)
        else:
            scheduled_tasks = range(n_tasks)

        for task_id in scheduled_tasks:
            if isinstance(task_set, VectorEnv):
                env = task_set.envs[task_id]
            else:
                env = task_set.get_task(task_id)
            replay_buffer.select_task(task_id)
            if task_selectables is not None:
                for ts in task_selectables:
                    ts.select_task(task_id)

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

            training_performances[task_id].extend(env_with_stats.return_queue)
            avg_training_performances[task_id] = np.mean(
                training_performances[task_id]
            )
            if (
                task_id in unsolved_pool
                and avg_training_performances[task_id] > solved_threshold
            ):
                unsolved_pool.remove(task_id)

            if logger is not None:
                logger.record_stat("task_id", task_id, step=global_step)
                logger.record_stat(
                    "task_performance",
                    avg_training_performances[task_id],
                    step=global_step,
                )
                logger.record_stat(
                    "pool_size_unsolvable",
                    len(unsolved_pool),
                    step=global_step,
                )

            if global_step >= b_total:
                break

    return result_st
