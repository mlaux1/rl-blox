from collections.abc import Callable

import numpy as np
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    SchedulerResult,
    TaskSelectionMixin,
    get_n_tasks,
    get_task_env,
    n_warmup_selections,
    train_task,
    warn_if_state_not_bound,
)
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


def train_uts(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    *,
    total_timesteps: int,
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    """Uniform task sampling (UTS).

    A basic task scheduler for multi-task reinforcement learning. It samples
    tasks uniformly at random.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    total_timesteps : int
        Total number of environment steps.

    episodes_per_task : int, optional
        Number of episodes per task selection.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`.

    replay_buffer : MultiTaskReplayBuffer, optional
        Replay buffer of all tasks. If given, ``train_st`` gets the view of
        the selected task. Otherwise, the replay buffer bound to ``train_st``
        is used.

    task_selectables : list, optional
        Objects that are informed about the selected task.

    seed : int, optional
        Random seed.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Show a progress bar.

    Returns
    -------
    result : SchedulerResult
        Result of the last call to ``train_st``, training steps per task, and
        no evaluation steps.

    Raises
    ------
    ValueError
        If ``total_timesteps`` or ``episodes_per_task`` is not positive.

    Warns
    -----
    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    Logging

    * ``task_id`` - task selected for training
    """
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    warn_if_state_not_bound(train_st, replay_buffer is not None)

    n_tasks = get_n_tasks(task_set)
    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    rng = np.random.default_rng(seed)
    training_steps = np.zeros(n_tasks, dtype=int)

    global_step = 0
    n_selections = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    while global_step < total_timesteps:
        if n_selections < n_warmup:
            task_id = n_selections % n_tasks
        else:
            task_id = int(rng.integers(n_tasks))
        n_selections += 1

        result, steps, _ = train_task(
            train_st,
            get_task_env(task_set, task_id),
            task_id,
            episodes_per_task,
            total_timesteps,
            global_step,
            seed,
            task_selectables,
            logger,
            replay_buffer,
        )
        global_step += steps
        training_steps[task_id] += steps
        progress.update(steps)

    progress.close()
    return SchedulerResult(result, training_steps, 0)
