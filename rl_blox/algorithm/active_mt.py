from collections.abc import Callable

import gymnasium as gym
import numpy as np
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    DUCBGeneralized,
    RoundRobinSelector,
    TaskSelectionMixin,
    TaskSelector,
)
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase

TASK_SELECTORS = {
    "Round Robin": (RoundRobinSelector, {}),
    "1-step Progress": (
        DUCBGeneralized,
        {
            "baseline": "last",
            "op": None,
        },
    ),
    "Monotonic Progress": (
        DUCBGeneralized,
        {
            "baseline": "max",
            "op": "max-with-0",
        },
    ),
    "Best Reward": (
        DUCBGeneralized,
        {
            "baseline": None,
            "op": None,
        },
    ),
    "Diversity": (
        DUCBGeneralized,
        {
            "baseline": None,
            "op": "neg",
        },
    ),
}


def train_active_mt(
    mt_def: DiscreteTaskSet,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    r_max: float,
    ducb_gamma: float = 0.95,
    xi: float = 0.002,
    task_selector: TaskSelector | str = "Monotonic Progress",
    total_timesteps: int = 1_000_000,
    scheduling_interval: int = 1,
    learning_starts: int = 5_000,
    seed: int = 0,
    task_selectables: list[TaskSelectionMixin] | None = None,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple:
    """Active Multi-Task Training.

    A multi-task extension of deep reinforcement learning similar to the method
    proposed by Fabisch and Metzen [1]_ for contextual policy search.

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

    r_max : float
        Upper bound for task selection reward.

    ducb_gamma : float
        Discount factor for D-UCB.

    xi : float, optional
        Padding function strength.

    task_selector : TaskSelector or str, default="Monotonic Progress"
        The task selection strategy. Can be an instance of ``TaskSelector`` or
        one of the predefined strategies:

        - "Round Robin": Cycles through tasks in order.
        - "1-step Progress": Selects tasks based on immediate progress.
        - "Monotonic Progress": Selects tasks based on the strictly positive
          immediate progress.
        - "Best Reward": Chooses tasks with the highest rewards.
        - "Diversity": Chooses tasks with the lowest rewards.

    total_timesteps : int
        The number of environment sets to train for.

    scheduling_interval : int
        Number of episodes after which the task scheduling is performed.

    learning_starts : int
        Number of steps to wait before starting training per task.

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

    References
    ----------
    .. [1] Fabisch, A., Metzen, J. H. (2014). Active Contextual Policy Search.
       Journal of Machine Learning Research, 15(97), 3371-3399.
       https://jmlr.org/papers/v15/fabisch14a.html
    """
    global_step = 0
    training_steps = np.zeros(len(mt_def), dtype=int)
    progress = tqdm(total=total_timesteps, disable=not progress_bar)

    if isinstance(task_selector, str):
        assert task_selector in TASK_SELECTORS, (
            f"task_selector must be one of {list(TASK_SELECTORS.keys())}"
            " or an instance of TaskSelector."
        )
        selector_class, selector_kwargs = TASK_SELECTORS[task_selector]
        hparams = {
            "upper_bound": r_max,
            "ducb_gamma": ducb_gamma,
            "zeta": xi,
        }
        hparams.update(selector_kwargs)
        task_selector = selector_class(tasks=np.arange(len(mt_def)), **hparams)

    while global_step < total_timesteps:
        task_id = task_selector.select()
        if logger is not None:
            logger.record_stat("task_id", task_id, step=global_step)

        env = mt_def.get_task(task_id)
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
            total_timesteps=total_timesteps,
            total_episodes=scheduling_interval,
            replay_buffer=replay_buffer,
            seed=seed + global_step,
            logger=logger,
            global_step=global_step,
            progress_bar=False,
        )

        if len(env_with_stats.return_queue) != scheduling_interval:
            # limit total_timesteps reached
            unlogged_steps = total_timesteps - global_step
            training_steps[task_id] += unlogged_steps
            progress.update(unlogged_steps)
            break

        assert len(env_with_stats.return_queue) > 0
        mean_return = np.mean(env_with_stats.return_queue)
        task_selector.feedback(mean_return)

        steps = sum(env_with_stats.length_queue)
        training_steps[task_id] += steps
        global_step += steps

        progress.update(steps)

    return result_st, training_steps
