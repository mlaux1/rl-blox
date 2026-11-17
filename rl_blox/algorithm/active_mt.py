from collections.abc import Callable

import numpy as np
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    DUCBGeneralized,
    RoundRobinSelector,
    SchedulerResult,
    TaskSelectionMixin,
    TaskSelector,
    get_n_tasks,
    get_task_env,
    n_warmup_selections,
    train_task,
    warn_if_state_not_bound,
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
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    r_max: float,
    *,
    total_timesteps: int,
    ducb_gamma: float = 0.95,
    xi: float = 0.002,
    task_selector: TaskSelector | str = "Monotonic Progress",
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    """Active multi-task training (AMT).

    A task scheduler for multi-task reinforcement learning that treats task
    selection as a non-stationary multi-armed bandit problem, similar to the
    method proposed by Fabisch and Metzen [1]_ for contextual policy search.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    r_max : float
        Upper bound of the reward of the bandit, i.e., of the mean return of
        a task selection after the transformation of ``task_selector``.

    total_timesteps : int
        Total number of environment steps.

    ducb_gamma : float, optional
        Discount factor of D-UCB.

    xi : float, optional
        Strength of the padding function of D-UCB.

    task_selector : TaskSelector or str, optional
        Task selection strategy. Either an instance of ``TaskSelector`` or
        one of the predefined strategies:

        - "Round Robin": cycles through the tasks in order.
        - "1-step Progress": rewards the change of the mean return since the
          last selection of the task.
        - "Monotonic Progress": rewards only positive changes of the mean
          return since its best selection of the task.
        - "Best Reward": rewards high mean returns.
        - "Diversity": rewards low mean returns.

    episodes_per_task : int, optional
        Number of episodes per task selection.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`. The task
        selector does not learn from the warm-up.

    replay_buffer : MultiTaskReplayBuffer, optional
        Replay buffer of all tasks. If given, ``train_st`` gets the view of
        the selected task. Otherwise, the replay buffer bound to ``train_st``
        is used.

    task_selectables : list, optional
        Objects that are informed about the selected task.

    seed : int, optional
        Random seed of the backbone.

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
        If ``task_selector`` is unknown or ``total_timesteps`` or
        ``episodes_per_task`` is not positive.

    Warns
    -----
    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    After each task selection, the task selector gets the mean return of the
    selection's episodes as feedback.

    Logging

    * ``task_id`` - task selected for training

    References
    ----------
    .. [1] Fabisch, A., Metzen, J. H. (2014). Active Contextual Policy Search.
       Journal of Machine Learning Research, 15(97), 3371-3399.
       https://jmlr.org/papers/v15/fabisch14a.html
    """
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if isinstance(task_selector, str) and task_selector not in TASK_SELECTORS:
        raise ValueError(
            f"task_selector must be one of {list(TASK_SELECTORS)} or an "
            "instance of TaskSelector"
        )
    warn_if_state_not_bound(train_st, replay_buffer is not None)

    n_tasks = get_n_tasks(task_set)
    if isinstance(task_selector, str):
        selector_class, selector_kwargs = TASK_SELECTORS[task_selector]
        task_selector = selector_class(
            tasks=np.arange(n_tasks),
            upper_bound=r_max,
            ducb_gamma=ducb_gamma,
            zeta=xi,
            **selector_kwargs,
        )
    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    training_steps = np.zeros(n_tasks, dtype=int)

    global_step = 0
    n_selections = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    while global_step < total_timesteps:
        warmup = n_selections < n_warmup
        if warmup:
            task_id = n_selections % n_tasks
        else:
            task_id = int(task_selector.select())
        n_selections += 1

        result, steps, returns = train_task(
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

        # The selection is incomplete if the budget ran out
        if not warmup and len(returns) == episodes_per_task:
            task_selector.feedback(float(np.mean(returns)))

    progress.close()
    return SchedulerResult(result, training_steps, 0)
