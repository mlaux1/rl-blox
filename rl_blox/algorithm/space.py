from collections.abc import Callable

import jax.numpy as jnp
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


def space_grow(mean_value: float, last_mean_value: float, eta: float) -> bool:
    r"""Check if SPaCE increases the size of the instance set.

    Parameters
    ----------
    mean_value : float
        Mean absolute start-state value :math:`V_t` of the current instance
        set after the last iteration.

    last_mean_value : float
        Mean absolute start-state value :math:`V_{t-1}` after the iteration
        before.

    eta : float
        Relative threshold :math:`\eta`.

    Returns
    -------
    grow : bool
        True if :math:`V_t \in [(1 - \eta) V_{t-1}, (1 + \eta) V_{t-1}]`,
        i.e., if the value estimates stagnated.
    """
    return (
        (1.0 - eta) * last_mean_value
        <= mean_value
        <= (1.0 + eta) * last_mean_value
    )


def space_instance_set(
    values: np.ndarray, last_values: np.ndarray, size: int
) -> np.ndarray:
    r"""Select the instances with the highest performance improvement capacity.

    The performance improvement capacity (PIC) of instance :math:`i` is

    .. math::

        d_t(i) = V_t(s_0, c_i) - V_{t-1}(s_0, c_i).

    Parameters
    ----------
    values : array, shape (n_tasks,)
        Start-state values after the last iteration.

    last_values : array, shape (n_tasks,)
        Start-state values after the iteration before.

    size : int
        Size of the instance set.

    Returns
    -------
    instance_set : array, shape (size,)
        Indices of the selected instances in descending order of their PIC.
        Ties are broken by the lower index.
    """
    pic = np.asarray(values) - np.asarray(last_values)
    return np.argsort(-pic, kind="stable")[:size]


def train_space(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    value_function: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    total_timesteps: int,
    eta: float = 0.05,
    kappa: int = 1,
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    r"""Self-Paced Context Evaluation (SPaCE).

    A task scheduler for multi-task reinforcement learning [1]_. It trains on
    a growing set of tasks (instances). The set grows once the backbone's
    value estimates of the start states stagnate, and it consists of the
    tasks whose start-state values changed most in the last iteration.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    value_function : callable
        Maps a batch of observations to the state values of the backbone,
        e.g., ``partial(greedy_value, q_net)`` with
        :func:`~rl_blox.blox.state_value.greedy_value` or
        :func:`~rl_blox.blox.state_value.policy_value`. For MR.Q, use
        :func:`~rl_blox.algorithm.mrq.mrq_value`. It must use the networks
        that ``train_st`` updates.

    total_timesteps : int
        Total number of environment steps.

    eta : float, optional
        Relative threshold :math:`\eta` of the change of the mean start-state
        value below which the instance set grows. Must be in (0, 1). The
        default is the value that the paper recommends.

    kappa : int, optional
        Number of tasks :math:`\kappa` that are added when the instance set
        grows. Must be positive.

    episodes_per_task : int, optional
        Number of episodes per task in each iteration.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up before
        the curriculum starts, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`. The paper has no
        warm-up.

    replay_buffer : MultiTaskReplayBuffer, optional
        Replay buffer of all tasks. If given, ``train_st`` gets the view of
        the selected task. Otherwise, the replay buffer bound to ``train_st``
        is used.

    task_selectables : list, optional
        Objects that are informed about the selected task, also while its
        start-state value is computed.

    seed : int, optional
        Random seed of the first task, the backbone, and the start states.

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
        If ``total_timesteps``, ``episodes_per_task`` or ``kappa`` is not
        positive, or ``eta`` is not in (0, 1).

    Warns
    -----
    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    SPaCE follows Algorithm 1 of [1]_. The instance set starts with a random
    task. Each iteration trains every task of the instance set for
    ``episodes_per_task`` episodes, in descending order of their PIC. After
    the iteration, SPaCE computes the start-state value
    :math:`V_t(s_0, c_i)` of every task and the mean absolute value
    :math:`V_t` of the tasks it just trained. The instance set grows by
    ``kappa`` tasks if

    .. math::

        V_t \in [(1 - \eta) V_{t-1}, (1 + \eta) V_{t-1}],

    with :math:`V_0 = 0`, see :func:`space_grow`. Then it consists of the
    tasks with the highest performance improvement capacity
    :math:`d_t(i) = V_t(s_0, c_i) - V_{t-1}(s_0, c_i)`, see
    :func:`space_instance_set`. :math:`V_0(s_0, c_i)` are the values before
    the first iteration. Once the instance set contains all tasks, SPaCE
    trains them in turns.

    As in the paper, each task has a single start state. It is sampled once
    by resetting the task's environment with a fixed seed. Computing values
    needs no environment steps.

    The paper does not define how long an iteration is. The authors' code
    [2]_ trains for 100 steps per task in the instance set, ranks tasks by
    their start-state value instead of the PIC in its experiments (default
    ``--spl_mode absolute``), and compares the change of the mean value
    with :math:`\eta` times the new mean.

    Logging

    * ``task_id`` - task selected for training
    * ``instance_set_size`` - size of the instance set
    * ``mean_value`` - mean absolute start-state value :math:`V_t`
    * ``task_value_{i}`` - start-state value of task i
    * ``task_pic_{i}`` - performance improvement capacity of task i

    References
    ----------
    .. [1] Eimer, T., Biedenkapp, A., Hutter, F., Lindauer, M. (2021).
       Self-Paced Context Evaluation for Contextual Reinforcement Learning.
       In Proceedings of the 38th International Conference on Machine
       Learning, PMLR 139. https://arxiv.org/abs/2106.05110

    .. [2] https://github.com/automl/SPaCE
    """
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must be in (0, 1)")
    if kappa < 1:
        raise ValueError("kappa must be positive")
    warn_if_state_not_bound(train_st, replay_buffer is not None)

    n_tasks = get_n_tasks(task_set)
    rng = np.random.default_rng(seed)
    training_steps = np.zeros(n_tasks, dtype=int)
    start_observations = [
        get_task_env(task_set, task_id).reset(seed=seed + task_id)[0]
        for task_id in range(n_tasks)
    ]

    global_step = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)

    def train(task_id):
        nonlocal global_step, result
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

    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    for i in range(n_warmup):
        if global_step >= total_timesteps:
            break
        train(i % n_tasks)

    values = _start_values(value_function, start_observations, task_selectables)
    mean_value = 0.0
    instance_set = [int(rng.integers(n_tasks))]
    while global_step < total_timesteps:
        for task_id in instance_set:
            if global_step >= total_timesteps:
                break
            train(task_id)
        if global_step >= total_timesteps:
            break

        last_values, last_mean_value = values, mean_value
        values = _start_values(
            value_function, start_observations, task_selectables
        )
        mean_value = float(np.mean(np.abs(values[instance_set])))
        size = len(instance_set)
        if space_grow(mean_value, last_mean_value, eta):
            size = min(size + kappa, n_tasks)
        instance_set = [
            int(i) for i in space_instance_set(values, last_values, size)
        ]

        if logger is not None:
            logger.record_stat("instance_set_size", size, step=global_step)
            logger.record_stat("mean_value", mean_value, step=global_step)
            for i in range(n_tasks):
                logger.record_stat(
                    f"task_value_{i}", values[i], step=global_step
                )
                logger.record_stat(
                    f"task_pic_{i}",
                    values[i] - last_values[i],
                    step=global_step,
                )

    progress.close()
    return SchedulerResult(result, training_steps, 0)


def _start_values(value_function, start_observations, task_selectables):
    values = np.empty(len(start_observations))
    for task_id, observation in enumerate(start_observations):
        if task_selectables is not None:
            for ts in task_selectables:
                ts.select_task(task_id)
        values[task_id] = float(
            value_function(jnp.asarray(observation)[jnp.newaxis])[0]
        )
    return values
