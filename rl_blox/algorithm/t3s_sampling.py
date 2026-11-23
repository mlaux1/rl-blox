from collections.abc import Callable
from functools import partial
from operator import attrgetter

import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from scipy.special import softmax
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    EvalEpisode,
    SchedulerResult,
    TaskSelectionMixin,
    evaluate_tasks,
    get_n_tasks,
    get_task_env,
    mean_task_scores,
    n_warmup_selections,
    next_evaluation_step,
    prepare_evaluation,
    train_task,
    warn_if_state_not_bound,
)
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


def success(episode: EvalEpisode) -> float:
    """Score an episode by success.

    Parameters
    ----------
    episode : EvalEpisode
        Evaluation episode.

    Returns
    -------
    score : float
        1 if ``info["success"]`` was set at any step of the episode,
        otherwise 0.

    Raises
    ------
    KeyError
        If the environment does not report ``info["success"]``.
    """
    try:
        return float(any(info["success"] for info in episode.infos))
    except KeyError:
        raise KeyError(
            "The evaluation environment does not report info['success']. "
            "Pass another episode_score, e.g., "
            "partial(normalized_return, low=..., high=...)."
        ) from None


def normalized_return(
    episode: EvalEpisode,
    low: float | np.ndarray,
    high: float | np.ndarray,
) -> float:
    """Score an episode by its return, scaled from [low, high] to [0, 1].

    Parameters
    ----------
    episode : EvalEpisode
        Evaluation episode.

    low : float or array, shape (n_tasks,)
        Return that corresponds to a score of 0, per task if an array.

    high : float or array, shape (n_tasks,)
        Return that corresponds to a score of 1, per task if an array.

    Returns
    -------
    score : float
        Clipped, normalized return.

    Raises
    ------
    ValueError
        If ``high`` is not greater than ``low``.

    Notes
    -----
    Bind the bounds with :func:`functools.partial` before passing this
    function as ``episode_score`` to :func:`train_t3s`.

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> score = partial(normalized_return, low=-1600.0, high=0.0)
    >>> score(EvalEpisode(task_id=0, rewards=np.full(200, -2.0), infos=[]))
    0.75
    """
    if np.ndim(low) > 0:
        low = low[episode.task_id]
    if np.ndim(high) > 0:
        high = high[episode.task_id]
    if high <= low:
        raise ValueError("high must be greater than low")

    ret = np.sum(episode.rewards)
    return float(np.clip((ret - low) / (high - low), 0.0, 1.0))


def train_t3s(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    *,
    total_timesteps: int,
    tau: float = 0.1,
    learning_speed_weight: float = 0.5,
    progress_weight: float = 0.5,
    n_average: int = 3,
    tasks_per_iteration: int = 5,
    sample_tasks_with_replacement: bool = False,
    weight_replay_batch: bool = True,
    eval_freq: int = 25_000,
    n_eval_episodes: int = 3,
    max_eval_episode_steps: int | None = None,
    episode_score: Callable[[EvalEpisode], float] = success,
    get_policy: Callable[..., nnx.Module] = attrgetter("policy"),
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    r"""T3S task scheduler.

    A task scheduler for multi-task reinforcement learning [1]_. It evaluates
    the policy on all tasks regularly and prefers tasks with low performance
    (task progress) and tasks whose performance dropped (learning speed).

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    eval_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks used for evaluation. Must contain the same number of
        tasks as ``task_set`` in the same order.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    total_timesteps : int
        Total number of environment steps, including evaluation steps.

    tau : float, optional
        Temperature of the softmax distributions. Smaller values concentrate
        the sampling on fewer tasks. Must be positive.

    learning_speed_weight : float, optional
        Weight of the learning speed distribution. 0 reproduces the authors'
        code, in which the learning speed term is commented out.

    progress_weight : float, optional
        Weight of the task progress distribution. Must sum up to 1 with
        ``learning_speed_weight``.

    n_average : int, optional
        Number of recent evaluations whose mean performance is used for task
        progress.

    tasks_per_iteration : int, optional
        Number of tasks sampled from the current distribution before checking
        whether an evaluation is due. Corresponds to :math:`K` in [1]_.

    sample_tasks_with_replacement : bool, optional
        Sample the tasks of an iteration with replacement. Without replacement,
        ``tasks_per_iteration`` must not exceed the number of tasks.

    weight_replay_batch : bool, optional
        Weight the tasks in replay batches by their sampling probabilities, as
        the authors' code does. Only applies if ``replay_buffer`` is given and
        replaces its ``task_weighting``. As in the authors' code, the weights
        are uniform if each iteration trains all tasks once.

    eval_freq : int, optional
        Number of environment steps between evaluations. Evaluations are only
        performed between iterations, but a delayed evaluation does not delay
        later ones, see :func:`~rl_blox.blox.multitask.next_evaluation_step`.

    n_eval_episodes : int, optional
        Number of evaluation episodes per task in each evaluation.

    max_eval_episode_steps : int, optional
        Maximum length of an evaluation episode. Used to skip evaluations that
        would exceed ``total_timesteps``. If not set, it is inferred from the
        evaluation environments, see
        :func:`~rl_blox.blox.multitask.max_evaluation_steps`. If it cannot be
        inferred, evaluations may exceed ``total_timesteps``.

    episode_score : callable, optional
        Maps an :class:`~rl_blox.blox.multitask.EvalEpisode` to a score in
        [0, 1]. The performance of a task is the mean score of its evaluation
        episodes. Use :func:`success` for environments that report
        ``info["success"]`` or :func:`normalized_return` with bound ``low``
        and ``high`` otherwise.

    get_policy : callable, optional
        Maps the result of ``train_st`` to the evaluated policy, an
        ``nnx.Module`` that maps an observation to an action, see
        :func:`~rl_blox.blox.multitask.act`. The default returns the field
        ``policy`` (SAC, DDPG, TD3).

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
        Objects that are informed about the selected or evaluated task.

    seed : int, optional
        Random seed of the task sampling, the backbone, and the evaluation
        environments.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Show a progress bar.

    Returns
    -------
    result : SchedulerResult
        Result of the last call to ``train_st``, training steps per task, and
        evaluation steps. Training and evaluation steps sum up to
        ``total_timesteps``.

    Raises
    ------
    ValueError
        If an argument is invalid, ``eval_set`` does not have as many tasks as
        ``task_set``, or ``episode_score`` returns a value outside of [0, 1].

    Warns
    -----
    UserWarning
        If the episode length of the evaluation environments cannot be
        inferred and ``max_eval_episode_steps`` is not set.

    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    After each evaluation, the task sampling probabilities are

    .. math::

        P(i) = w_p \operatorname{softmax}\left(\frac{1 - \bar{p}_i}{\tau}
        \right) + w_s \operatorname{softmax}\left(
        \frac{-(p_i - p_i^{\text{prev}})}{\tau}\right),

    where :math:`w_p` (``progress_weight``) and :math:`w_s`
    (``learning_speed_weight``) weight task progress and learning speed,
    :math:`\bar{p}_i` is the mean performance of task :math:`i` over the last
    ``n_average`` evaluations, and :math:`p_i` and :math:`p_i^{\text{prev}}`
    are its performances at the last two evaluations (learning speed 0 after
    the first one). Before the first evaluation, tasks are sampled uniformly.
    Evaluations take the deterministic action of the policy.

    The defaults of ``tau``, ``n_average``, ``tasks_per_iteration`` and
    ``n_eval_episodes`` come from the authors' code [2]_, the weights from
    the paper. The authors' code does not count evaluation steps toward the
    budget; here, they count, as in all task schedulers of rl-blox.

    Logging

    * ``task_id`` - task selected for training
    * ``eval_steps`` - number of environment steps used for evaluation so
      far
    * ``task_performance_{i}`` - performance of task i at the last evaluation
    * ``task_progress_{i}`` - task progress metric of task i
    * ``task_learning_speed_{i}`` - learning speed metric of task i
    * ``task_probability_{i}`` - sampling probability of task i

    References
    ----------
    .. [1] Yu, Y., Yang, T., Lv, Y., Zheng, Y., Hao, J. (2023). T3S: Improving
       Multi-Task Reinforcement Learning with Task-Specific Feature Selector
       and Scheduler. In 2023 International Joint Conference on Neural
       Networks (IJCNN), pp. 1-8.
       https://doi.org/10.1109/IJCNN54540.2023.10191536

    .. [2] https://github.com/metaqiang/t3s-multitask-rl
    """
    n_tasks = get_n_tasks(task_set)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not np.isclose(progress_weight + learning_speed_weight, 1.0):
        raise ValueError("metric weights must sum up to 1")
    if n_average < 1:
        raise ValueError("n_average must be positive")
    if tasks_per_iteration < 1:
        raise ValueError("tasks_per_iteration must be positive")
    if not sample_tasks_with_replacement and tasks_per_iteration > n_tasks:
        raise ValueError(
            "tasks_per_iteration cannot exceed number of tasks when sampling "
            "without replacement"
        )
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    warn_if_state_not_bound(train_st, replay_buffer is not None)
    max_eval_steps = prepare_evaluation(
        eval_set,
        n_tasks,
        eval_freq,
        n_eval_episodes,
        max_eval_episode_steps,
        total_timesteps,
        seed,
    )

    # The authors' code weights replay batches uniformly if all tasks are
    # trained in each iteration
    weight_batches = (
        weight_replay_batch
        and replay_buffer is not None
        and (sample_tasks_with_replacement or tasks_per_iteration < n_tasks)
    )
    task_probs = np.full(n_tasks, 1.0 / n_tasks)
    if weight_batches:
        replay_buffer.set_task_weights(task_probs)
    if logger is not None:
        for i in range(n_tasks):
            logger.record_stat(f"task_probability_{i}", task_probs[i], step=0)

    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    rng = np.random.default_rng(seed)
    training_steps = np.zeros(n_tasks, dtype=int)
    performances = []

    global_step = 0
    eval_steps = 0
    next_eval_step = eval_freq
    n_selections = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    while global_step < total_timesteps:
        if n_selections < n_warmup:
            task_ids = [n_selections % n_tasks]
        else:
            task_ids = rng.choice(
                n_tasks,
                size=tasks_per_iteration,
                replace=sample_tasks_with_replacement,
                p=task_probs,
            )

        for task_id in task_ids:
            if global_step >= total_timesteps:
                break
            task_id = int(task_id)
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

        if (
            global_step >= next_eval_step
            and global_step + max_eval_steps <= total_timesteps
        ):
            episodes, steps, _ = evaluate_tasks(
                get_policy(result),
                eval_set,
                n_eval_episodes,
                task_selectables=task_selectables,
            )
            global_step += steps
            eval_steps += steps
            progress.update(steps)
            next_eval_step = next_evaluation_step(
                next_eval_step, eval_freq, global_step
            )

            performances.append(
                mean_task_scores(episodes, partial(_score, episode_score))
            )
            progress_metric = 1.0 - np.mean(performances[-n_average:], axis=0)
            if len(performances) < 2:
                learning_speed_metric = np.zeros(n_tasks)
            else:
                # Tasks whose performance dropped get more weight
                learning_speed_metric = -(performances[-1] - performances[-2])
            task_probs = progress_weight * softmax(
                progress_metric / tau
            ) + learning_speed_weight * softmax(learning_speed_metric / tau)
            if weight_batches:
                replay_buffer.set_task_weights(task_probs)

            if logger is not None:
                logger.record_stat("eval_steps", eval_steps, step=global_step)
                for i in range(n_tasks):
                    for name, value in (
                        ("performance", performances[-1][i]),
                        ("progress", progress_metric[i]),
                        ("learning_speed", learning_speed_metric[i]),
                        ("probability", task_probs[i]),
                    ):
                        logger.record_stat(
                            f"task_{name}_{i}", value, step=global_step
                        )

    progress.close()
    return SchedulerResult(result, training_steps, eval_steps)


def _score(
    episode_score: Callable[[EvalEpisode], float], episode: EvalEpisode
) -> float:
    score = episode_score(episode)
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"episode_score must return a value in [0, 1], got {score} for "
            f"task {episode.task_id}"
        )
    return score
