import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from scipy.special import softmax
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet, warn_if_state_not_bound
from ..logging.logger import LoggerBase


@nnx.jit
def act(policy, observation):
    out = policy(observation)
    return out[0] if isinstance(out, tuple) else out


def _get_env(task_set, i):
    if isinstance(task_set, SyncVectorEnv):
        return task_set.envs[i]
    else:
        return task_set.get_task(i)


def _max_episode_steps(env):
    while env is not None:
        if isinstance(env, TimeLimit):
            return env._max_episode_steps
        if env.spec is not None and env.spec.max_episode_steps is not None:
            return env.spec.max_episode_steps
        env = getattr(env, "env", None)
    return None


class EvalEpisode(NamedTuple):
    """Evaluation episode passed to the ``episode_score`` function of T3S.

    Parameters
    ----------
    task_id : int
        Index of the evaluated task.

    rewards : array, shape (n_steps,)
        Rewards obtained in each step of the episode.

    infos : list of dict
        Info dictionaries returned by the environment in each step.
    """

    task_id: int
    rewards: np.ndarray
    infos: list[dict]


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
    tau: float = 0.5,
    learning_speed_weight: float = 0.5,
    progress_weight: float = 0.5,
    n_eval_episodes: int = 10,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    tasks_per_iteration: int = 3,
    sample_tasks_with_replacement: bool = False,
    eval_freq: int = 25_000,
    max_eval_episode_steps: int | None = None,
    episode_score: Callable[[EvalEpisode], float] = success,
    seed: int = 1,
    exploring_starts: int = 2_000,
    progress_bar: bool = True,
    logger: LoggerBase = None,
) -> tuple:
    r"""T3S task sampling.

    A task scheduling method for multi-task reinforcement learning. Given
    a set of tasks, it samples a task to train on based on a combination of
    task progress and task learning speed by performing regular evaluation
    rollouts on each task in the set [1]_.

    Parameters
    ----------
    task_set : DiscreteTaskSet | SyncVectorEnv
        The set of tasks available for training.

    eval_set : DiscreteTaskSet | SyncVectorEnv
        The set of tasks used for evaluation. Must contain the same number of
        tasks as ``task_set`` in the same order.

    train_st : Callable
        The training function of the backbone algorithm, e.g., ``train_sac``
        with the backbone's state bound via ``functools.partial``. It is called
        once per scheduled task and must accept the keyword arguments
        ``seed``, ``total_timesteps``, ``total_episodes``, ``learning_starts``,
        ``progress_bar``, ``logger``, ``global_step``, and ``bar``, and return
        a namedtuple with the fields ``policy`` and ``global_step``.

    tau : float, optional (default: 0.5)
        Temperature of the softmax distributions. Smaller values concentrate
        the sampling on fewer tasks. Must be positive.

    learning_speed_weight : float, optional (default: 0.5)
        Weight of the learning speed distribution.

    progress_weight : float, optional (default: 0.5)
        Weight of the task progress distribution. Must sum up to 1 with
        ``learning_speed_weight``.

    n_eval_episodes : int, optional (default: 10)
        Number of evaluation episodes per task in each evaluation.

    total_timesteps : int
        The number of total environment steps, including training and
        evaluation steps.

    episodes_per_task : int
        The number of episodes to train the policy on the scheduled task for.

    tasks_per_iteration : int, optional (default: 3)
        Number of tasks sampled from the current distribution before checking
        whether an evaluation is due.

    sample_tasks_with_replacement : bool, optional (default: False)
        Sample the tasks of an iteration with replacement. Without replacement,
        ``tasks_per_iteration`` must not exceed the number of tasks.

    eval_freq : int, optional (default: 25_000)
        Number of environment steps between evaluations. Evaluations are only
        performed between iterations, so the actual interval can be slightly
        larger.

    max_eval_episode_steps : int, optional
        Maximum length of an evaluation episode. Used to skip evaluations that
        would exceed ``total_timesteps``. If not set, it is inferred from the
        ``TimeLimit`` wrapper or spec of the evaluation environments. If it
        cannot be inferred, evaluations may exceed ``total_timesteps``.

    episode_score : callable, optional (default: success)
        Maps an :class:`EvalEpisode` to a score in [0, 1]. The performance of
        a task is the mean score of its evaluation episodes. Use
        :func:`success` for environments that report ``info["success"]`` or
        :func:`normalized_return` with bound ``low`` and ``high`` otherwise.

    seed : int
        The random seed for task sampling, the backbone, and the evaluation
        environments.

    exploring_starts : int
        The number of random exploration steps to be performed at the beginning
        of training.

    progress_bar : bool
        Flag to enable/disable the tqdm progress bar.

    logger : Logger
        Experiment logger.

    Returns
    -------
    result : namedtuple
        The result of the last call to ``train_st``, extended by the fields
        ``train_steps`` and ``eval_steps``. Both training and evaluation steps
        count towards ``total_timesteps``, hence ``global_step`` is their sum.

    Raises
    ------
    ValueError
        If ``tau`` is not positive, the metric weights do not sum up to 1,
        ``tasks_per_iteration`` or ``n_eval_episodes`` is invalid, or
        ``episode_score`` returns a value outside of [0, 1].

    Warns
    -----
    UserWarning
        If the episode length of the evaluation environments cannot be
        inferred and ``max_eval_episode_steps`` is not set.

    UserWarning
        If the backbone's replay buffer, target networks, or entropy control
        are not bound to ``train_st``.

    Notes
    -----
    After each evaluation, the performance :math:`p_i \in [0, 1]` of each task
    :math:`i` is used to update the task sampling probabilities

    .. math::

        P(i) = w_p \operatorname{softmax}\left(\frac{1 - p_i}{\tau}\right)
        + w_s \operatorname{softmax}\left(
        \frac{-(p_i - p_i^{\text{prev}})}{\tau}\right),

    where :math:`w_p` (``progress_weight``) and :math:`w_s`
    (``learning_speed_weight``) weight task progress and learning speed, and
    :math:`p_i^{\text{prev}}` is the performance at the previous evaluation.
    Before the first evaluation, tasks are sampled uniformly.

    All tasks are trained with the same backbone state, i.e., the policy,
    critics, optimizers, replay buffer, and entropy coefficient that are bound
    to ``train_st``. The backbone's state must be bound explicitly, e.g.,
    ``partial(train_sac, replay_buffer=..., q_target=..., ...)``. Otherwise,
    the backbone creates a new replay buffer and new target networks for each
    scheduled task.

    Logging

    * ``task_id`` - task scheduled for training
    * ``train_steps`` - number of environment steps used for training
    * ``eval_steps`` - number of environment steps used for evaluation
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
    """
    assert total_timesteps > 0
    warn_if_state_not_bound(train_st)

    if tau <= 0:
        raise ValueError("tau must be > 0")

    if not np.isclose(
        progress_weight + learning_speed_weight,
        1.0,
    ):
        raise ValueError("metric weights must sum up to 1")

    # global_step counts both training and evaluation steps
    global_step = 0
    train_steps = 0
    eval_steps = 0
    next_eval_step = eval_freq
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    key = jax.random.key(seed)

    if isinstance(task_set, SyncVectorEnv):
        n_tasks = task_set.num_envs
        assert eval_set.num_envs == n_tasks
    else:
        n_tasks = len(task_set)
        assert len(eval_set) == n_tasks

    if tasks_per_iteration < 1:
        raise ValueError("tasks_per_iteration must be positive")
    if not sample_tasks_with_replacement and tasks_per_iteration > n_tasks:
        raise ValueError(
            "tasks_per_iteration cannot exceed number of tasks when sampling "
            "without replacement"
        )
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be positive")

    # Seed evaluation environments for reproducible evaluations
    for i in range(n_tasks):
        _get_env(eval_set, i).reset(seed=seed + i)

    # Upper bound on the number of steps a full evaluation can take
    if max_eval_episode_steps is not None:
        episode_limits = [max_eval_episode_steps] * n_tasks
    else:
        episode_limits = [
            _max_episode_steps(_get_env(eval_set, i)) for i in range(n_tasks)
        ]
    if None in episode_limits:
        if eval_freq < total_timesteps:
            warnings.warn(
                "Could not infer the episode length of all evaluation "
                "environments, evaluations may exceed total_timesteps. Set "
                "max_eval_episode_steps to avoid this.",
                stacklevel=2,
            )
        max_eval_steps = 0
    else:
        max_eval_steps = n_eval_episodes * sum(episode_limits)

    # Assign initial sampling probabilities
    task_probs = np.ones(n_tasks) / n_tasks
    previous_task_performance = None

    if logger is not None:
        for i in range(n_tasks):
            logger.record_stat(
                f"task_probability_{i}",
                task_probs[i],
                step=global_step,
            )

    while global_step < total_timesteps:
        key, skey = jax.random.split(key)
        task_ids = np.asarray(
            jax.random.choice(
                skey,
                n_tasks,
                shape=(tasks_per_iteration,),
                replace=sample_tasks_with_replacement,
                p=task_probs,
            )
        )

        for task_id in task_ids:
            if global_step >= total_timesteps:
                break
            task_id = int(task_id)
            if logger is not None:
                logger.record_stat("task_id", task_id, step=global_step)

            st_result = train_st(
                _get_env(task_set, task_id),
                seed=seed + global_step,
                total_timesteps=total_timesteps,
                total_episodes=episodes_per_task,
                learning_starts=exploring_starts,
                progress_bar=progress_bar,
                logger=logger,
                global_step=global_step,
                bar=progress,
            )

            train_steps += st_result.global_step - global_step
            global_step = st_result.global_step

        if (
            global_step >= next_eval_step
            and global_step + max_eval_steps <= total_timesteps
        ):
            # Evaluate and compute task progress and task learning speed
            task_performance = np.zeros_like(task_probs)

            for i in range(n_tasks):
                env = _get_env(eval_set, i)
                scores = []

                for _ in range(n_eval_episodes):
                    obs, _ = env.reset()
                    rewards = []
                    infos = []
                    done = False
                    while not done:
                        action = np.asarray(act(st_result.policy, obs))
                        eval_steps += 1
                        global_step += 1
                        progress.update()
                        obs, reward, terminated, truncated, info = env.step(
                            action
                        )
                        rewards.append(reward)
                        infos.append(info)
                        done = terminated or truncated

                    score = episode_score(
                        EvalEpisode(i, np.asarray(rewards), infos)
                    )
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(
                            f"episode_score must return a value in [0, 1], "
                            f"got {score} for task {i}"
                        )
                    scores.append(score)

                task_performance[i] = np.mean(scores)

            # Compute new task probabilities
            task_progress_metric = 1.0 - task_performance

            if previous_task_performance is None:
                task_learning_speed_metric = np.zeros(n_tasks)
            else:
                # Negative change in performance as in the T3S paper: tasks
                # whose performance dropped get more weight
                task_learning_speed_metric = -(
                    task_performance - previous_task_performance
                )
            task_probs = progress_weight * softmax(
                task_progress_metric / tau
            ) + learning_speed_weight * softmax(
                task_learning_speed_metric / tau
            )

            if logger is not None:
                logger.record_stat("train_steps", train_steps, step=global_step)
                logger.record_stat("eval_steps", eval_steps, step=global_step)
                for i in range(n_tasks):
                    logger.record_stat(
                        f"task_performance_{i}",
                        task_performance[i],
                        step=global_step,
                    )
                    logger.record_stat(
                        f"task_progress_{i}",
                        task_progress_metric[i],
                        step=global_step,
                    )
                    logger.record_stat(
                        f"task_learning_speed_{i}",
                        task_learning_speed_metric[i],
                        step=global_step,
                    )
                    logger.record_stat(
                        f"task_probability_{i}",
                        task_probs[i],
                        step=global_step,
                    )

            previous_task_performance = task_performance.copy()

            next_eval_step = eval_freq + global_step

    progress.close()

    if logger is not None:
        logger.record_stat("train_steps", train_steps, step=global_step)
        logger.record_stat("eval_steps", eval_steps, step=global_step)

    T3SResult = namedtuple(
        "T3SResult", st_result._fields + ("train_steps", "eval_steps")
    )
    return T3SResult(
        *st_result._replace(global_step=global_step), train_steps, eval_steps
    )
