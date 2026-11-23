import warnings
from collections import deque
from collections.abc import Callable
from functools import partial
from operator import attrgetter

import jax
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    EvalEpisode,
    SchedulerResult,
    TaskSelectionMixin,
    evaluate_tasks,
    get_n_tasks,
    get_task_env,
    is_stochastic,
    mean_task_scores,
    n_warmup_selections,
    next_evaluation_step,
    prepare_evaluation,
    train_task,
    warn_if_state_not_bound,
)
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


def train_smt(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    *,
    total_timesteps: int,
    b1_fraction: float = 0.85,
    solved_threshold: float = 4000.0,
    unsolvable_threshold: float = 1000.0,
    kappa: float = 0.8,
    K: int = 3,
    n_average: int = 5,
    eval_freq: int = 500_000,
    n_eval_episodes: int = 5,
    max_eval_episode_steps: int | None = None,
    gamma: float = 0.99,
    get_policy: Callable[..., nnx.Module] = attrgetter("policy"),
    reset_st: Callable[[int], None] | None = None,
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    r"""Scheduled Multi-Task (SMT) training.

    Multi-task RL faces the challenge of varying task difficulties, often
    leading to negative transfer when simpler tasks overshadow the learning of
    more complex ones. To overcome this challenge, SMT strategically prioritizes
    more challenging tasks, thereby enhancing overall learning efficiency [1]_.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    eval_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks used for evaluation. Must contain the same number of
        tasks as ``task_set`` in the same order. Must not share environment
        instances with ``task_set``.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    total_timesteps : int
        Total number of environment steps of both stages, including
        evaluation steps. Corresponds to :math:`B_1 + B_2` in [1]_.

    b1_fraction : float, optional
        Share of ``total_timesteps`` that the first stage uses. The first
        stage uses :math:`B_1` = ``round(b1_fraction * total_timesteps)``
        steps, the second stage the rest. The default corresponds to
        :math:`B_1` = 17M and :math:`B_2` = 3M in [1]_.

    solved_threshold : float, optional
        A task is solved once its training performance (see ``n_average``)
        is greater than this value. In the first stage, a solved task returns
        to the main pool if its undiscounted evaluation return is not greater
        than this value. Corresponds to :math:`M` in [1]_. The default is set
        for Meta-World with episodes of 500 steps.

    unsolvable_threshold : float, optional
        When a task has used up its budget, it is unsolvable if its training
        performance is less than this value. Otherwise, it returns to the main
        pool. Corresponds to :math:`m` in [1]_. The default is set for
        Meta-World with episodes of 500 steps.

    kappa : float, optional
        Share of the remaining budget of the first stage that the tasks in
        the training pool may use. When a task enters the training pool, its
        budget is set to :math:`\kappa (B_1 - B_{used}) / K`, where
        :math:`B_{used}` is the number of steps used so far (Appendix G.1 of
        [1]_).

    K : int, optional
        Size of the training pool, i.e., number of tasks trained in each
        iteration of the first stage.

    n_average : int, optional
        Number of recent training episodes of a task whose mean return is the
        task's training performance. It is used to check if a task is solved
        or unsolvable.

    eval_freq : int, optional
        Number of environment steps between evaluations in the first stage.
        Corresponds to :math:`T_{eval}` in [1]_. Evaluations are only
        performed between iterations, but a delayed evaluation does not delay
        later ones, see :func:`~rl_blox.blox.multitask.next_evaluation_step`.

    n_eval_episodes : int, optional
        Number of evaluation episodes per task in each evaluation.
        Corresponds to :math:`n_{eval}` in [1]_.

    max_eval_episode_steps : int, optional
        Maximum length of an evaluation episode. Used to skip evaluations that
        would exceed :math:`B_1`. If not set, it is inferred from the
        evaluation environments, see
        :func:`~rl_blox.blox.multitask.max_evaluation_steps`. If it cannot be
        inferred, evaluations may exceed :math:`B_1`.

    gamma : float, optional
        Discount factor of the evaluation metric. Should be the same as the
        discount factor of the backbone.

    get_policy : callable, optional
        Maps the result of ``train_st`` to the evaluated policy, an
        ``nnx.Module`` that maps an observation to an action. Policies with a
        method ``sample_and_log_prob(observation, key)`` are sampled, all
        others are treated as deterministic, see
        :func:`~rl_blox.blox.multitask.evaluate_tasks`. The default returns
        the field ``policy`` (SAC, DDPG, TD3).

    reset_st : callable, optional
        Reinitializes the backbone's networks after each evaluation in the
        first stage. It is called with a seed and must reinitialize the state
        bound to ``train_st`` in place, except for the replay buffer. If not
        set, the networks are not reset.

    episodes_per_task : int, optional
        Number of episodes per task selection.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up at the
        start of the first stage, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`.

    replay_buffer : MultiTaskReplayBuffer, optional
        Replay buffer of all tasks. If given, ``train_st`` gets the view of
        the selected task. Otherwise, the replay buffer bound to ``train_st``
        is used.

    task_selectables : list, optional
        Objects that are informed about the selected or evaluated task.

    seed : int, optional
        Random seed of the task selection, the backbone, the evaluation
        environments, and the evaluated policy.

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
        If an argument is invalid, :math:`B_1` is 0, or ``eval_set`` does
        not have as many tasks as ``task_set``.

    Warns
    -----
    UserWarning
        If ``K`` equals the number of tasks, so there is no task scheduling.

    UserWarning
        If all tasks are solved in the first stage, so that the second stage
        trains all tasks round robin.

    UserWarning
        If the evaluated policy is deterministic, so that the entropy term of
        the evaluation metric is omitted.

    UserWarning
        If the episode length of the evaluation environments cannot be
        inferred and ``max_eval_episode_steps`` is not set.

    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    In contrast to [1]_, any RL algorithm can be used as a backbone, tasks
    can be distinguished by a context vector instead of a recurrent task
    encoder, and networks are only reset if ``reset_st`` is given. If no
    task is unsolvable after the first stage, the second stage trains all
    unsolved tasks instead, or all tasks if every task is solved, so that
    SMT always uses the full budget.

    In the first stage, SMT evaluates the policy on all tasks every
    ``eval_freq`` steps and fills vacancies in the training pool with the
    task from the main pool that has the lowest evaluation performance,
    i.e., the mean of

    .. math::

        \sum_{t=1}^H \gamma^{t-1} r_t - \sum_{t=1}^H \log \pi(a_t|s_t)

    over ``n_eval_episodes`` episodes (equation (5) of [1]_). The entropy
    term is omitted for deterministic policies. Before the first evaluation,
    tasks are selected at random. A solved task whose mean undiscounted
    evaluation return is not greater than ``solved_threshold`` returns to
    the main pool and has to be solved again by new training episodes.
    Evaluation steps count towards :math:`B_1`. After each evaluation,
    ``reset_st`` is called. Tasks that are solved after the warm-up start
    in the solved pool.

    The second stage trains the unsolvable tasks round robin until
    :math:`B_2` is used up, even once they are solved.

    Logging

    * ``task_id`` - task selected for training
    * ``task_performance`` - training performance of the selected task
    * ``pool_size_{main,solved,unsolvable}`` - sizes of the task pools in
      the first stage
    * ``worst task`` - task that enters the training pool
    * ``worst performance`` - evaluation performance of that task
    * ``eval_steps`` - number of environment steps used for evaluation so
      far
    * ``eval_performance_{i}`` - evaluation performance of task i
    * ``reset`` - 1 whenever the networks are reset

    References
    ----------
    .. [1] Cho, M., Park, J., Lee, S., Sung, Y. (2024). Hard tasks first:
       multi-task reinforcement learning through task scheduling. In
       Proceedings of the 41st International Conference on Machine Learning,
       Vol. 235. JMLR.org, Article 340, 8556–8577.
       https://icml.cc/virtual/2024/poster/33388
    """
    n_tasks = get_n_tasks(task_set)
    b1 = round(b1_fraction * total_timesteps)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if not 0 < b1_fraction <= 1:
        raise ValueError("b1_fraction must be in (0, 1]")
    if b1 == 0:
        raise ValueError("b1_fraction * total_timesteps must be at least 1")
    if not 1 <= K <= n_tasks:
        raise ValueError(
            f"K must be between 1 and the number of tasks ({n_tasks})"
        )
    if not 0 < kappa <= 1:
        raise ValueError("kappa must be in (0, 1]")
    if n_average < 1:
        raise ValueError("n_average must be positive")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    warn_if_state_not_bound(train_st, replay_buffer is not None)
    if K == n_tasks:
        warnings.warn(
            f"K={K} equals the number of tasks ({n_tasks}). All tasks are "
            "trained at the same time, so there is no task scheduling.",
            stacklevel=2,
        )
    max_eval_steps = prepare_evaluation(
        eval_set,
        n_tasks,
        eval_freq,
        n_eval_episodes,
        max_eval_episode_steps,
        b1,
        seed,
    )

    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    training_steps = np.zeros(n_tasks, dtype=int)
    training_returns = [deque(maxlen=n_average) for _ in range(n_tasks)]
    training_performances = np.full(n_tasks, -np.finfo(float).max)

    global_step = 0
    eval_steps = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)

    def train(task_id, budget):
        nonlocal global_step, result
        result, steps, returns = train_task(
            train_st,
            get_task_env(task_set, task_id),
            task_id,
            episodes_per_task,
            budget,
            global_step,
            seed,
            task_selectables,
            logger,
            replay_buffer,
        )
        global_step += steps
        training_steps[task_id] += steps
        progress.update(steps)
        training_returns[task_id].extend(returns)
        if len(training_returns[task_id]) > 0:
            training_performances[task_id] = np.mean(training_returns[task_id])
        if logger is not None:
            logger.record_stat(
                "task_performance",
                training_performances[task_id],
                step=global_step,
            )
        return steps

    # Stage 1
    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    for i in range(n_warmup):
        if global_step >= b1:
            break
        train(i % n_tasks, b1)

    solved_pool = {
        task_id
        for task_id in range(n_tasks)
        if training_performances[task_id] > solved_threshold
    }
    unsolvable_pool = set()
    main_pool = sorted(set(range(n_tasks)) - solved_pool)
    training_pool = {
        int(task_id)
        for task_id in rng.choice(
            main_pool, size=min(K, len(main_pool)), replace=False
        )
    }
    main_pool = set(main_pool) - training_pool
    task_budgets = np.full(n_tasks, kappa * (b1 - global_step) / K)
    # A task's budget is spent from the moment it enters the training pool.
    steps_in_training_pool = np.zeros(n_tasks, dtype=int)
    # All tasks tie before the first evaluation, so they are picked at random.
    eval_performances = np.full(n_tasks, -np.inf)
    next_eval_step = eval_freq
    warned_deterministic = False

    while global_step < b1:
        updated_training_pool = set(training_pool)
        for task_id in sorted(training_pool):
            steps_in_training_pool[task_id] += train(task_id, b1)

            if logger is not None:
                for name, pool in (
                    ("main", main_pool),
                    ("solved", solved_pool),
                    ("unsolvable", unsolvable_pool),
                ):
                    logger.record_stat(
                        f"pool_size_{name}", len(pool), step=global_step
                    )

            if training_performances[task_id] > solved_threshold:
                solved_pool.add(task_id)
                updated_training_pool.remove(task_id)
            elif steps_in_training_pool[task_id] >= task_budgets[task_id]:
                if training_performances[task_id] < unsolvable_threshold:
                    unsolvable_pool.add(task_id)
                else:
                    main_pool.add(task_id)
                updated_training_pool.remove(task_id)

            if global_step >= b1:
                break

        if global_step >= next_eval_step and global_step + max_eval_steps <= b1:
            policy = get_policy(result)
            if not is_stochastic(policy) and not warned_deterministic:
                warnings.warn(
                    "The evaluated policy is deterministic, so the entropy "
                    "term of the SMT evaluation metric is omitted.",
                    stacklevel=2,
                )
                warned_deterministic = True
            episodes, steps, key = evaluate_tasks(
                policy, eval_set, n_eval_episodes, key, task_selectables
            )
            global_step += steps
            eval_steps += steps
            progress.update(steps)
            next_eval_step = next_evaluation_step(
                next_eval_step, eval_freq, global_step
            )

            eval_performances = mean_task_scores(
                episodes, partial(_evaluation_metric, gamma=gamma)
            )
            eval_returns = mean_task_scores(
                episodes, lambda episode: np.sum(episode.rewards)
            )
            if logger is not None:
                logger.record_stat("eval_steps", eval_steps, step=global_step)
                for i in range(n_tasks):
                    logger.record_stat(
                        f"eval_performance_{i}",
                        eval_performances[i],
                        step=global_step,
                    )
            # Solved tasks that are no longer solved return to the main pool
            # and have to be solved again by new training episodes.
            for task_id in sorted(solved_pool):
                if eval_returns[task_id] <= solved_threshold:
                    solved_pool.remove(task_id)
                    main_pool.add(task_id)
                    training_returns[task_id].clear()
            if reset_st is not None:
                reset_st(seed + global_step)
                if logger is not None:
                    logger.record_stat("reset", 1, step=global_step)

        while len(updated_training_pool) < K and main_pool:
            # Select the task from the main pool with the lowest evaluation
            # performance. Ties, e.g., before the first evaluation, are broken
            # at random.
            candidates = np.array(sorted(main_pool))
            performances = eval_performances[candidates]
            worst = int(
                rng.choice(candidates[performances == performances.min()])
            )
            updated_training_pool.add(worst)
            main_pool.remove(worst)
            task_budgets[worst] = kappa * (b1 - global_step) / K
            steps_in_training_pool[worst] = 0

            if logger is not None:
                logger.record_stat("worst task", worst, step=global_step)
                logger.record_stat(
                    "worst performance",
                    eval_performances[worst],
                    step=global_step,
                )

        training_pool = updated_training_pool
        if not training_pool:  # All tasks are solved or unsolvable
            break

    # Stage 2
    if unsolvable_pool:
        scheduled_tasks = sorted(unsolvable_pool)
    elif main_pool or training_pool:
        # Unlike the authors' code, train all unsolved tasks if no task is
        # unsolvable
        scheduled_tasks = sorted(main_pool | training_pool)
    else:
        warnings.warn(
            "All tasks are solved after the first SMT stage. The second stage "
            "trains all tasks round robin.",
            stacklevel=2,
        )
        scheduled_tasks = list(range(n_tasks))

    i = 0
    while global_step < total_timesteps:
        train(scheduled_tasks[i % len(scheduled_tasks)], total_timesteps)
        i += 1

    progress.close()
    return SchedulerResult(result, training_steps, eval_steps)


def _evaluation_metric(episode: EvalEpisode, gamma: float) -> float:
    """Discounted return plus entropy of an episode (equation (5) of SMT)."""
    discounts = gamma ** np.arange(len(episode.rewards))
    metric = float(np.sum(discounts * episode.rewards))
    if episode.log_probs is not None:
        metric -= float(np.sum(episode.log_probs))
    return metric
