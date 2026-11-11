import copy
import warnings
from collections import deque
from collections.abc import Callable
from operator import attrgetter

import gymnasium as gym
import jax
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import (
    DiscreteTaskSet,
    TaskSelectionMixin,
    act,
    get_n_tasks,
    get_task_env,
    max_episode_steps,
)
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase


def train_smt(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    b1: int = 17_000_000,
    b2: int = 3_000_000,
    solved_threshold: float = 4000.0,
    unsolvable_threshold: float = 1000.0,
    scheduling_interval: int = 1,
    kappa: float = 0.8,
    K: int = 3,
    n_average: int = 5,
    eval_freq: int = 500_000,
    n_eval_episodes: int = 5,
    max_eval_episode_steps: int | None = None,
    gamma: float = 0.99,
    get_policy: Callable[..., nnx.Module] = attrgetter("policy"),
    reset_st: Callable[[int], None] | None = None,
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
    task_set : DiscreteTaskSet or SyncVectorEnv
        The multi-task environment definition.

    eval_set : DiscreteTaskSet or SyncVectorEnv
        The set of tasks used for evaluation. Must contain the same number of
        tasks as ``task_set`` in the same order. Must not share environment
        instances with ``task_set``.

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
        is greater than this value. In the first stage, a solved task returns
        to the main pool if its undiscounted evaluation return is not greater
        than this value. Corresponds to :math:`M` in the paper [1]_.
        The default is set for Meta-World with episodes of 500 steps.

    unsolvable_threshold : float
        When a task has used up its budget, it is unsolvable if its training
        performance is less than this value. Otherwise, it returns to the main
        pool. Corresponds to :math:`m` in the paper [1]_. The default is set
        for Meta-World with episodes of 500 steps.

    scheduling_interval : int
        Number of episodes after which the task scheduling is performed.

    kappa : float
        Share of the remaining budget of the first stage that the tasks in
        the training pool may use. When a task enters the training pool, its
        budget is set to :math:`\kappa (B_1 - B_{used}) / K`, where
        :math:`B_{used}` is the number of steps used so far (Appendix G.1 of
        [1]_).

    K : int
        Number of tasks to train in each iteration.

    n_average : int
        Number of recent training episodes of a task whose mean return is the
        task's training performance. It is used to check if a task is solved
        or unsolvable.

    eval_freq : int
        Number of environment steps between evaluations in the first stage.
        Corresponds to :math:`T_{eval}` in the paper [1]_. Evaluations are
        only performed between scheduling iterations, so the actual interval
        can be slightly larger.

    n_eval_episodes : int
        Number of evaluation episodes per task in each evaluation.
        Corresponds to :math:`n_{eval}` in the paper [1]_.

    max_eval_episode_steps : int, optional
        Maximum length of an evaluation episode. Used to skip evaluations that
        would exceed ``b1``. If not set, it is inferred from the ``TimeLimit``
        wrapper or spec of the evaluation environments. If it cannot be
        inferred, evaluations may exceed ``b1``.

    gamma : float
        Discount factor of the evaluation return. Should be the same as the
        discount factor of the backbone.

    get_policy : callable
        Maps the result of ``train_st`` to the policy that is evaluated, an
        ``nnx.Module`` that maps an observation to an action. Policies with a
        method ``sample_and_log_prob(observation, key)`` are sampled, all
        others are treated as deterministic. The default returns the field
        ``policy`` (SAC, DDPG, TD3).

    reset_st : callable, optional
        Reinitializes the backbone's networks after each evaluation in the
        first stage. It is called with a seed and must reinitialize the state
        bound to ``train_st`` in place, except for the replay buffer. If not
        set, the networks are not reset.

    learning_starts : int
        Number of steps at the start of SMT in which the backbone only takes
        random actions and does not train, counted over all tasks.

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

    eval_steps : int
        Number of environment steps used for evaluation. Together with
        ``training_steps``, they sum up to :math:`B_1 + B_2`.

    Raises
    ------
    ValueError
        If ``K`` is not between 1 and the number of tasks, ``kappa`` is not in
        (0, 1], ``b1`` is not positive, ``b2`` is negative, ``eval_set`` does
        not have the same number of tasks as ``task_set``, or ``eval_freq``
        or ``n_eval_episodes`` is not positive.

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

    Notes
    -----
    In contrast to [1]_, any RL algorithm can be used as a backbone, tasks
    can be distinguished by a context vector instead of a recurrent task
    encoder, and networks are only reset if ``reset_st`` is given. If no
    task is unsolvable after the first stage, the second stage trains all
    unsolved tasks instead, or all tasks if every task is solved, so that
    SMT always uses the full budget :math:`B_1 + B_2`.

    The backbone's target networks and entropy control must be bound to
    ``train_st``, e.g., ``partial(train_sac, q_target=..., ...)``, so that
    all tasks share them.

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
    ``reset_st`` is called.

    The second stage trains the unsolvable tasks round robin until
    :math:`B_2` is used up, even once they are solved.

    Logging

    * ``eval_steps`` - number of environment steps used for evaluation
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
    rng = np.random.default_rng(seed)
    n_tasks = get_n_tasks(task_set)

    if not 1 <= K <= n_tasks:
        raise ValueError(
            f"K must be between 1 and the number of tasks ({n_tasks})"
        )
    if not 0 < kappa <= 1:
        raise ValueError("kappa must be in (0, 1]")
    if b1 <= 0:
        raise ValueError("b1 must be positive")
    if b2 < 0:
        raise ValueError("b2 must not be negative")
    n_eval_tasks = get_n_tasks(eval_set)
    if n_eval_tasks != n_tasks:
        raise ValueError(
            f"eval_set has {n_eval_tasks} tasks, task_set has {n_tasks}"
        )
    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes must be positive")

    if K == n_tasks:
        warnings.warn(
            f"K={K} equals the number of tasks ({n_tasks}). All tasks are "
            "trained at the same time, so there is no task scheduling.",
            stacklevel=2,
        )

    b_total = b1 + b2
    global_step = 0
    training_steps = np.zeros(n_tasks, dtype=int)
    training_performances = [deque(maxlen=n_average) for _ in range(n_tasks)]
    avg_training_performances = np.full(n_tasks, -np.finfo(float).max)
    progress = tqdm(total=b_total, disable=not progress_bar)

    # Seed evaluation environments for reproducible evaluations
    for i in range(n_tasks):
        get_task_env(eval_set, i).reset(seed=seed + i)
    max_eval_steps = max_evaluation_steps(
        eval_set, n_eval_episodes, max_eval_episode_steps
    )
    if max_eval_steps is None:
        if eval_freq < b1:
            warnings.warn(
                "Could not infer the episode length of all evaluation "
                "environments, evaluations may exceed b1. Set "
                "max_eval_episode_steps to avoid this.",
                stacklevel=2,
            )
        max_eval_steps = 0

    global_step, result_st, unsolvable_pool, eval_steps = smt_stage1(
        task_set,
        eval_set,
        eval_freq,
        max_eval_steps,
        n_eval_episodes,
        gamma,
        get_policy,
        reset_st,
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

    return result_st, training_steps, avg_training_performances, eval_steps


def smt_stage1(
    task_set,
    eval_set,
    eval_freq,
    max_eval_steps,
    n_eval_episodes,
    gamma,
    get_policy,
    reset_st,
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
    n_tasks = get_n_tasks(task_set)
    training_pool = set(rng.choice(n_tasks, size=K, replace=False))
    main_pool = set(range(n_tasks)) - training_pool
    solved_pool = set()
    unsolvable_pool = set()
    task_budgets = np.full(n_tasks, kappa * b1 / K)
    # A task's budget is spent from the moment it enters the training pool.
    steps_in_training_pool = np.zeros(n_tasks, dtype=int)
    # All tasks tie before the first evaluation, so they are picked at random.
    eval_performances = np.full(n_tasks, -np.inf)
    next_eval_step = eval_freq
    key = jax.random.key(seed)
    warned_deterministic = False
    total_eval_steps = 0

    while global_step < b1:
        updated_training_pool = copy.deepcopy(training_pool)
        for task_id in training_pool:
            env = get_task_env(task_set, task_id)
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

        if global_step >= next_eval_step and global_step + max_eval_steps <= b1:
            policy = get_policy(result_st)
            if not is_stochastic(policy) and not warned_deterministic:
                warnings.warn(
                    "The evaluated policy is deterministic, so the entropy "
                    "term of the SMT evaluation metric is omitted.",
                    stacklevel=3,
                )
                warned_deterministic = True
            eval_performances, eval_returns, eval_steps, key = evaluate_tasks(
                policy,
                eval_set,
                n_eval_episodes,
                gamma,
                key,
                task_selectables,
            )
            global_step += eval_steps
            total_eval_steps += eval_steps
            progress.update(eval_steps)
            next_eval_step = global_step + eval_freq
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
                    training_performances[task_id].clear()
            if reset_st is not None:
                reset_st(seed + global_step)
                if logger is not None:
                    logger.record_stat("reset", 1, step=global_step)

        early_stop_stage = False
        while len(updated_training_pool) < K:
            if len(main_pool) == 0:
                # All tasks outside the training pool are solved or unsolvable.
                if len(updated_training_pool) == 0:  # No tasks left to train.
                    early_stop_stage = True
                # else: Continue until training pool is solved or unsolvable.
                break

            # Select task with the lowest evaluation performance (from main
            # pool). Ties, e.g., before the first evaluation, are broken at
            # random.
            main_pool_indices = np.array(sorted(main_pool))
            performances = eval_performances[main_pool_indices]
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
                    eval_performances[worst],
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

    return global_step, result_st, unsolvable_pool, total_eval_steps


def smt_stage2(
    task_set,
    train_st,
    replay_buffer,
    task_selectables,
    unsolvable_pool,
    training_steps,
    training_performances,
    avg_training_performances,
    global_step,
    scheduling_interval,
    b_total,
    learning_starts,
    logger,
    seed,
    progress,
    result_st,
):
    """SMT trains the unsolvable tasks round robin in stage 2.

    They are trained until the budget is used up, even once they are solved.
    Without unsolvable tasks, all tasks are trained round robin.
    """
    if len(unsolvable_pool) > 0:
        scheduled_tasks = sorted(unsolvable_pool)
    else:
        scheduled_tasks = range(len(training_steps))
    while global_step < b_total:
        for task_id in scheduled_tasks:
            env = get_task_env(task_set, task_id)
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

            if logger is not None:
                logger.record_stat("task_id", task_id, step=global_step)
                logger.record_stat(
                    "task_performance",
                    avg_training_performances[task_id],
                    step=global_step,
                )

            if global_step >= b_total:
                break

    return result_st


def is_stochastic(policy: nnx.Module) -> bool:
    """Check if a policy can sample actions with their log probabilities.

    Parameters
    ----------
    policy : nnx.Module
        Policy.

    Returns
    -------
    stochastic : bool
        True if the policy has a method ``sample_and_log_prob``.
    """
    return hasattr(policy, "sample_and_log_prob")


def max_evaluation_steps(
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    n_eval_episodes: int,
    max_eval_episode_steps: int | None,
) -> int | None:
    """Upper bound on the number of steps of an evaluation of all tasks.

    Parameters
    ----------
    eval_set : DiscreteTaskSet or SyncVectorEnv
        The set of tasks used for evaluation.

    n_eval_episodes : int
        Number of evaluation episodes per task.

    max_eval_episode_steps : int or None
        Maximum length of an evaluation episode. Inferred from the evaluation
        environments if None.

    Returns
    -------
    max_steps : int or None
        Upper bound on the number of steps. None if the episode length of an
        evaluation environment cannot be inferred.
    """
    n_tasks = get_n_tasks(eval_set)

    if max_eval_episode_steps is not None:
        episode_limits = [max_eval_episode_steps] * n_tasks
    else:
        episode_limits = [
            max_episode_steps(get_task_env(eval_set, i)) for i in range(n_tasks)
        ]
    if None in episode_limits:
        return None
    return n_eval_episodes * sum(episode_limits)


def evaluate_tasks(
    policy: nnx.Module,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    n_eval_episodes: int,
    gamma: float,
    key: jax.Array,
    task_selectables: list[TaskSelectionMixin] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, jax.Array]:
    r"""Evaluate a policy on all tasks with equation (5) of the SMT paper.

    Parameters
    ----------
    policy : nnx.Module
        Policy that maps an observation to an action. If it is stochastic
        (see :func:`is_stochastic`), actions are sampled from it.

    eval_set : DiscreteTaskSet or SyncVectorEnv
        The set of tasks used for evaluation.

    n_eval_episodes : int
        Number of evaluation episodes per task.

    gamma : float
        Discount factor of the evaluation return.

    key : jax.Array
        Pseudo random number generator key for sampling actions.

    task_selectables : list, optional
        Objects that are informed about the evaluated task.

    Returns
    -------
    performances : np.ndarray
        Mean evaluation metric of each task.

    returns : np.ndarray
        Mean undiscounted return of each task.

    n_steps : int
        Number of environment steps used for the evaluation.

    key : jax.Array
        Updated pseudo random number generator key.

    Notes
    -----
    The evaluation metric of an episode is

    .. math::

        \sum_{t=1}^H \gamma^{t-1} r_t - \sum_{t=1}^H \log \pi(a_t|s_t).

    For deterministic policies, the entropy term is omitted.
    """
    n_tasks = get_n_tasks(eval_set)
    stochastic = is_stochastic(policy)

    performances = np.zeros(n_tasks)
    returns = np.zeros(n_tasks)
    n_steps = 0
    for task_id in range(n_tasks):
        if task_selectables is not None:
            for ts in task_selectables:
                ts.select_task(task_id)
        env = get_task_env(eval_set, task_id)

        scores = []
        episode_returns = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            discounted_return = 0.0
            entropy = 0.0
            discount = 1.0
            done = False
            while not done:
                if stochastic:
                    key, sampling_key = jax.random.split(key)
                    action, log_prob = policy.sample_and_log_prob(
                        obs, sampling_key
                    )
                    entropy -= float(log_prob)
                else:
                    action = act(policy, obs)
                obs, reward, terminated, truncated, _ = env.step(
                    np.asarray(action)
                )
                episode_return += float(reward)
                discounted_return += discount * float(reward)
                discount *= gamma
                n_steps += 1
                done = terminated or truncated
            scores.append(discounted_return + entropy)
            episode_returns.append(episode_return)

        performances[task_id] = np.mean(scores)
        returns[task_id] = np.mean(episode_returns)

    return performances, returns, n_steps, key
