from collections.abc import Callable

import numpy as np
from gymnasium.vector import SyncVectorEnv
from numpy.typing import ArrayLike
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
from ..blox.rollout import RecordSuccess
from ..logging.logger import LoggerBase


def drats_return_gaps(
    returns: ArrayLike, reference_returns: ArrayLike, random_returns: ArrayLike
) -> np.ndarray:
    r"""Normalized return gaps of DRATS.

    Parameters
    ----------
    returns : array, shape (n_tasks,)
        Return estimates :math:`\hat{J}_i`.

    reference_returns : float or array, shape (n_tasks,)
        Reference returns :math:`J^{ref}_i`.

    random_returns : float or array, shape (n_tasks,)
        Returns of a random policy :math:`J^{rand}_i`.

    Returns
    -------
    gaps : array, shape (n_tasks,)
        Return gaps

        .. math::

            g_i = \frac{J^{ref}_i - \hat{J}_i}{J^{ref}_i - J^{rand}_i},

        clipped to [0, 1]. 1 for tasks whose reference return is not greater
        than the return of a random policy.
    """
    returns = np.asarray(returns, dtype=float)
    reference_returns = np.broadcast_to(reference_returns, returns.shape)
    random_returns = np.broadcast_to(random_returns, returns.shape)
    scale = reference_returns - random_returns
    valid = scale > 0.0
    gaps = np.ones_like(returns)
    gaps[valid] = (reference_returns[valid] - returns[valid]) / scale[valid]
    return np.clip(gaps, 0.0, 1.0)


def drats_mirror_ascent(
    probabilities: np.ndarray, gaps: np.ndarray, eta: float, alpha: float
) -> np.ndarray:
    r"""Mirror ascent step of DRATS on the task distribution.

    Parameters
    ----------
    probabilities : array, shape (n_tasks,)
        Task distribution :math:`q_t`. All entries must be positive.

    gaps : array, shape (n_tasks,)
        Return gaps :math:`g_i`, see :func:`drats_return_gaps`.

    eta : float
        Inverse temperature :math:`\eta`.

    alpha : float
        Step size :math:`\alpha \in (0, \eta]`.

    Returns
    -------
    probabilities : array, shape (n_tasks,)
        Task distribution :math:`q_{t+1}`.

    Notes
    -----
    The step [1]_ (Algorithm 2) ascends
    :math:`L(q) = \mathbb{E}_{i \sim q}[g_i] - \frac{1}{\eta} KL(q \| p_0)`
    with the uniform distribution :math:`p_0` and the negative entropy as
    mirror map:

    .. math::

        h_i = g_i - \frac{1}{\eta} (\log(k q_{t,i}) + 1), \quad
        q_{t+1,i} = \frac{q_{t,i} \exp(\alpha h_i)}
        {\sum_j q_{t,j} \exp(\alpha h_j)}.

    :math:`\alpha = \eta` gives the best response
    :math:`\operatorname{softmax}(\eta g)`.

    References
    ----------
    .. [1] Corrado, N. E., Huang, W., Hanna, J. P. (2026). Distributionally
       Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling.
       https://arxiv.org/abs/2605.14350
    """
    log_probabilities = np.log(probabilities)
    h = gaps - (np.log(len(probabilities)) + log_probabilities + 1.0) / eta
    logits = log_probabilities + alpha * h
    unnormalized = np.exp(logits - np.max(logits))
    return unnormalized / np.sum(unnormalized)


def project_min_probability(
    probabilities: np.ndarray, min_probability: float
) -> np.ndarray:
    r"""KL projection of a distribution onto a minimum probability.

    Parameters
    ----------
    probabilities : array, shape (n,)
        Distribution :math:`p`.

    min_probability : float
        Minimum probability :math:`\epsilon` of each entry. Must not be
        greater than 1 / n.

    Returns
    -------
    probabilities : array, shape (n,)
        The distribution :math:`q` with :math:`q_i \geq \epsilon` that
        minimizes :math:`KL(q \| p)`.

    Notes
    -----
    The solution is :math:`q_i = \max(\epsilon, c p_i)`, where :math:`c` is
    chosen so that :math:`q` sums up to 1. The entries clipped to
    :math:`\epsilon` are found by fixing them one round after another, at
    most n rounds.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.zeros(len(probabilities), dtype=bool)
    while True:
        scale = (1.0 - min_probability * np.sum(clipped)) / np.sum(
            probabilities[~clipped]
        )
        new_clipped = clipped | (scale * probabilities < min_probability)
        if np.array_equal(new_clipped, clipped):
            break
        clipped = new_clipped
    return np.where(clipped, min_probability, scale * probabilities)


def train_drats(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    *,
    total_timesteps: int,
    episodes_per_update: int,
    eta: float = 8.0,
    alpha: float | None = None,
    min_probability: float | None = None,
    reference_returns: ArrayLike | None = None,
    success_threshold: float | None = None,
    random_returns: ArrayLike = 0.0,
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    replay_buffer: MultiTaskReplayBuffer | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    r"""Distributionally Robust Adaptive Task Sampling (DRATS).

    A task scheduler for multi-task reinforcement learning [1]_. It samples
    tasks from a distribution that prefers the tasks furthest from a
    reference return, and moves it toward them after each batch of
    episodes.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    total_timesteps : int
        Total number of environment steps.

    episodes_per_update : int
        Number of episodes in a batch, after which the task distribution is
        updated. The paper uses 200 episodes on MetaWorld-MT10 and 1000 on
        MT50.

    eta : float, optional
        Inverse temperature :math:`\eta`. Larger values concentrate the task
        distribution on the tasks with the largest return gaps. The default
        is the paper's value for MetaWorld-MT10, the best one in its
        ablation.

    alpha : float, optional
        Step size :math:`\alpha` of the mirror ascent, in (0, eta]. Defaults
        to 0.5 * eta, as in the paper's experiments.

    min_probability : float, optional
        Minimum probability :math:`\epsilon` of each task, in (0, 1 /
        n_tasks]. Defaults to 1 / (4 n_tasks), as in the paper.

    reference_returns : float or array, shape (n_tasks,), optional
        Reference returns :math:`J^{ref}_i`, e.g., the maximum return of each
        task. If None, the maximum return observed in a single episode is
        used. With ``success_threshold``, these are the initial reference
        returns.

    success_threshold : float, optional
        If given, the reference return of a task becomes its maximum
        observed return once its success rate in a batch exceeds this
        threshold, in (0, 1]. The paper uses 0.5. Needs
        ``reference_returns`` and environments that report
        ``info["success"]``.

    random_returns : float or array, shape (n_tasks,), optional
        Returns of a random policy :math:`J^{rand}_i`.

    episodes_per_task : int, optional
        Number of episodes per task selection.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`. The paper has
        no warm-up.

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
        If ``total_timesteps``, ``episodes_per_update``,
        ``episodes_per_task`` or ``eta`` is not positive, ``alpha``,
        ``min_probability`` or ``success_threshold`` is out of range,
        ``success_threshold`` is given without ``reference_returns``, or a
        reference return is not greater than the return of a random policy.

    Warns
    -----
    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    DRATS follows Algorithm 1 of [1]_. The task distribution :math:`q`
    starts uniform. Each selection samples a task from :math:`q` and trains
    on it for ``episodes_per_task`` episodes. Once the batch holds
    ``episodes_per_update`` episodes, DRATS estimates the return
    :math:`\hat{J}_i` of each task, computes the return gaps
    :math:`g_i` (see :func:`drats_return_gaps`), takes a mirror ascent step
    (see :func:`drats_mirror_ascent`), and projects :math:`q` onto
    :math:`q_i \geq \epsilon` (see :func:`project_min_probability`).
    Episodes of the warm-up count toward the first batch.

    The paper leaves some details open. Here,

    * :math:`\hat{J}_i` is the mean return of the task's completed episodes
      in the batch, and 0 for tasks without episodes in the batch, as in
      Appendix F.2 of the paper;
    * the gaps are clipped to [0, 1], and they are 1 while a task's
      reference return is not greater than ``random_returns``, e.g., before
      a return was observed;
    * the paper does not say how :math:`J^{rand}` is obtained, so it is an
      argument;
    * with ``success_threshold``, the success rate is measured over the
      task's episodes in the batch; the reference return switches once and
      then follows the maximum observed return;
    * :math:`\alpha` is in (0, eta] as in Algorithm 2 of the paper, while
      its Algorithm 1 states [0, 1];
    * the projection onto the minimum probability is the exact KL
      projection.

    The paper samples on-policy batches. With off-policy backbones, the
    replay buffer is sampled as configured, not weighted by :math:`q`.

    Logging

    * ``task_id`` - task selected for training
    * ``task_probability_{i}`` - probability of task i
    * ``task_return_estimate_{i}`` - return estimate of task i in the batch
    * ``task_gap_{i}`` - return gap of task i
    * ``reference_return_{i}`` - reference return of task i

    References
    ----------
    .. [1] Corrado, N. E., Huang, W., Hanna, J. P. (2026). Distributionally
       Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling.
       https://arxiv.org/abs/2605.14350
    """
    n_tasks = get_n_tasks(task_set)
    if alpha is None:
        alpha = 0.5 * eta
    if min_probability is None:
        min_probability = 1.0 / (4.0 * n_tasks)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be positive")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if not 0.0 < alpha <= eta:
        raise ValueError("alpha must be in (0, eta]")
    if not 0.0 < min_probability <= 1.0 / n_tasks:
        raise ValueError("min_probability must be in (0, 1 / n_tasks]")
    if success_threshold is not None:
        if reference_returns is None:
            raise ValueError("success_threshold needs reference_returns")
        if not 0.0 < success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
    random_returns = np.broadcast_to(
        np.asarray(random_returns, dtype=float), (n_tasks,)
    )
    if reference_returns is not None:
        initial_references = np.broadcast_to(
            np.asarray(reference_returns, dtype=float), (n_tasks,)
        )
        if np.any(initial_references <= random_returns):
            raise ValueError(
                "reference_returns must be greater than random_returns"
            )
    warn_if_state_not_bound(train_st, replay_buffer is not None)

    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_tasks, 1.0 / n_tasks)
    max_returns = np.full(n_tasks, -np.inf)
    switched = np.zeros(n_tasks, dtype=bool)
    batch_returns = [[] for _ in range(n_tasks)]
    batch_successes = [[] for _ in range(n_tasks)]
    n_batch_episodes = 0
    training_steps = np.zeros(n_tasks, dtype=int)
    if logger is not None:
        for i in range(n_tasks):
            logger.record_stat(
                f"task_probability_{i}", probabilities[i], step=0
            )

    global_step = 0
    n_selections = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    while global_step < total_timesteps:
        if n_selections < n_warmup:
            task_id = n_selections % n_tasks
        else:
            task_id = int(rng.choice(n_tasks, p=probabilities))
        n_selections += 1

        env = get_task_env(task_set, task_id)
        if success_threshold is not None:
            env = RecordSuccess(env)
        result, steps, returns = train_task(
            train_st,
            env,
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

        batch_returns[task_id].extend(returns)
        if success_threshold is not None:
            batch_successes[task_id].extend(env.successes())
        if len(returns) > 0:
            max_returns[task_id] = max(max_returns[task_id], np.max(returns))
        n_batch_episodes += len(returns)
        if n_batch_episodes < episodes_per_update:
            continue

        return_estimates = np.array(
            [np.mean(r) if len(r) > 0 else 0.0 for r in batch_returns]
        )
        if reference_returns is None:
            references = max_returns
        else:
            if success_threshold is not None:
                switched |= np.array(
                    [
                        len(s) > 0 and np.mean(s) > success_threshold
                        for s in batch_successes
                    ]
                )
            references = np.where(switched, max_returns, initial_references)
        gaps = drats_return_gaps(return_estimates, references, random_returns)
        probabilities = project_min_probability(
            drats_mirror_ascent(probabilities, gaps, eta, alpha),
            min_probability,
        )
        batch_returns = [[] for _ in range(n_tasks)]
        batch_successes = [[] for _ in range(n_tasks)]
        n_batch_episodes = 0

        if logger is not None:
            for i in range(n_tasks):
                for name, value in (
                    ("task_probability", probabilities[i]),
                    ("task_return_estimate", return_estimates[i]),
                    ("task_gap", gaps[i]),
                    ("reference_return", references[i]),
                ):
                    logger.record_stat(f"{name}_{i}", value, step=global_step)

    progress.close()
    return SchedulerResult(result, training_steps, 0)
