from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.gae import compute_gae
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
from ..blox.rollout import RecordTransitions, Transitions, pad_transitions
from ..logging.logger import LoggerBase


def plr_score(
    transitions: Transitions,
    value_function: Callable[[jnp.ndarray], jnp.ndarray],
    gamma: float,
    gae_lambda: float,
) -> float:
    """Score transitions by the magnitude of their advantage estimates.

    Parameters
    ----------
    transitions : Transitions
        Transitions of one or more episodes. The last transition ends an
        episode even if it is not terminated or truncated.

    value_function : callable
        Maps a batch of observations to their state values.

    gamma : float
        Discount factor.

    gae_lambda : float
        Smoothing factor of the generalized advantage estimation.

    Returns
    -------
    score : float
        Mean absolute generalized advantage estimate over all transitions.
    """
    padded, mask = pad_transitions(transitions)
    n_steps = int(np.sum(mask))
    dones = padded.termination | padded.truncation | ~mask
    dones[n_steps - 1] = True

    values = value_function(jnp.asarray(padded.observation))
    next_values = value_function(jnp.asarray(padded.next_observation))
    advantages, _ = compute_gae(
        jnp.asarray(padded.reward),
        values,
        next_values,
        jnp.asarray(padded.termination),
        gamma,
        gae_lambda,
        dones=jnp.asarray(dones),
    )
    return float(jnp.mean(jnp.abs(advantages[:n_steps])))


def plr_replay_distribution(
    scores: np.ndarray,
    last_selections: np.ndarray,
    n_selections: int,
    seen: np.ndarray,
    beta: float,
    rho: float,
) -> np.ndarray:
    r"""Replay distribution of prioritized level replay.

    Parameters
    ----------
    scores : array, shape (n_tasks,)
        Score of each task.

    last_selections : array, shape (n_tasks,)
        Selection count at which each task was selected last.

    n_selections : int
        Number of selections so far. Must not be smaller than the entries of
        ``last_selections`` that belong to seen tasks.

    seen : array, shape (n_tasks,)
        Flags indicating which tasks were selected before.

    beta : float
        Temperature of the score-prioritized distribution.

    rho : float
        Weight of the staleness-prioritized distribution.

    Returns
    -------
    probabilities : array, shape (n_tasks,)
        Probability of each task. Tasks that were not seen have probability 0.

    Notes
    -----
    The replay distribution [1]_ over seen tasks is

    .. math::

        P(i) = (1 - \rho) \frac{h(S_i)^{1 / \beta}}{\sum_j h(S_j)^{1 / \beta}}
        + \rho \frac{c - C_i}{\sum_j (c - C_j)},

    where :math:`h(S_i) = 1 / \operatorname{rank}(S_i)` is the reciprocal
    rank of the score :math:`S_i` in descending order, :math:`c` is the
    number of selections so far, and :math:`C_i` is the number of selections
    up to and including the last selection of task :math:`i`. Hence, the task
    selected last has staleness 0, as in the official implementation. If it
    is the only seen task, it has probability 1.

    References
    ----------
    .. [1] Jiang, M., Grefenstette, E., Rocktäschel, T. (2021). Prioritized
       Level Replay. In Proceedings of the 38th International Conference on
       Machine Learning, PMLR 139. https://arxiv.org/abs/2010.03934
    """
    seen_ids = np.flatnonzero(seen)

    ranks = np.empty(len(seen_ids))
    ranks[np.argsort(-scores[seen_ids], kind="stable")] = np.arange(
        1, len(seen_ids) + 1
    )
    score_probabilities = (1.0 / ranks) ** (1.0 / beta)
    score_probabilities /= np.sum(score_probabilities)

    staleness = n_selections - last_selections[seen_ids]
    if np.any(staleness > 0):
        staleness_probabilities = staleness / np.sum(staleness)
    else:  # only the task selected last was seen
        staleness_probabilities = score_probabilities

    probabilities = np.zeros(len(scores))
    probabilities[seen_ids] = (
        1.0 - rho
    ) * score_probabilities + rho * staleness_probabilities
    return probabilities


def train_plr(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    value_function: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    total_timesteps: int,
    episodes_per_task: int = 1,
    warmup_episodes: int = 0,
    beta: float = 0.1,
    rho: float = 0.1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    task_selectables: list[TaskSelectionMixin] | None = None,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> SchedulerResult:
    r"""Prioritized level replay (PLR).

    A task scheduler for multi-task reinforcement learning [1]_. It prefers
    tasks whose transitions had large advantage estimates when they were
    trained last, i.e., tasks with a high learning potential.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`~rl_blox.blox.multitask.train_task`.

    value_function : callable
        Maps a batch of observations to the state values of the backbone,
        in the units of the rewards, e.g.,
        ``partial(greedy_value, q_net)`` with
        :func:`~rl_blox.blox.state_value.greedy_value` or
        :func:`~rl_blox.blox.state_value.policy_value`. For MR.Q, use
        ``partial(mrq_value, q, policy_with_encoder, replay_buffer)`` with
        :func:`~rl_blox.algorithm.mrq.mrq_value`. It must use the networks
        that ``train_st`` updates.

    total_timesteps : int
        Total number of environment steps.

    episodes_per_task : int, optional
        Number of episodes per task selection.

    warmup_episodes : int, optional
        Minimum number of episodes per task in a round-robin warm-up, see
        :func:`~rl_blox.blox.multitask.n_warmup_selections`.

    beta : float, optional
        Temperature of the score-prioritized distribution. Must be positive.

    rho : float, optional
        Weight of the staleness-prioritized distribution. Must be in [0, 1].

    gamma : float, optional
        Discount factor of the advantage estimates. Should be the discount
        factor of the backbone.

    gae_lambda : float, optional
        Smoothing factor of the advantage estimates. 0 scores tasks by their
        absolute one-step TD errors.

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
        If ``beta`` is not positive or ``rho`` is not in [0, 1].

    Warns
    -----
    UserWarning
        If the backbone's state is not bound to ``train_st``.

    Notes
    -----
    While some tasks were not seen, an unseen task is selected uniformly
    with probability :math:`1 - n_{seen} / n_{tasks}`. Otherwise, a seen task
    is sampled from the replay distribution, see
    :func:`plr_replay_distribution`. Each selection increases the selection
    count :math:`c`. After training on task :math:`i`, its score :math:`S_i`
    is set to the mean absolute generalized advantage estimate of the
    transitions, see :func:`plr_score`, and its timestamp :math:`C_i` to
    :math:`c`.

    The paper counts episodes instead of selections, which is the same if
    each selection has the same number of episodes. The official
    implementation [2]_ replays only after all tasks were seen, which
    ``warmup_episodes=1`` reproduces, except that the first round visits the
    tasks in order. It scores trajectories with the values that were
    predicted while collecting them. Here, the value function is evaluated
    after ``train_st`` trained on the transitions, because off-policy
    backbones update it during the episode.

    Logging

    * ``task_id`` - task selected for training
    * ``task_score_{i}`` - score of task i after it was trained

    References
    ----------
    .. [1] Jiang, M., Grefenstette, E., Rocktäschel, T. (2021). Prioritized
       Level Replay. In Proceedings of the 38th International Conference on
       Machine Learning, PMLR 139. https://arxiv.org/abs/2010.03934

    .. [2] https://github.com/facebookresearch/level-replay
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    warn_if_state_not_bound(train_st)

    n_tasks = get_n_tasks(task_set)
    n_warmup = n_warmup_selections(n_tasks, warmup_episodes, episodes_per_task)
    rng = np.random.default_rng(seed)

    scores = np.zeros(n_tasks)
    last_selections = np.zeros(n_tasks, dtype=int)
    seen = np.zeros(n_tasks, dtype=bool)
    training_steps = np.zeros(n_tasks, dtype=int)

    global_step = 0
    n_selections = 0
    result = None
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    while global_step < total_timesteps:
        if n_selections < n_warmup:
            task_id = n_selections % n_tasks
        elif rng.random() < np.mean(seen):
            probabilities = plr_replay_distribution(
                scores, last_selections, n_selections, seen, beta, rho
            )
            task_id = int(rng.choice(n_tasks, p=probabilities))
        else:
            task_id = int(rng.choice(np.flatnonzero(~seen)))
        n_selections += 1

        env = RecordTransitions(get_task_env(task_set, task_id))
        result, steps, _ = train_task(
            train_st,
            env,
            task_id,
            episodes_per_task,
            total_timesteps,
            global_step,
            seed,
            task_selectables,
            logger,
        )
        global_step += steps
        training_steps[task_id] += steps
        progress.update(steps)

        scores[task_id] = plr_score(
            env.transitions(), value_function, gamma, gae_lambda
        )
        last_selections[task_id] = n_selections
        seen[task_id] = True
        if logger is not None:
            logger.record_stat(
                f"task_score_{task_id}", scores[task_id], step=global_step
            )

    progress.close()
    return SchedulerResult(result, training_steps, 0)
