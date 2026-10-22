from collections.abc import Callable

import jax
from gymnasium.vector import SyncVectorEnv
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet, warn_if_state_not_bound
from ..logging.logger import LoggerBase


def train_uts(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    seed: int = 1,
    exploring_starts: int = 1_000,
    progress_bar: bool = True,
    logger: LoggerBase = None,
) -> tuple:
    """Uniform task sampling.

    A basic task scheduling method for multi-task reinforcement learning. Given
    a set of tasks, it uniformly samples a task on which a given backbone
    algorithm is trained for ``episodes_per_task`` episodes.

    Parameters
    ----------
    task_set : DiscreteTaskSet | SyncVectorEnv
        The set of tasks available for training.

    train_st : Callable
        The training function of the backbone algorithm, e.g., ``train_sac``
        with the backbone's state bound via ``functools.partial``. It is called
        once per scheduled task.

    total_timesteps : int
        The number of total environment steps to train for.

    episodes_per_task : int
        The number of episodes to train the policy on the scheduled task for.

    seed : int
        The random seed.

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
        The result of the last call to ``train_st``.

    Warns
    -----
    UserWarning
        If the backbone's replay buffer, target networks, or entropy control
        are not bound to ``train_st``.

    Notes
    -----
    All tasks are trained with the same backbone state, i.e., the policy,
    critics, optimizers, replay buffer, and entropy coefficient that are bound
    to ``train_st``. The backbone's state must be bound explicitly, e.g.,
    ``partial(train_sac, replay_buffer=..., q_target=..., ...)``. Otherwise,
    the backbone creates a new replay buffer and new target networks for each
    scheduled task.

    Logging

    * ``task_id`` - task scheduled for training
    """
    assert total_timesteps > 0
    warn_if_state_not_bound(train_st)

    global_step = 0
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    key = jax.random.key(seed)

    if isinstance(task_set, SyncVectorEnv):
        n_tasks = task_set.num_envs
    else:
        n_tasks = len(task_set)

    while global_step < total_timesteps:
        key, skey = jax.random.split(key)
        task_id = int(jax.random.choice(skey, n_tasks))
        if logger is not None:
            logger.record_stat("task_id", task_id, step=global_step)

        if isinstance(task_set, SyncVectorEnv):
            env = task_set.envs[task_id]
        else:
            env = task_set.get_task(task_id)

        st_result = train_st(
            env,
            seed=seed + global_step,
            total_timesteps=total_timesteps,
            total_episodes=episodes_per_task,
            learning_starts=exploring_starts,
            progress_bar=progress_bar,
            logger=logger,
            global_step=global_step,
            bar=progress,
        )

        global_step = st_result.global_step

    progress.close()

    return st_result
