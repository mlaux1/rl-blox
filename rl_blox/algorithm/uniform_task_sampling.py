from collections.abc import Callable

import jax
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet
from ..logging.logger import LoggerBase


def train_uts(
    task_set: DiscreteTaskSet,
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
    algorithm is trained on for one episode.

    Parameters
    ----------

    task_set : DiscreteTaskSet
        The set of tasks available for training.

    train_st : Callable
        The training step of the backbone algorithm.

    total_timesteps : int
        The number of total environment steps to train for.

    episodes_per_task : int
        The number of episodes to train the policy on the scheduled task for.

    seed : int
        The random seed.

    exploring_starts : int
        The number of random exploration steps to be performed at the beginning
        of training.

    progress_par : bool
        Flag to enable/disable the tqdm progress bar.

    logger : Logger
        Experiment logger.

    """
    global_step = 0
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    key = jax.random.key(seed)

    n_tasks = len(task_set)

    while global_step < total_timesteps:
        key, skey = jax.random.split(key)
        task_id = jax.random.choice(skey, n_tasks)
        env = task_set.get_task(task_id)
        st_result = train_st(
            env,
            seed=seed + global_step,
            total_timesteps=total_timesteps,
            total_episodes=episodes_per_task,
            learning_starts=exploring_starts,
            progress_bar=False,
            logger=logger,
            global_step=global_step,
        )

        _, _, _, _, _, _, _, new_global_step = st_result

        it_steps = new_global_step - global_step

        progress.update(it_steps)
        global_step = new_global_step

    return st_result
