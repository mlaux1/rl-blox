from collections.abc import Callable

import jax
import numpy as np
from gymnasium.vector import VectorEnv
from scipy.special import softmax
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet
from ..logging.logger import LoggerBase


def train_t3s(
    task_set: DiscreteTaskSet | VectorEnv,
    train_st: Callable,
    tau: float = 1.0,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    seed: int = 1,
    exploring_starts: int = 1_000,
    progress_bar: bool = True,
    logger: LoggerBase = None,
) -> tuple:
    """T3S task sampling.

    A task scheduling method for multi-task reinforcement learning. Given
    a set of tasks, it samples a task to train on based on a combination of
    task progress and task learning speed by performing regular evaluation
    rollouts on each task in the set.

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

    if isinstance(task_set, VectorEnv):
        n_tasks = task_set.num_envs
    else:
        n_tasks = len(task_set)

    # Assign initial sampling probabilities
    task_probs = np.ones(n_tasks) / n_tasks
    print(f"{task_probs=}")

    while global_step < total_timesteps:
        key, skey = jax.random.split(key)
        task_id = jax.random.choice(skey, n_tasks, p=task_probs)
        print(f"Sampled task {task_id=}")

        if isinstance(task_set, VectorEnv):
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

        # Perform evaluation and compute task progress and task learning speed
        task_success_rates = np.zeros_like(task_probs)
        task_learning_progress = np.zeros_like(task_probs)
        last_task_learning_progress = np.zeros_like(task_probs)

        # Compute new task probabilities
        task_probs = 0.5 * softmax(task_success_rates / tau) + 0.5 * softmax(
            task_learning_progress / tau
        )

        print(f"{task_probs=}")

    return st_result
