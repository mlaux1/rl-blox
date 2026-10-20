from collections.abc import Callable

import jax
import numpy as np
from gymnasium.vector import SyncVectorEnv
from scipy.special import softmax
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet
from ..logging.logger import LoggerBase


def train_t3s(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    tau: float = 1.0,
    learning_speed_weight: float = 0.5,
    progress_weight: float = 0.5,
    n_eval_episodes: int = 10,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    eval_freq: int = 25_000,
    seed: int = 1,
    exploring_starts: int = 2_000,
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

    task_set : DiscreteTaskSet | SyncVectorEnv
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
    assert total_timesteps > 0

    if tau <= 0:
        raise ValueError("tau must be > 0")

    if not np.isclose(
        progress_weight + learning_speed_weight,
        1.0,
    ):
        raise ValueError("metric weights must sum up to 1")

    global_step = 0
    eval_step = 0
    next_eval_step = eval_freq
    progress = tqdm(total=total_timesteps, disable=not progress_bar)
    key = jax.random.key(seed)

    if isinstance(task_set, SyncVectorEnv):
        n_tasks = task_set.num_envs
    else:
        n_tasks = len(task_set)

    # Assign initial sampling probabilities
    task_probs = np.ones(n_tasks) / n_tasks
    print(f"{task_probs=}")
    previous_task_performance = None

    while global_step + eval_step < total_timesteps:
        key, skey = jax.random.split(key)
        task_id = int(jax.random.choice(skey, n_tasks, p=task_probs))
        print(f"Sampled task {task_id=}")

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

        if global_step + eval_step >= next_eval_step:
            # Perform evaluation and compute task progress and task learning speed
            task_performance = np.zeros_like(task_probs)

            for i in range(n_tasks):
                if isinstance(task_set, SyncVectorEnv):
                    env = task_set.envs[i]
                else:
                    env = task_set.get_task(i)

                n_successes = 0

                for _ in range(n_eval_episodes):
                    obs, _ = env.reset()
                    success = False
                    done = False
                    while not done:
                        action = st_result.policy(obs)[0]
                        eval_step += 1
                        global_step += 1
                        progress.update()
                        obs, reward, terminated, truncated, info = env.step(
                            action
                        )
                        success |= bool(info["success"])
                        done = terminated or truncated
                    n_successes += int(success)

                task_performance[i] = n_successes / n_eval_episodes

            # Compute new task probabilities
            task_progress_metric = 1.0 - task_performance

            if previous_task_performance is None:
                task_learning_speed_metric = np.zeros(n_tasks)
            else:
                task_learning_speed_metric = -(
                    task_performance - previous_task_performance
                )

            print(
                f"{task_performance=},{task_progress_metric=},{
                    task_learning_speed_metric=}"
            )

            task_probs = progress_weight * softmax(
                task_progress_metric / tau
            ) + learning_speed_weight * softmax(
                task_learning_speed_metric / tau
            )

            print(f"{task_probs=}")
            previous_task_performance = task_performance.copy()

            next_eval_step = eval_freq + global_step

    return st_result
