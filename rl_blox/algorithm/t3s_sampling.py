import warnings
from collections.abc import Callable

import jax
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from scipy.special import softmax
from tqdm.rich import tqdm

from ..blox.multitask import DiscreteTaskSet
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


def train_t3s(
    task_set: DiscreteTaskSet | SyncVectorEnv,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    train_st: Callable,
    tau: float = 1.0,
    learning_speed_weight: float = 0.5,
    progress_weight: float = 0.5,
    n_eval_episodes: int = 10,
    total_timesteps: int = 100_000,
    episodes_per_task: int = 1,
    tasks_per_iteration: int = 3,
    sample_tasks_with_replacement: bool = False,
    eval_freq: int = 25_000,
    max_eval_episode_steps: int | None = None,
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

    max_eval_episode_steps : int, optional
        Maximum length of an evaluation episode. Used to skip evaluations that
        would exceed ``total_timesteps``. If not set, it is inferred from the
        ``TimeLimit`` wrapper or spec of the evaluation environments. If it
        cannot be inferred, evaluations may exceed ``total_timesteps``.

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
        assert eval_set.num_envs == n_tasks
    else:
        n_tasks = len(task_set)
        assert len(eval_set) == n_tasks

    if tasks_per_iteration < 1:
        raise ValueError("tasks_per_iteration must be positive")
    if not sample_tasks_with_replacement and tasks_per_iteration > n_tasks:
        raise ValueError(
            "tasks_per_iteration cannot exceed number of tasks whan sampling without replacement"
        )

    # Upper bound on the number of steps a full evaluation can take
    if max_eval_episode_steps is not None:
        episode_limits = [max_eval_episode_steps] * n_tasks
    else:
        episode_limits = [
            _max_episode_steps(_get_env(eval_set, i)) for i in range(n_tasks)
        ]
    if None in episode_limits:
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

            global_step = st_result.global_step

        if (
            global_step >= next_eval_step
            and global_step + max_eval_steps <= total_timesteps
        ):
            # Perform evaluation and compute task progress and task learning speed
            task_performance = np.zeros_like(task_probs)

            for i in range(n_tasks):
                env = _get_env(eval_set, i)
                n_successes = 0

                for _ in range(n_eval_episodes):
                    obs, _ = env.reset()
                    success = False
                    done = False
                    while not done:
                        action = np.asarray(act(st_result.policy, obs))
                        eval_step += 1
                        global_step += 1
                        progress.update()
                        obs, _, terminated, truncated, info = env.step(action)
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
            task_probs = progress_weight * softmax(
                task_progress_metric / tau
            ) + learning_speed_weight * softmax(
                task_learning_speed_metric / tau
            )

            if logger is not None:
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

    return st_result
