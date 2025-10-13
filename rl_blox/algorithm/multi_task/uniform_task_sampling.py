from collections.abc import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers import TransformObservation
from jax.typing import ArrayLike
from tqdm.rich import tqdm

from ...logging.logger import LoggerBase


class TaskSet:
    """A collection of tasks (environments)."""

    def __init__(
        self, contexts: list[ArrayLike], envs: list[gym.Env], context_aware=True
    ):
        assert len(contexts) == len(envs)
        self.contexts = contexts
        self.task_envs = envs
        self.context_aware = context_aware

        if context_aware:
            for i in range(len(contexts)):
                assert isinstance(
                    self.task_envs[i].observation_space, gym.spaces.Box
                )
                new_low = np.concatenate(
                    [self.task_envs[i].observation_space.low, contexts[0]]
                )
                new_high = np.concatenate(
                    [
                        self.task_envs[i].observation_space.high,
                        contexts[-1],
                    ]
                )
                new_obs_space = gym.spaces.Box(low=new_low, high=new_high)
                ctx_i = np.asarray(self.contexts[i])
                self.task_envs[i] = TransformObservation(
                    self.task_envs[i],
                    lambda obs, ctx=ctx_i: np.concatenate([obs, ctx]),
                    new_obs_space,
                )

    def get_context(self, task_id: int) -> jnp.ndarray:
        assert 0 <= task_id < len(self.contexts)
        return self.contexts[task_id]

    def get_task_env(self, task_id: int) -> gym.Env:
        assert 0 <= task_id < len(self.contexts)
        return self.task_envs[task_id]

    def get_task_and_context(self, task_id: int) -> tuple[gym.Env, jnp.ndarray]:
        assert 0 <= task_id < len(self.contexts)
        return self.task_envs[task_id], self.contexts[task_id]

    def __len__(self) -> int:
        return len(self.contexts)


class PrioritisedTaskSampler:
    """A sampler that returns envs from a TaskSet given priorities."""

    def __init__(
        self,
        task_set: TaskSet,
        priorities: ArrayLike | None = None,
    ):
        self.task_set = task_set
        self.priorities = priorities

    def sample(self, key) -> tuple[gym.Env, jnp.ndarray]:
        env_id = jax.random.choice(
            key, jnp.arange(len(self.task_set)), p=self.priorities
        ).item()
        return self.task_set.get_task_and_context(env_id)

    def update_priorities(self, priorities) -> None:
        self.priorities = priorities


def train_uts(
    envs: TaskSet,
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

    envs : TaskSet
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
    task_sampler = PrioritisedTaskSampler(envs)

    while global_step < total_timesteps:
        key, skey = jax.random.split(key)
        env, context = task_sampler.sample(skey)
        st_result = train_st(
            env,
            seed=seed + global_step,
            total_timesteps=total_timesteps,
            max_episodes=episodes_per_task,
            learning_starts=exploring_starts,
            progress_bar=False,
            logger=logger,
            global_step=global_step,
        )

        _, _, _, _, _, _, _, new_global_step = st_result

        progress.update(new_global_step - global_step)
        global_step = new_global_step

    return st_result
