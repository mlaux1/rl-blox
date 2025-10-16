from abc import ABCMeta, abstractmethod

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers import TransformObservation
from numpy.typing import ArrayLike

from ..blox.mapb import DUCB


class TaskSelectionMixin:
    task_id: int
    """Current task ID."""

    def __init__(self):
        self.task_id = 0

    def select_task(self, task_id: int) -> None:
        """Selects the task.

        Parameters
        ----------
        task_id : int
            ID of the task to select, usually an index.
        """
        self.task_id = task_id


class DiscreteTaskSet(metaclass=ABCMeta):
    """Defines a discrete set of environments for multi-task RL."""

    def __init__(self, contexts: ArrayLike, context_aware: bool):
        self.contexts = contexts
        self.context_high = jnp.max(contexts, axis=0)
        self.context_low = jnp.min(contexts, axis=0)
        self.context_aware = context_aware

    def get_task(self, task_id: int) -> gym.Env:
        """Returns the task environment for the given task ID."""
        assert 0 <= task_id < len(self.contexts)
        context = self.contexts[task_id]
        st_env = self._get_env(context)
        if self.context_aware:
            new_obs_space = gym.spaces.Box(
                low=np.concatenate(
                    (self.context_low, st_env.observation_space.low), axis=0
                ),
                high=np.concatenate(
                    (self.context_high, st_env.observation_space.high), axis=0
                ),
                dtype=st_env.observation_space.dtype,
            )

            return TransformObservation(
                st_env,
                lambda obs, ctx=context: np.concatenate((context, obs)),
                new_obs_space,
            )
        else:
            return st_env

    def get_context(self, task_id: int) -> np.ndarray:
        """Returns the task context for the given task ID."""
        return self.contexts[task_id]

    @abstractmethod
    def _get_env(self, context: ArrayLike) -> gym.Env:
        """Returns the base environment without context."""

    def __len__(self) -> int:
        return len(self.contexts)


class TaskSelector:
    def __init__(self, tasks):
        self.tasks = tasks
        self.waiting_for_reward = False

    def select(self) -> int:
        assert (
            not self.waiting_for_reward
        ), "You have to provide a reward for the last target"
        self.waiting_for_reward = True
        return 0

    def feedback(self, reward: float):
        assert self.waiting_for_reward, "Cannot assign reward to any target"
        self.waiting_for_reward = False


class DUCBGeneralized(TaskSelector):
    def __init__(
        self,
        tasks: ArrayLike,
        upper_bound: float,
        ducb_gamma: float,
        zeta: float,
        baseline: str | None,
        op: str | None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(tasks)
        self.baseline = baseline
        self.op = op
        self.verbose = verbose
        self.heuristic_params = kwargs

        self.n_contexts = tasks.shape[0]
        self.ducb = DUCB(
            n_arms=self.n_contexts,
            upper_bound=upper_bound,
            gamma=ducb_gamma,
            zeta=zeta,
        )
        self.last_rewards = [[] for _ in range(self.n_contexts)]
        self.chosen_arm = -1

    def select(self) -> int:
        super().select()
        self.chosen_arm = self.ducb.choose_arm()
        return self.tasks[self.chosen_arm]

    def feedback(self, reward: float):
        last_rewards = np.array(self.last_rewards[self.chosen_arm])[::-1]

        if len(last_rewards) == 0:
            self.ducb.chosen_arms = self.ducb.chosen_arms[:-1]
        else:
            if self.baseline == "max":
                b = np.max(last_rewards)
            elif self.baseline == "avg":
                b = np.mean(last_rewards)
            elif self.baseline == "davg":
                gamma = self.heuristic_params["heuristic_gamma"]
                b = np.sum(
                    last_rewards * gamma ** np.arange(1, len(last_rewards) + 1)
                ) * (1.0 / gamma - 1.0)
            elif self.baseline == "last":
                b = last_rewards[0]
            else:
                b = 0.0
            intrinsic_reward = reward - b
            if self.op == "max-with-0":
                intrinsic_reward = np.maximum(0.0, intrinsic_reward)
            elif self.op == "abs":
                intrinsic_reward = np.abs(intrinsic_reward)
            elif self.op == "neg":
                intrinsic_reward *= -1
            self.ducb.reward(intrinsic_reward)

        super().feedback(reward)

        self.last_rewards[self.chosen_arm].append(reward)


class RoundRobinSelector(TaskSelector):
    def __init__(self, tasks, **kwargs):
        super().__init__(tasks)
        self.i = 0

    def select(self) -> int:
        super().select()
        self.i += 1
        return self.tasks[self.i % len(self.tasks)]

    def feedback(self, reward: float):
        super().feedback(reward)


class PrioritisedTaskSampler:
    """A sampler that returns envs from a TaskSet given priorities."""

    def __init__(
        self,
        task_set: DiscreteTaskSet,
        priorities: ArrayLike | None = None,
    ):
        self.task_set = task_set
        self.priorities = priorities

    def sample(self, key) -> tuple[gym.Env, jnp.ndarray]:
        env_id = jax.random.choice(
            key, jnp.arange(len(self.task_set)), p=self.priorities
        ).item()
        return self.task_set.get_task(env_id), self.task_set.contexts[env_id]

    def update_priorities(self, priorities) -> None:
        self.priorities = priorities
