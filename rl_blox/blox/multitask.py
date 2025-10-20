from collections.abc import Callable

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


class DiscreteTaskSet:
    """Defines a discrete set of environments for multi-task RL."""

    def __init__(
        self,
        base_env: gym.Env,
        set_context: Callable[[gym.Env], None],
        contexts: ArrayLike,
        context_aware: bool,
    ):
        self.base_env = base_env
        self.contexts = contexts
        self.context_aware = context_aware
        self.set_context = set_context

        context_high = np.max(contexts, axis=0).ravel()
        context_low = np.min(contexts, axis=0).ravel()
        self.contextual_obs_space = gym.spaces.Box(
            low=np.concatenate(
                (context_low, self.base_env.observation_space.low),
                axis=0,
            ),
            high=np.concatenate(
                (context_high, self.base_env.observation_space.high),
                axis=0,
            ),
            dtype=self.base_env.observation_space.dtype,
        )

    def get_task(self, task_id: int) -> gym.Env:
        """Returns the task environment for the given task ID."""
        assert 0 <= task_id < len(self.contexts)
        context = np.asarray(self.contexts[task_id])
        self.set_context(self.base_env, context)
        if self.context_aware:
            return TransformObservation(
                self.base_env,
                lambda obs, ctx=context: np.concatenate(
                    (context, np.ravel(obs)), axis=0
                ),
                self.contextual_obs_space,
            )
        else:
            return self.base_env

    def get_context(self, task_id: int) -> np.ndarray:
        """Returns the task context for the given task ID."""
        return self.contexts[task_id]

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
