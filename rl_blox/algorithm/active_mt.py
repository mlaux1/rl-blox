import warnings
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike
from tqdm.rich import tqdm

from ..blox.mapb import DUCB
from ..blox.replay_buffer import MultiTaskReplayBuffer
from ..logging.logger import LoggerBase
from .smt import ContextualMultiTaskDefinition


class TaskSelector:
    def __init__(self, tasks):
        self.tasks = tasks
        self.waiting_for_reward = False

    def select(self):
        assert (
            not self.waiting_for_reward
        ), "You have to provide a reward for the last target"
        self.waiting_for_reward = True

    def feedback(self, reward):
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


TASK_SELECTORS = {
    "Round Robin": (RoundRobinSelector, {}),
    "1-step Progress": (
        DUCBGeneralized,
        {
            "baseline": "last",
            "op": None,
        },
    ),
    "Monotonic Progress": (
        DUCBGeneralized,
        {
            "baseline": "max",
            "op": "max-with-0",
        },
    ),
    "Best Reward": (
        DUCBGeneralized,
        {
            "baseline": None,
            "op": None,
        },
    ),
    "Diversity": (
        DUCBGeneralized,
        {
            "baseline": None,
            "op": "neg",
        },
    ),
}


def train_active_mt(
    mt_def: ContextualMultiTaskDefinition,
    train_st: Callable,
    replay_buffer: MultiTaskReplayBuffer,
    r_max: float,
    ducb_gamma: float = 0.95,
    xi: float = 0.002,
    task_selector: TaskSelector | str = "Monotonic Progress",
    total_timesteps: int = 1_000_000,
    scheduling_interval: int = 1,
    learning_starts: int = 5_000,
    seed: int = 0,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple:
    """Active Multi-Task Training.

    A multi-task extension of deep reinforcement learning similar to the method
    proposeed by Fabisch and Metzen [1]_ for contextual policy search.

    Parameters
    ----------
    mt_def
        The multi-task environment definition.

    train_st : callable
        The single-task training algorithm. The training function should accept
        the following parameters:

        - ``env``: The environment to train on.
        - ``learning_starts``: Number of steps to wait before starting training.
        - ``total_timesteps``: Total number of timesteps to train the agent.
        - ``replay_buffer``: The replay buffer to use for training.
        - ``seed``: Seed for random number generation.
        - ``logger``: Logger to record training statistics.
        - ``global_step``: Current global step in the training process.
        - ``progress_bar``: Flag to enable/disable the tqdm progress bar.

        No other parameters will be passed to the training function by SMT.
        Hence, for any other parameters, the training function should be
        prepared with ``functools.partial`` or similar.

    replay_buffer : MultiTaskReplayBuffer
        Replay buffer.

    r_max : float
        Upper bound for task selection reward.

    ducb_gamma : float
        Discount factor for D-UCB.

    xi : float, optional
        Padding function strength.

    task_selector : TaskSelector or str, default="Monotonic Progress"
        The task selection strategy. Can be an instance of ``TaskSelector`` or
        one of the predefined strategies:

        - "Round Robin": Cycles through tasks in order.
        - "1-step Progress": Selects tasks based on immediate progress.
        - "Monotonic Progress": Selects tasks based on the strictly positive
          immediate progress.
        - "Best Reward": Chooses tasks with the highest rewards.
        - "Diversity": Chooses tasks with the lowest rewards.

    total_timesteps : int
        The number of environment sets to train for.

    scheduling_interval : int
        Number of episodes after which the task scheduling is performed.

    learning_starts : int
        Number of steps to wait before starting training per task.

    seed : int
        Seed for random number generation.

    logger : LoggerBase, optional
        Experiment logger.

    progress_bar : bool, optional
        Flag to enable/disable the tqdm progressbar.

    Returns
    -------
    result
        The training result. Same as the result of the ``train_st`` function.

    training_steps : np.ndarray
        Number of training steps for each task.

    References
    ----------
    .. [1] Fabisch, A., Metzen, J. H. (2014). Active Contextual Policy Search.
       Journal of Machine Learning Research, 15(97), 3371-3399.
       https://jmlr.org/papers/v15/fabisch14a.html
    """
    global_step = 0
    training_steps = np.zeros(len(mt_def), dtype=int)
    progress = tqdm(total=total_timesteps, disable=not progress_bar)

    if isinstance(task_selector, str):
        assert task_selector in TASK_SELECTORS, (
            f"task_selector must be one of {list(TASK_SELECTORS.keys())}"
            " or an instance of TaskSelector."
        )
        selector_class, selector_kwargs = TASK_SELECTORS[task_selector]
        hparams = {
            "upper_bound": r_max,
            "ducb_gamma": ducb_gamma,
            "zeta": xi,
        }
        hparams.update(selector_kwargs)
        task_selector = selector_class(
            tasks=np.arange(len(mt_def)), **hparams
        )

    while global_step < total_timesteps:
        task_id = task_selector.select()
        if logger is not None:
            logger.record_stat("task_id", task_id, step=global_step)

        env = mt_def.get_task(task_id)
        env_with_stats = gym.wrappers.RecordEpisodeStatistics(
            env, buffer_length=scheduling_interval
        )
        replay_buffer.select_task(task_id)

        result_st = train_st(
            env=env_with_stats,
            learning_starts=learning_starts,
            total_timesteps=total_timesteps,
            total_episodes=scheduling_interval,
            replay_buffer=replay_buffer,
            seed=seed + global_step,
            logger=logger,
            global_step=global_step,
            progress_bar=False,
        )

        if len(env_with_stats.return_queue) != scheduling_interval:
            # limit total_timesteps reached
            unlogged_steps = total_timesteps - global_step
            training_steps[task_id] += unlogged_steps
            progress.update(unlogged_steps)
            break

        assert len(env_with_stats.return_queue) > 0
        mean_return = np.mean(env_with_stats.return_queue)
        task_selector.feedback(mean_return)

        steps = sum(env_with_stats.length_queue)
        training_steps[task_id] += steps
        global_step += steps

        progress.update(steps)

    return result_st, training_steps
