import inspect
import math
import warnings
from collections import namedtuple
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit, TransformObservation
from numpy.typing import ArrayLike

from ..blox.mapb import DUCB
from ..logging.logger import LoggerBase

if TYPE_CHECKING:
    from .replay_buffer import MultiTaskReplayBuffer


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


BACKBONE_STATE = (
    "replay_buffer",
    "q_target",
    "q_target_net",
    "policy_target",
    "actor_target",
    "critic_target",
    "policy_with_encoder_target",
    "entropy_control",
)
"""Arguments of backbone algorithms that are created anew if not passed."""


def warn_if_state_not_bound(
    train_st: Callable, replay_buffer_passed: bool = False
) -> None:
    """Warn if the backbone's state is not bound to its training function.

    Task schedulers call ``train_st`` once per scheduled task. Backbone
    algorithms create a new replay buffer, new target networks, or a new
    entropy control if these are not passed, which discards their state
    whenever a new task is scheduled.

    Parameters
    ----------
    train_st : callable
        Training function of the backbone algorithm. Only instances of
        :class:`functools.partial` can be checked, other callables are
        ignored.

    replay_buffer_passed : bool, optional
        The scheduler passes the replay buffer to ``train_st`` itself, so it
        does not have to be bound.

    Warns
    -----
    UserWarning
        If an argument from :data:`BACKBONE_STATE` that the backbone accepts
        is not bound to ``train_st``.
    """
    if not isinstance(train_st, partial):
        return

    accepted = inspect.signature(train_st.func).parameters
    unbound = [
        name
        for name in BACKBONE_STATE
        if name in accepted
        and train_st.keywords.get(name) is None
        and not (name == "replay_buffer" and replay_buffer_passed)
    ]
    if unbound:
        warnings.warn(
            f"{', '.join(unbound)} not bound to train_st, the backbone will "
            "create new ones for each scheduled task. Bind them with "
            "functools.partial to share them across tasks.",
            stacklevel=3,
        )


@nnx.jit
def act(policy: nnx.Module, observation: jnp.ndarray) -> jnp.ndarray:
    """Compute the deterministic action of a policy.

    Parameters
    ----------
    policy : nnx.Module
        Policy. If it returns a tuple, e.g., the mean and standard deviation
        of a Gaussian policy, the first element is the action.

    observation : array
        Observation.

    Returns
    -------
    action : array
        Action.
    """
    out = policy(observation)
    return out[0] if isinstance(out, tuple) else out


def get_task_env(
    task_set: "DiscreteTaskSet | SyncVectorEnv", task_id: int
) -> gym.Env:
    """Get the environment of a task.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    task_id : int
        Index of the task.

    Returns
    -------
    env : gym.Env
        Environment of the task.
    """
    if isinstance(task_set, SyncVectorEnv):
        return task_set.envs[task_id]
    else:
        return task_set.get_task(task_id)


def get_n_tasks(task_set: "DiscreteTaskSet | SyncVectorEnv") -> int:
    """Get the number of tasks.

    Parameters
    ----------
    task_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks.

    Returns
    -------
    n_tasks : int
        Number of tasks.
    """
    if isinstance(task_set, SyncVectorEnv):
        return task_set.num_envs
    else:
        return len(task_set)


def max_episode_steps(env: gym.Env) -> int | None:
    """Infer the maximum episode length of an environment.

    Parameters
    ----------
    env : gym.Env
        Environment.

    Returns
    -------
    max_episode_steps : int or None
        Maximum episode length from the ``TimeLimit`` wrapper or the spec of
        the environment or any environment that it wraps. None if it cannot
        be inferred.
    """
    while env is not None:
        if isinstance(env, TimeLimit):
            return env._max_episode_steps
        if env.spec is not None and env.spec.max_episode_steps is not None:
            return env.spec.max_episode_steps
        env = getattr(env, "env", None)
    return None


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
        task = self.tasks[self.i % len(self.tasks)]
        self.i += 1
        return task

    def feedback(self, reward: float):
        super().feedback(reward)


SchedulerResult = namedtuple(
    "SchedulerResult", ["result", "training_steps", "eval_steps"]
)
"""Result of a task scheduler: the result of the last call to ``train_st``,
the number of environment steps used to train on each task, and the number of
environment steps used for evaluations."""


def train_task(
    train_st: Callable,
    env: gym.Env,
    task_id: int,
    n_episodes: int,
    total_timesteps: int,
    global_step: int,
    seed: int,
    task_selectables: list[TaskSelectionMixin] | None = None,
    logger: LoggerBase | None = None,
    replay_buffer: "MultiTaskReplayBuffer | None" = None,
) -> tuple:
    """Train the backbone on one task for one scheduling interval.

    Parameters
    ----------
    train_st : callable
        Training function of the backbone with its state bound, see
        :func:`warn_if_state_not_bound`. It is called with the keyword
        arguments ``env``, ``seed``, ``total_timesteps``, ``total_episodes``,
        ``global_step``, ``logger``, and ``progress_bar``, and with
        ``replay_buffer`` if one is given.

    env : gym.Env
        Environment of the task, e.g., from :func:`get_task_env`.

    task_id : int
        Index of the task.

    n_episodes : int
        Number of episodes to train for.

    total_timesteps : int
        Total budget of the scheduler. Training stops when it runs out.

    global_step : int
        Number of environment steps used so far.

    seed : int
        Random seed of the scheduler. ``train_st`` is called with
        ``seed + global_step``.

    task_selectables : list, optional
        Objects that are informed about the task before training.

    logger : LoggerBase, optional
        Experiment logger.

    replay_buffer : MultiTaskReplayBuffer, optional
        Replay buffer of all tasks. ``train_st`` gets the view of the task,
        see :meth:`~rl_blox.blox.replay_buffer.MultiTaskReplayBuffer.task`.

    Returns
    -------
    result : namedtuple
        Result of ``train_st``.

    steps : int
        Number of environment steps of the interval. If the total budget ran
        out during the interval, the remaining budget.

    returns : array, shape (n_episodes,)
        Returns of the completed episodes. Fewer than ``n_episodes`` if the
        total budget ran out.

    Notes
    -----
    Logging

    * ``task_id`` - task scheduled for training
    """
    if logger is not None:
        logger.record_stat("task_id", task_id, step=global_step)
    if task_selectables is not None:
        for ts in task_selectables:
            ts.select_task(task_id)

    # Count steps independently of the global_step returned by train_st
    env = gym.wrappers.RecordEpisodeStatistics(
        env, buffer_length=n_episodes, stats_key="scheduler_episode"
    )
    kwargs = {}
    if replay_buffer is not None:
        kwargs["replay_buffer"] = replay_buffer.task(task_id)
    result = train_st(
        env=env,
        seed=seed + global_step,
        total_timesteps=total_timesteps,
        total_episodes=n_episodes,
        global_step=global_step,
        logger=logger,
        progress_bar=False,
        **kwargs,
    )

    returns = np.asarray(env.return_queue, dtype=float)
    if len(returns) < n_episodes:
        steps = total_timesteps - global_step
    else:
        steps = int(np.sum(env.length_queue))
    return result, steps, returns


def n_warmup_selections(
    n_tasks: int, warmup_episodes: int, episodes_per_task: int
) -> int:
    """Number of task selections of a round-robin warm-up.

    The warm-up consists of whole rounds over all tasks, so that each task is
    trained for at least ``warmup_episodes`` episodes.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.

    warmup_episodes : int
        Minimum number of warm-up episodes per task.

    episodes_per_task : int
        Number of episodes per task selection.

    Returns
    -------
    n_selections : int
        Number of task selections of the warm-up.
    """
    return n_tasks * math.ceil(warmup_episodes / episodes_per_task)


EvalEpisode = namedtuple(
    "EvalEpisode",
    ["task_id", "rewards", "infos", "log_probs"],
    defaults=(None,),
)
"""Episode of an evaluation by a task scheduler: index of the task, reward
and info dictionary of each step, and log probability of each action if
actions were sampled, otherwise None."""


def is_stochastic(policy: nnx.Module) -> bool:
    """Check if a policy can sample actions with their log probabilities.

    Parameters
    ----------
    policy : nnx.Module
        Policy.

    Returns
    -------
    stochastic : bool
        True if the policy has a method ``sample_and_log_prob``.
    """
    return hasattr(policy, "sample_and_log_prob")


@nnx.jit
def _sample_and_log_prob(policy, observation, key):
    return policy.sample_and_log_prob(observation, key)


def max_evaluation_steps(
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    n_eval_episodes: int,
    max_eval_episode_steps: int | None,
) -> int | None:
    """Upper bound on the number of steps of an evaluation of all tasks.

    Parameters
    ----------
    eval_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks used for evaluation.

    n_eval_episodes : int
        Number of evaluation episodes per task.

    max_eval_episode_steps : int or None
        Maximum length of an evaluation episode. Inferred from the evaluation
        environments if None, see :func:`max_episode_steps`.

    Returns
    -------
    max_steps : int or None
        Upper bound on the number of steps. None if the episode length of an
        evaluation environment cannot be inferred.
    """
    n_tasks = get_n_tasks(eval_set)
    if max_eval_episode_steps is not None:
        episode_limits = [max_eval_episode_steps] * n_tasks
    else:
        episode_limits = [
            max_episode_steps(get_task_env(eval_set, i)) for i in range(n_tasks)
        ]
    if None in episode_limits:
        return None
    return n_eval_episodes * sum(episode_limits)


def evaluate_tasks(
    policy: nnx.Module,
    eval_set: DiscreteTaskSet | SyncVectorEnv,
    n_eval_episodes: int,
    key: jax.Array | None = None,
    task_selectables: list[TaskSelectionMixin] | None = None,
) -> tuple[list[list[EvalEpisode]], int, jax.Array | None]:
    """Run evaluation episodes on all tasks.

    Parameters
    ----------
    policy : nnx.Module
        Policy that maps an observation to an action.

    eval_set : DiscreteTaskSet or SyncVectorEnv
        Set of tasks used for evaluation.

    n_eval_episodes : int
        Number of evaluation episodes per task.

    key : jax.Array, optional
        Key for sampling actions. If it is given and the policy is stochastic
        (see :func:`is_stochastic`), actions are sampled. Otherwise, the
        deterministic action is taken, see :func:`act`.

    task_selectables : list, optional
        Objects that are informed about the evaluated task.

    Returns
    -------
    episodes : list of list of EvalEpisode
        Evaluation episodes of each task.

    n_steps : int
        Number of environment steps of the evaluation.

    key : jax.Array or None
        Updated key.
    """
    sample = key is not None and is_stochastic(policy)

    episodes = []
    n_steps = 0
    for task_id in range(get_n_tasks(eval_set)):
        if task_selectables is not None:
            for ts in task_selectables:
                ts.select_task(task_id)
        env = get_task_env(eval_set, task_id)

        task_episodes = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            rewards, infos, log_probs = [], [], []
            done = False
            while not done:
                if sample:
                    key, sampling_key = jax.random.split(key)
                    action, log_prob = _sample_and_log_prob(
                        policy, obs, sampling_key
                    )
                    log_probs.append(float(log_prob))
                else:
                    action = act(policy, obs)
                obs, reward, terminated, truncated, info = env.step(
                    np.asarray(action)
                )
                rewards.append(float(reward))
                infos.append(info)
                done = terminated or truncated
            n_steps += len(rewards)
            task_episodes.append(
                EvalEpisode(
                    task_id,
                    np.asarray(rewards),
                    infos,
                    np.asarray(log_probs) if sample else None,
                )
            )
        episodes.append(task_episodes)

    return episodes, n_steps, key


def next_evaluation_step(
    next_eval_step: int, eval_freq: int, global_step: int
) -> int:
    """Step at which the next evaluation is due.

    Evaluations are due every ``eval_freq`` steps. Due steps that already
    passed, e.g., during a long evaluation, are skipped, so that delays do
    not shift later evaluations.

    Parameters
    ----------
    next_eval_step : int
        Step at which the last evaluation was due.

    eval_freq : int
        Number of environment steps between evaluations.

    global_step : int
        Number of environment steps used so far.

    Returns
    -------
    next_eval_step : int
        First due step after ``global_step``.
    """
    n_passed = max(global_step - next_eval_step, 0) // eval_freq
    return next_eval_step + (n_passed + 1) * eval_freq
