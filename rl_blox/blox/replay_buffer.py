import copy
from collections import OrderedDict, namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


class ReplayBuffer:
    """Replay buffer that returns jax arrays.

    For each quantity, we store all samples in NumPy array that will be
    preallocated once the size of the quantities is know, that is, when the
    first transition sample is added. This makes sampling faster than when
    we use a deque.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'termination']. These names have to be used as key word arguments when
        adding a sample. When sampling a batch, the arrays will be returned in
        this order.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.
    """

    buffer: OrderedDict[str, npt.NDArray[float]]
    Batch: type
    buffer_size: int
    current_len: int
    insert_idx: int

    def __init__(
        self,
        buffer_size: int,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "termination",
            ]
        if dtypes is None:
            dtypes = [
                float,
                int if discrete_actions else float,
                float,
                float,
                int,
            ]
        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.Batch = namedtuple("Batch", self.buffer)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape,
                    dtype=self.buffer[k].dtype,
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        batch : Batch
            Named tuple with order defined by keys. Content is also accessible
            via names, e.g., ``batch.observation``.
        """
        indices = rng.integers(0, self.current_len, batch_size)
        return self.Batch(
            **{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer}
        )

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["Batch"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.Batch = namedtuple("Batch", self.buffer)


class SubtrajectoryReplayBuffer:
    """Replay buffer for sampling batches of subtrajectories.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    horizon : int, optional
        Maximum length of the horizon.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'terminated', 'truncated']. These names have to be used as key word
        arguments when adding a sample. When sampling a batch, the arrays will
        be returned in this order. Must contain at least 'observation',
        'next_observation', 'terminated', and 'truncated'.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'terminated' and 'truncated'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.
    """

    def __init__(
        self,
        buffer_size: int,
        horizon: int = 1,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        assert buffer_size > 0
        assert horizon > 0

        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminated",
                "truncated",
            ]
        if dtypes is None:
            dtypes = [
                float,
                int if discrete_actions else float,
                float,
                float,
                int,
                int,
            ]
        for key in [
            "observation",
            "next_observation",
            "terminated",
            "truncated",
        ]:
            if key not in keys:
                raise ValueError(f"'{key}' must be in keys")

        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.Batch = namedtuple("Batch", self.buffer)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

        self.episode_timesteps = 0
        # track if there are any terminal transitions in the buffer
        self.environment_terminates = False
        self.horizon = horizon
        self.mask_ = np.zeros(self.buffer_size, dtype=int)

    def add_sample(self, **sample) -> list[int]:
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape,
                    dtype=self.buffer[k].dtype,
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v

        self.current_len = min(self.current_len + 1, self.buffer_size)
        self.episode_timesteps += 1
        if sample["terminated"]:
            self.environment_terminates = True

        self.mask_[self.insert_idx] = 0
        if self.episode_timesteps > self.horizon:
            self.mask_[(self.insert_idx - self.horizon) % self.buffer_size] = 1

        inserted_at = [self.insert_idx]
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size

        if sample["terminated"] or sample["truncated"]:
            for k in self.buffer:
                if k == "reward":
                    self.buffer[k][self.insert_idx] = 0.0
                else:
                    self.buffer[k][self.insert_idx] = sample[k]
            self.buffer["observation"][self.insert_idx] = sample[
                "next_observation"
            ]

            self.mask_[self.insert_idx % self.buffer_size] = 0
            past_idx = (
                self.insert_idx
                - np.arange(min(self.episode_timesteps, self.horizon))
                - 1
            ) % self.buffer_size
            self.mask_[past_idx] = (
                0 if sample["truncated"] else 1
            )  # mask out truncated subtrajectories

            inserted_at += [self.insert_idx]
            self.insert_idx = (self.insert_idx + 1) % self.buffer_size
            self.current_len = min(self.current_len + 1, self.buffer_size)

            self.episode_timesteps = 0

        return inserted_at

    def sample_batch(
        self,
        batch_size: int,
        horizon: int,
        include_intermediate: bool,
        rng: np.random.Generator,
    ) -> tuple[jnp.ndarray]:
        """Sample a batch of transitions from the replay buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples to be returned.

        horizon : int
            Horizon for the sampled transitions.

        include_intermediate : bool
            Whether to include intermediate states in the sampled transitions.

        rng : np.random.Generator
            Random number generator for sampling.

        Returns
        -------
        batch : tuple[jnp.ndarray]
            A tuple containing the sampled observations, actions, rewards,
            next observations, and terminations.
        """
        assert batch_size > 0
        assert horizon > 0

        indices = self._sample_idx(batch_size, rng)
        # TODO % self.current_len or % self.buffer_size?
        # - maybe self.buffer_size is possible because of the mask?
        indices = (
            indices[:, np.newaxis] + np.arange(horizon)[np.newaxis]
        ) % self.current_len

        if include_intermediate:
            # sample subtrajectories (with horizon dimension) for unrolling
            # dynamics
            batch = self.Batch(
                **{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer}
            )
        else:
            # sample at specific horizon (used for multistep rewards)
            batch = {}
            for k in self.buffer:
                if k in ["observation", "action"]:
                    indices_without_intermediate = indices[:, 0]
                elif k == "next_observation":
                    indices_without_intermediate = indices[:, -1]
                else:
                    indices_without_intermediate = indices
                batch[k] = jnp.asarray(
                    self.buffer[k][indices_without_intermediate]
                )
            batch = self.Batch(**batch)

        return batch

    def _sample_idx(
        self, batch_size: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        nz = np.nonzero(self.mask_)[0]
        indices = rng.integers(0, len(nz), size=batch_size)
        return nz[indices]

    def reward_scale(self, eps: float = 1e-8):
        assert "reward" in self.buffer
        # very hacky way of computing the reward scale...
        return max(
            np.abs(self.buffer["reward"][: self.current_len]).mean(), eps
        )

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["Batch"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.Batch = namedtuple("Batch", self.buffer)


class PriorityBuffer:
    max_priority: float
    priority: npt.NDArray[float]
    sampled_indices: npt.NDArray[int]

    def __init__(self, buffer_size: int):
        self.max_priority = 1.0
        self.priority = np.empty(buffer_size, dtype=float)
        self.sampled_indices = np.empty(0, dtype=int)

    def initialize_priority(self, insert_idx: npt.NDArray[int]):
        """Initialize the priority of the next inserted sample."""
        self.priority[insert_idx] = self.max_priority

    def prioritized_sampling(
        self,
        current_len: int,
        batch_size: int,
        rng: np.random.Generator,
        mask: npt.NDArray[int] | None = None,
    ) -> npt.NDArray[int]:
        """Sample indices based on the priority distribution."""
        priority = self.priority[:current_len]
        if mask is not None:
            priority = priority * mask[:current_len]
        probabilities = np.cumsum(priority)
        random_uniforms = rng.uniform(0, 1, size=batch_size) * probabilities[-1]
        self.sampled_indices = np.searchsorted(probabilities, random_uniforms)
        return self.sampled_indices

    def update_priority(self, priority: npt.ArrayLike):
        """Update the priority of the sampled indices."""
        self.priority[self.sampled_indices] = priority
        self.max_priority = max(np.max(priority), self.max_priority)

    def reset_max_priority(self, current_len: int):
        """Recalculate the maximum priority."""
        if current_len > 0:
            self.max_priority = np.max(self.priority[:current_len])


class LAP(ReplayBuffer):
    r"""Prioritized replay buffer.

    This replay buffer can be used for loss-adjusted PER (LAP) [1]_ and
    prioritized experience replay (PER) [2]_. PER is a sampling scheme for
    replay buffers, in which transitions are sampled in proportion to their
    temporal-difference (TD) error. The intuitive argument behind PER is that
    training on the highest error samples will result in the largest
    performance gain.

    PER changes the traditional uniformly sampled replay buffers. The
    probability of sampling a transition i is proportional to the absolute TD
    error :math:`|\delta_i|`, set to the power of a hyper-parameter
    :math:`\alpha` to smooth out extremes:

    .. math::

        p(i)
        =
        \frac{|\delta_i|^{\alpha} + \epsilon}
        {\sum_j |\delta_j|^{\alpha} + \epsilon},

    where a small constant :math:`\epsilon` is added to ensure each transition
    is sampled with non-zero probability. This is necessary as often the
    current TD error is approximated by the TD error when i was last sampled.

    LAP changes this to (:func:`lap_priority`)

    .. math::

        p(i)
        =
        \frac{\max(|\delta_i|^{\alpha}, 1)}
        {\sum_j \max(|\delta_j|^{\alpha}, 1)},

    which leads to uniform sampling of transitions with a TD error smaller than
    1 to avoid the bias introduced from using MSE and prioritization. A LAP
    replay buffer is supposed to be paired with a Huber loss with a threshold
    of 1 to switch between MSE and L1 loss.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'termination']. These names have to be used as key word arguments when
        adding a sample. When sampling a batch, the arrays will be returned in
        this order.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    .. [2] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """

    priority: PriorityBuffer

    def __init__(
        self,
        buffer_size: int,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        super().__init__(buffer_size, keys, dtypes, discrete_actions)
        self.priority = PriorityBuffer(buffer_size)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        self.priority.initialize_priority(self.insert_idx)
        super().add_sample(**sample)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        batch : Batch
            Named tuple with order defined by keys. Content is also accessible
            via names, e.g., ``batch.observation``.
        """
        indices = self.priority.prioritized_sampling(
            self.current_len, batch_size, rng
        )
        return self.Batch(
            **{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer}
        )

    def update_priority(self, priority):
        self.priority.update_priority(priority)

    def reset_max_priority(self):
        self.priority.reset_max_priority(self.current_len)


class SubtrajectoryReplayBufferPER(SubtrajectoryReplayBuffer):
    """Replay buffer for sampling batches of subtrajectories with PER or LAP.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    horizon : int, optional
        Maximum length of the horizon.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'terminated', 'truncated']. These names have to be used as key word
        arguments when adding a sample. When sampling a batch, the arrays will
        be returned in this order. Must contain at least 'observation',
        'next_observation', 'terminated', and 'truncated'.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float for everything except
        'terminated' and 'truncated'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.
    """

    priority: PriorityBuffer

    def __init__(
        self,
        buffer_size: int,
        horizon: int = 1,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
    ):
        super().__init__(buffer_size, horizon, keys, dtypes, discrete_actions)
        self.priority = PriorityBuffer(buffer_size)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        inserted_at = super().add_sample(**sample)
        self.priority.initialize_priority(inserted_at)

    def _sample_idx(
        self, batch_size: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        return self.priority.prioritized_sampling(
            self.current_len, batch_size, rng, self.mask_
        )

    def update_priority(self, priority):
        self.priority.update_priority(priority)

    def reset_max_priority(self):
        self.priority.reset_max_priority(self.current_len)


@partial(jax.jit, static_argnames=["min_priority", "alpha"])
def lap_priority(
    abs_td_error: jnp.ndarray, min_priority: float, alpha: float
) -> jnp.ndarray:
    r"""Compute sample priority for loss-adjusted PER (LAP).

    Loss-adjusted prioritized experience replay (LAP) [1]_ is based on
    prioritized experience replay (PER) [2]_. LAP uses the priority

    .. math::

        p(i)
        =
        \frac{\max(|\delta_i|^{\alpha}, p_{\min})}
        {\sum_j \max(|\delta_j|^{\alpha}, p_{\min})},

    which leads to uniform sampling of transitions with a TD error smaller than
    :math:`p_{\min}` (usually 1) to avoid the bias introduced from using MSE
    and prioritization. A LAP replay buffer is supposed to be paired with a
    Huber loss with a threshold of :math:`p_{\min}` to switch between MSE and
    L1 loss.

    Parameters
    ----------
    abs_td_error : array
        A batch of :math:`|\delta_i|`.

    min_priority : float
        Minimum priority :math:`p_{\min}`.

    alpha : float
        Smoothing exponent :math:`\alpha`.

    Returns
    -------
    p : array
        A batch of priorities :math:`p(i)`.

    References
    ----------
    .. [1] Fujimoto, S., Meger, D., Precup, D. (2020). An Equivalence between
       Loss Functions and Non-Uniform Sampling in Experience Replay. In
       Advances in Neural Information Processing Systems 33.
       https://papers.nips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html

    .. [2] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized
       Experience Replay. In International Conference on Learning
       Representations. https://arxiv.org/abs/1511.05952
    """
    return jnp.maximum(abs_td_error, min_priority) ** alpha


class MultiTaskReplayBuffer:
    """Replay buffer for discrete set of tasks.

    For each task we will create a separate replay buffer. During sampling,
    we will store samples in the corresponding task's replay buffer. We will
    sample batches from all replay buffers.

    Parameters
    ----------
    replay_buffer : object
        Replay buffer instance that will be copied for each task.

    n_tasks : int
        Number of tasks.
    """

    def __init__(self, replay_buffer, n_tasks: int):
        self.buffers = [replay_buffer]
        for _ in range(n_tasks - 1):
            self.buffers.append(copy.deepcopy(replay_buffer))
        self.selected_task = 0
        self.active_buffers = set()

    def select_task(self, task_id: int):
        """Select the task for which samples will be added."""
        if 0 <= task_id < len(self.buffers):
            self.selected_task = task_id
        else:
            raise ValueError(
                f"Invalid task id: {task_id}. Must be in [0, {len(self.buffers) - 1}]."
            )

    def add_sample(self, *args, **kwargs):
        """Add transition sample to the replay buffer for a specific task."""
        self.buffers[self.selected_task].add_sample(*args, **kwargs)
        self.active_buffers.add(self.selected_task)

    def sample_batch(self, *args, **kwargs):
        """Sample a batch of transitions from all replay buffers."""
        args = list(args)
        if "rng" in kwargs:
            rng = kwargs.pop("rng")
        elif len(kwargs) == 0:
            rng = args[-1]
            del args[-1]
        else:
            raise ValueError("No rng provided.")

        self.sampled_task_idx = int(
            rng.choice(list(self.active_buffers), size=1)
        )
        return self.buffers[self.sampled_task_idx].sample_batch(
            *args, rng=rng, **kwargs
        )

    def reward_scale(self, eps: float = 1e-8):
        """Compute the reward scale for all tasks."""
        n_samples = 0
        accumulated_reward = 0.0
        for buffer in self.buffers:
            if len(buffer) == 0:
                continue
            n_samples += len(buffer)
            accumulated_reward += buffer.reward_scale(eps) * len(buffer)
        return accumulated_reward / n_samples

    def update_priority(self, priority):
        """Update the priority of previous samples."""
        self.buffers[self.sampled_task_idx].update_priority(priority)

    def reset_max_priority(self):
        """Recalculate the maximum priority for all tasks."""
        for buffer in self.buffers:
            buffer.reset_max_priority()

    @property
    def environment_terminates(self):
        return any(
            hasattr(buffer, "environment_terminates")
            and buffer.environment_terminates
            for buffer in self.buffers
        )

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)
