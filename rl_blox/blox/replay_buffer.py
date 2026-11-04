import copy
from collections import OrderedDict, namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


def _select_indices(indices, mode):
    """Select the first or last step of subtrajectory indices."""
    if mode == "first":
        return indices[:, 0]
    if mode == "last":
        return indices[:, -1]
    return indices


@partial(jax.jit, static_argnames=("layout", "select"))
def _gather_on_device(data, indices, layout, select):
    fields = []
    for (start, stop, shape, dtype), mode in zip(layout, select, strict=True):
        rows = _select_indices(indices, mode)
        values = data[rows, start:stop].reshape(rows.shape + shape)
        fields.append(values.astype(dtype))
    return fields


@partial(jax.jit, donate_argnums=0)
def _scatter_on_device(data, rows, values):
    return data.at[rows].set(values)


class DeviceCopy:
    """Copy of replay buffer data on the default JAX device.

    All quantities are packed as float32 columns of one array, so that changed
    rows are copied in one transfer. Integers must be smaller than 2**24.

    Parameters
    ----------
    n_rows : int
        Number of rows.

    arrays : list of arrays
        Quantities of a replay buffer, used for their shapes and dtypes.
    """

    def __init__(self, n_rows: int, arrays: list[npt.NDArray]):
        layout = []
        start = 0
        for array in arrays:
            shape = array.shape[1:]
            stop = start + int(np.prod(shape))
            dtype = jax.dtypes.canonicalize_dtype(array.dtype).name
            layout.append((start, stop, shape, dtype))
            start = stop
        self.layout = tuple(layout)
        self.data = jnp.zeros((n_rows, start), dtype=jnp.float32)

    def update(self, rows: npt.NDArray[int], values: list[npt.NDArray]):
        """Copy values of all quantities to rows."""
        packed = np.concatenate(
            [v.reshape(len(rows), -1) for v in values], axis=1
        ).astype(np.float32)
        self.data = _scatter_on_device(self.data, rows, packed)

    def gather(self, indices: npt.NDArray[int], select: tuple[str]):
        """Gather all quantities at indices, see :func:`_select_indices`."""
        return _gather_on_device(
            self.data, jnp.asarray(indices), self.layout, select
        )


class _BatchStorage:
    """Gathers batches from host memory or from a copy on the device."""

    def _init_storage(self, on_device: bool):
        self.on_device = on_device
        self.device_copy = None
        self.changed_rows = []

    def _mark_changed(self, rows: int | list[int]):
        if self.on_device:
            self.changed_rows.append(np.atleast_1d(rows))

    def _gather(self, indices, select=None):
        if select is None:
            select = ("all",) * len(self.buffer)
        return self.Batch(
            **{
                k: self.buffer[k][_select_indices(indices, mode)]
                for k, mode in zip(self.buffer, select, strict=True)
            }
        )

    def _transfer(self, indices, select=None):
        if not self.on_device:
            return jax.device_put(self._gather(indices, select))
        if select is None:
            select = ("all",) * len(self.buffer)
        self._update_device_copy()
        return self.Batch(*self.device_copy.gather(indices, select))

    def _update_device_copy(self):
        arrays = list(self.buffer.values())
        if self.device_copy is None:
            self.device_copy = DeviceCopy(self.buffer_size, arrays)
            rows = np.arange(self.current_len)
        elif self.changed_rows:
            rows = np.unique(np.concatenate(self.changed_rows))
        else:
            return
        self.changed_rows = []
        self.device_copy.update(rows, [a[rows] for a in arrays])

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["Batch"]
        d["device_copy"] = None
        d["changed_rows"] = []
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.Batch = namedtuple("Batch", self.buffer)


class ReplayBuffer(_BatchStorage):
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
        dtype used for each buffer. Defaults to float32 for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    on_device : bool, optional
        Keep a copy of the data on the default JAX device and gather batches
        there. Integers must be smaller than 2**24.
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
        on_device: bool = False,
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
                np.float32,
                int if discrete_actions else np.float32,
                np.float32,
                np.float32,
                int,
            ]
        self.buffer = OrderedDict()
        for k, t in zip(keys, dtypes, strict=True):
            self.buffer[k] = np.empty(0, dtype=t)
        self.Batch = namedtuple("Batch", self.buffer)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0
        self._init_storage(on_device)

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
        self._mark_changed(self.insert_idx)
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
        return self._transfer(*self._sample_indices(batch_size, rng))

    def sample_arrays(self, batch_size: int, rng: np.random.Generator):
        """Like :meth:`sample_batch`, but returns NumPy arrays."""
        return self._gather(*self._sample_indices(batch_size, rng))

    def _sample_indices(self, batch_size: int, rng: np.random.Generator):
        return rng.integers(0, self.current_len, batch_size), None

    def can_sample(self) -> bool:
        """Check if a batch can be sampled."""
        return self.current_len > 0

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len


def dilate_right(a, n):
    """Given a mask 'a' of 0s and 1s, return a mask that is created from 'a'
    by setting all n values to the right of every 1 in 'a' to 1 too. Wrap
    around at the end.

    Examples
    --------
    a=[0, 1, 0, 0, 1, 0]
    - n=0 -> [0, 1, 0, 0, 1, 0]
    - n=1 -> [0, 1, 1, 0, 1, 1]
    - n=2 -> [1, 1, 1, 1, 1, 1]
    """
    for _ in range(n):
        a_shifted_right = np.concatenate([a[-1:], a[:-1]])
        a = np.bitwise_or(a, a_shifted_right)
    return a


class SubtrajectoryReplayBuffer(_BatchStorage):
    """Replay buffer for sampling batches of subtrajectories.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    horizon : int, optional
        Maximum length of the horizon. During sampling, a shorter horizon may
        be specified.

    keys : list[str], optional
        Names of the quantities that should be stored in the replay buffer.
        Defaults to ['observation', 'action', 'reward', 'next_observation',
        'terminated', 'truncated']. These names have to be used as key word
        arguments when adding a sample. When sampling a batch, the arrays will
        be returned in this order. Must contain at least 'observation',
        'next_observation', 'terminated', and 'truncated'.

    dtypes : list[dtype], optional
        dtype used for each buffer. Defaults to float32 for everything except
        'terminated' and 'truncated'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    mrq_mode : bool, optional
        If True, termination and truncation are handled the way the original
        MRQ implementation of this replay buffer did, which may or may not be
        considered incorrect. If False, the following changes are included:
        - No subtrajectories will be generated that include transitions from
          more than one episode. With mrq_mode, this is possible for
          subtrajectories that start close to the end of a terminated episode.
        - There is no delay in added transitions for being eligible to be
          sampled any more. Originally, the most recently added transition
          would in general not be part of the sample_batch() return value.
        - If sample_batch() is called with a horizon shorter than the horizon
          specified on construction, the mask of allowed start indices for
          sample subtrajectories will be extended to the right by the
          difference.

        For MRQ, subtrajectories that start in one episode and end in the next
        are expected and required for best performance, for two reasons:
        1. MRQ benefits from having all transitions up to the last transition of
        an episode as the first transition in a subtrajectory. For the last
        horizon-1 transitions this is achieved by sampling subtrajectories that
        end in next episode.
        2. These trans-episode trajectories do not corrupt the learning process
        because MRQ discounts rewards of transitions following a termination
        transition, effectively ignoring the second episode in two-episode
        subtrajectories.

        Note that if mrq_mode is True, add_sample() returns a list[int].
        If mrq_mode is False, add_sample() returns just an int.

    on_device : bool, optional
        Keep a copy of the data on the default JAX device and gather batches
        there. Integers must be smaller than 2**24.

    Notes
    -----
    This is implemented as a ring buffer of size buffer_size to which
    transition samples (or just samples) will be added at self._insert_idx.
    This self._insert_idx is incremented after each sample addition and wraps
    around at the end, overwriting previously added samples.
    """

    def __init__(
        self,
        buffer_size: int,
        horizon: int = 1,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
        mrq_mode: bool = True,
        on_device: bool = False,
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
                np.float32,
                int if discrete_actions else np.float32,
                np.float32,
                np.float32,
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
        # mask that is 1 at an index if that index in the buffer can be the
        # first index of a subtrajectory to be sampled with sample_batch
        # if that is called with horizon==self.horizon, or 0 otherwise.
        self.mask_ = np.zeros(self.buffer_size, dtype=int)
        self._mrq_mode = mrq_mode
        self._init_storage(on_device)

    def add_sample(self, **sample) -> int | list[int]:
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.

        Returns
        -------
        int | list[int]
            Index in the buffer at which the sample was inserted.
            Note that if mrq_mode was set to True on construction,
            this is a list of indices (multiple samples may be inserted).
        """
        if self._mrq_mode:
            inserted_at = self._add_sample_mrq(**sample)
        else:
            inserted_at = self._add_sample(**sample)
        self._mark_changed(inserted_at)
        return inserted_at

    def _add_sample(self, **sample) -> int:
        """Add transition sample to the replay buffer.

        Notes
        -----
        This is the updated version of _add_sample(). See documentation
        for __init__(): mrq_mode.
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
        if self.episode_timesteps >= self.horizon and not sample["truncated"]:
            # mask in the index that the new subtrajectory starts with
            self.mask_[
                (self.insert_idx - self.horizon + 1) % self.buffer_size
            ] = 1

        inserted_at = self.insert_idx
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size

        if sample["terminated"] or sample["truncated"]:
            self.episode_timesteps = 0

        return inserted_at

    def _add_sample_mrq(self, **sample) -> list[int]:
        """Add transition sample to the replay buffer.

        Notes
        -----
        This is the mrq version of _add_sample(). See documentation
        for __init__(): mrq_mode.
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

            # mask out truncated subtrajectories or mask in the last indices
            # before termination to ensure that these transitions may appear
            # as the first transitions in sampled subtrajectories
            past_idx = (
                self.insert_idx
                - np.arange(min(self.episode_timesteps, self.horizon))
                - 1
            ) % self.buffer_size
            self.mask_[past_idx] = 0 if sample["truncated"] else 1

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
        return self._transfer(
            *self._sample_indices(
                batch_size, horizon, include_intermediate, rng
            )
        )

    def sample_arrays(
        self,
        batch_size: int,
        horizon: int,
        include_intermediate: bool,
        rng: np.random.Generator,
    ):
        """Like :meth:`sample_batch`, but returns NumPy arrays."""
        return self._gather(
            *self._sample_indices(
                batch_size, horizon, include_intermediate, rng
            )
        )

    def _sample_indices(
        self,
        batch_size: int,
        horizon: int,
        include_intermediate: bool,
        rng: np.random.Generator,
    ):
        assert batch_size > 0
        assert horizon > 0

        indices = self._sample_idx(batch_size, horizon, rng)
        # TODO % self.current_len or % self.buffer_size?
        # - maybe self.buffer_size is possible because of the mask?
        indices = (
            indices[:, np.newaxis] + np.arange(horizon)[np.newaxis]
        ) % self.current_len

        if include_intermediate:
            # sample subtrajectories (with horizon dimension) for unrolling
            # dynamics
            return indices, None
        # sample at specific horizon (used for multistep rewards)
        select = tuple(
            (
                "first"
                if k in ["observation", "action"]
                else "last" if k == "next_observation" else "all"
            )
            for k in self.buffer
        )
        return indices, select

    def _mask_for_horizon(self, horizon: int):
        """Get the mask for starting indices to be sampled under the given
        sampling horizon.

        If mrq_mode is True, this is just self.mask_.

        If mrq_mode is False, and the given sampling horizon is the same as
        self.horizon, this is also just self.mask_. If however mrq_mode is
        False and the sampling horizon is shorter, recalculate the mask
        so as to allow for sampling from later starting indices enabled by
        the shorter horizon. This recalculation is performed by dilate_right().

        If, for example, self.horizon is 5, self.mask_[0] is 1 and
        self.mask_[1:] is 0, this means transitions [0,5) may be sampled.
        For a sampling horizon of 5, this means only one subtrajectory may
        be sampled, namely the one starting at index 0. It has length 5 and
        contains all transitions at indices in [0,5). For a sampling horizon
        of 3, however, this means subtrajectories may be sampled starting at
        either 0, 1, or 2, as all these subtrajectories (of length 3) only
        contain transitions in [0,5).
        """
        if self._mrq_mode:
            return self.mask_
        assert horizon >= 1 and horizon <= self.horizon
        if horizon == self.horizon:
            return self.mask_
        difference = self.horizon - horizon
        return dilate_right(self.mask_, difference)

    def _sample_idx(
        self, batch_size: int, horizon: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        mask = self._mask_for_horizon(horizon)
        # Redraw invalid starts instead of searching all valid starts
        indices = rng.integers(0, self.current_len, size=batch_size)
        invalid = np.flatnonzero(mask[indices] == 0)
        for _ in range(10):
            if len(invalid) == 0:
                return indices
            indices[invalid] = rng.integers(0, self.current_len, len(invalid))
            invalid = invalid[mask[indices[invalid]] == 0]
        valid_starts = np.flatnonzero(mask)
        indices[invalid] = rng.choice(valid_starts, len(invalid))
        return indices

    def can_sample(self) -> bool:
        """Check if a subtrajectory can be sampled."""
        return bool(np.any(self.mask_))

    def reward_scale(self, eps: float = 1e-8):
        assert "reward" in self.buffer
        # very hacky way of computing the reward scale...
        return max(
            np.abs(self.buffer["reward"][: self.current_len]).mean(), eps
        )

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len


class SumTree:
    """Binary tree of sums for sampling indices in proportion to values.

    Parameters
    ----------
    capacity : int
        Number of values.
    """

    def __init__(self, capacity: int):
        self.n_leaves = 1
        while self.n_leaves < capacity:
            self.n_leaves *= 2
        self.nodes = np.zeros(2 * self.n_leaves)

    def total(self) -> float:
        """Sum of all values."""
        return self.nodes[1]

    def update(self, indices: npt.ArrayLike, values: npt.ArrayLike):
        """Set values at indices."""
        nodes = np.asarray(indices) + self.n_leaves
        self.nodes[nodes] = values
        # All leaves have the same depth, so the parents reach the root at once
        parents = nodes // 2
        while parents[0] > 0:
            self.nodes[parents] = (
                self.nodes[2 * parents] + self.nodes[2 * parents + 1]
            )
            parents //= 2

    def sample(self, cumulative_values: npt.ArrayLike) -> npt.NDArray[int]:
        """Find the indices at which the cumulative sum of values is reached.

        Parameters
        ----------
        cumulative_values : array, shape (n_samples,)
            Values in [0, total()).

        Returns
        -------
        indices : array, shape (n_samples,)
            Indices of values that are not zero.
        """
        remaining = np.array(cumulative_values, dtype=float)
        nodes = np.ones(len(remaining), dtype=int)
        while nodes[0] < self.n_leaves:
            left = 2 * nodes
            left_sum = self.nodes[left]
            go_right = (remaining >= left_sum) & (self.nodes[left + 1] > 0)
            remaining = np.where(go_right, remaining - left_sum, remaining)
            nodes = left + go_right
        return nodes - self.n_leaves


class PriorityBuffer:
    """Priorities of the samples in a replay buffer.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the buffer.

    mask : array, shape (buffer_size,), optional
        Samples where the mask is 0 are never sampled. Call :meth:`refresh`
        after changing it.
    """

    max_priority: float
    priority: npt.NDArray[float]
    sampled_indices: npt.NDArray[int]

    def __init__(self, buffer_size: int, mask: npt.NDArray[int] | None = None):
        self.max_priority = 1.0
        self.priority = np.zeros(buffer_size)
        self.mask = mask
        self.sum_tree = SumTree(buffer_size)
        self.sampled_indices = np.empty(0, dtype=int)

    def initialize_priority(self, insert_idx: int | list[int]):
        """Initialize the priority of the next inserted sample."""
        self.priority[insert_idx] = self.max_priority
        self.refresh(np.atleast_1d(insert_idx))

    def refresh(self, indices: npt.ArrayLike):
        """Update the sampling weights of indices."""
        weights = self.priority[indices]
        if self.mask is not None:
            weights = weights * self.mask[indices]
        self.sum_tree.update(indices, weights)

    def total(self) -> float:
        """Sum of the sampling weights."""
        return self.sum_tree.total()

    def prioritized_sampling(
        self,
        current_len: int,
        batch_size: int,
        rng: np.random.Generator,
        mask: npt.NDArray[int] | None = None,
    ) -> npt.NDArray[int]:
        """Sample indices based on the priority distribution."""
        if mask is not None and mask is not self.mask:
            priority = self.priority[:current_len] * mask[:current_len]
            probabilities = np.cumsum(priority)
            random_uniforms = (
                rng.uniform(0, 1, size=batch_size) * probabilities[-1]
            )
            self.sampled_indices = np.searchsorted(
                probabilities, random_uniforms
            )
        else:
            random_uniforms = rng.uniform(0, 1, size=batch_size) * self.total()
            self.sampled_indices = self.sum_tree.sample(random_uniforms)
        return self.sampled_indices

    def update_priority(self, priority: npt.ArrayLike):
        """Update the priority of the sampled indices."""
        self.priority[self.sampled_indices] = priority
        self.refresh(self.sampled_indices)
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
        dtype used for each buffer. Defaults to float32 for everything except
        'termination'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    on_device : bool, optional
        See :class:`ReplayBuffer`.

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
        on_device: bool = False,
    ):
        super().__init__(buffer_size, keys, dtypes, discrete_actions, on_device)
        self.priority = PriorityBuffer(buffer_size)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        self.priority.initialize_priority(self.insert_idx)
        super().add_sample(**sample)

    def _sample_indices(self, batch_size: int, rng: np.random.Generator):
        indices = self.priority.prioritized_sampling(
            self.current_len, batch_size, rng
        )
        return indices, None

    def update_priority(self, priority):
        self.priority.update_priority(priority)

    def reset_max_priority(self):
        self.priority.reset_max_priority(self.current_len)


class PrioritizedReplayBuffer(LAP):
    def prioritized_sampling_stratified(
        self, batch_size: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        """Sample one index from each of batch_size equal priority segments."""
        segment = self.priority.total() / batch_size

        # sample one uniform value per segment
        random_points = rng.uniform(
            low=np.arange(batch_size) * segment,
            high=(np.arange(batch_size) + 1) * segment,
            size=batch_size,
        )

        self.priority.sampled_indices = self.priority.sum_tree.sample(
            random_points
        )
        return self.priority.sampled_indices

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator, beta: float = 0.4
    ) -> list[jnp.ndarray]:
        """Sample a batch of transitions along with their importance ratio.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        rng : np.random.Generator
            Random number generator.

        beta: float
            Importance-sampling correction exponent.

        Returns
        -------
        batch : Batch
            Named tuple with order defined by keys. Content is also accessible
            via names, e.g., ``batch.observation``.

        importance_ratio : array
            Importance sampling ratio corresponding to the samples in batch.
        """
        indices = self.prioritized_sampling_stratified(batch_size, rng)
        importance_ratio = self.compute_importance_ratio(indices, beta)
        return self._transfer(indices), jax.device_put(
            importance_ratio / importance_ratio.max()
        )

    def sample_arrays(
        self, batch_size: int, rng: np.random.Generator, beta: float = 0.4
    ):
        """Like :meth:`sample_batch`, but returns NumPy arrays and
        unnormalized importance ratios."""
        # TODO: Using stratified prioritized sampling or the prioritized sampling
        #       implementation in PriorityBuffer class makes almost no difference
        #       in training. Consider removing prioritized_sampling_stratified()
        #       and merging PrioritizedReplayBuffer class with LAP class.
        indices = self.prioritized_sampling_stratified(batch_size, rng)
        # indices = self.priority.prioritized_sampling(
        #     self.current_len, batch_size, rng
        # )
        importance_ratio = self.compute_importance_ratio(indices, beta)
        return self._gather(indices), importance_ratio

    def compute_importance_ratio(
        self, indices: npt.NDArray[int], beta: float
    ) -> npt.NDArray[float]:
        r"""Compute unnormalized importance sampling weights.

        Parameters
        ----------
        indices : array, shape (n_samples,)
            Indices of the sampled transitions.

        beta : float
            Importance-sampling correction exponent.

        Returns
        -------
        weights : array, shape (n_samples,)
            Weights :math:`w_i = (N P(i))^{-\beta}`, where :math:`P(i)` is the
            sampling probability of transition :math:`i` and :math:`N` the
            number of transitions.
        """
        probability = self.priority.priority[indices] / self.priority.total()
        return (self.current_len * probability) ** (-beta)


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
        dtype used for each buffer. Defaults to float32 for everything except
        'terminated' and 'truncated'.

    discrete_actions : bool, optional
        Changes the default dtype for actions to int.

    mrq_mode : bool, optional
        See documentation for SubtrajectoryReplayBuffer.__init__(): mrq_mode.

    on_device : bool, optional
        See :class:`SubtrajectoryReplayBuffer`.
    """

    priority: PriorityBuffer

    def __init__(
        self,
        buffer_size: int,
        horizon: int = 1,
        keys: list[str] | None = None,
        dtypes: list[npt.DTypeLike] | None = None,
        discrete_actions: bool = False,
        mrq_mode: bool = True,
        on_device: bool = False,
    ):
        super().__init__(
            buffer_size,
            horizon,
            keys,
            dtypes,
            discrete_actions,
            mrq_mode,
            on_device,
        )
        self.priority = PriorityBuffer(buffer_size, mask=self.mask_)

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer.

        Note that the individual arguments have to be passed as keyword
        arguments with keys matching the ones passed to the constructor or
        the default keys respectively.
        """
        previous_insert_idx = self.insert_idx
        inserted_at = super().add_sample(**sample)
        self.priority.initialize_priority(inserted_at)
        # Adding a sample changes the mask from horizon steps before it
        n_changed = (self.insert_idx - previous_insert_idx) % self.buffer_size
        changed = (
            previous_insert_idx
            - self.horizon
            + np.arange(n_changed + self.horizon)
        )
        self.priority.refresh(changed % self.buffer_size)

    def _sample_idx(
        self, batch_size: int, horizon: int, rng: np.random.Generator
    ) -> npt.NDArray[int]:
        mask = self._mask_for_horizon(horizon)
        return self.priority.prioritized_sampling(
            self.current_len, batch_size, rng, mask
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


@partial(jax.jit, static_argnames=["alpha"])
def per_priority(
    abs_td_error: jnp.ndarray, alpha: float, epsion: float
) -> jnp.ndarray:
    r"""Compute sample priority for PER.

    Prioritized experience replay (PER) [2]_ has two variants of priority, i.e., proportional
    and rank-based prioritization. This implementation uses the more common proportional variant
    as defined in [1]_:

    .. math::

        p(i)
        =
        |\delta_i|^{\alpha} + \epsilon.

    Parameters
    ----------
    abs_td_error : array
        A batch of :math:`|\delta_i|`.

    alpha : float
        Prioritization exponent :math:`\alpha`.

    epsilon : float
        Small constant :math:`\epsilon` to ensure probability is non-zero.

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
    return abs_td_error**alpha + epsion


# TODO: rank-based prioritization
# @partial(jax.jit, static_argnames=["alpha"])
# def per_rank_priority(
#     abs_td_error: jnp.ndarray, alpha: float
# ) -> jnp.ndarray:
#     return None


class MultiTaskReplayBuffer:
    """Replay buffer for discrete set of tasks.

    For each task we will create a separate replay buffer. Samples are added
    through the view returned by :meth:`task`. Batches are sampled from all
    tasks that have data.

    Parameters
    ----------
    replay_buffer : object
        Replay buffer instance that will be copied for each task.

    n_tasks : int
        Number of tasks.

    task_weighting : str, optional
        Share of each task in a batch. 'uniform': equal shares. 'proportional':
        shares proportional to the number of samples of the task.

    Raises
    ------
    ValueError
        If ``task_weighting`` is unknown or ``replay_buffer`` is a
        :class:`PrioritizedReplayBuffer` with ``on_device``.

    Notes
    -----
    If ``replay_buffer`` has ``on_device``, the data of all tasks is copied to
    one array on the device, from which batches are gathered.
    """

    def __init__(
        self, replay_buffer, n_tasks: int, task_weighting: str = "uniform"
    ):
        if task_weighting not in ("uniform", "proportional"):
            raise ValueError(f"Unknown task weighting: {task_weighting}")
        if replay_buffer.on_device and isinstance(
            replay_buffer, PrioritizedReplayBuffer
        ):
            raise ValueError(
                "on_device is not supported for a PrioritizedReplayBuffer in "
                "a MultiTaskReplayBuffer"
            )
        self.buffers = [replay_buffer]
        for _ in range(n_tasks - 1):
            self.buffers.append(copy.deepcopy(replay_buffer))
        self.task_weighting = task_weighting
        # Tasks that can be sampled from. A task stays sampleable once it is.
        self.sampleable_tasks = set()
        self.sampled_counts = {}
        self.device_copy = None

    def task(self, task_id: int) -> "TaskReplayBuffer":
        """View of this replay buffer that adds samples to one task.

        Parameters
        ----------
        task_id : int
            Index of the task.

        Returns
        -------
        view : TaskReplayBuffer
            View that can be passed to a single-task algorithm.

        Raises
        ------
        ValueError
            If the task index is invalid.
        """
        if not 0 <= task_id < len(self.buffers):
            raise ValueError(
                f"Invalid task id: {task_id}. "
                f"Must be in [0, {len(self.buffers) - 1}]."
            )
        return TaskReplayBuffer(self, task_id)

    def add_sample(self, task_id: int, **sample):
        """Add transition sample to the replay buffer of a task.

        Parameters
        ----------
        task_id : int
            Index of the task.

        **sample
            Transition, see the ``add_sample`` of the task's replay buffer.
        """
        self.buffers[task_id].add_sample(**sample)
        if task_id not in self.sampleable_tasks:
            if self.buffers[task_id].can_sample():
                self.sampleable_tasks.add(task_id)

    def sample_batch(self, batch_size: int, *args, **kwargs):
        """Sample a batch of transitions from all tasks.

        Parameters
        ----------
        batch_size : int
            Size of the sampled batch.

        *args, **kwargs
            Further arguments of the ``sample_batch`` of the task's replay
            buffer, including the random number generator.

        Returns
        -------
        batch
            Batches of the tasks, concatenated along the first axis.
        """
        if self.buffers[0].on_device:
            return self._sample_on_device(batch_size, *args, **kwargs)
        arrays = self.sample_arrays(batch_size, *args, **kwargs)
        if isinstance(self.buffers[0], PrioritizedReplayBuffer):
            batch, importance_ratio = arrays
            arrays = batch, importance_ratio / importance_ratio.max()
        return jax.device_put(arrays)

    def sample_arrays(self, batch_size: int, *args, **kwargs):
        """Like :meth:`sample_batch`, but returns NumPy arrays."""
        self._sample_counts(batch_size, *args, **kwargs)
        batches = [
            self.buffers[task_id].sample_arrays(count, *args, **kwargs)
            for task_id, count in self.sampled_counts.items()
        ]
        # Each task's buffer has its own Batch type, so concatenate the leaves
        fields = zip(*(jax.tree.leaves(batch) for batch in batches))
        return jax.tree.unflatten(
            jax.tree.structure(batches[0]),
            [np.concatenate(field) for field in fields],
        )

    def _sample_on_device(self, batch_size: int, *args, **kwargs):
        self._sample_counts(batch_size, *args, **kwargs)
        buffer_size = self.buffers[0].buffer_size
        indices = []
        for task_id, count in self.sampled_counts.items():
            task_indices, select = self.buffers[task_id]._sample_indices(
                count, *args, **kwargs
            )
            indices.append(task_indices + task_id * buffer_size)
        self._update_device_copy()
        first_task = self.buffers[next(iter(self.sampled_counts))]
        if select is None:
            select = ("all",) * len(first_task.buffer)
        return first_task.Batch(
            *self.device_copy.gather(np.concatenate(indices), select)
        )

    def _update_device_copy(self):
        buffer_size = self.buffers[0].buffer_size
        if self.device_copy is None:
            arrays = list(
                self.buffers[min(self.sampleable_tasks)].buffer.values()
            )
            self.device_copy = DeviceCopy(
                len(self.buffers) * buffer_size, arrays
            )
            for buffer in self.buffers:
                buffer.changed_rows = [np.arange(buffer.current_len)]
        rows = []
        values = []
        for task_id, buffer in enumerate(self.buffers):
            if not buffer.changed_rows:
                continue
            task_rows = np.unique(np.concatenate(buffer.changed_rows))
            buffer.changed_rows = []
            rows.append(task_rows + task_id * buffer_size)
            values.append([a[task_rows] for a in buffer.buffer.values()])
        if rows:
            self.device_copy.update(
                np.concatenate(rows),
                [np.concatenate(field) for field in zip(*values)],
            )

    def _sample_counts(self, batch_size: int, *args, **kwargs):
        rng = kwargs.get("rng")
        if rng is None:
            rng = next(a for a in args if isinstance(a, np.random.Generator))

        tasks = sorted(self.sampleable_tasks)
        if self.task_weighting == "uniform":
            counts = np.full(len(tasks), batch_size // len(tasks))
            remainder = batch_size % len(tasks)
            counts[rng.choice(len(tasks), remainder, replace=False)] += 1
        else:
            n_samples = np.array([len(self.buffers[t]) for t in tasks])
            counts = rng.multinomial(batch_size, n_samples / n_samples.sum())
        self.sampled_counts = {
            task_id: int(count)
            for task_id, count in zip(tasks, counts, strict=True)
            if count > 0
        }

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
        """Update the priority of the samples of the last batch."""
        boundaries = np.cumsum(list(self.sampled_counts.values()))[:-1]
        task_priorities = np.split(np.asarray(priority), boundaries)
        for task_id, task_priority in zip(
            self.sampled_counts, task_priorities, strict=True
        ):
            self.buffers[task_id].update_priority(task_priority)

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


class TaskReplayBuffer:
    """View of a :class:`MultiTaskReplayBuffer` for one task.

    Samples are added to the replay buffer of the task. All other methods
    use the whole multi-task replay buffer.

    Parameters
    ----------
    replay_buffer : MultiTaskReplayBuffer
        Multi-task replay buffer.

    task_id : int
        Index of the task.
    """

    def __init__(self, replay_buffer: MultiTaskReplayBuffer, task_id: int):
        self.replay_buffer = replay_buffer
        self.task_id = task_id

    def add_sample(self, **sample):
        """Add transition sample to the replay buffer of the task."""
        self.replay_buffer.add_sample(self.task_id, **sample)

    def sample_batch(self, *args, **kwargs):
        """Sample a batch of transitions from all tasks."""
        return self.replay_buffer.sample_batch(*args, **kwargs)

    def reward_scale(self, eps: float = 1e-8):
        """Compute the reward scale for all tasks."""
        return self.replay_buffer.reward_scale(eps)

    def update_priority(self, priority):
        """Update the priority of previous samples."""
        self.replay_buffer.update_priority(priority)

    def reset_max_priority(self):
        """Recalculate the maximum priority for all tasks."""
        self.replay_buffer.reset_max_priority()

    @property
    def environment_terminates(self):
        return self.replay_buffer.environment_terminates

    def __len__(self):
        return len(self.replay_buffer)
