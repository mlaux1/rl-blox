from collections import namedtuple, OrderedDict
from typing import Type

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
    Batch: Type
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
        self.Batch = namedtuple("Batch", keys)
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
        """
        indices = rng.integers(0, self.current_len, batch_size)
        return self.Batch(**{k: jnp.asarray(self.buffer[k][indices]) for k in self.buffer})

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len
