import random
from collections import OrderedDict, deque, namedtuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

Transition = namedtuple(
    "Transition",
    ("observation", "action", "reward", "next_observation", "terminated"),
)


class ReplayBuffer:  # TODO remove
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def push(self, *args):
        """Stores the transition."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ReplayBufferJax:
    buffer: deque[tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

    def __init__(self, n_samples):
        self.buffer = deque(maxlen=n_samples)

    def add_samples(self, observation, action, reward, next_observation, done):
        for i in range(len(done)):
            self.buffer.append(
                (
                    observation[i],
                    action[i],
                    reward[i],
                    next_observation[i],
                    done[i],
                )
            )

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        observations = jnp.vstack([self.buffer[i][0] for i in indices])
        actions = jnp.stack([self.buffer[i][1] for i in indices])
        rewards = jnp.hstack([self.buffer[i][2] for i in indices])
        next_observations = jnp.vstack([self.buffer[i][3] for i in indices])
        dones = jnp.hstack([self.buffer[i][4] for i in indices])
        return observations, actions, rewards, next_observations, dones


class ReplayBuffer:
    buffer: OrderedDict[str, np.typing.NDArray[float]]

    def __init__(self, buffer_size: int, keys: list[str] | None = None):
        if keys is None:
            keys = [
                "observation",
                "action",
                "reward",
                "next_observation",
                "termination",
            ]
        self.buffer = OrderedDict()
        for k in keys:
            self.buffer[k] = np.empty(0, dtype=float)
        self.buffer_size = buffer_size
        self.current_len = 0
        self.insert_idx = 0

    def add_sample(self, **sample):
        if self.current_len == 0:
            for k, v in sample.items():
                assert k in self.buffer, f"{k} not in {self.buffer.keys()}"
                self.buffer[k] = np.empty(
                    (self.buffer_size,) + np.asarray(v).shape, dtype=float
                )
        for k, v in sample.items():
            self.buffer[k][self.insert_idx] = v
        self.insert_idx = (self.insert_idx + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> list[jnp.ndarray]:
        indices = rng.integers(0, self.current_len, batch_size)
        return [jnp.asarray(self.buffer[k][indices]) for k in self.buffer]

    def __len__(self):
        return self.current_len
