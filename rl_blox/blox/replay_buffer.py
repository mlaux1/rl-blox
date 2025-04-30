import random
from collections import deque, namedtuple
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

Transition = namedtuple(
    "Transition",
    ("observation", "action", "reward", "next_observation", "terminated"),
)


class ReplayBuffer:
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
    buffer: deque[Tuple[ArrayLike, ArrayLike, float, ArrayLike, bool]]

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = rng.integers(0, len(self.buffer), batch_size)
        observations = jnp.vstack([self.buffer[i][0] for i in indices])
        actions = jnp.stack([self.buffer[i][1] for i in indices])
        rewards = jnp.hstack([self.buffer[i][2] for i in indices])
        next_observations = jnp.vstack([self.buffer[i][3] for i in indices])
        dones = jnp.hstack([self.buffer[i][4] for i in indices])
        return observations, actions, rewards, next_observations, dones
