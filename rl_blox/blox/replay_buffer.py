from collections import OrderedDict

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


def discounted_reward_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """Computes the discounted return for each step.

    Parameters
    ----------
    rewards : list
        Rewards of one episode.

    gamma : float
        Discount factor.

    Returns
    -------
    discounted_returns : array
        Discounted return until the end of the episode.
    """
    discounted_returns = []
    accumulated_return = 0.0
    for r in reversed(rewards):
        accumulated_return *= gamma
        accumulated_return += r
        discounted_returns.append(accumulated_return)
    return np.array(list(reversed(discounted_returns)))


class EpisodeBuffer:
    """Collects samples batched in episodes."""

    episodes: list[list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]]]

    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self.episodes.append([])

    def add_sample(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        next_observation: jnp.ndarray,
        reward: float,
    ):
        assert len(self.episodes) > 0
        self.episodes[-1].append(
            (observation, action, next_observation, reward)
        )

    def _indices(self) -> list[int]:
        indices = []
        for episode in self.episodes:
            indices.extend([t for t in range(len(episode))])
        return indices

    def _observations(self) -> list:
        observations = []
        for episode in self.episodes:
            observations.extend([o for o, _, _, _ in episode])
        return observations

    def _actions(self) -> list:
        actions = []
        for episode in self.episodes:
            actions.extend([a for _, a, _, _ in episode])
        return actions

    def _nest_observations(self) -> list:
        next_observations = []
        for episode in self.episodes:
            next_observations.extend([s for _, _, s, _ in episode])
        return next_observations

    def _rewards(self) -> list[list[float]]:
        rewards = []
        for episode in self.episodes:
            rewards.append([r for _, _, _, r in episode])
        return rewards

    def __len__(self) -> int:
        return sum(map(len, self.episodes))

    def average_return(self) -> float:
        return sum(
            [sum([r for _, _, _, r in episode]) for episode in self.episodes]
        ) / len(self.episodes)

    def prepare_policy_gradient_dataset(
        self, action_space: gym.spaces.Space, gamma: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        observations = jnp.array(self._observations())
        actions = jnp.array(self._actions())
        next_observations = jnp.array(self._nest_observations())
        if isinstance(action_space, gym.spaces.Discrete):
            actions -= action_space.start
        returns = jnp.hstack(
            [discounted_reward_to_go(R, gamma) for R in self._rewards()]
        )
        gamma_discount = gamma ** jnp.hstack(self._indices())
        return observations, actions, next_observations, returns, gamma_discount


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
    ) -> list[jnp.ndarray]:
        """Sample a batch of transitions.

        Note that the individual quantities will be returned in the same order
        as the keys were given to the constructor or the default order
        respectively.
        """
        indices = rng.integers(0, self.current_len, batch_size)
        return [jnp.asarray(self.buffer[k][indices]) for k in self.buffer]

    def __len__(self):
        """Return current number of stored transitions in the replay buffer."""
        return self.current_len
