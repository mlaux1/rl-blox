from collections import namedtuple

import gymnasium as gym
import numpy as np

Transitions = namedtuple(
    "Transitions",
    [
        "observation",
        "action",
        "reward",
        "next_observation",
        "termination",
        "truncation",
    ],
)
"""Transitions in the order in which they were collected, with the number of
steps as first dimension. At the end of an episode, ``next_observation`` is
the last observation of the episode, not the observation after a reset."""


class RecordTransitions(gym.Wrapper):
    """Record all transitions of an environment.

    Use this wrapper when another function interacts with the environment,
    e.g., the training function of a backbone algorithm.

    Parameters
    ----------
    env : gym.Env
        Environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._observation = None
        self._samples = []

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = observation
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._samples.append(
            (
                self._observation,
                action,
                reward,
                observation,
                terminated,
                truncated,
            )
        )
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def transitions(self) -> Transitions:
        """Return the recorded transitions.

        Returns
        -------
        transitions : Transitions
            Recorded transitions as arrays.
        """
        return Transitions(
            *(np.asarray(field) for field in zip(*self._samples))
        )


class RecordSuccess(gym.Wrapper):
    """Record whether the completed episodes of an environment succeeded.

    An episode succeeded if ``info["success"]`` was set at any of its steps.
    Use this wrapper when another function interacts with the environment,
    e.g., the training function of a backbone algorithm.

    Parameters
    ----------
    env : gym.Env
        Environment that reports ``info["success"]`` at every step.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._success = False
        self._successes = []

    def reset(self, *, seed=None, options=None):
        self._success = False
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "success" not in info:
            raise KeyError("The environment does not report info['success'].")
        self._success = self._success or bool(info["success"])
        if terminated or truncated:
            self._successes.append(self._success)
        return observation, reward, terminated, truncated, info

    def successes(self) -> np.ndarray:
        """Return the success flags of the completed episodes.

        Returns
        -------
        successes : array, shape (n_episodes,)
            True for each completed episode that succeeded.
        """
        return np.asarray(self._successes, dtype=bool)


def pad_transitions(
    transitions: Transitions, size: int | None = None
) -> tuple[Transitions, np.ndarray]:
    """Pad transitions with zeros to a fixed number of steps.

    Jitted functions are compiled again for each new number of steps. Padding
    to a few sizes avoids this.

    Parameters
    ----------
    transitions : Transitions
        Transitions.

    size : int, optional
        Number of steps after padding. Defaults to the next power of two.

    Returns
    -------
    padded : Transitions
        Padded transitions.

    mask : array, shape (size,)
        True for the original steps, False for padding.

    Raises
    ------
    ValueError
        If ``size`` is smaller than the number of steps.
    """
    n_steps = len(transitions.reward)
    if size is None:
        size = 1 << (n_steps - 1).bit_length()
    if size < n_steps:
        raise ValueError(
            f"size ({size}) is smaller than the number of steps ({n_steps})"
        )

    padded = Transitions(
        *(
            np.pad(field, [(0, size - n_steps)] + [(0, 0)] * (field.ndim - 1))
            for field in transitions
        )
    )
    mask = np.arange(size) < n_steps
    return padded, mask
