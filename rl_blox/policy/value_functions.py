import abc
import logging

import numpy as np
import numpy.typing as npt
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.utils import flatdim

from ..util import gymtools


class ValueFunction(abc.ABC):
    """Base value function class interface."""

    @abc.abstractmethod
    def get_state_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, observations, values) -> None:
        pass


class TabularValueFunction(ValueFunction):
    """Tabular state value function."""

    def __init__(self, observation_space: Discrete, initial_value: float = 0.0):
        self.values = np.full(
            shape=flatdim(observation_space), fill_value=initial_value
        )

    def get_state_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.values[observation]

    def update(self, observations, step) -> None:
        self.values[observations] += step


class QFunction(abc.ABC):
    """Base Q function class."""

    @abc.abstractmethod
    def get_action_value(
        self, observation: npt.ArrayLike, action: npt.ArrayLike
    ) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, observations, actions, step) -> None:
        pass


class TabularQFunction(QFunction):
    """Tabular action value function."""

    def __init__(
        self,
        observation_space: Discrete,
        action_space: Discrete,
        initial_value: float = 0.0,
    ):
        obs_shape = gymtools.space_shape(observation_space)
        act_shape = (flatdim(action_space),)
        self.values = np.full(
            shape=obs_shape + act_shape, fill_value=initial_value
        )

    def get_action_value(
        self, observation: npt.ArrayLike, action: npt.ArrayLike
    ) -> npt.ArrayLike:
        return self.values[observation, action]

    def update(self, observations, actions, step):
        self.values[observations, actions] += step

        logging.debug(f"Updating Q Table: {observations=}, {actions=}, {step=}")
        logging.debug(f"New Q table: {self.values}")
