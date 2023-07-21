import abc

import numpy as np
import numpy.typing as npt
from gymnasium.spaces.discrete import Discrete


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
        self.values = np.full(shape=observation_space.n, fill_value=initial_value)

    def get_state_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.values[observation]

    def update(self, observations, step) -> None:
        self.values[observations] += step


class QFunction(abc.ABC):
    """Base Q function class."""

    @abc.abstractmethod
    def get_action_value(self, observation: npt.ArrayLike, action: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, observations, actions, step) -> None:
        pass


class TabularQFunction(QFunction):
    """Tabular action value function."""

    def __init__(self, observation_space: Discrete, action_space: Discrete, initial_value: float = 0.0):
        self.values = np.full(shape=(observation_space.n, action_space.n), fill_value=initial_value)

    def get_action_value(self, observation: npt.ArrayLike, action: npt.ArrayLike) -> npt.ArrayLike:
        return self.values[observation, action]

    def update(self, observations, actions, step) -> None:
        self.values[observations, actions] += step


