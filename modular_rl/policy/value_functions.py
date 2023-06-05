import abc

import numpy as np
import numpy.typing as npt
from gymnasium.spaces.discrete import Discrete


class ValueFunction(abc.ABC):
    """Base value function class interface."""

    @abc.abstractmethod
    def get_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, data) -> None:
        pass


class TabularValueFunction(ValueFunction):
    """Tabular state value function."""

    def __init__(self, observation_space: Discrete, initial_value: float = 0.0):

        self.values = np.full(shape=observation_space.n, fill_value=initial_value)

    def get_value(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.values[observation]

    def update(self, data) -> None:
        pass

