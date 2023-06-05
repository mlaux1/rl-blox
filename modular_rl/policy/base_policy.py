import abc
import numpy.typing as npt


class BasePolicy(abc.ABC):
    """Base policy class interface."""

    @abc.abstractmethod
    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, data) -> None:
        pass

class ValueBasedPolicy(BasePolicy):
    """Simple Value Based Policy."""

    def __init__(self, value):
        self.value_function = value_function

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
