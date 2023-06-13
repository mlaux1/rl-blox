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


class UniformRandomPolicy(BasePolicy):
    """Simple Random Policy."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.action_space.sample()

    def update(self, data) -> None:
        pass
