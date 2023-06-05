import abc
import numpy.typing as npt


class BaseModel(abc.ABC):
    """Base model class interface."""

    @abc.abstractmethod
    def get_output(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, data) -> None:
        pass
