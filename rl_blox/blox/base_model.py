import abc
import random

import numpy.typing as npt


class BaseModel(abc.ABC):
    """Base model class interface."""

    @abc.abstractmethod
    def get_output(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, data) -> None:
        pass


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
