import abc
import random
from collections import deque, namedtuple

import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


class BaseModel(abc.ABC):
    """Base model class interface."""

    @abc.abstractmethod
    def get_output(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def update(self, data) -> None:
        pass


Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation")
)


class ReplayBuffer(object):
    def __init__(self, size: int):
        self.buffer = deque([], maxlen=size)

    def push(self, *args):
        """Stores the transition."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


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
