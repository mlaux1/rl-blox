import abc
import numpy as np
import numpy.typing as npt


class BasePolicy(abc.ABC):
    """Base policy class interface."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        pass


class UniformRandomPolicy(BasePolicy):
    """Simple Random Policy."""

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.action_space.sample()

    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        return 1.0 / self.action_space.n


class GreedyQPolicy(BasePolicy):
    """
    Greedy policy that selects the action that maximises the Q-function.
    """

    def __init__(self, observation_space, action_space, initial_value=0.0):
        super().__init__(observation_space, action_space)

        self.q_table = np.full(shape=(observation_space.n, action_space.n), fill_value=initial_value)

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return np.random.choice(np.flatnonzero(self.q_table[observation] == self.q_table[observation].max()))

    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        return 1.0 / np.count_nonzero(self.q_table[observation] == self.q_table[observation].max())

    def update(self, idx, value) -> None:
        self.q_table[idx] = value
