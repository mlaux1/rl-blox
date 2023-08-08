import abc
import numpy as np
import numpy.typing as npt

from modular_rl.policy.value_functions import TabularValueFunction, TabularQFunction
from numpy.random import default_rng


class BasePolicy(abc.ABC):
    """Base policy class interface."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.rng = default_rng(42)

    @abc.abstractmethod
    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        pass


class StateValueBasedPolicy(BasePolicy):
    """Base policy class for state-value-based policies."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        self.value_function = TabularValueFunction(observation_space)


class QValueBasedPolicy(BasePolicy):
    """Base policy class for q-value-based policies."""

    def __init__(self, observation_space, action_space, initial_value=0.0):
        super().__init__(observation_space, action_space)

        self.value_function = TabularQFunction(observation_space, action_space, initial_value)

    @abc.abstractmethod
    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        pass

    def update(self, observations, actions, values) -> None:
        self.value_function.update(observations, actions, values)


class UniformRandomPolicy(BasePolicy):
    """Simple Random Policy."""

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.action_space.sample()

    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        return 1.0 / self.action_space.n


class GreedyQPolicy(QValueBasedPolicy):
    """
    Greedy policy that selects the action that maximises the Q-function.
    """

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        return self.rng.choice(np.flatnonzero(self.value_function.values[observation] == self.value_function.values[observation].max()))

    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        return 1.0 / np.count_nonzero(self.value_function.values[observation] == self.value_function.values[observation].max())


class EpsilonGreedyPolicy(QValueBasedPolicy):
    """
    Epsilon-Greedy policy that selects the action that maximises the Q-function in 1-epsilon probability and
    performs a random action otherwise.
    """

    def __init__(self, observation_space, action_space, epsilon: float):
        super().__init__(observation_space, action_space)

        self.epsilon = epsilon

    def get_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:

        if default_rng(42).random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.random.choice(np.flatnonzero(self.value_function.values[observation] == self.value_function.values[observation].max()))

    def get_action_probability(self, action: npt.ArrayLike, observation: npt.ArrayLike) -> float:
        return 1.0 / np.count_nonzero(self.value_function.values[observation] == self.value_function.values[observation].max())
