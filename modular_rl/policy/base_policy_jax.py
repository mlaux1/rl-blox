import abc

import jax.numpy as jnp
import jax.random
from jax import Array, random

from modular_rl.policy.value_functions import (TabularQFunction,
                                               TabularValueFunction)


class BasePolicy(abc.ABC):
    """Base policy class interface."""

    def __init__(self, observation_space, action_space, key=42):
        self.observation_space = observation_space
        self.action_space = action_space

        self.key = jax.random.PRNGKey(key)

    @abc.abstractmethod
    def get_action(self, observation: Array, key: int) -> Array:
        pass

    @abc.abstractmethod
    def get_action_probability(
            self,
            action: Array,
            observation: Array
    ) -> float:
        pass


class StateValueBasedPolicy(BasePolicy):
    """Base policy class for state-value-based policies."""

    def __init__(self, observation_space, action_space, key):
        super().__init__(observation_space, action_space, key)

        self.value_function = TabularValueFunction(observation_space)


class QValueBasedPolicy(BasePolicy):
    """Base policy class for q-value-based policies."""

    def __init__(self, observation_space, action_space, initial_value=0.0):
        super().__init__(observation_space, action_space)

        self.value_function = TabularQFunction(
            observation_space, action_space, initial_value
        )

    @abc.abstractmethod
    def get_action(self, observation: Array, key: int) -> Array:
        pass

    @abc.abstractmethod
    def get_action_probability(
        self, action: Array, observation: Array
    ) -> float:
        pass

    def update(self, observations, actions, values) -> None:
        self.value_function.update(observations, actions, values)


class UniformRandomPolicy(BasePolicy):
    """Simple Random Policy."""

    def get_action(self, observation: Array, key: int) -> Array:
        return self.action_space.sample()

    def get_action_probability(
        self, action: Array, observation: Array
    ) -> float:
        return 1.0 / self.action_space.n


class GreedyQPolicy(QValueBasedPolicy):
    """
    Greedy policy that selects the action that maximises the Q-function.
    """

    def get_action(self, observation: Array, key: int) -> Array:
        return self.rng.choice(
            jnp.flatnonzero(
                self.value_function.values[observation]
                == self.value_function.values[observation].max()
            )
        )

    def get_action_probability(
        self, action: Array, observation: Array
    ) -> float:
        return 1.0 / jnp.count_nonzero(
            self.value_function.values[observation]
            == self.value_function.values[observation].max()
        )


class EpsilonGreedyPolicy(QValueBasedPolicy):
    """
    Epsilon-Greedy policy that selects the action that maximises the Q-function
    in 1-epsilon probability and performs a random action otherwise.
    """

    def __init__(self, observation_space, action_space, epsilon: float):
        super().__init__(observation_space, action_space)

        self.epsilon = epsilon

    def get_action(
            self,
            observation: Array,
            key: int
    ) -> Array:
        if random.uniform(self.key) < self.epsilon:
            return self.action_space.sample()
        else:
            return random.choice(
                self.key, jnp.flatnonzero(
                    self.value_function.values[observation]
                    == self.value_function.values[observation].max()
                )
            )

    def get_greedy_action(self, observation, key):
        return random.choice(
            key, jnp.flatnonzero(
                self.value_function.values[observation]
                == self.value_function.values[observation].max()
            )
        )

    def get_action_probability(
        self, action: Array, observation: Array
    ) -> float:
        return 1.0 / jnp.count_nonzero(
            self.value_function.values[observation]
            == self.value_function.values[observation].max()
        )
