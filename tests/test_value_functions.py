import numpy as np
from gymnasium.spaces.discrete import Discrete

from rl_blox.blox.value_functions import TabularQFunction, TabularValueFunction


class TestValueFunction:
    def test_creation(self):
        test_discrete = Discrete(3)
        test_instance = TabularValueFunction(test_discrete, 0.1)

        assert test_instance.values.shape == (3,)
        assert np.all(test_instance.values == 0.1)

    def test_update(self):
        test_discrete = Discrete(5)
        test_instance = TabularValueFunction(test_discrete)

        test_instance.update(0, 42)

        assert test_instance.values[0] == 42

        test_instance.update([3, 4], [666, -123])
        assert test_instance.values[3] == 666
        assert test_instance.values[4] == -123


class TestQFunction:
    def test_creation(self):
        test_discrete_obs = Discrete(2)
        test_discrete_acs = Discrete(3)
        test_instance = TabularQFunction(
            test_discrete_obs, test_discrete_acs, 0.1
        )

        assert test_instance.values.shape == (2, 3)
        assert np.all(test_instance.values == 0.1)

    def test_update(self):
        test_discrete_obs = Discrete(4)
        test_discrete_acs = Discrete(5)
        test_instance = TabularQFunction(test_discrete_obs, test_discrete_acs)

        test_instance.update(0, 1, 42)

        assert test_instance.values[0, 1] == 42

        test_instance.update([3, 3], [2, 4], [666, -123])
        assert test_instance.values[3, 2] == 666
        assert test_instance.values[3, 4] == -123
