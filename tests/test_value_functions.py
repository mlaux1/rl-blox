import numpy as np

from modular_rl.policy.value_functions import TabularValueFunction, TabularQFunction
from gymnasium.spaces.discrete import Discrete


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

        test_instance.update([3, 4], [666, 666])
        assert test_instance.values[3] == 666
        assert test_instance.values[4] == 666

class TestQFunction:
    def test_creation(self):

        test_discrete_obs = Discrete(2)
        test_discrete_acs = Discrete(3)
        test_instance = TabularQFunction(test_discrete_obs, test_discrete_acs, 0.1)

        assert test_instance.values.shape == (2, 3)
        assert np.all(test_instance.values == 0.1)

    def test_update(self):
        test_discrete_obs = Discrete(4)
        test_discrete_acs = Discrete(5)
        test_instance = TabularQFunction(test_discrete_obs, test_discrete_acs)

        test_instance.update(0, 1, 42)

        assert test_instance.values[0, 1] == 42

        test_instance.update([3, 3], [2, 4], [666, 666])
        assert test_instance.values[3, 2] == 666
        assert test_instance.values[3, 4] == 666
