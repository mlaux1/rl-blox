from modular_rl.policy.value_functions import TabularValueFunction
from gymnasium.spaces.discrete import Discrete


class TestTabularValueFunction:

    def test_creation(self):

        test_discrete = Discrete(3)
        test_instance = TabularValueFunction(test_discrete, 0.1)

        assert test_instance.values.shape == (3,)
