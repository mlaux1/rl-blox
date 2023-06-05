import numpy as np
import numpy.typing as npt

from modular_rl.policy.base_model import BaseModel


class TabularModel(BaseModel):

    def __init__(self, env, init_value: float = 0.0):

        self.table = np.full(shape=(env.observation_space.n, env.action_space.n), fill_value=init_value)

    def get_output(self, inputs: npt.ArrayLike) -> npt.ArrayLike:
        return self.table[inputs]

    def update(self, data) -> None:
        pass

