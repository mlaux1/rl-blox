from jax import jit
from jax.typing import ArrayLike


@jit
def td_error(
    reward: ArrayLike, gamma: float, value: ArrayLike, next_value: ArrayLike
):
    return reward + gamma * next_value - value
