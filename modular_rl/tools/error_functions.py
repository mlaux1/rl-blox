from jax.typing import ArrayLike
from jax import jit


@jit
def td_error(
        reward: ArrayLike,
        gamma: float,
        value: ArrayLike,
        next_value: ArrayLike):
    return reward + gamma * next_value - value
