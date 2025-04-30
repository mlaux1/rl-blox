import gymnasium as gym
from gymnasium.spaces.utils import flatdim


def space_shape(space):
    """Get shape for value function table from space."""
    if isinstance(  # this is only one simple hack so far, but... YAGNI!
        space, gym.spaces.Tuple
    ) and isinstance(space[0], gym.spaces.Discrete):
        return tuple([space.n for space in space])
    return (flatdim(space),)
