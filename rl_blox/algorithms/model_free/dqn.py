import flax
import flax.linen as nn

from ..policy.replay_buffer import ReplayBuffer
from .ddpg import MlpQNetwork


def dqn(
    q_network: nn.Module,
    epsilon: float,
    buffer_size: int = 1e6,
    batch_size: int = 64,
    totral_time_steps: int = 1e4,
    gradient_steps: int = 1,
    seed: int = 1,
):
    """Deep Q-Networks.

    This algorithm is an off-policy value-function based RL algorithm. It uses a
    neural network to approximate the Q-function.
    """
    # initialise episode
    # for each step:
    # select epsilon greedy action
    # execute action
    # store transition in replay buffer
    # sample minibatch from replay buffer
    # perform gradient descent step based on minibatch

    raise NotImplemented
