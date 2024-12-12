import jax
import flax
import flax.linen as nn
import optax

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
    
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    
    key, q_key = jax.random.split(key, 2)
    q_state = TargetTrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs, action_space.sample()),
        target_params=q_network.init(q_key, obs, action_space.sample()),
        tx=optax.adam(learning_rate=learning_rate),
    )  


    # initialise episode
    # for each step:
    # select epsilon greedy action
    # execute action
    # store transition in replay buffer
    # sample minibatch from replay buffer
    # perform gradient descent step based on minibatch

    raise NotImplemented
