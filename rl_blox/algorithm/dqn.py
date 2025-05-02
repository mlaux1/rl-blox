import gymnasium
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.typing import ArrayLike
from tqdm import tqdm

from ..blox.replay_buffer import ReplayBuffer, Transition


class MLP(nnx.Module):
    """Basic Multi-layer Perceptron with two hidden layers."""

    def __init__(self, din, dhidden, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dhidden, rngs=rngs)
        self.linear3 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def linear_schedule(
    total_timesteps: int,
    start: float = 1.0,
    end: float = 0.1,
    fraction: float = 0.1,
) -> jnp.ndarray:
    transition_steps = int(
        total_timesteps * fraction
    )  # Number of steps for decay
    schedule = jnp.ones(total_timesteps) * end  # Default value after decay

    schedule.at[:transition_steps].set(
        jnp.linspace(start, end, transition_steps)
    )

    return schedule


@jax.jit
def _extract(
    batch: list,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Extracts the arrays of the given list of transitions.

    Parameters
    ----------
    batch : list[Transition]
        The batch of transitions

    Returns
    -------
    observation : ArrayLike
        All observations of the given batch as a stacked array.
    reward : ArrayLike
        All rewards of the given batch as a stacked array.
    action : ArrayLike
        All actions of the given batch as a stacked array.
    terminated : ArrayLike
        All terminations of the given batch as a stacked array.
    next_observation : ArrayLike
        All next_observations of the given batch as a stacked array.

    """
    observation = jnp.stack([t.observation for t in batch])
    reward = jnp.stack([t.reward for t in batch])
    action = jnp.stack([t.action for t in batch])
    terminated = jnp.stack([t.terminated for t in batch])
    next_obs = jnp.stack([t.next_observation for t in batch])
    return observation, reward, action, terminated, next_obs


@nnx.jit
def critic_loss(
    q_net: MLP,
    batch: list[Transition],
    gamma: float = 0.99,
) -> float:
    """Calculates the loss of the given Q-net for a given minibatch of
    transitions.

    Parameters
    ----------
    q_net : MLP
        The Q-network to compute the loss for.
    batch : list[Transition]
        The minibatch of transitions.
    gamma : float, default=0.99
        The discount factor.

    Returns
    -------
    loss : float
        The computed loss for the given minibatch.
    """
    obs, reward, action, terminated, next_obs = _extract(batch)

    next_q = q_net(next_obs)
    max_next_q = jnp.max(next_q, axis=1)

    target = jnp.array(reward) + (1 - terminated) * gamma * max_next_q

    pred = q_net(obs)
    pred = pred[jnp.arange(len(pred)), action]

    loss = optax.squared_error(pred, target).mean()

    return loss


@nnx.jit
def _train_step(
    q_net: MLP,
    optimizer: nnx.Optimizer,
    batch: ArrayLike,
) -> None:
    """Performs a single training step to optimise the Q-network.

    Parameters
    ----------
    q_net : MLP
        The MLP to be updated.
    optimizer : nnx.Optimizer
        The optimizer to be used.
    batch :
        The minibatch of transitions to compute the update from.
    """
    grad_fn = nnx.value_and_grad(critic_loss)
    loss, grads = grad_fn(q_net, batch)
    optimizer.update(grads)


@nnx.jit
def greedy_policy(
    q_net: MLP,
    obs: ArrayLike,
) -> int:
    """Greedy policy.

    Selects the greedy action for a given observation based on the given
    Q-Network by choosing the action that maximises the Q-Value.

    Parameters
    ----------
    q_net : MLP
        The Q-Network to be used for greedy action selection.
    obs : ArrayLike
        The observation for which to select an action.

    Returns
    -------
    action : int
        The selected greedy action.

    """
    q_vals = q_net([obs])
    return jnp.argmax(q_vals)


def train_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    replay_buffer: ReplayBuffer,
    optimizer: nnx.Optimizer,
    batch_size: int = 32,
    total_timesteps: int = 1e4,
    gamma: float = 0.99,
    seed: int = 1,
) -> tuple[MLP, nnx.Optimizer]:
    """Deep Q Learning with Experience Replay

    Implements the most basic version of DQN with experience replay as described
    in Mnih et al. (2013) [1]_, which is an off-policy value-based RL algorithm.
    It uses a neural network to approximate the Q-function and samples
    minibatches from the replay buffer to calculate updates.

    This implementation aims to be as close as possible to the original algorithm
    described in the paper while remaining not overly engineered towards a
    specific environment. For example, this implementation uses the same linear
    schedule to decrease epsilon from 1.0 to 0.1 over the first ten percent of
    training steps, but does not impose any architecture on the used Q-net or
    requires a specific preprocessing of observations as is done in the original
    paper to solve the Atari use case.

    Parameters
    ----------
    q_net : MLP
        The Q-network to be optimised.
    env: gymnasium
        The envrionment to train the Q-network on.
    replay_buffer : ReplayBuffer
        The replay buffer used for storing collected transitions.
    optimizer : nnx.Optimizer
        The optimiser for the Q-Network.
    total_timesteps : int
        The number of environment sets to train for.
    learning_rate : float
        The learning rate for updating the weights of the Q-net.
    gamma : float
        The discount factor.
    seed : int
        The random seed, which can be set to reproduce results.

    Returns
    -------
    q_net : MLP
        The trained Q-network.
    optimizer : nnx.Optimizer
        The Q-net optimiser.

    References
    ----------
    .. [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I.,
       Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep
       reinforcement learning. arXiv preprint arXiv:1312.5602.
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    key = jax.random.key(seed)

    # initialise episode
    obs, _ = env.reset(seed=seed)

    epsilon = linear_schedule(total_timesteps)

    # for each step:
    for step in tqdm(range(total_timesteps)):
        key, subkey = jax.random.split(key)
        roll = jax.random.uniform(subkey)
        if roll < epsilon[step]:
            action = env.action_space.sample()
        else:
            action = greedy_policy(q_net, obs)

        next_obs, reward, terminated, truncated, info = env.step(int(action))
        replay_buffer.push(obs, action, reward, next_obs, terminated)

        # sample minibatch from replay buffer
        if step > batch_size:
            transition_batch = replay_buffer.sample(batch_size)

            _train_step(q_net, optimizer, transition_batch)

        # housekeeping
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return q_net, optimizer
