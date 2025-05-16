import gymnasium
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.typing import ArrayLike
from tqdm import tqdm

from ..blox.function_approximator.mlp import MLP
from ..blox.replay_buffer import ReplayBuffer, Transition
from ..blox.schedules import linear_schedule
from .dqn import _extract, greedy_policy


@nnx.jit
def critic_loss(
    q_net: MLP,
    q_target: MLP,
    batch: list[Transition],
    gamma: float = 0.99,
) -> float:
    """Calculates the loss of the given Q-net for a given minibatch of
    transitions.

    Parameters
    ----------
    q_net : MLP
        The Q-network to compute the loss for.
    q_target : MLP
        The target Q-Network.
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

    next_q = q_target(next_obs)
    max_next_q = jnp.max(next_q, axis=1)

    target = jnp.array(reward) + (1 - terminated) * gamma * max_next_q

    pred = q_net(obs)
    pred = pred[jnp.arange(len(pred)), action]

    loss = optax.squared_error(pred, target).mean()

    return loss


@nnx.jit
def _train_step(
    q_net: MLP,
    q_target: MLP,
    optimizer: nnx.Optimizer,
    batch: ArrayLike,
    gamma: float = 0.99,
) -> None:
    """Performs a single training step to optimise the Q-network.

    Parameters
    ----------
    q_net : MLP
        The MLP to be updated.
    q_target : MLP
        The target Q-Network.
    optimizer : nnx.Optimizer
        The optimizer to be used.
    batch : ArrayLike
        The minibatch of transitions to compute the update from.
    gamma : float, optional
        The discount factor.
    """
    grad_fn = nnx.value_and_grad(critic_loss)
    loss, grads = grad_fn(q_net, q_target, batch, gamma)
    optimizer.update(grads)


def train_nature_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    replay_buffer: ReplayBuffer,
    optimizer: nnx.Optimizer,
    batch_size: int = 64,
    total_timesteps: int = 1e4,
    gamma: float = 0.99,
    update_frequency: int = 4,
    target_update_frequency: int = 1000,
    q_target_net: MLP | None = None,
    seed: int = 1,
) -> tuple[MLP, nnx.Optimizer]:
    """Deep Q Learning with Experience Replay

    Implements the most common version of DQN with experience replay as described
    in Mnih et al. (2015) [1]_, which is an off-policy value-based RL algorithm.
    It uses a neural network to approximate the Q-function and samples
    minibatches from the replay buffer to calculate updates as well as target
    networks that are copied regularly from the current Q-network.

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
        The environment to train the Q-network on.
    replay_buffer : ReplayBuffer
        The replay buffer used for storing collected transitions.
    optimizer : nnx.Optimizer
        The optimiser for the Q-Network.
    update_frequency : int, optional
        The number of time steps after which the Q-net is updated.
    target_update_frequency : int, optional
        The number of time steps after which the target net is updated.
    total_timesteps : int
        The number of environment sets to train for.
    gamma : float
        The discount factor.
    q_target_net : MLP, optional
        The target Q-network. Only needed when continuing prior training.
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
    [1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control
    through deep reinforcement learning. Nature 518, 529–533 (2015).
    https://doi.org/10.1038/nature14236
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    key = jax.random.key(seed)

    # intialise the target network
    if q_target_net is None:
        q_target_net = nnx.clone(q_net)

    # initialise episode
    obs, _ = env.reset(seed=seed)

    epsilon = linear_schedule(total_timesteps)

    key, subkey = jax.random.split(key)
    epsilon_rolls = jax.random.uniform(subkey, (total_timesteps,))

    for step in tqdm(range(total_timesteps)):
        if epsilon_rolls[step] < epsilon[step]:
            action = env.action_space.sample()
        else:
            action = greedy_policy(q_net, obs)

        next_obs, reward, terminated, truncated, info = env.step(int(action))
        replay_buffer.push(obs, action, reward, next_obs, terminated)

        if step > batch_size:
            if step % update_frequency == 0:
                transition_batch = replay_buffer.sample(batch_size)
                _train_step(
                    q_net, q_target_net, optimizer, transition_batch, gamma
                )

            if step % target_update_frequency == 0:
                q_net = q_target_net
                q_target_net = nnx.clone(q_net)

        # housekeeping
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return q_net, optimizer
