from functools import partial
from typing import List, Tuple

import flax
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import jit, vmap
from tqdm import tqdm

from ...policy.replay_buffer import ReplayBuffer


class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dhidden, rngs=rngs)
        self.linear3 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


@jit
def _extract(batch):
    observation = jnp.stack([t.observation for t in batch])
    reward = jnp.stack([t.reward for t in batch])
    action = jnp.stack([t.action for t in batch])
    terminated = jnp.stack([t.terminated for t in batch])
    next_obs = jnp.stack([t.next_observation for t in batch])
    return observation, reward, action, terminated, next_obs


@nnx.jit
def _critic_loss(q_net, batch, gamma=0.99):
    obs, reward, action, terminated, next_obs = _extract(batch)

    next_q = q_net(next_obs)
    max_next_q = jnp.max(next_q, axis=1)

    target = jnp.array(reward) + (1 - terminated) * gamma * max_next_q

    pred = q_net(obs)
    pred = pred[jnp.arange(len(pred)), action]

    loss = optax.squared_error(pred, target).mean()

    return loss


@nnx.jit
def _train_step(q_net, optimizer, batch):
    grad_fn = nnx.value_and_grad(_critic_loss)
    loss, grads = grad_fn(q_net, batch)
    optimizer.update(grads)


def train_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    buffer_size: int = 3_000,
    batch_size: int = 32,
    total_timesteps: int = 1e4,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    seed: int = 1,
):
    """Deep Q Learning with Experience Replay

    Implements the most basic version of DQN with experience replay as described
    in Mnih et al. (2013), which is an off-policy value-based RL algorithm. It
    uses a neural network to approximate the Q-function and samples minibatches
    from the replay buffer to calculate updates.

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
    buffer_size : int
        The maximum size of the replay buffer.
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
    """

    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "DQN only supports discrete action spaces"

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # initialise optimiser
    optimizer = nnx.Optimizer(q_net, optax.adam(learning_rate))

    # initialise episode
    obs, _ = env.reset(seed=seed)

    rb = ReplayBuffer(buffer_size)

    # for each step:
    for step in tqdm(range(total_timesteps)):
        # select epsilon greedy action
        key, subkey = jax.random.split(key)
        roll = jax.random.uniform(subkey)
        if roll < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = q_net([obs])
            action = jnp.argmax(q_vals)

        # execute action
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        # store transition in replay buffer
        rb.push(obs, action, reward, next_obs, terminated)

        # sample minibatch from replay buffer
        if step > batch_size:
            transition_batch = rb.sample(batch_size)

            # perform gradient descent step based on minibatch
            _train_step(q_net, optimizer, transition_batch)

        # housekeeping
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        epsilon = epsilon - (0.9 * (step / (total_timesteps / 10.0)))

    return q_net
