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
        # x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


@jit
def extract_obs(batch):
    return jnp.stack([t.observation for t in batch])


@jit
def extract_rew(batch):
    return jnp.stack([t.reward for t in batch])


@jit
def extract_act(batch):
    return jnp.stack([t.action for t in batch])


@jit
def extract(batch):
    terminated = jnp.stack([t.terminated for t in batch])
    next_obs = jnp.stack([t.next_observation for t in batch])
    return terminated, next_obs


def critic_loss(q_net, batch, gamma=0.99):
    obs = extract_obs(batch)
    rew = extract_rew(batch)
    act = extract_act(batch)
    terminated, next_obs = extract(batch)

    next_q = q_net(next_obs)
    max_next_q = jnp.max(next_q, axis=1)

    target = jnp.array(rew) + (1 - terminated) * gamma * max_next_q

    pred = q_net(obs)
    pred = pred[jnp.arange(len(pred)), act]

    loss = optax.squared_error(pred, target).mean()

    return loss


@nnx.jit
def train_step(q_net, optimizer, batch):
    grad_fn = nnx.value_and_grad(critic_loss)
    loss, grads = grad_fn(q_net, batch)
    optimizer.update(grads)


def train_dqn(
    q_net: MLP,
    env: gymnasium.Env,
    epsilon: float,
    decay: float,
    buffer_size: int = 3_000,
    batch_size: int = 32,
    total_timesteps: int = 1e4,
    gradient_steps: int = 1,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    tau: float = 0.05,
    seed: int = 1,
):
    """Deep Q-Networks.

    This algorithm is an off-policy value-function based RL algorithm. It uses a
    neural network to approximate the Q-function.
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
            train_step(q_net, optimizer, transition_batch)

        # housekeeping
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        epsilon = epsilon - (0.9 * (step / total_timesteps))

    return q_net
