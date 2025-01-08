from functools import partial
from typing import List, Tuple

import flax
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

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


def critic_loss(q_net, batch, gamma=0.9999):
    obs, action, reward, next_obs, terminated = batch[0]

    target = reward
    if not terminated:
        target += gamma * jnp.max(q_net([next_obs]))

    pred = q_net([obs])[action]
    loss = optax.squared_error(pred, target)

    return loss


def train_dqn(
    env: gymnasium.Env,
    epsilon: float,
    buffer_size: int = 1_000_000,
    batch_size: int = 1,
    total_timesteps: int = 1e4,
    gradient_steps: int = 1,
    learning_rate: float = 1e-4,
    gamma: float = 0.9999,
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
    nnx_rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    q_net = MLP(1, 64, 4, nnx_rngs)

    # initialise optimiser
    optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate, 0.9))

    # initialise episode
    obs, _ = env.reset(seed=seed)

    rb = ReplayBuffer(buffer_size)

    # for each step:
    for _ in range(total_timesteps):
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
        transition_batch = rb.sample(batch_size)

        # perform gradient descent step based on minibatch

        grad_fn = nnx.value_and_grad(critic_loss)
        loss, grads = grad_fn(q_net, transition_batch, gamma)

        optimizer.update(grads)

        # housekeeping
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return q_net
