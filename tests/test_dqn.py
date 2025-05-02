import gymnasium as gym
import optax
from flax import nnx

from rl_blox.algorithm.dqn import MLP, train_dqn
from rl_blox.blox.replay_buffer import ReplayBuffer


def test_q_learning():
    env = gym.make("CartPole-v1")
    seed = 42
    nnx_rngs = nnx.Rngs(seed)

    rb = ReplayBuffer(100)

    q_net = MLP(4, 10, env.action_space.n, nnx_rngs)

    optimizer = nnx.Optimizer(q_net, optax.rprop(0.0003))

    train_dqn(
        q_net,
        env,
        rb,
        optimizer,
        seed=seed,
        total_timesteps=10,
    )

    env.close()
