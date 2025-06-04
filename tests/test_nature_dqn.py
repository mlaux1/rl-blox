import gymnasium as gym
import optax
from flax import nnx

from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer


def test_nature_dqn():
    env = gym.make("CartPole-v1")
    seed = 42

    rb = ReplayBuffer(100)

    q_net = MLP(
        env.observation_space.shape[0],
        int(env.action_space.n),
        [10],
        "relu",
        nnx.Rngs(seed),
    )

    optimizer = nnx.Optimizer(q_net, optax.rprop(0.0003))

    train_nature_dqn(
        q_net,
        env,
        rb,
        optimizer,
        seed=seed,
        total_timesteps=10,
    )

    env.close()
