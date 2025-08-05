import optax
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer


def test_ddqn(cart_pole_env):
    seed = 42

    rb = ReplayBuffer(100)

    q_net = MLP(
        cart_pole_env.observation_space.shape[0],
        int(cart_pole_env.action_space.n),
        [10],
        "relu",
        nnx.Rngs(seed),
    )

    optimizer = nnx.Optimizer(q_net, optax.rprop(0.0003), wrt=nnx.Param)

    train_ddqn(
        q_net,
        cart_pole_env,
        rb,
        optimizer,
        seed=seed,
        total_timesteps=10,
    )
