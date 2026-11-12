import jax
import optax
from flax import nnx
from numpy.testing import assert_array_equal

from rl_blox.algorithm.per import train_ddqn_per
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import PrioritizedReplayBuffer


def test_ddqn_per_same_seed_gives_same_result(cart_pole_env):
    def train():
        q_net = MLP(4, 2, [10], "relu", nnx.Rngs(0))
        optimizer = nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param)
        train_ddqn_per(
            q_net,
            cart_pole_env,
            PrioritizedReplayBuffer(100, discrete_actions=True),
            optimizer,
            batch_size=8,
            total_timesteps=100,
            seed=1,
            progress_bar=False,
        )
        return jax.tree.leaves(nnx.state(q_net, nnx.Param))

    for a, b in zip(train(), train(), strict=True):
        assert_array_equal(a, b)
