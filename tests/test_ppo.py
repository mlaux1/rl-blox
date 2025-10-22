import gymnasium as gym
import jax
import optax
from flax import nnx

from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import SoftmaxPolicy


def test_ppo(cart_pole_envs):
    seed = 42
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    features = envs.observation_space.shape[1]
    actions = int(envs.single_action_space.n)

    actor = MLP(
        features,
        actions,
        [10],
        "relu",
        nnx.Rngs(seed),
    )
    actor = SoftmaxPolicy(actor)

    critic = MLP(
        features,
        1,
        [10],
        "relu",
        nnx.Rngs(seed),
    )

    optimizer_actor = nnx.Optimizer(actor, optax.rprop(0.0003), wrt=nnx.Param)
    optimizer_critic = nnx.Optimizer(critic, optax.rprop(0.0003), wrt=nnx.Param)

    actor, critic, optimizer_actor, optimizer_critic = train_ppo(
        envs, actor, critic, optimizer_actor, optimizer_critic, iterations=5
    )
