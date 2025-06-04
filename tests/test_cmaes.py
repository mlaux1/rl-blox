import gymnasium as gym
from flax import nnx

from rl_blox.algorithm.cmaes import train_cmaes
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    DeterministicTanhPolicy,
)


def test_cmaes():
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    seed = 1
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    hparams_model = dict(
        hidden_nodes=[64, 64],
        activation="relu",
    )
    hparams_algorithm = dict(
        n_samples_per_update=None,
        variance=0.3,
        active=False,
        total_episodes=1,
        seed=seed,
    )
    policy_net = MLP(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        **hparams_model,
        rngs=nnx.Rngs(seed),
    )
    policy = DeterministicTanhPolicy(policy_net, env.action_space)

    policy, _, _ = train_cmaes(env, policy, **hparams_algorithm)
    env.close()
