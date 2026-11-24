import math

import pytest
from flax import nnx

from rl_blox.algorithm.cmaes import CMAESConfig, train_cmaes
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    DeterministicTanhPolicy,
)


def test_cmaes(pendulum_env):
    seed = 1
    pendulum_env.action_space.seed(seed)
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
        pendulum_env.observation_space.shape[0],
        pendulum_env.action_space.shape[0],
        **hparams_model,
        rngs=nnx.Rngs(seed),
    )
    policy = DeterministicTanhPolicy(policy_net, pendulum_env.action_space)

    policy, _, _ = train_cmaes(pendulum_env, policy, **hparams_algorithm)


@pytest.mark.parametrize("n_params", [4, 10, 4417])
def test_cmaes_rank_mu_learning_rate(n_params):
    config = CMAESConfig.create(
        active=False,
        bounds=None,
        maximize=True,
        min_variance=0.0,
        min_fitness_dist=0.0,
        max_condition=1e7,
        n_params=n_params,
        n_samples_per_update=None,
    )
    # Hansen (2016), The CMA Evolution Strategy: A Tutorial, Table 1
    n_samples = 4 + int(3 * math.log(n_params))
    weights = [
        math.log(n_samples / 2 + 0.5) - math.log(i + 1)
        for i in range(n_samples // 2)
    ]
    mueff = sum(weights) ** 2 / sum(w**2 for w in weights)
    c1 = 2 / ((n_params + 1.3) ** 2 + mueff)
    cmu = min(
        1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n_params + 2) ** 2 + mueff)
    )
    assert config.cmu == pytest.approx(cmu, rel=1e-5)
