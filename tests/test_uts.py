from functools import partial

import gymnasium as gym
import jax.numpy as jnp
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.multi_task.uniform_task_sampling import (
    TaskSet,
    train_uts,
)
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac


def test_uts():
    seed = 1
    env_name = "Pendulum-v1"

    train_contexts = jnp.linspace(5, 15, 2)[:, jnp.newaxis]
    train_envs = [
        RecordEpisodeStatistics(gym.make(env_name, g=5)),
        RecordEpisodeStatistics(gym.make(env_name, g=15)),
    ]

    train_set = TaskSet(train_contexts, train_envs)

    hparams_models = dict(
        q_hidden_nodes=[512, 512],
        q_learning_rate=1e-3,
        policy_learning_rate=1e-3,
        policy_hidden_nodes=[128, 128],
        seed=seed,
    )

    hparams_alg = dict(
        total_timesteps=100,
        exploring_starts=50,
        episodes_per_task=1,
    )

    state = create_sac_state(train_set.get_task_env(0), **hparams_models)
    entropy_control = EntropyControl(train_set.get_task_env(0), 0.2, True, 1e-3)

    train_st = partial(
        train_sac,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        entropy_control=entropy_control,
    )

    _ = train_uts(
        train_set,
        train_st,
        **hparams_alg,
    )
