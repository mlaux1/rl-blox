from functools import partial

import gymnasium as gym
import numpy as np

from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.multitask import DiscreteTaskSet


def test_uts():
    seed = 1
    env_name = "Pendulum-v1"

    def set_context(env: gym.Env, context):
        env.unwrapped.g = context

    base_env = gym.make(env_name)
    contexts = np.linspace(0, 20, 21)[:, np.newaxis]

    train_set = DiscreteTaskSet(
        base_env, set_context, contexts, context_aware=True
    )

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

    env = train_set.get_task(0)

    state = create_sac_state(env, **hparams_models)
    entropy_control = EntropyControl(env, 0.2, True, 1e-3)

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
