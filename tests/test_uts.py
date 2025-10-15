from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.multitask import DiscreteTaskSet


class MultiTaskPendulum(DiscreteTaskSet):
    def __init__(self, render_mode=None):
        super().__init__(
            contexts=np.linspace(5, 15, 11)[:, np.newaxis],
            context_aware=True,
        )
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)

    def _get_env(self, context):
        self.env.unwrapped.g = context[0]
        return self.env

    def get_solved_threshold(self, task_id: int) -> float:
        return -100.0

    def get_unsolvable_threshold(self, task_id: int) -> float:
        return -1000.0

    def close(self):
        self.env.close()


def test_uts():
    seed = 1

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

    train_set = MultiTaskPendulum()

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
