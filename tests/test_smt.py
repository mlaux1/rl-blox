from functools import partial

import gymnasium as gym
import numpy as np
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.smt import ContextualMultiTaskDefinition, train_smt
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer


class MultiTaskPendulum(ContextualMultiTaskDefinition):
    def __init__(self, render_mode=None):
        super().__init__(
            contexts=np.linspace(5, 15, 11)[:, np.newaxis],
            context_in_observation=True,
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


def test_smt():
    seed = 1

    mt_def = MultiTaskPendulum()
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=1_000),
        len(mt_def),
    )

    env = mt_def.get_task(0)
    state = create_ddpg_state(env, seed=seed)
    policy_target = nnx.clone(state.policy)
    q_target = nnx.clone(state.q)

    train_st = partial(
        train_ddpg,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        policy_target=policy_target,
        q_target=q_target,
    )

    train_smt(
        mt_def,
        train_st,
        replay_buffer,
        b1=200,
        b2=200,
        learning_starts=50,
        scheduling_interval=1,
        seed=seed,
    )
    mt_def.close()
