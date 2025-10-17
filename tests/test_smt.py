from functools import partial

import gymnasium as gym
import numpy as np
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.smt import train_smt
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer


def test_smt():
    seed = 1
    env_name = "Pendulum-v1"

    def set_context(env: gym.Env, context):
        env.unwrapped.g = context

    base_env = gym.make(env_name)
    contexts = np.linspace(0, 20, 21)[:, np.newaxis]

    train_set = DiscreteTaskSet(
        base_env, set_context, contexts, context_aware=True
    )

    env = train_set.get_task(0)
    state = create_ddpg_state(env, seed=seed)
    policy_target = nnx.clone(state.policy)
    q_target = nnx.clone(state.q)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=1000),
        len(train_set),
    )

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
        train_set,
        train_st,
        replay_buffer,
        b1=200,
        b2=200,
        learning_starts=50,
        scheduling_interval=1,
        seed=seed,
    )
