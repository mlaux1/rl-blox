from functools import partial

import gymnasium as gym
import numpy as np
from flax import nnx

from rl_blox.algorithm.td3 import create_td3_state
from rl_blox.algorithm.td3_lap import train_td3_lap
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import LAP

EPISODE_LENGTH = 20


def make_train_st(env):
    state = create_td3_state(
        env, policy_hidden_nodes=[8], q_hidden_nodes=[8], seed=0
    )
    return partial(
        train_td3_lap,
        **state._asdict(),
        policy_target=nnx.clone(state.policy),
        q_target=nnx.clone(state.q),
        batch_size=8,
        learning_starts=20,
        replay_buffer=LAP(1_000),
        progress_bar=False,
    )


def test_td3_lap_stops_after_total_episodes():
    env = gym.wrappers.RecordEpisodeStatistics(
        gym.make("Pendulum-v1", max_episode_steps=EPISODE_LENGTH),
        buffer_length=10,
    )

    result = make_train_st(env)(
        env=env, total_timesteps=10_000, total_episodes=3
    )

    assert len(env.length_queue) == 3
    assert result.global_step == 3 * EPISODE_LENGTH


def test_td3_lap_returns_number_of_steps():
    env = gym.make("Pendulum-v1", max_episode_steps=EPISODE_LENGTH)

    result = make_train_st(env)(env=env, total_timesteps=50, global_step=20)

    assert result.global_step == 50


def test_td3_lap_with_task_scheduler():
    def set_gravity(env, context):
        env.unwrapped.g = context[0]

    task_set = DiscreteTaskSet(
        gym.make("Pendulum-v1", max_episode_steps=EPISODE_LENGTH),
        set_gravity,
        np.array([[5.0], [10.0]]),
        context_aware=True,
    )

    result = train_uts(
        task_set,
        make_train_st(task_set.get_task(0)),
        total_timesteps=100,
        progress_bar=False,
    )

    assert np.sum(result.training_steps) == 100
