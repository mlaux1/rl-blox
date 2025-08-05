import os
import pickle

import gymnasium as gym
import numpy as np
from numpy.testing import assert_array_almost_equal

from rl_blox.blox.replay_buffer import ReplayBuffer, SubtrajectoryReplayBuffer


def test_pickle_replay_buffer():
    rb = ReplayBuffer(10)
    rb.add_sample(
        observation=np.arange(3),
        action=np.arange(4),
        reward=5.0,
        next_observation=np.zeros(3),
        termination=False,
    )
    rb.add_sample(
        observation=np.arange(3) + 1,
        action=np.arange(4) + 1,
        reward=6.0,
        next_observation=np.ones(3),
        termination=True,
    )

    filename = "/tmp/replay_buffer.pkl"
    try:
        with open(filename, "wb") as f:
            pickle.dump(rb, f)
        with open(filename, "rb") as f:
            rb_loaded = pickle.load(f)

        assert rb.buffer_size == rb_loaded.buffer_size
        assert len(rb) == len(rb_loaded)
        assert rb.insert_idx == rb_loaded.insert_idx

        o1, a1, r1, no1, t1 = rb.sample_batch(2, np.random.default_rng(0))
        o2, a2, r2, no2, t2 = rb_loaded.sample_batch(
            2, np.random.default_rng(0)
        )
        assert_array_almost_equal(o1, o2)
        assert_array_almost_equal(a1, a2)
        assert_array_almost_equal(r1, r2)
        assert_array_almost_equal(no1, no2)
        assert_array_almost_equal(t1, t2)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_episodic_replay_buffer():
    buffer = SubtrajectoryReplayBuffer(buffer_size=10_000, horizon=5)

    n_steps = 2_000

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset()
    n_episodes = 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
        )
        if terminated or truncated:
            obs, _ = env.reset()
            n_episodes += 1
        else:
            obs = next_obs
    env.close()

    assert not buffer.environment_terminates

    # Terminated / truncated states are counted as samples
    assert len(buffer) == n_steps + n_episodes

    rng = np.random.default_rng(42)
    batch = buffer.sample_batch(32, 5, True, rng)
    assert batch.observation.shape[0] == 32
    assert batch.observation.shape[1] == 5
    assert batch.observation.shape[2] == 3

    assert np.count_nonzero(buffer.mask_) == n_steps - n_episodes * 5
