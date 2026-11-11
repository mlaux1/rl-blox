import copy
import os
import pickle

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from rl_blox.blox.replay_buffer import (
    LAP,
    MultiTaskReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    SubtrajectoryReplayBuffer,
    SubtrajectoryReplayBufferPER,
    SumTree,
    dilate_right,
)


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


def test_subtrajectory_replay_buffer():
    buffer = SubtrajectoryReplayBuffer(
        buffer_size=10_000, horizon=5, mrq_mode=False
    )

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

    assert len(buffer) == n_steps

    rng = np.random.default_rng(42)
    batch = buffer.sample_batch(32, 5, True, rng)
    assert batch.observation.shape[0] == 32
    assert batch.observation.shape[1] == 5
    assert batch.observation.shape[2] == 3

    assert np.count_nonzero(buffer.mask_) == n_steps - n_episodes * 5


def add(buffer, obs, next_obs, term, trunc):
    buffer.add_sample(
        observation=float(obs),
        action=12,
        reward=7,
        next_observation=float(next_obs),
        terminated=term,
        truncated=trunc,
    )


@pytest.fixture
def mask():
    return np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])


def test_dilate_right_0(mask):
    assert (dilate_right(mask, 0) == mask).all()


def test_dilate_right_1(mask):
    target = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    assert (dilate_right(mask, 1) == target).all()


def test_dilate_right_2(mask):
    target = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    assert (dilate_right(mask, 2) == target).all()


def test_dilate_right_3(mask):
    target = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert (dilate_right(mask, 3) == target).all()


@pytest.fixture
def empty_strb():
    return SubtrajectoryReplayBuffer(buffer_size=10, horizon=3, mrq_mode=False)


@pytest.fixture
def strb_for_masking(empty_strb):
    buf = empty_strb
    for i in range(4):
        add(buf, obs=i, next_obs=i + 1, term=False, trunc=False)
    return buf


@pytest.fixture
def full_strb(empty_strb):
    buf = empty_strb
    for i in range(10):
        add(buf, obs=i, next_obs=i + 1, term=False, trunc=False)
    return buf


def test_subtrajectory_replay_buffer_no_end(strb_for_masking):
    buf = strb_for_masking
    add(buf, obs=4, next_obs=5, term=False, trunc=False)
    assert buf.current_len == 5
    assert (buf.mask_ == jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all()


def test_subtrajectory_replay_buffer_termination(strb_for_masking):
    buf = strb_for_masking
    add(buf, obs=4, next_obs=5, term=True, trunc=False)
    assert buf.current_len == 5
    assert (buf.mask_ == jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all()


def test_subtrajectory_replay_buffer_truncation(strb_for_masking):
    buf = strb_for_masking
    add(buf, obs=4, next_obs=5, term=False, trunc=True)
    assert buf.current_len == 5
    assert (buf.mask_ == jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])).all()


def test_subtrajectory_replay_buffer_has_horizon_as_min_episode_len():
    buf = SubtrajectoryReplayBuffer(buffer_size=10, horizon=3, mrq_mode=False)
    for i in range(3):
        add(buf, obs=i, next_obs=i + 1, term=False, trunc=False)
    assert buf.mask_.any()
    # ensure subtrajectories can be generated from a minimum-length episode
    # otherwise an error will be thrown from inside sample_batch
    _ = buf.sample_batch(
        batch_size=1,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )


def test_subtrajectory_replay_buffer_respects_episode_boundaries_on_termination():
    buf = SubtrajectoryReplayBuffer(buffer_size=10, horizon=3, mrq_mode=False)
    for i in range(4):  # all non-negative
        add(buf, obs=i, next_obs=i + 1, term=False, trunc=False)
    add(buf, obs=4, next_obs=5, term=True, trunc=False)  # still non-negative

    for i in range(-4, 0):  # next episode, all negative
        add(buf, obs=i, next_obs=i - 1, term=False, trunc=False)

    obs_subseqs, *_ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    for obs_subseq in obs_subseqs:
        assert (obs_subseq >= 0).all() or (obs_subseq <= 0).all()


def test_subtrajectory_replay_buffer_termination_transitions_are_always_last_without_next_ep(
    strb_for_masking,
):
    buf = strb_for_masking
    # add termination transition as the last transition
    add(buf, obs=4, next_obs=5, term=True, trunc=False)

    _, _, _, _, terminated, _ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not terminated[:, :-1].any() and terminated[:, -1].any()


def test_subtrajectory_replay_buffer_termination_transitions_are_always_last_with_next_ep(
    strb_for_masking,
):
    buf = strb_for_masking
    # add termination transition before the next episode
    add(buf, obs=4, next_obs=5, term=True, trunc=False)
    add(buf, obs=5, next_obs=6, term=False, trunc=False)
    add(buf, obs=6, next_obs=7, term=False, trunc=False)

    _, _, _, _, terminated, _ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not terminated[:, :-1].any() and terminated[:, -1].any()


def test_subtrajectory_replay_buffer_may_not_produce_truncation_transitions_without_next_ep(
    strb_for_masking,
):
    buf = strb_for_masking
    # add truncation transition as the last transition
    add(buf, obs=4, next_obs=5, term=False, trunc=True)

    _, _, _, _, _, truncated = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not truncated.any()


def test_subtrajectory_replay_buffer_may_not_produce_truncation_transitions_with_next_ep(
    strb_for_masking,
):
    buf = strb_for_masking
    # add truncation transition before the next episode
    add(buf, obs=4, next_obs=5, term=False, trunc=True)
    add(buf, obs=5, next_obs=6, term=False, trunc=False)
    add(buf, obs=6, next_obs=7, term=False, trunc=False)

    _, _, _, _, _, truncated = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not truncated.any()


# From here: Same tests as before, just on an already full buffer
# (as far as they apply here). They are arranged so that generally
# the most interesting index for each test respectively (i.e., newest
# transition, termination, truncation, ...) is at buffer index 0.


def test_subtrajectory_replay_buffer_full_no_end(full_strb):
    buf = full_strb
    add(buf, obs=10, next_obs=11, term=False, trunc=False)
    assert buf.current_len == 10
    assert (buf.mask_ == jnp.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])).all()


def test_subtrajectory_replay_buffer_full_termination(full_strb):
    buf = full_strb
    add(buf, obs=10, next_obs=11, term=True, trunc=False)
    assert buf.current_len == 10
    assert (buf.mask_ == jnp.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])).all()


def test_subtrajectory_replay_buffer_full_truncation(full_strb):
    buf = full_strb
    add(buf, obs=10, next_obs=11, term=False, trunc=True)
    assert buf.current_len == 10
    assert (buf.mask_ == jnp.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0])).all()


def test_subtrajectory_replay_buffer_full_respects_episode_boundaries_on_termination():
    buf = SubtrajectoryReplayBuffer(buffer_size=10, horizon=3, mrq_mode=False)
    for i in range(10):  # all non-negative
        add(buf, obs=i, next_obs=i + 1, term=False, trunc=False)
    add(buf, obs=10, next_obs=11, term=True, trunc=False)  # still non-negative

    for i in range(-4, 0):  # next episode, all negative
        add(buf, obs=i, next_obs=i - 1, term=False, trunc=False)

    obs_subseqs, *_ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    for obs_subseq in obs_subseqs:
        assert (obs_subseq >= 0).all() or (obs_subseq <= 0).all()


def test_subtrajectory_replay_buffer_full_termination_transitions_are_always_last_without_next_ep(
    full_strb,
):
    buf = full_strb
    # add termination transition as the last transition
    add(buf, obs=10, next_obs=11, term=True, trunc=False)

    _, _, _, _, terminated, _ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not terminated[:, :-1].any() and terminated[:, -1].any()


def test_subtrajectory_replay_buffer_full_termination_transitions_are_always_last_with_next_ep(
    full_strb,
):
    buf = full_strb
    # add termination transition before the next episode
    add(buf, obs=10, next_obs=11, term=True, trunc=False)
    add(buf, obs=11, next_obs=12, term=False, trunc=False)
    add(buf, obs=12, next_obs=13, term=False, trunc=False)

    _, _, _, _, terminated, _ = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not terminated[:, :-1].any() and terminated[:, -1].any()


def test_subtrajectory_replay_buffer_full_may_not_produce_truncation_transitions_without_next_ep(
    full_strb,
):
    buf = full_strb
    # add truncation transition as the last transition
    add(buf, obs=10, next_obs=11, term=False, trunc=True)

    _, _, _, _, _, truncated = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not truncated.any()


def test_subtrajectory_replay_buffer_full_may_not_produce_truncation_transitions_with_next_ep(
    full_strb,
):
    buf = full_strb
    # add truncation transition before the next episode
    add(buf, obs=10, next_obs=11, term=False, trunc=True)
    add(buf, obs=11, next_obs=12, term=False, trunc=False)
    add(buf, obs=12, next_obs=13, term=False, trunc=False)

    _, _, _, _, _, truncated = buf.sample_batch(
        batch_size=100,
        horizon=3,
        include_intermediate=True,
        rng=np.random.default_rng(0),
    )
    assert not truncated.any()


def add_transition(replay_buffer, value):
    replay_buffer.add_sample(
        observation=np.full(2, value),
        action=np.zeros(1),
        reward=0.0,
        next_observation=np.full(2, value),
        termination=0,
    )


def test_multitask_task_view_adds_samples_to_its_task():
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 3)

    add_transition(replay_buffer.task(1), 1.0)
    add_transition(replay_buffer.task(1), 1.0)
    add_transition(replay_buffer.task(2), 2.0)

    assert [len(buffer) for buffer in replay_buffer.buffers] == [0, 2, 1]


def test_multitask_task_view_behaves_like_multitask_buffer():
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2)
    view = replay_buffer.task(0)

    add_transition(view, 0.0)

    assert len(view) == len(replay_buffer) == 1
    batch = view.sample_batch(4, np.random.default_rng(0))
    assert batch.observation.shape == (4, 2)


def test_multitask_task_view_rejects_invalid_task():
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2)

    with pytest.raises(ValueError):
        replay_buffer.task(2)


def multitask_buffer(n_samples_per_task, **kwargs):
    """Multi-task buffer whose observations contain the task index."""
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100), len(n_samples_per_task), **kwargs
    )
    for task_id, n_samples in enumerate(n_samples_per_task):
        for _ in range(n_samples):
            add_transition(replay_buffer.task(task_id), task_id)
    return replay_buffer


def sampled_tasks(batch):
    return np.asarray(batch.observation[:, 0]).astype(int)


def test_multitask_uniform_weighting_splits_batch_evenly():
    replay_buffer = multitask_buffer([5, 50, 5])

    batch = replay_buffer.sample_batch(7, np.random.default_rng(0))

    counts = np.bincount(sampled_tasks(batch), minlength=3)
    assert counts.sum() == 7
    assert set(counts.tolist()) == {2, 3}


def test_multitask_proportional_weighting_follows_amount_of_data():
    replay_buffer = multitask_buffer([10, 90], task_weighting="proportional")

    batch = replay_buffer.sample_batch(10_000, np.random.default_rng(0))

    share_of_task_0 = np.mean(sampled_tasks(batch) == 0)
    assert share_of_task_0 == pytest.approx(0.1, abs=0.02)


def test_multitask_rejects_unknown_task_weighting():
    with pytest.raises(ValueError):
        MultiTaskReplayBuffer(
            ReplayBuffer(buffer_size=10), 2, task_weighting="random"
        )


def test_multitask_samples_only_tasks_with_data():
    replay_buffer = multitask_buffer([0, 5, 0])

    batch = replay_buffer.sample_batch(6, np.random.default_rng(0))

    assert set(sampled_tasks(batch).tolist()) == {1}


def test_multitask_skips_subtrajectory_buffers_without_valid_start():
    replay_buffer = MultiTaskReplayBuffer(
        SubtrajectoryReplayBuffer(buffer_size=10, horizon=3, mrq_mode=False),
        2,
    )
    # A full episode of 4 steps in task 0, an episode of 2 steps in task 1
    for i in range(4):
        add(
            replay_buffer.task(0),
            obs=i,
            next_obs=i + 1,
            term=i == 3,
            trunc=False,
        )
    for i in range(2):
        add(
            replay_buffer.task(1),
            obs=10 + i,
            next_obs=11 + i,
            term=i == 1,
            trunc=False,
        )

    batch = replay_buffer.sample_batch(8, 3, True, np.random.default_rng(0))

    assert batch.observation.shape == (8, 3)
    assert np.all(batch.observation < 10)


def test_multitask_routes_priorities_to_sampled_tasks():
    replay_buffer = MultiTaskReplayBuffer(LAP(buffer_size=10), 2)
    for task_id in range(2):
        for _ in range(5):
            add_transition(replay_buffer.task(task_id), task_id)

    batch = replay_buffer.sample_batch(4, np.random.default_rng(0))
    priorities = 10.0 + sampled_tasks(batch)
    replay_buffer.update_priority(priorities)

    for task_id, buffer in enumerate(replay_buffer.buffers):
        sampled = buffer.priority.sampled_indices
        assert len(sampled) == 2
        assert np.all(buffer.priority.priority[sampled] == 10.0 + task_id)


@pytest.mark.parametrize(
    "replay_buffer",
    [ReplayBuffer(buffer_size=10), SubtrajectoryReplayBuffer(buffer_size=10)],
)
def test_replay_buffer_stores_floats_as_float32(replay_buffer):
    assert replay_buffer.buffer["observation"].dtype == np.float32
    assert replay_buffer.buffer["reward"].dtype == np.float32


def filled_flat_buffers():
    buffers = [ReplayBuffer(buffer_size=10), LAP(buffer_size=10)]
    for replay_buffer in buffers:
        for i in range(10):
            add_transition(replay_buffer, i)
    return buffers + [multitask_buffer([5, 5])]


@pytest.mark.parametrize("replay_buffer", filled_flat_buffers())
def test_sample_arrays_returns_numpy_version_of_sample_batch(replay_buffer):
    arrays = replay_buffer.sample_arrays(6, np.random.default_rng(0))
    batch = replay_buffer.sample_batch(6, np.random.default_rng(0))

    assert all(isinstance(a, np.ndarray) for a in arrays)
    assert all(isinstance(b, jax.Array) for b in batch)
    for a, b in zip(arrays, batch, strict=True):
        assert_array_almost_equal(a, b)


@pytest.mark.parametrize("include_intermediate", [False, True])
def test_subtrajectory_sample_arrays_returns_numpy_version_of_sample_batch(
    full_strb, include_intermediate
):
    arrays = full_strb.sample_arrays(
        6, 3, include_intermediate, np.random.default_rng(0)
    )
    batch = full_strb.sample_batch(
        6, 3, include_intermediate, np.random.default_rng(0)
    )

    assert all(isinstance(a, np.ndarray) for a in arrays)
    for a, b in zip(arrays, batch, strict=True):
        assert_array_almost_equal(a, b)


def test_sum_tree_samples_proportional_to_values():
    values = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    tree = SumTree(len(values))
    tree.update(np.arange(len(values)), values)

    rng = np.random.default_rng(0)
    indices = tree.sample(rng.uniform(size=100_000) * tree.total())

    frequencies = np.bincount(indices, minlength=len(values)) / len(indices)
    assert tree.total() == pytest.approx(10.0)
    assert_array_almost_equal(frequencies, values / values.sum(), decimal=2)
    assert frequencies[0] == frequencies[3] == 0.0


def test_lap_sampling_matches_cumulative_sum_sampling():
    lap = LAP(buffer_size=1000)
    for i in range(1000):
        add_transition(lap, i)
    rng = np.random.default_rng(0)
    lap.sample_batch(1000, rng)
    lap.update_priority(rng.uniform(0.1, 3.0, 1000))

    indices = lap.priority.prioritized_sampling(
        1000, 10_000, np.random.default_rng(1)
    )

    cumulative = np.cumsum(lap.priority.priority)
    uniforms = np.random.default_rng(1).uniform(0, 1, size=10_000)
    expected = np.searchsorted(cumulative, uniforms * cumulative[-1])
    assert np.mean(indices == expected) > 0.999


def test_subtrajectory_per_never_samples_masked_out_starts():
    buffer = SubtrajectoryReplayBufferPER(buffer_size=50, horizon=3)
    rng = np.random.default_rng(0)
    for i in range(120):  # overwrite the ring buffer more than twice
        add(buffer, obs=i, next_obs=i + 1, term=i % 9 == 8, trunc=i % 13 == 12)
        buffer.sample_batch(16, 3, False, rng)
        buffer.update_priority(rng.uniform(0.1, 3.0, 16))

    starts = buffer.priority.prioritized_sampling(
        buffer.current_len, 10_000, rng
    )

    assert np.all(buffer.mask_[starts] == 1)


def test_multitask_normalizes_importance_ratios_over_whole_batch():
    replay_buffer = MultiTaskReplayBuffer(
        PrioritizedReplayBuffer(buffer_size=100), 2
    )
    for task_id in range(2):
        for _ in range(100):
            add_transition(replay_buffer.task(task_id), task_id)
    rng = np.random.default_rng(0)
    # Task 0 keeps equal priorities, task 1 gets very unequal priorities
    batch, _ = replay_buffer.sample_batch(200, rng)
    tasks = sampled_tasks(batch)
    replay_buffer.update_priority(
        np.where(tasks == 0, 1.0, rng.uniform(0.01, 10.0, 200))
    )

    batch, importance_ratio = replay_buffer.sample_batch(200, rng, 0.4)

    tasks = sampled_tasks(batch)
    importance_ratio = np.asarray(importance_ratio)
    assert importance_ratio.max() == pytest.approx(1.0)
    assert np.all(importance_ratio[tasks == 0] < 1.0)


@pytest.mark.parametrize("episode_length", [4, 40])
def test_subtrajectory_samples_valid_starts_uniformly(episode_length):
    # Short episodes leave few valid starts, which triggers the fallback
    buffer = SubtrajectoryReplayBuffer(
        buffer_size=1000, horizon=3, mrq_mode=False
    )
    for i in range(1000):
        add(
            buffer,
            obs=i,
            next_obs=i + 1,
            term=False,
            trunc=i % episode_length == episode_length - 1,
        )
    valid_starts = np.flatnonzero(buffer.mask_)

    starts = buffer._sample_idx(100_000, 3, np.random.default_rng(0))

    assert np.all(buffer.mask_[starts] == 1)
    frequencies = np.bincount(starts, minlength=1000)[valid_starts]
    expected = len(starts) / len(valid_starts)
    assert np.all(np.abs(frequencies - expected) < 5 * np.sqrt(expected))


def add_step(replay_buffer, i, subtrajectory):
    transition = dict(
        observation=np.full(2, i, dtype=float),
        action=np.full(1, -i, dtype=float),
        reward=float(i),
        next_observation=np.full(2, i + 1, dtype=float),
    )
    if subtrajectory:
        replay_buffer.add_sample(
            **transition, terminated=i % 7 == 6, truncated=i % 11 == 10
        )
    else:
        replay_buffer.add_sample(**transition, termination=i % 7 == 6)


DEVICE_CASES = {
    "ReplayBuffer": (lambda **kw: ReplayBuffer(50, **kw), (), False),
    "LAP": (lambda **kw: LAP(50, **kw), (), False),
    "PrioritizedReplayBuffer": (
        lambda **kw: PrioritizedReplayBuffer(50, **kw),
        (0.4,),
        False,
    ),
    "Subtrajectory": (
        lambda **kw: SubtrajectoryReplayBuffer(50, horizon=3, **kw),
        (3, False),
        True,
    ),
    "SubtrajectoryPER with horizon": (
        lambda **kw: SubtrajectoryReplayBufferPER(50, horizon=3, **kw),
        (3, True),
        True,
    ),
}


def sample(replay_buffer, extra_args, seed):
    rng = np.random.default_rng(seed)
    if isinstance(replay_buffer, MultiTaskReplayBuffer):
        single_task_buffer = replay_buffer.buffers[0]
    else:
        single_task_buffer = replay_buffer
    if isinstance(single_task_buffer, SubtrajectoryReplayBuffer):
        return replay_buffer.sample_batch(16, *extra_args, rng)
    return replay_buffer.sample_batch(16, rng, *extra_args)


def assert_same_batches(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("case", DEVICE_CASES)
def test_on_device_sampling_equals_host_sampling(case):
    create, extra_args, subtrajectory = DEVICE_CASES[case]
    host, device = create(), create(on_device=True)
    # Sample between additions and wrap around the ring buffer
    for start, stop in [(0, 30), (30, 45), (45, 120)]:
        for i in range(start, stop):
            add_step(host, i, subtrajectory)
            add_step(device, i, subtrajectory)
        assert_same_batches(
            sample(host, extra_args, stop), sample(device, extra_args, stop)
        )


def test_on_device_buffer_can_be_pickled():
    replay_buffer = ReplayBuffer(50, on_device=True)
    for i in range(30):
        add_step(replay_buffer, i, False)
    sample(replay_buffer, (), 0)

    restored = pickle.loads(pickle.dumps(replay_buffer))
    add_step(restored, 30, False)
    add_step(replay_buffer, 30, False)

    assert_same_batches(sample(replay_buffer, (), 1), sample(restored, (), 1))


@pytest.mark.parametrize(
    "create, extra_args, subtrajectory",
    [
        (lambda **kw: ReplayBuffer(50, **kw), (), False),
        (
            lambda **kw: SubtrajectoryReplayBufferPER(50, 3, **kw),
            (3, True),
            True,
        ),
    ],
)
def test_on_device_multitask_sampling_equals_host_sampling(
    create, extra_args, subtrajectory
):
    host = MultiTaskReplayBuffer(create(), 3)
    device = MultiTaskReplayBuffer(create(on_device=True), 3)
    # Sample while later tasks are still empty
    for task_id in range(3):
        for i in range(40 * task_id, 40 * task_id + 25):
            add_step(host.task(task_id), i, subtrajectory)
            add_step(device.task(task_id), i, subtrajectory)
        assert_same_batches(
            sample(host, extra_args, task_id),
            sample(device, extra_args, task_id),
        )


@pytest.mark.parametrize(
    "restore",
    [lambda b: pickle.loads(pickle.dumps(b)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_on_device_multitask_buffer_can_be_restored(restore):
    host = MultiTaskReplayBuffer(ReplayBuffer(50), 2)
    device = MultiTaskReplayBuffer(ReplayBuffer(50, on_device=True), 2)
    for i in range(10):
        add_step(host.task(i % 2), i, False)
        add_step(device.task(i % 2), i, False)
    sample(device, (), 0)
    # Not on the device yet
    for i in range(10, 40):
        add_step(host.task(i % 2), i, False)
        add_step(device.task(i % 2), i, False)

    restored = restore(device)

    assert_same_batches(sample(host, (), 1), sample(restored, (), 1))


def test_on_device_multitask_rejects_prioritized_replay_buffer():
    with pytest.raises(ValueError, match="on_device"):
        MultiTaskReplayBuffer(PrioritizedReplayBuffer(50, on_device=True), 2)
