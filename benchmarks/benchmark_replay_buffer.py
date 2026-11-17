import argparse
import time

import jax
import numpy as np

from rl_blox.blox.replay_buffer import (
    LAP,
    MultiTaskReplayBuffer,
    ReplayBuffer,
    SubtrajectoryReplayBuffer,
    SubtrajectoryReplayBufferPER,
)

OBS_DIM = 39  # Meta-World
ACTION_DIM = 4
EPISODE_LENGTH = 500
BATCH_SIZE = 256
ENCODER_BATCH_SIZE = 256 * 250  # MR.Q: batch_size * target_delay


def fill(replay_buffer, n_samples, rng, terminations=True):
    """Add random episodes with truncated or terminated last transition."""
    for i in range(n_samples):
        episode_end = (i + 1) % EPISODE_LENGTH == 0
        sample = dict(
            observation=rng.standard_normal(OBS_DIM),
            action=rng.uniform(-1, 1, ACTION_DIM),
            reward=rng.standard_normal(),
            next_observation=rng.standard_normal(OBS_DIM),
        )
        if terminations:
            sample.update(terminated=False, truncated=episode_end)
        else:
            sample.update(termination=0)
        replay_buffer.add_sample(**sample)
    return replay_buffer


def measure_fill(replay_buffer, n_samples, rng, terminations=True):
    """Fill a replay buffer and return it with the mean time per sample."""
    start = time.perf_counter()
    fill(replay_buffer, n_samples, rng, terminations)
    return replay_buffer, (time.perf_counter() - start) / n_samples * 1e3


def measure(fn, n_repeats):
    """Mean wall-clock time of fn in ms, including the transfer to device."""
    jax.block_until_ready(fn())  # warm-up
    start = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / n_repeats * 1e3


def main(buffer_size, n_tasks, n_repeats, on_device):
    rng = np.random.default_rng(0)
    results = {}

    replay_buffer, results["ReplayBuffer add_sample"] = measure_fill(
        ReplayBuffer(buffer_size, on_device=on_device),
        buffer_size,
        rng,
        terminations=False,
    )
    results["ReplayBuffer sample_batch"] = measure(
        lambda: replay_buffer.sample_batch(BATCH_SIZE, rng), n_repeats
    )

    lap, results["LAP add_sample"] = measure_fill(
        LAP(buffer_size, on_device=on_device),
        buffer_size,
        rng,
        terminations=False,
    )
    results["LAP sample_batch"] = measure(
        lambda: lap.sample_batch(BATCH_SIZE, rng), n_repeats
    )
    priority = rng.uniform(0.1, 2.0, BATCH_SIZE)
    results["LAP update_priority"] = measure(
        lambda: lap.update_priority(priority), n_repeats
    )

    subtrajectory = fill(
        SubtrajectoryReplayBuffer(buffer_size, horizon=5, on_device=on_device),
        buffer_size,
        rng,
    )
    results["Subtrajectory sample_batch (h=3)"] = measure(
        lambda: subtrajectory.sample_batch(BATCH_SIZE, 3, False, rng),
        n_repeats,
    )

    mrq, results["MR.Q add_sample"] = measure_fill(
        SubtrajectoryReplayBufferPER(
            buffer_size, horizon=5, on_device=on_device
        ),
        buffer_size,
        rng,
    )
    results["MR.Q critic sample_batch (h=3)"] = measure(
        lambda: mrq.sample_batch(BATCH_SIZE, 3, False, rng), n_repeats
    )
    results["MR.Q encoder sample_batch (h=5)"] = measure(
        lambda: mrq.sample_batch(ENCODER_BATCH_SIZE, 5, True, rng),
        max(n_repeats // 20, 1),
    )
    mrq.sample_batch(BATCH_SIZE, 3, False, rng)
    results["MR.Q update_priority"] = measure(
        lambda: mrq.update_priority(priority), n_repeats
    )
    results["MR.Q reset_max_priority"] = measure(
        mrq.reset_max_priority, n_repeats
    )

    multitask = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size // n_tasks, on_device=on_device), n_tasks
    )
    for task_id in range(n_tasks):
        fill(
            multitask.task(task_id),
            buffer_size // n_tasks,
            rng,
            terminations=False,
        )
    results[f"MultiTask ({n_tasks} tasks) sample_batch"] = measure(
        lambda: multitask.sample_batch(BATCH_SIZE, rng), n_repeats
    )

    print(
        f"buffer size {buffer_size}, observation dimension {OBS_DIM}, "
        f"on_device {on_device}"
    )
    for name, ms in results.items():
        print(f"{name:45s} {ms:9.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay buffer benchmark.")
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--on-device", action="store_true")
    args = parser.parse_args()
    main(args.buffer_size, args.n_tasks, args.n_repeats, args.on_device)
