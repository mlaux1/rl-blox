import timeit

import numpy as np

from rl_blox.blox.replay_buffer import ReplayBuffer


def fill_replay_buffer(
    buffer_size, observations, actions, rewards, next_observations, terminations
):
    replay_buffer = ReplayBuffer(buffer_size)
    for i in range(buffer_size):
        replay_buffer.add_sample(
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            next_observation=next_observations[i],
            termination=terminations[i],
        )
    return replay_buffer


def sample_replay_buffer(replay_buffer, batch_size, rng):
    replay_buffer.sample_batch(batch_size, rng)


buffer_size = 10_000
observations = np.zeros((buffer_size, 5))
actions = np.ones((buffer_size, 3))
rewards = np.zeros(buffer_size)
next_observations = np.ones((buffer_size, 5))
terminations = np.zeros(buffer_size, dtype=int)

benchmark_fill_replay_buffer = lambda: fill_replay_buffer(
    buffer_size, observations, actions, rewards, next_observations, terminations
)

times = timeit.repeat(benchmark_fill_replay_buffer, repeat=10, number=1)
print(f"mean: {np.mean(times):.5f} s, std. dev.: {np.std(times):.5f}")

replay_buffer = benchmark_fill_replay_buffer()
rng = np.random.default_rng(0)
benchmark_sample_replay_buffer = lambda: sample_replay_buffer(
    replay_buffer, 256, rng
)

times = timeit.repeat(benchmark_sample_replay_buffer, repeat=10, number=100)
print(f"mean: {np.mean(times):.5f} s, std. dev.: {np.std(times):.5f}")
