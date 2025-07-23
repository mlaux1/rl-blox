import gymnasium as gym
import numpy as np

from rl_blox.algorithm.mrq import EpisodicReplayBuffer


def test_episodic_replay_buffer():
    buffer = EpisodicReplayBuffer(buffer_size=10_000, horizon=5)

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset()
    for _ in range(2_000):
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
        else:
            obs = next_obs
    env.close()

    assert len(buffer) == 2_000

    rng = np.random.default_rng(42)
    batch = buffer.sample_batch(32, 5, True, rng)
    assert batch.observations.shappe[0] == 32
    assert batch.observations.shappe[1] == 5
    assert batch.observations.shappe[2] == 3
