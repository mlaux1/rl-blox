import gymnasium as gym
import numpy as np

from rl_blox.algorithm.mrq import EpisodicReplayBuffer


def test_episodic_replay_buffer():
    buffer = EpisodicReplayBuffer(buffer_size=10_000, horizon=5)

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
