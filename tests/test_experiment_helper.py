import gymnasium as gym
import pytest

from rl_blox.util.experiment_helper import generate_rollout


class CountingEnv(gym.Env):
    """Ends the episode after ``n_steps`` steps."""

    observation_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, n_steps, terminate):
        self.n_steps = n_steps
        self.terminate = terminate

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.t, {}

    def step(self, action):
        if self.t >= self.n_steps:
            raise RuntimeError("step after the end of the episode")
        self.t += 1
        done = self.t == self.n_steps
        return (
            self.t,
            1.0,
            done and self.terminate,
            done and not self.terminate,
            {},
        )


@pytest.mark.parametrize("terminate", [True, False])
def test_generate_rollout_stops_at_episode_end(terminate):
    obs, actions, rewards = generate_rollout(
        CountingEnv(3, terminate), lambda observation, key: 0
    )
    assert obs.tolist() == [0, 1, 2, 3]
    assert actions.tolist() == [0, 0, 0]
    assert rewards.tolist() == [1.0, 1.0, 1.0]
