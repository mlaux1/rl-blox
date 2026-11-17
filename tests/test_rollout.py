import gymnasium as gym
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from rl_blox.blox.rollout import (
    RecordTransitions,
    Transitions,
    pad_transitions,
)


def make_transitions(n_steps):
    return Transitions(
        observation=np.ones((n_steps, 3)),
        action=np.ones(n_steps, dtype=int),
        reward=np.ones(n_steps),
        next_observation=np.ones((n_steps, 3)),
        termination=np.ones(n_steps, dtype=bool),
        truncation=np.zeros(n_steps, dtype=bool),
    )


def test_record_transitions_keeps_order_and_episode_ends():
    env = RecordTransitions(gym.make("Pendulum-v1", max_episode_steps=3))
    env.action_space.seed(0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            reset_observation, _ = env.reset()

    transitions = env.transitions()

    assert transitions.observation.shape == (5, 3)
    assert transitions.action.shape == (5, 1)
    assert transitions.reward.shape == (5,)
    assert_array_equal(transitions.truncation, [0, 0, 1, 0, 0])
    assert not transitions.termination.any()
    # Within an episode, the next observation is the following observation
    assert_array_equal(
        transitions.next_observation[:2], transitions.observation[1:3]
    )
    assert_array_equal(
        transitions.next_observation[3], transitions.observation[4]
    )
    # At the end of an episode, it is the last observation of the episode
    assert_array_equal(transitions.observation[3], reset_observation)
    assert not np.array_equal(
        transitions.next_observation[2], transitions.observation[3]
    )


def test_pad_transitions_pads_to_next_power_of_two():
    padded, mask = pad_transitions(make_transitions(5))

    assert padded.observation.shape == (8, 3)
    assert padded.termination.dtype == bool
    assert_array_equal(padded.reward, [1, 1, 1, 1, 1, 0, 0, 0])
    assert_array_equal(mask, [1, 1, 1, 1, 1, 0, 0, 0])


def test_pad_transitions_rejects_too_small_size():
    with pytest.raises(ValueError, match="smaller than the number of steps"):
        pad_transitions(make_transitions(5), size=4)
