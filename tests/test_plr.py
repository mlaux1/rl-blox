from functools import partial

import gymnasium as gym
import numpy as np
import optax
import pytest
from flax import nnx
from numpy.testing import assert_allclose

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.plr import (
    plr_replay_distribution,
    plr_score,
    train_plr,
)
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet, SchedulerResult
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.blox.rollout import Transitions
from rl_blox.blox.state_value import greedy_value
from rl_blox.logging.logger import MemoryLogger


def make_task_set():
    def set_force(env, context):
        env.unwrapped.force_mag = context[0]

    contexts = np.array([[5.0], [10.0], [15.0]])
    return DiscreteTaskSet(
        gym.make("CartPole-v1"), set_force, contexts, context_aware=True
    )


def make_transitions():
    # Two episodes: the first is truncated, the second ends with the interval
    return Transitions(
        observation=np.array([[0.0], [1.0], [5.0]]),
        action=np.zeros(3, dtype=int),
        reward=np.ones(3),
        next_observation=np.array([[1.0], [2.0], [6.0]]),
        termination=np.zeros(3, dtype=bool),
        truncation=np.array([False, True, False]),
    )


def value_function(observation):
    return observation[:, 0]


def test_plr_score_without_lambda_is_mean_absolute_td_error():
    score = plr_score(make_transitions(), value_function, 0.5, 0.0)

    # TD errors 1.5, 1.0 (bootstrapped despite the truncation) and -1.0
    assert score == pytest.approx(3.5 / 3.0)


def test_plr_score_does_not_accumulate_across_episodes():
    score = plr_score(make_transitions(), value_function, 0.5, 1.0)

    # Advantages 1.5 + 0.5 * 1.0, 1.0 and -1.0
    assert score == pytest.approx(4.0 / 3.0)


def test_plr_replay_distribution_mixes_rank_and_staleness():
    scores = np.array([3.0, 1.0, 2.0, 5.0])
    last_selections = np.array([1, 2, 3, 0])
    seen = np.array([True, True, True, False])

    probabilities = plr_replay_distribution(
        scores, last_selections, 3, seen, beta=1.0, rho=0.5
    )

    score_probabilities = np.array([1.0, 1.0 / 3.0, 1.0 / 2.0]) / (11.0 / 6.0)
    # The task selected last has staleness 0
    staleness_probabilities = np.array([2.0, 1.0, 0.0]) / 3.0
    expected = 0.5 * score_probabilities + 0.5 * staleness_probabilities
    assert_allclose(probabilities[:3], expected)
    assert probabilities[3] == 0.0


def test_plr_replay_distribution_with_one_seen_task():
    probabilities = plr_replay_distribution(
        np.array([0.0, 2.0]),
        np.array([0, 1]),
        1,
        np.array([False, True]),
        beta=0.1,
        rho=0.1,
    )

    assert_allclose(probabilities, np.array([0.0, 1.0]))


def test_plr_replay_distribution_temperature_sharpens_ranks():
    probabilities = plr_replay_distribution(
        np.array([1.0, 2.0]),
        np.array([1, 1]),
        2,
        np.array([True, True]),
        beta=0.1,
        rho=0.0,
    )

    assert_allclose(probabilities, np.array([2.0**-10, 1.0]) / (1 + 2.0**-10))


def test_train_plr():
    task_set = make_task_set()
    q_net = MLP(5, 2, [16], "relu", rngs=nnx.Rngs(0))
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        replay_buffer=ReplayBuffer(buffer_size=1_000, discrete_actions=True),
        optimizer=nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param),
        q_target_net=nnx.clone(q_net),
        batch_size=32,
    )
    logger = MemoryLogger()

    result = train_plr(
        task_set,
        train_st,
        partial(greedy_value, q_net),
        total_timesteps=500,
        warmup_episodes=1,
        logger=logger,
        progress_bar=False,
    )

    assert isinstance(result, SchedulerResult)
    assert np.sum(result.training_steps) == 500
    assert result.eval_steps == 0
    # Round-robin warm-up, then sampling
    assert logger.stats["task_id"][:3] == [0, 1, 2]
    assert len(logger.stats["task_id"]) > 3
    for task_id in range(3):
        assert np.all(np.asarray(logger.stats[f"task_score_{task_id}"]) >= 0)


@pytest.mark.parametrize("kwargs", [{"beta": 0.0}, {"rho": 1.5}])
def test_train_plr_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        train_plr(
            make_task_set(),
            None,
            None,
            total_timesteps=100,
            progress_bar=False,
            **kwargs,
        )
