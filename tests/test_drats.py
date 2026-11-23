from functools import partial

import gymnasium as gym
import numpy as np
import optax
import pytest
from flax import nnx
from scipy.optimize import minimize

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.drats import (
    drats_mirror_ascent,
    drats_return_gaps,
    project_min_probability,
    train_drats,
)
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet, SchedulerResult
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 10


class ScriptedEnv(gym.Env):
    """Episodes of 10 steps with a fixed reward per step and success flag."""

    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def __init__(self, report_success=True):
        self.report_success = report_success
        self.reward = 0.0
        self.success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        info = {"success": self.success} if self.report_success else {}
        observation = np.zeros(1, dtype=np.float32)
        return observation, self.reward, False, self.t >= EPISODE_LENGTH, info


def set_context(env, context):
    env.reward = float(context[0])
    env.success = bool(context[1])


def make_task_set(rewards=(1.0, 0.5, 0.0), successes=None, **kwargs):
    if successes is None:
        successes = [0] * len(rewards)
    return DiscreteTaskSet(
        ScriptedEnv(**kwargs),
        set_context,
        np.column_stack((rewards, successes)),
        context_aware=False,
    )


def random_actions_st(
    env, total_timesteps, total_episodes, global_step, seed, **kwargs
):
    """Backbone stub that only acts randomly for the scheduled episodes."""
    env.action_space.seed(seed)
    step = global_step
    for episode in range(total_episodes):
        env.reset(seed=seed + episode)
        done = False
        while not done and step < total_timesteps:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            step += 1
    return step


def run_drats(task_set=None, **kwargs):
    logger = MemoryLogger()
    result = train_drats(
        make_task_set() if task_set is None else task_set,
        random_actions_st,
        logger=logger,
        progress_bar=False,
        **kwargs,
    )
    return result, logger


def last_stats(logger, name, n_tasks=3):
    return np.array([logger.stats[f"{name}_{i}"][-1] for i in range(n_tasks)])


def test_drats_return_gaps_normalizes_and_clips():
    gaps = drats_return_gaps([0.0, 5.0, 10.0, 12.0, -1.0], 10.0, 0.0)

    assert gaps.tolist() == [1.0, 0.5, 0.0, 0.0, 1.0]


def test_drats_return_gaps_per_task_random_returns():
    gaps = drats_return_gaps([5.0, 5.0], [10.0, 10.0], [0.0, 5.0])

    assert gaps.tolist() == [0.5, 1.0]


def test_drats_return_gaps_without_valid_reference():
    gaps = drats_return_gaps([5.0, 5.0], [-np.inf, 0.0], 0.0)

    assert gaps.tolist() == [1.0, 1.0]


def test_drats_mirror_ascent_keeps_uniform_with_equal_gaps():
    probabilities = drats_mirror_ascent(np.full(4, 0.25), np.ones(4), 8.0, 4.0)

    assert np.allclose(probabilities, 0.25)


def test_drats_mirror_ascent_with_full_step_is_best_response():
    gaps = np.array([0.1, 0.5, 0.9])
    eta = 4.0

    probabilities = drats_mirror_ascent(
        np.array([0.6, 0.3, 0.1]), gaps, eta, eta
    )

    expected = np.exp(eta * gaps) / np.sum(np.exp(eta * gaps))
    assert np.allclose(probabilities, expected)


def test_drats_mirror_ascent_moves_mass_to_larger_gaps():
    probabilities = drats_mirror_ascent(
        np.full(3, 1 / 3), np.array([0.0, 0.5, 1.0]), 8.0, 0.5
    )

    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert np.isclose(np.sum(probabilities), 1.0)


def test_project_min_probability_keeps_feasible_distribution():
    probabilities = np.array([0.5, 0.3, 0.2])

    assert np.allclose(
        project_min_probability(probabilities, 0.1), [0.5, 0.3, 0.2]
    )


@pytest.mark.parametrize(
    "probabilities, min_probability",
    [
        ([0.97, 0.02, 0.01], 0.1),
        ([0.9, 0.05, 0.03, 0.02], 0.0625),
        ([0.5, 0.45, 0.04, 0.01], 0.2),
        ([0.999, 0.0005, 0.0005], 1 / 3),
    ],
)
def test_project_min_probability_is_kl_projection(
    probabilities, min_probability
):
    probabilities = np.array(probabilities)

    projection = project_min_probability(probabilities, min_probability)

    assert np.all(projection >= min_probability - 1e-12)
    assert np.isclose(np.sum(projection), 1.0)

    def kl(q):
        return np.sum(q * np.log(q / probabilities))

    numerical = minimize(
        kl,
        np.full(len(probabilities), 1.0 / len(probabilities)),
        bounds=[(min_probability, 1.0)] * len(probabilities),
        constraints={"type": "eq", "fun": lambda q: np.sum(q) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    assert kl(projection) <= numerical.fun + 1e-7
    assert np.allclose(projection, numerical.x, atol=1e-4)


def test_train_drats_updates_after_full_batches():
    result, logger = run_drats(
        total_timesteps=12 * EPISODE_LENGTH,
        episodes_per_update=6,
        reference_returns=10.0,
    )

    assert isinstance(result, SchedulerResult)
    assert result.training_steps.sum() == 12 * EPISODE_LENGTH
    assert result.eval_steps == 0
    steps, _ = logger.get_stat("task_probability_0", x_key="step")
    assert steps.tolist() == [0, 6 * EPISODE_LENGTH, 12 * EPISODE_LENGTH]


def test_train_drats_prefers_tasks_with_larger_gaps():
    # Returns 10, 5 and 0 with reference return 10: gaps 0, 0.5 and 1. With
    # eta = 1, the best response softmax(g) stays above the minimum
    # probability.
    _, logger = run_drats(
        total_timesteps=600 * EPISODE_LENGTH,
        episodes_per_update=30,
        eta=1.0,
        reference_returns=10.0,
    )

    probabilities = last_stats(logger, "task_probability")
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert np.allclose(
        probabilities,
        np.exp([0.0, 0.5, 1.0]) / np.sum(np.exp([0.0, 0.5, 1.0])),
        atol=0.02,
    )
    assert np.allclose(last_stats(logger, "task_gap"), [0.0, 0.5, 1.0])


def test_train_drats_estimates_zero_return_for_tasks_without_episodes():
    _, logger = run_drats(
        task_set=make_task_set(rewards=(1.0, 1.0, 1.0)),
        total_timesteps=EPISODE_LENGTH,
        episodes_per_update=1,
        reference_returns=10.0,
    )

    task_id = logger.stats["task_id"][0]
    estimates = last_stats(logger, "task_return_estimate")
    gaps = last_stats(logger, "task_gap")
    assert estimates[task_id] == 10.0 and gaps[task_id] == 0.0
    others = [i for i in range(3) if i != task_id]
    assert np.all(estimates[others] == 0.0)
    assert np.all(gaps[others] == 1.0)


def test_train_drats_uses_maximum_observed_returns():
    _, logger = run_drats(
        total_timesteps=3 * EPISODE_LENGTH,
        episodes_per_update=3,
        warmup_episodes=1,
    )

    assert last_stats(logger, "reference_return").tolist() == [10.0, 5.0, 0.0]
    # Task 2's reference return equals the random return: gap 1
    assert last_stats(logger, "task_gap").tolist() == [0.0, 0.0, 1.0]


def test_train_drats_switches_reference_return_after_success():
    _, logger = run_drats(
        task_set=make_task_set(successes=(1, 0, 0)),
        total_timesteps=3 * EPISODE_LENGTH,
        episodes_per_update=3,
        warmup_episodes=1,
        reference_returns=100.0,
        success_threshold=0.5,
    )

    assert last_stats(logger, "reference_return").tolist() == [
        10.0,
        100.0,
        100.0,
    ]


def test_train_drats_success_threshold_needs_success_info():
    with pytest.raises(KeyError, match="success"):
        run_drats(
            task_set=make_task_set(report_success=False),
            total_timesteps=3 * EPISODE_LENGTH,
            episodes_per_update=3,
            reference_returns=100.0,
            success_threshold=0.5,
        )


def test_train_drats_warmup_trains_tasks_round_robin():
    _, logger = run_drats(
        total_timesteps=4 * EPISODE_LENGTH,
        episodes_per_update=3,
        warmup_episodes=1,
        reference_returns=10.0,
    )

    assert logger.stats["task_id"][:3] == [0, 1, 2]


def test_train_drats_passes_views_of_multitask_replay_buffer():
    received_buffers = []

    def record_buffer_st(replay_buffer, **kwargs):
        received_buffers.append(replay_buffer)
        return random_actions_st(**kwargs)

    result = train_drats(
        make_task_set(),
        record_buffer_st,
        total_timesteps=3 * EPISODE_LENGTH,
        episodes_per_update=3,
        warmup_episodes=1,
        reference_returns=10.0,
        replay_buffer=MultiTaskReplayBuffer(ReplayBuffer(buffer_size=100), 3),
        progress_bar=False,
    )

    assert result.training_steps.sum() == 3 * EPISODE_LENGTH
    assert [buffer.task_id for buffer in received_buffers] == [0, 1, 2]


def test_train_drats_with_ddqn():
    def set_force(env, context):
        env.unwrapped.force_mag = context[0]

    task_set = DiscreteTaskSet(
        gym.make("CartPole-v1"),
        set_force,
        np.array([[5.0], [10.0], [15.0]]),
        context_aware=True,
    )
    q_net = MLP(5, 2, [16], "relu", rngs=nnx.Rngs(0))
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        optimizer=nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param),
        q_target_net=nnx.clone(q_net),
        batch_size=32,
    )
    logger = MemoryLogger()

    result = train_drats(
        task_set,
        train_st,
        total_timesteps=500,
        episodes_per_update=3,
        reference_returns=500.0,
        replay_buffer=MultiTaskReplayBuffer(
            ReplayBuffer(buffer_size=1_000, discrete_actions=True), 3
        ),
        logger=logger,
        progress_bar=False,
    )

    assert np.sum(result.training_steps) == 500
    probabilities = last_stats(logger, "task_probability")
    assert np.isclose(np.sum(probabilities), 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_timesteps": 0},
        {"episodes_per_update": 0},
        {"episodes_per_task": 0},
        {"eta": 0.0},
        {"alpha": 0.0},
        {"alpha": 9.0},
        {"min_probability": 0.0},
        {"min_probability": 0.5},
        {"success_threshold": 0.5, "reference_returns": None},
        {"success_threshold": 0.0},
        {"reference_returns": 0.0},
    ],
)
def test_train_drats_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        train_drats(
            make_task_set(),
            None,
            **(
                {
                    "total_timesteps": 100,
                    "episodes_per_update": 3,
                    "reference_returns": 10.0,
                }
                | kwargs
            ),
            progress_bar=False,
        )
