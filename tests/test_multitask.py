import warnings
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.vector import SyncVectorEnv

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import (
    EvalEpisode,
    RoundRobinSelector,
    TaskSelectionMixin,
    evaluate_tasks,
    max_evaluation_steps,
    mean_task_scores,
    n_warmup_selections,
    next_evaluation_step,
    prepare_evaluation,
    select_task,
    train_task,
    warn_if_state_not_bound,
)
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 200  # fixed length of Pendulum episodes


def make_train_st(env):
    state = create_sac_state(
        env, policy_hidden_nodes=[8], q_hidden_nodes=[8], seed=0
    )
    # Random actions without updates keep the tests fast
    return partial(
        train_sac,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_target=nnx.clone(state.q),
        q_optimizer=state.q_optimizer,
        entropy_control=EntropyControl(env, 0.2, False, 1e-3),
        replay_buffer=ReplayBuffer(buffer_size=1_000),
        learning_starts=10_000,
    )


def test_train_task_counts_steps_of_all_episodes():
    env = gym.make("Pendulum-v1")

    _, steps, returns = train_task(
        make_train_st(env),
        env,
        task_id=0,
        n_episodes=2,
        total_timesteps=10_000,
        global_step=100,
        seed=0,
    )

    assert steps == 2 * EPISODE_LENGTH
    assert returns.shape == (2,)


def test_train_task_stops_when_budget_runs_out():
    env = gym.make("Pendulum-v1")

    _, steps, returns = train_task(
        make_train_st(env),
        env,
        task_id=0,
        n_episodes=2,
        total_timesteps=1_300,
        global_step=1_000,
        seed=0,
    )

    assert steps == 300
    assert returns.shape == (1,)


def test_train_task_informs_task_selectables_and_logger():
    env = gym.make("Pendulum-v1")
    selectable = TaskSelectionMixin()
    logger = MemoryLogger()

    train_task(
        make_train_st(env),
        env,
        task_id=3,
        n_episodes=1,
        total_timesteps=10_000,
        global_step=0,
        seed=0,
        task_selectables=[selectable],
        logger=logger,
    )

    assert selectable.task_id == 3
    assert logger.stats["task_id"] == [3]


def test_n_warmup_selections_covers_whole_rounds():
    assert n_warmup_selections(10, 0, 1) == 0
    assert n_warmup_selections(10, 2, 1) == 20
    # At least 3 episodes per task with 2 episodes per selection
    assert n_warmup_selections(10, 3, 2) == 20


def test_train_task_passes_view_of_multitask_replay_buffer():
    env = gym.make("Pendulum-v1")
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=1_000), 3)

    train_task(
        make_train_st(env),
        env,
        task_id=2,
        n_episodes=1,
        total_timesteps=10_000,
        global_step=0,
        seed=0,
        replay_buffer=replay_buffer,
    )

    assert [len(buffer) for buffer in replay_buffer.buffers] == [0, 0, 200]


def make_ddqn_train_st(**kwargs):
    q_net = MLP(4, 2, [8], "relu", rngs=nnx.Rngs(0))
    return partial(
        train_ddqn,
        q_net=q_net,
        optimizer=nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param),
        **kwargs,
    )


def test_warn_if_state_not_bound_checks_target_networks():
    with pytest.warns(UserWarning, match="q_target_net"):
        warn_if_state_not_bound(
            make_ddqn_train_st(replay_buffer=ReplayBuffer(buffer_size=10))
        )


def test_warn_if_state_not_bound_accepts_replay_buffer_from_scheduler():
    q_net = MLP(4, 2, [8], "relu", rngs=nnx.Rngs(0))
    train_st = make_ddqn_train_st(q_target_net=q_net)

    with pytest.warns(UserWarning, match="replay_buffer"):
        warn_if_state_not_bound(train_st)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_state_not_bound(train_st, replay_buffer_passed=True)


def test_round_robin_selector_starts_with_first_task():
    selector = RoundRobinSelector(np.arange(3))
    selected = []
    for _ in range(4):
        selected.append(int(selector.select()))
        selector.feedback(0.0)

    assert selected == [0, 1, 2, 0]


class ConstantPolicy(nnx.Module):
    def __call__(self, observation):
        return jnp.zeros(1)


def make_eval_set(n_tasks=2, episode_length=5):
    return SyncVectorEnv(
        [
            lambda: gym.make("Pendulum-v1", max_episode_steps=episode_length)
            for _ in range(n_tasks)
        ]
    )


def test_evaluate_tasks_with_deterministic_policy():
    selectable = TaskSelectionMixin()

    episodes, n_steps, key = evaluate_tasks(
        ConstantPolicy(), make_eval_set(), 3, task_selectables=[selectable]
    )

    assert n_steps == 2 * 3 * 5
    assert key is None
    assert [len(task_episodes) for task_episodes in episodes] == [3, 3]
    episode = episodes[1][0]
    assert episode.task_id == 1
    assert episode.rewards.shape == (5,)
    assert len(episode.infos) == 5
    assert episode.log_probs is None
    assert selectable.task_id == 1


def test_evaluate_tasks_samples_stochastic_policy():
    env = gym.make("Pendulum-v1")
    policy = create_sac_state(env, policy_hidden_nodes=[8], seed=0).policy
    key = jax.random.key(0)

    episodes, _, new_key = evaluate_tasks(policy, make_eval_set(1), 1, key)

    assert episodes[0][0].log_probs.shape == (5,)
    assert not jnp.array_equal(
        jax.random.key_data(new_key), jax.random.key_data(key)
    )


def test_max_evaluation_steps():
    assert max_evaluation_steps(make_eval_set(), 3, None) == 2 * 3 * 5
    assert max_evaluation_steps(make_eval_set(), 3, 7) == 2 * 3 * 7


def test_select_task_informs_all_task_selectables():
    selectables = [TaskSelectionMixin(), TaskSelectionMixin()]

    select_task(selectables, 2)
    select_task(None, 3)

    assert [ts.task_id for ts in selectables] == [2, 2]


def test_prepare_evaluation_seeds_tasks_and_returns_bound():
    eval_sets = [make_eval_set() for _ in range(3)]

    max_eval_steps = [
        prepare_evaluation(eval_set, 2, 10, 3, None, 100, seed)
        for eval_set, seed in zip(eval_sets, [4, 4, 5], strict=True)
    ]

    assert max_eval_steps == [2 * 3 * 5] * 3
    observations = [
        [eval_set.envs[i].reset()[0] for i in range(2)]
        for eval_set in eval_sets
    ]
    assert np.array_equal(observations[0], observations[1])
    assert not np.array_equal(observations[0], observations[2])


@pytest.mark.parametrize(
    "args",
    [
        (2, 0, 3),  # eval_freq
        (2, 10, 0),  # n_eval_episodes
        (3, 10, 3),  # number of tasks
    ],
)
def test_prepare_evaluation_rejects_invalid_arguments(args):
    n_tasks, eval_freq, n_eval_episodes = args
    with pytest.raises(ValueError):
        prepare_evaluation(
            make_eval_set(), n_tasks, eval_freq, n_eval_episodes, None, 100, 0
        )


def test_prepare_evaluation_warns_if_bound_is_unknown():
    eval_set = SyncVectorEnv([PendulumEnv, PendulumEnv])

    with pytest.warns(UserWarning, match="infer the episode length"):
        assert prepare_evaluation(eval_set, 2, 10, 3, None, 100, 0) == 0

    # No evaluation takes place within the budget
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert prepare_evaluation(eval_set, 2, 100, 3, None, 100, 0) == 0
    # The episode length is given
    assert prepare_evaluation(eval_set, 2, 10, 3, 7, 100, 0) == 2 * 3 * 7


def test_mean_task_scores():
    episodes = [
        [EvalEpisode(0, np.ones(2), [{}, {}]), EvalEpisode(0, np.ones(4), [])],
        [EvalEpisode(1, np.zeros(3), [])],
    ]

    scores = mean_task_scores(episodes, lambda e: np.sum(e.rewards))

    assert scores.tolist() == [3.0, 0.0]


def test_next_evaluation_step_skips_due_steps_that_passed():
    assert next_evaluation_step(100, 100, 130) == 200
    # The evaluation took longer than eval_freq
    assert next_evaluation_step(100, 100, 250) == 300
    assert next_evaluation_step(100, 100, 300) == 400
