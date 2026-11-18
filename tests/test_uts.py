from functools import partial

import gymnasium as gym
import numpy as np
import optax
import pytest
from flax import nnx

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet, SchedulerResult
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import MemoryLogger

EPISODE_LENGTH = 200  # fixed length of Pendulum episodes


def make_task_set():
    def set_gravity(env, context):
        env.unwrapped.g = context[0]

    contexts = np.array([[5.0], [10.0], [15.0]])
    return DiscreteTaskSet(
        gym.make("Pendulum-v1"), set_gravity, contexts, context_aware=True
    )


def make_train_st(task_set, **kwargs):
    env = task_set.get_task(0)
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
        learning_starts=10_000,
        **kwargs,
    )


def test_train_uts():
    task_set = make_task_set()
    logger = MemoryLogger()

    result = train_uts(
        task_set,
        make_train_st(task_set, replay_buffer=ReplayBuffer(buffer_size=2_000)),
        total_timesteps=5 * EPISODE_LENGTH,
        warmup_episodes=1,
        logger=logger,
        progress_bar=False,
    )

    assert isinstance(result, SchedulerResult)
    assert np.sum(result.training_steps) == 5 * EPISODE_LENGTH
    assert result.eval_steps == 0
    # Round-robin warm-up, then sampling
    assert logger.stats["task_id"][:3] == [0, 1, 2]
    assert len(logger.stats["task_id"]) == 5


def test_train_uts_passes_views_of_multitask_replay_buffer():
    task_set = make_task_set()
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=1_000), 3)

    result = train_uts(
        task_set,
        make_train_st(task_set),
        total_timesteps=3 * EPISODE_LENGTH,
        warmup_episodes=1,
        replay_buffer=replay_buffer,
        progress_bar=False,
    )

    assert [len(buffer) for buffer in replay_buffer.buffers] == [
        EPISODE_LENGTH
    ] * 3
    assert result.training_steps.tolist() == [EPISODE_LENGTH] * 3


def test_train_uts_with_backbone_whose_first_parameter_is_not_env():
    task_set = DiscreteTaskSet(
        gym.make("CartPole-v1"),
        lambda env, context: None,
        np.zeros((2, 1)),
        context_aware=False,
    )
    q_net = MLP(4, 2, [8], "relu", rngs=nnx.Rngs(0))
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        replay_buffer=ReplayBuffer(buffer_size=1_000, discrete_actions=True),
        optimizer=nnx.Optimizer(q_net, optax.adam(1e-3), wrt=nnx.Param),
        q_target_net=nnx.clone(q_net),
    )

    result = train_uts(
        task_set, train_st, total_timesteps=100, progress_bar=False
    )

    assert np.sum(result.training_steps) == 100


@pytest.mark.parametrize(
    "kwargs", [{"total_timesteps": 0}, {"episodes_per_task": 0}]
)
def test_train_uts_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        train_uts(
            make_task_set(),
            None,
            **({"total_timesteps": 100} | kwargs),
            progress_bar=False,
        )
