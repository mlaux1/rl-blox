import warnings
from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from flax import nnx
from gymnasium.envs.classic_control import PendulumEnv

from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.t3s_sampling import (
    EvalEpisode,
    normalized_return,
    train_t3s,
)
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import ReplayBuffer

pendulum_score = partial(normalized_return, low=-1600.0, high=0.0)

# no gradient updates and an evaluation after the first training episode
quick_hparams = dict(
    total_timesteps=1_000,
    exploring_starts=10_000,
    tasks_per_iteration=1,
    n_eval_episodes=1,
    eval_freq=1,
)


class ManualTruncation(gym.Wrapper):
    """Truncates after 200 steps without a TimeLimit wrapper or spec."""

    def reset(self, **kwargs):
        self.n_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        self.n_steps += 1
        return obs, reward, terminated, self.n_steps >= 200, info


def set_context(env: gym.Env, context):
    env.unwrapped.g = context[0]


def make_task_sets(make_env=lambda: gym.make("Pendulum-v1")):
    contexts = np.linspace(0, 20, 3)[:, np.newaxis]
    train_set = DiscreteTaskSet(
        make_env(), set_context, contexts, context_aware=True
    )
    eval_set = DiscreteTaskSet(
        make_env(), set_context, contexts, context_aware=True
    )
    return train_set, eval_set


def make_train_st(env, seed=1, bind_state=True):
    state = create_sac_state(
        env,
        q_hidden_nodes=[64, 64],
        q_learning_rate=1e-3,
        policy_learning_rate=1e-3,
        policy_hidden_nodes=[64, 64],
        seed=seed,
    )
    kwargs = dict(
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
    )
    if bind_state:
        kwargs.update(
            q_target=nnx.clone(state.q),
            entropy_control=EntropyControl(env, 0.2, True, 1e-3),
            replay_buffer=ReplayBuffer(buffer_size=1_400),
        )
    return partial(train_sac, **kwargs)


def test_t3s():
    train_set, eval_set = make_task_sets()
    train_st = make_train_st(train_set.get_task(0))

    result = train_t3s(
        train_set,
        eval_set,
        train_st,
        total_timesteps=1_400,
        exploring_starts=100,
        episodes_per_task=1,
        tasks_per_iteration=1,
        n_eval_episodes=1,
        eval_freq=200,
        episode_score=pendulum_score,
    )

    # one evaluation of 3 tasks x 1 episode x 200 steps fits into the budget
    assert result.global_step == 1_400
    assert result.eval_steps == 600
    assert result.train_steps == 800


def test_t3s_score_out_of_range():
    train_set, eval_set = make_task_sets()
    train_st = make_train_st(train_set.get_task(0))

    with pytest.raises(ValueError, match=r"must return a value in \[0, 1\]"):
        train_t3s(
            train_set,
            eval_set,
            train_st,
            episode_score=lambda episode: float(np.sum(episode.rewards)),
            **quick_hparams,
        )


def test_t3s_missing_success_info():
    train_set, eval_set = make_task_sets()
    train_st = make_train_st(train_set.get_task(0))

    with pytest.raises(KeyError, match="does not report info"):
        train_t3s(train_set, eval_set, train_st, **quick_hparams)


def test_t3s_unknown_episode_length():
    train_set, eval_set = make_task_sets(
        lambda: ManualTruncation(PendulumEnv())
    )
    train_st = make_train_st(train_set.get_task(0))

    with pytest.warns(UserWarning, match="infer the episode length"):
        train_t3s(
            train_set,
            eval_set,
            train_st,
            episode_score=pendulum_score,
            **quick_hparams,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = train_t3s(
            train_set,
            eval_set,
            train_st,
            episode_score=pendulum_score,
            max_eval_episode_steps=200,
            **quick_hparams,
        )
    assert not any("infer the episode length" in str(w.message) for w in caught)
    assert result.global_step == quick_hparams["total_timesteps"]


def test_t3s_unbound_state():
    train_set, eval_set = make_task_sets()
    train_st = make_train_st(train_set.get_task(0), bind_state=False)

    with pytest.warns(UserWarning, match="not bound to train_st") as caught:
        train_t3s(
            train_set,
            eval_set,
            train_st,
            episode_score=pendulum_score,
            **quick_hparams,
        )
    message = next(
        str(w.message) for w in caught if "not bound" in str(w.message)
    )
    assert "replay_buffer, q_target, entropy_control" in message


def test_normalized_return():
    def episode(task_id, episode_return):
        return EvalEpisode(task_id, np.full(200, episode_return / 200), [])

    assert pendulum_score(episode(0, -400.0)) == pytest.approx(0.75)
    assert pendulum_score(episode(0, -2000.0)) == 0.0
    assert pendulum_score(episode(0, 10.0)) == 1.0

    per_task_score = partial(
        normalized_return, low=[-1600.0, -1000.0], high=[0.0, 0.0]
    )
    assert per_task_score(episode(0, -800.0)) == pytest.approx(0.5)
    assert per_task_score(episode(1, -800.0)) == pytest.approx(0.2)

    with pytest.raises(ValueError, match="high must be greater than low"):
        normalized_return(episode(0, 0.0), low=0.0, high=0.0)
