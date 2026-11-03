from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.smt import train_smt
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import MemoryLogger


def test_smt():
    seed = 1
    env_name = "Pendulum-v1"

    def set_context(env: gym.Env, context):
        env.unwrapped.g = context[0]

    base_env = gym.make(env_name)
    contexts = np.linspace(0, 20, 21)[:, np.newaxis]

    train_set = DiscreteTaskSet(
        base_env, set_context, contexts, context_aware=True
    )

    env = train_set.get_task(0)
    state = create_ddpg_state(env, seed=seed)
    policy_target = nnx.clone(state.policy)
    q_target = nnx.clone(state.q)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=1000),
        len(train_set),
    )

    train_st = partial(
        train_ddpg,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        policy_target=policy_target,
        q_target=q_target,
    )

    train_smt(
        train_set,
        train_st,
        replay_buffer,
        b1=200,
        b2=200,
        learning_starts=50,
        scheduling_interval=1,
        seed=seed,
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


def test_smt_budget_counts_from_entering_training_pool():
    base_env = gym.make("Pendulum-v1")  # every episode has 200 steps
    task_set = DiscreteTaskSet(
        base_env,
        lambda env, context: None,
        np.zeros((2, 1)),
        context_aware=False,
    )
    logger = MemoryLogger()

    train_smt(
        task_set,
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
        b1=8_000,
        b2=200,
        solved_threshold=np.inf,  # no task is ever solved
        unsolvable_threshold=-np.inf,  # no task is ever unsolvable
        kappa=0.5,
        K=1,
        logger=logger,
        progress_bar=False,
    )

    # Each task returns to the main pool once its budget is used up and
    # tasks re-enter the training pool, so the training pool is refilled
    # whenever a budget of kappa times the remaining budget of stage 1
    # (8000 steps at the start) divided by K = 1 has been used since the
    # task entered it: 4000, 2000, 1000, 500 -> 600, 200, 100 -> 200.
    refill_steps, _ = logger.get_stat("worst task", x_key="step")
    assert refill_steps.tolist() == [4000, 6000, 7000, 7600, 7800, 8000]


class ConstantRewardEnv(gym.Env):
    """Episodes of 10 steps with the same reward in every step."""

    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def __init__(self):
        self.reward = 0.0
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        truncated = self.t >= 10
        return np.zeros(1, dtype=np.float32), self.reward, False, truncated, {}


def constant_reward_task_set(rewards):
    """Tasks that differ only in their reward per step."""

    def set_reward(env, context):
        env.reward = float(context[0])

    return DiscreteTaskSet(
        ConstantRewardEnv(),
        set_reward,
        np.asarray(rewards, dtype=float)[:, np.newaxis],
        context_aware=False,
    )


def test_smt_stage2_trains_tasks_left_in_training_pool():
    # Tasks 0 and 1 are solved in their first episode (return 10). Task 2
    # is never solved (return 0) and never unsolvable. Whenever it uses up
    # its budget, it is the only task in the main pool and re-enters the
    # training pool at once, so it is still in the training pool when
    # stage 1 ends, while the main pool and the unsolvable pool are empty.
    task_set = constant_reward_task_set([1.0, 1.0, 0.0])

    _, training_steps, _ = train_smt(
        task_set,
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
        b1=1_000,
        b2=1_000,
        solved_threshold=5.0,
        unsolvable_threshold=-np.inf,
        kappa=0.9,
        K=2,
        progress_bar=False,
    )

    assert training_steps.sum() == 2_000
    assert training_steps[2] == 2_000 - training_steps[:2].sum()


def test_smt_warns_if_all_tasks_are_in_training_pool():
    task_set = constant_reward_task_set([0.0, 0.0])

    with pytest.warns(UserWarning, match="no task scheduling"):
        train_smt(
            task_set,
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
            b1=100,
            b2=100,
            K=2,
            progress_bar=False,
        )


class ImprovingRewardEnv(ConstantRewardEnv):
    """Rewards each step with 1 once a task has been trained long enough.

    The context of a task is the number of episodes after which it is solved.
    """

    def __init__(self):
        super().__init__()
        self.episodes_to_solve = 0
        self.episodes = {}

    def reset(self, seed=None, options=None):
        n_episodes = self.episodes.get(self.episodes_to_solve, 0) + 1
        self.episodes[self.episodes_to_solve] = n_episodes
        self.reward = float(n_episodes > self.episodes_to_solve)
        return super().reset(seed=seed, options=options)


class TaskRecorder:
    """Records the task that SMT selects for each scheduling interval."""

    def __init__(self):
        self.selected_tasks = []

    def select_task(self, task_id):
        self.selected_tasks.append(task_id)


def test_smt_stage2_trains_unsolvable_tasks_until_solved():
    def set_episodes_to_solve(env, context):
        env.episodes_to_solve = int(context[0])

    # Task 0 needs 30 episodes, but its budget of at most 20 steps (2
    # episodes) ends in stage 1, so it becomes unsolvable. Task 1 is solved
    # at once.
    task_set = DiscreteTaskSet(
        ImprovingRewardEnv(),
        set_episodes_to_solve,
        np.array([[30.0], [0.0]]),
        context_aware=False,
    )
    recorder = TaskRecorder()

    _, training_steps, _ = train_smt(
        task_set,
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
        b1=200,
        b2=1_000,
        solved_threshold=5.0,
        unsolvable_threshold=0.5,
        kappa=0.1,
        K=1,
        n_average=3,
        task_selectables=[recorder],
        progress_bar=False,
    )

    assert training_steps.sum() == 1_200
    # Stage 1 ends after 3 episodes. Task 0 then trains alone for 30 more
    # episodes, until the mean of its last 3 returns reaches 5 (episodes 31
    # and 32 have a return of 10). The rest of the 120 episodes alternate.
    stage2_tasks = recorder.selected_tasks[3:]
    assert stage2_tasks == [0] * 30 + [0, 1] * 43 + [0]


def test_smt_round_robin_if_all_tasks_solved_in_stage1():
    task_set = constant_reward_task_set([1.0, 1.0])

    with pytest.warns(UserWarning, match="round robin"):
        _, training_steps, _ = train_smt(
            task_set,
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
            b1=100,
            b2=100,
            solved_threshold=5.0,
            K=1,
            progress_bar=False,
        )

    # Each task is solved in its first episode in stage 1, then the
    # remaining 180 steps are split evenly between both tasks.
    assert training_steps.tolist() == [100, 100]


def test_smt_without_stage2_budget():
    task_set = constant_reward_task_set([0.0, 0.0])

    _, training_steps, _ = train_smt(
        task_set,
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
        b1=100,
        b2=0,
        solved_threshold=np.inf,  # no task is ever solved
        unsolvable_threshold=-np.inf,  # no task is ever unsolvable
        K=1,
        progress_bar=False,
    )

    assert training_steps.sum() == 100


def test_smt_picks_untrained_tasks_at_random():
    # Each task returns to the main pool after one episode, as its budget
    # (at most 6 steps) is shorter than an episode (10 steps). So the first
    # 5 refills pick each of the other, untrained tasks once.
    def untrained_refill_order(seed):
        logger = MemoryLogger()
        train_smt(
            constant_reward_task_set([0.0] * 6),
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 6),
            b1=60,
            b2=10,
            solved_threshold=np.inf,  # no task is ever solved
            unsolvable_threshold=-np.inf,  # no task is ever unsolvable
            kappa=0.1,
            K=1,
            seed=seed,
            logger=logger,
            progress_bar=False,
        )
        _, refilled_tasks = logger.get_stat("worst task", x_key="step")
        return refilled_tasks[:5].tolist()

    orders = [untrained_refill_order(seed) for seed in range(5)]

    assert all(len(set(order)) == 5 for order in orders)
    assert any(order != sorted(order) for order in orders)


def test_smt_budget_shares_stage1_budget_among_training_pool():
    # Paper settings for MT10, scaled down by 1000. Each task's budget is
    # kappa * (b1 - used steps) / K, so the 3 initial tasks get
    # 0.8 * 17000 / 3 = 4533.3 steps. Trained in turns with episodes of
    # 10 steps, all of them use it up in round 454, at step 3 * 4540.
    logger = MemoryLogger()

    train_smt(
        constant_reward_task_set([0.0] * 10),
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 10),
        b1=17_000,
        b2=3_000,
        solved_threshold=np.inf,  # no task is ever solved
        unsolvable_threshold=-np.inf,  # no task is ever unsolvable
        kappa=0.8,
        K=3,
        logger=logger,
        progress_bar=False,
    )

    refill_steps, _ = logger.get_stat("worst task", x_key="step")
    assert refill_steps[:3].tolist() == [13_620] * 3
