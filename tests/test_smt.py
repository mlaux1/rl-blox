import copy
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.smt import evaluate_tasks, train_smt
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
    eval_set = DiscreteTaskSet(
        gym.make(env_name), set_context, contexts, context_aware=True
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

    _, training_steps, training_performances, eval_steps = train_smt(
        train_set,
        eval_set,
        train_st,
        replay_buffer,
        b1=200,
        b2=200,
        learning_starts=50,
        scheduling_interval=1,
        seed=seed,
    )

    assert training_steps.shape == (len(train_set),)
    assert training_steps.sum() == 400
    assert eval_steps == 0
    assert training_performances.shape == (len(train_set),)
    assert len(replay_buffer) == 400


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
        copy.deepcopy(task_set),
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

    _, training_steps, _, _ = train_smt(
        task_set,
        copy.deepcopy(task_set),
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
            copy.deepcopy(task_set),
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

    _, training_steps, _, _ = train_smt(
        task_set,
        copy.deepcopy(task_set),
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
        _, training_steps, _, _ = train_smt(
            task_set,
            copy.deepcopy(task_set),
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

    _, training_steps, _, _ = train_smt(
        task_set,
        copy.deepcopy(task_set),
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


def test_smt_picks_tasks_at_random_before_first_evaluation():
    # Each task returns to the main pool after one episode, as its budget
    # (at most 6 steps) is shorter than an episode (10 steps). There is no
    # evaluation, so every refill picks a task from the main pool at random.
    def refill_order(seed):
        logger = MemoryLogger()
        train_smt(
            constant_reward_task_set([0.0] * 6),
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
        return tuple(refilled_tasks.tolist())

    orders = {refill_order(seed) for seed in range(5)}

    assert len(orders) > 1


def test_smt_budget_shares_stage1_budget_among_training_pool():
    # Paper settings for MT10, scaled down by 1000. Each task's budget is
    # kappa * (b1 - used steps) / K, so the 3 initial tasks get
    # 0.8 * 17000 / 3 = 4533.3 steps. Trained in turns with episodes of
    # 10 steps, all of them use it up in round 454, at step 3 * 4540.
    logger = MemoryLogger()

    train_smt(
        constant_reward_task_set([0.0] * 10),
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


def test_smt_thresholds_are_strict():
    # Every episode has a return of exactly 10, which equals both thresholds.
    # A task is only solved above M and only unsolvable below m, so both
    # tasks stay unsolved and keep returning to the main pool.
    logger = MemoryLogger()

    train_smt(
        constant_reward_task_set([1.0, 1.0]),
        constant_reward_task_set([1.0, 1.0]),
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2),
        b1=100,
        b2=0,
        solved_threshold=10.0,
        unsolvable_threshold=10.0,
        kappa=0.1,
        K=1,
        logger=logger,
        progress_bar=False,
    )

    _, n_solved = logger.get_stat("pool_size_solved", x_key="step")
    _, n_unsolvable = logger.get_stat("pool_size_unsolvable", x_key="step")
    assert n_solved.max() == 0
    assert n_unsolvable.max() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"K": 0},
        {"K": 3},
        {"kappa": 0.0},
        {"kappa": 1.5},
        {"b1": 0},
        {"b2": -1},
    ],
)
def test_smt_rejects_invalid_arguments(kwargs):
    arguments = {"b1": 100, "b2": 100, "K": 1, "progress_bar": False}
    arguments.update(kwargs)
    task_set = constant_reward_task_set([0.0, 0.0])

    with pytest.raises(ValueError):
        train_smt(
            task_set,
            copy.deepcopy(task_set),
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), len(task_set)),
            **arguments,
        )


def test_smt_rejects_eval_set_with_other_number_of_tasks():
    with pytest.raises(ValueError, match="eval_set"):
        train_smt(
            constant_reward_task_set([0.0, 0.0]),
            constant_reward_task_set([0.0, 0.0, 0.0]),
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2),
            b1=100,
            b2=100,
            K=1,
            progress_bar=False,
        )


@pytest.mark.parametrize("kwargs", [{"eval_freq": 0}, {"n_eval_episodes": 0}])
def test_smt_rejects_invalid_evaluation_arguments(kwargs):
    with pytest.raises(ValueError):
        train_smt(
            constant_reward_task_set([0.0, 0.0]),
            constant_reward_task_set([0.0, 0.0]),
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2),
            b1=100,
            b2=100,
            K=1,
            progress_bar=False,
            **kwargs,
        )


class ZeroPolicy(nnx.Module):
    """Deterministic policy that always takes the action 0."""

    def __call__(self, observation):
        return jnp.zeros(1)


class FixedStochasticPolicy(ZeroPolicy):
    """Stochastic policy that always samples 0 with the same probability."""

    def __init__(self, log_prob):
        self.log_prob = log_prob

    def sample_and_log_prob(self, observation, key):
        return jnp.zeros(1), jnp.asarray(self.log_prob)


def test_smt_refills_training_pool_with_worst_evaluated_task():
    # All tasks have the same training return, but task 1 has the lowest
    # evaluation return. Every scheduling iteration is followed by an
    # evaluation, so every refill picks task 1.
    logger = MemoryLogger()

    _, training_steps, _, eval_steps = train_smt(
        constant_reward_task_set([0.0, 0.0, 0.0]),
        constant_reward_task_set([0.0, -1.0, 1.0]),
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 3),
        b1=400,
        b2=0,
        solved_threshold=np.inf,  # no task is ever solved
        unsolvable_threshold=-np.inf,  # no task is ever unsolvable
        kappa=0.1,
        K=1,
        eval_freq=1,
        n_eval_episodes=1,
        max_eval_episode_steps=10,
        get_policy=lambda result: FixedStochasticPolicy(log_prob=0.0),
        logger=logger,
        progress_bar=False,
    )

    _, refilled_tasks = logger.get_stat("worst task", x_key="step")
    assert len(refilled_tasks) > 0
    assert set(refilled_tasks.tolist()) == {1}
    # Evaluation steps count towards b1
    _, logged_eval_steps = logger.get_stat("eval_steps", x_key="step")
    assert eval_steps == logged_eval_steps.sum() > 0
    assert training_steps.sum() + eval_steps == 400


def test_smt_evaluation_metric_adds_entropy_to_discounted_return():
    performances, n_steps, _ = evaluate_tasks(
        FixedStochasticPolicy(log_prob=-2.0),
        constant_reward_task_set([1.0]),
        n_eval_episodes=2,
        gamma=0.5,
        key=jax.random.key(0),
    )

    discounted_return = (1.0 - 0.5**10) / (1.0 - 0.5)
    entropy = 10 * 2.0
    assert performances[0] == pytest.approx(discounted_return + entropy)
    assert n_steps == 20


def test_smt_evaluation_of_deterministic_policy_omits_entropy():
    performances, _, _ = evaluate_tasks(
        ZeroPolicy(),
        constant_reward_task_set([1.0]),
        n_eval_episodes=1,
        gamma=0.5,
        key=jax.random.key(0),
    )

    assert performances[0] == pytest.approx((1.0 - 0.5**10) / (1.0 - 0.5))


def test_smt_warns_if_evaluated_policy_is_deterministic():
    with pytest.warns(UserWarning, match="entropy term"):
        train_smt(
            constant_reward_task_set([0.0, 0.0]),
            constant_reward_task_set([0.0, 0.0]),
            random_actions_st,
            MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2),
            b1=100,
            b2=0,
            K=1,
            eval_freq=1,
            n_eval_episodes=1,
            max_eval_episode_steps=10,
            get_policy=lambda result: ZeroPolicy(),
            progress_bar=False,
        )


def test_smt_resets_networks_after_each_evaluation_in_stage1():
    logger = MemoryLogger()
    reset_seeds = []

    train_smt(
        constant_reward_task_set([0.0, -1.0, 1.0]),
        constant_reward_task_set([0.0, -1.0, 1.0]),
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 3),
        b1=400,
        b2=200,
        solved_threshold=np.inf,  # no task is ever solved
        unsolvable_threshold=-np.inf,  # no task is ever unsolvable
        kappa=0.1,
        K=1,
        eval_freq=100,
        n_eval_episodes=1,
        max_eval_episode_steps=10,
        get_policy=lambda result: FixedStochasticPolicy(log_prob=0.0),
        reset_st=reset_seeds.append,
        logger=logger,
        progress_bar=False,
    )

    eval_steps, _ = logger.get_stat("eval_steps", x_key="step")
    reset_steps, _ = logger.get_stat("reset", x_key="step")
    assert len(reset_seeds) > 1
    assert len(set(reset_seeds)) == len(reset_seeds)
    # Each evaluation is followed by a reset, and both only happen in stage 1
    assert reset_steps.tolist() == eval_steps.tolist()
    assert max(reset_steps) <= 400


def test_smt_does_not_reset_without_reset_function():
    logger = MemoryLogger()

    train_smt(
        constant_reward_task_set([0.0, 0.0]),
        constant_reward_task_set([0.0, 0.0]),
        random_actions_st,
        MultiTaskReplayBuffer(ReplayBuffer(buffer_size=10), 2),
        b1=100,
        b2=0,
        K=1,
        eval_freq=1,
        n_eval_episodes=1,
        max_eval_episode_steps=10,
        get_policy=lambda result: FixedStochasticPolicy(log_prob=0.0),
        logger=logger,
        progress_bar=False,
    )

    assert "eval_steps" in logger.stats
    assert "reset" not in logger.stats
