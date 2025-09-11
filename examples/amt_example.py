from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.active_mt import train_active_mt
from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.mrq import create_mrq_state, train_mrq
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.smt import ContextualMultiTaskDefinition
from rl_blox.blox.replay_buffer import (
    MultiTaskReplayBuffer,
    ReplayBuffer,
    SubtrajectoryReplayBufferPER,
)
from rl_blox.logging.logger import AIMLogger


class MultiTaskPendulum(ContextualMultiTaskDefinition):
    def __init__(self, render_mode=None):
        super().__init__(
            contexts=np.linspace(5, 15, 11)[:, np.newaxis],
            context_in_observation=True,
        )
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)

    def _get_env(self, context):
        self.env.unwrapped.g = context[0]
        return self.env

    def get_solved_threshold(self, task_id: int) -> float:
        return -100.0

    def get_unsolvable_threshold(self, task_id: int) -> float:
        return -1000.0

    def close(self):
        self.env.close()


seed = 2
verbose = 2
backbone = "SAC"  # Backbone algorithm to use for SMT: "DDPG", "SAC", "MR.Q"

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name="Pendulum-v1",
    algorithm_name=f"AMT-{backbone}",
    hparams={},
)

mt_def = MultiTaskPendulum()

env = mt_def.get_task(0)
if backbone == "DDPG":
    state = create_ddpg_state(env, seed=seed)
    policy_target = nnx.clone(state.policy)
    q_target = nnx.clone(state.q)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100_000),
        len(mt_def),
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
elif backbone == "MR.Q":
    state = create_mrq_state(env, seed=seed)
    q_target = nnx.clone(state.q)
    policy_with_encoder_target = nnx.clone(state.policy_with_encoder)
    replay_buffer = MultiTaskReplayBuffer(
        SubtrajectoryReplayBufferPER(buffer_size=100_000, horizon=5),
        len(mt_def),
    )

    train_st = partial(
        train_mrq,
        policy_with_encoder=state.policy_with_encoder,
        encoder_optimizer=state.encoder_optimizer,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        the_bins=state.the_bins,
        q_target=q_target,
        policy_with_encoder_target=policy_with_encoder_target,
    )
else:
    assert backbone == "SAC", "Backbone must be either 'DDPG' or 'SAC'."
    state = create_sac_state(env, seed=seed)
    q_target = nnx.clone(state.q)
    entroy_control = EntropyControl(env, 0.2, True, 1e-3)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100_000),
        len(mt_def),
    )

    train_st = partial(
        train_sac,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        q_target=q_target,
        entropy_control=entroy_control,
    )

result = train_active_mt(
    mt_def,
    train_st,
    replay_buffer,
    task_selector="Monotonic Progress",
    learning_starts=1_000,
    scheduling_interval=1_000,
    total_timesteps=50_000,
    logger=logger,
    seed=seed,
)
mt_def.close()

# Evaluation
result_st = result[0]
if backbone == "MR.Q":
    policy = result_st.policy_with_encoder
else:
    policy = result_st.policy
q = result_st.q
mt_env = MultiTaskPendulum(render_mode="human")
for task_id in range(len(mt_env)):
    print(f"Evaluating task {task_id}")
    env = mt_env.get_task(task_id)
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        if backbone in ["DDPG", "MR.Q"]:
            action = np.asarray(policy(jnp.asarray(obs)))
        else:
            action = np.asarray(policy(jnp.asarray(obs))[0])
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        if verbose:
            q_value = q(jnp.concatenate((obs, action)))
            print(f"{q_value=}")
        obs = np.asarray(next_obs)
