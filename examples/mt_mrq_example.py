from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.active_mt import train_active_mt
from rl_blox.algorithm.mrq import train_mrq
from rl_blox.algorithm.smt import ContextualMultiTaskDefinition, train_smt
from rl_blox.blox.embedding.task_embedding import create_mt_mrq_state
from rl_blox.blox.replay_buffer import (
    MultiTaskReplayBuffer,
    SubtrajectoryReplayBufferPER,
)
from rl_blox.logging.logger import AIMLogger


class MultiTaskPendulum(ContextualMultiTaskDefinition):
    def __init__(self, render_mode=None):
        super().__init__(
            contexts=np.linspace(5, 15, 11)[:, np.newaxis],
            context_in_observation=False,
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
task_scheduling = "AMT"  # "AMT" or "SMT"

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name="Pendulum-v1",
    algorithm_name=f"{task_scheduling}-MT-MR.Q",
    hparams={},
)

mt_def = MultiTaskPendulum()

env = mt_def.get_task(0)
state = create_mt_mrq_state(env, len(mt_def), seed=seed)
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

if task_scheduling == "SMT":
    result = train_smt(
        mt_def,
        train_st,
        replay_buffer,
        task_selectables=[state.policy_with_encoder.encoder],
        b1=110_000,
        b2=10_000,
        learning_starts=1_000,
        scheduling_interval=1,
        logger=logger,
        seed=seed,
    )
elif task_scheduling == "AMT":
    result = train_active_mt(
        mt_def,
        train_st,
        replay_buffer,
        task_selectables=[state.policy_with_encoder.encoder],
        task_selector="Monotonic Progress",
        r_max=2_000,
        ducb_gamma=0.95,
        xi=0.002,
        learning_starts=11 * 200,
        scheduling_interval=1,
        total_timesteps=50_000,
        logger=logger,
        seed=seed,
    )
else:
    raise ValueError(f"Unknown task scheduler: {task_scheduling}")
mt_def.close()

# Evaluation
result_st = result[0]
policy = result_st.policy_with_encoder
mt_env = MultiTaskPendulum(render_mode="human")
for task_id in range(len(mt_env)):
    print(f"Evaluating task {task_id}")
    env = mt_env.get_task(task_id)
    policy.encoder.select_task(task_id)
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        if verbose:
            zsa = policy.encoder.encode_zsa(
                policy.encoder.encode_zs(jnp.asarray(obs)),
                jnp.asarray(action),
            )
            q_value = result_st.q(zsa)
            print(f"{q_value=}")
        obs = np.asarray(next_obs)
