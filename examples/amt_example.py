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
from rl_blox.algorithm.td3 import create_td3_state, train_td3
from rl_blox.algorithm.td7 import create_td7_state, train_td7
from rl_blox.blox.embedding.sale import DeterministicSALEPolicy
from rl_blox.blox.replay_buffer import (
    LAP,
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
# Backbone algorithm to use for Active MT: "SAC", "DDPG", "TD3", "TD7", "MR.Q"
backbone = "MR.Q"

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
elif backbone == "TD3":
    state = create_td3_state(env, seed=seed)
    q_target = nnx.clone(state.q)
    policy_target = nnx.clone(state.policy)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100_000),
        len(mt_def),
    )

    train_st = partial(
        train_td3,
        policy=state.policy,
        policy_optimizer=state.policy_optimizer,
        q=state.q,
        q_optimizer=state.q_optimizer,
        q_target=q_target,
        policy_target=policy_target,
    )
elif backbone == "TD7":
    state = create_td7_state(env, seed=seed)
    actor_target = nnx.clone(state.actor)
    critic_target = nnx.clone(state.critic)
    replay_buffer = MultiTaskReplayBuffer(
        LAP(buffer_size=100_000),
        len(mt_def),
    )

    train_st = partial(
        train_td7,
        embedding=state.embedding,
        embedding_optimizer=state.embedding_optimizer,
        actor=state.actor,
        actor_optimizer=state.actor_optimizer,
        critic=state.critic,
        critic_optimizer=state.critic_optimizer,
        critic_target=critic_target,
        actor_target=actor_target,
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
    task_selector="1-step Progress",
    learning_starts=11 * 200,
    scheduling_interval=1,
    total_timesteps=50_000,
    logger=logger,
    seed=seed,
)
mt_def.close()

# Evaluation
result_st = result[0]
if backbone == "MR.Q":
    policy = result_st.policy_with_encoder
elif backbone == "TD7":
    policy = DeterministicSALEPolicy(result_st.embedding, result_st.actor)
else:
    policy = result_st.policy
mt_env = MultiTaskPendulum(render_mode="human")
for task_id in range(len(mt_env)):
    print(f"Evaluating task {task_id}")
    env = mt_env.get_task(task_id)
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        if backbone == "SAC":
            action = np.asarray(policy(jnp.asarray(obs))[0])
        else:
            action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        if verbose:
            if backbone == "MR.Q":
                zsa = policy.encoder.encode_zsa(
                    policy.encoder.encode_zs(jnp.asarray(obs)),
                    jnp.asarray(action),
                )
                q_value = result_st.q(zsa)
            elif backbone == "TD7":
                zsa, zs = result_st.embedding(
                    jnp.asarray(obs),
                    jnp.asarray(action),
                )
                q_value = result_st.critic(
                    jnp.concatenate((obs, action)), zsa=zsa, zs=zs
                )
            else:
                q_value = result_st.q(jnp.concatenate((obs, action)))
            print(f"{q_value=}")
        obs = np.asarray(next_obs)
