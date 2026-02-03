from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.mrq import create_mrq_state, train_mrq
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.smt import train_smt
from rl_blox.algorithm.td3 import create_td3_state, train_td3
from rl_blox.algorithm.td7 import create_td7_state, train_td7
from rl_blox.blox.embedding.sale import DeterministicSALEPolicy
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import (
    LAP,
    MultiTaskReplayBuffer,
    ReplayBuffer,
    SubtrajectoryReplayBufferPER,
)
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"


def set_context(env: gym.Env, context):
    env.unwrapped.g = context[0]


base_env = gym.make(env_name)
contexts = np.linspace(0, 20, 21)[:, np.newaxis]

train_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)


seed = 2
verbose = False
# Backbone algorithm to use for SMT: "SAC", "DDPG", "TD3", "TD7", "MR.Q"
backbone = "SAC"

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name="Pendulum-v1",
    algorithm_name=f"SMT-{backbone}",
    hparams={},
)

env = train_set.get_task(0)
if backbone == "DDPG":
    state = create_ddpg_state(env, seed=seed)
    policy_target = nnx.clone(state.policy)
    q_target = nnx.clone(state.q)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100_000),
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
elif backbone == "TD3":
    state = create_td3_state(env, seed=seed)
    q_target = nnx.clone(state.q)
    policy_target = nnx.clone(state.policy)
    replay_buffer = MultiTaskReplayBuffer(
        ReplayBuffer(buffer_size=100_000),
        len(train_set),
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
        len(train_set),
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
        len(train_set),
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
        len(train_set),
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

result = train_smt(
    train_set,
    train_st,
    replay_buffer,
    solved_threshold=-100.0,
    unsolvable_threshold=-1000.0,
    b1=11_000,
    b2=10_000,
    learning_starts=1_000,
    scheduling_interval=1,
    logger=logger,
    seed=seed,
)

# Evaluation
result_st = result[0]
if backbone == "MR.Q":
    policy = result_st.policy_with_encoder
elif backbone == "TD7":
    policy = DeterministicSALEPolicy(result_st.embedding, result_st.actor)
else:
    policy = result_st.policy

base_env = gym.make(env_name, render_mode="human")
test_set = DiscreteTaskSet(base_env, set_context, contexts, context_aware=True)

for task_id in range(len(test_set)):
    print(f"Evaluating task {task_id}")
    env = test_set.get_task(task_id)
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
