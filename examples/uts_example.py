from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.uniform_task_sampling import train_uts
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList

env_name = "Pendulum-v1"
seed = 42
verbose = 1
backbone_algorithm = "SAC"


def set_g(env: gym.Env, context):
    env.unwrapped.g = context


base_env = gym.make(env_name)
contexts = np.linspace(0, 20, 21)[:, np.newaxis]

train_set = DiscreteTaskSet(base_env, set_g, contexts, context_aware=True)

hparams_models = dict(
    q_hidden_nodes=[512, 512],
    q_learning_rate=3e-4,
    policy_learning_rate=1e-3,
    policy_hidden_nodes=[128, 128],
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=11_000,
    exploring_starts=5_000,
    episodes_per_task=1,
)

logger = LoggerList([AIMLogger(), OrbaxCheckpointer()])
logger.define_experiment(
    env_name=env_name,
    algorithm_name=f"UTS-{backbone_algorithm}",
    hparams=hparams_models | hparams_algorithm,
)

env = train_set.get_task(0)

match backbone_algorithm:
    case "SAC":
        sac_state = create_sac_state(env, **hparams_models)

        q_target = nnx.clone(sac_state.q)
        replay_buffer = ReplayBuffer(buffer_size=11_000)
        entropy_control = EntropyControl(train_set.get_task(0), 0.2, True, 1e-3)

        train_st = partial(
            train_sac,
            policy=sac_state.policy,
            policy_optimizer=sac_state.policy_optimizer,
            q=sac_state.q,
            q_target=q_target,
            q_optimizer=sac_state.q_optimizer,
            entropy_control=entropy_control,
            replay_buffer=replay_buffer,
        )
    case "DDPG":
        state = create_ddpg_state(env, seed=seed)
        policy_target = nnx.clone(state.policy)
        q_target = nnx.clone(state.q)
        replay_buffer = ReplayBuffer(buffer_size=11_000)

        train_st = partial(
            train_ddpg,
            policy=state.policy,
            policy_target=policy_target,
            policy_optimizer=state.policy_optimizer,
            q=state.q,
            q_optimizer=state.q_optimizer,
            q_target=q_target,
            replay_buffer=replay_buffer,
        )
    case _:
        raise ValueError(
            f"Unsupported backbone algorithm: {backbone_algorithm}"
        )


uts_result = train_uts(
    train_set,
    train_st,
    **hparams_algorithm,
    logger=logger,
)

policy, _, q, _, _, _, _, _ = uts_result

base_env = gym.make(env_name, render_mode="human")
contexts = np.linspace(0, 20, 21)[:, np.newaxis]

test_set = DiscreteTaskSet(base_env, set_g, contexts, context_aware=True)

for i in range(len(test_set)):
    env = test_set.get_task(i)
    ep_return = 0.0
    done = False
    obs, _ = env.reset()
    while not done:
        if backbone_algorithm == "SAC":
            action = np.asarray(policy(jnp.asarray(obs))[0])
        else:
            action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, info = env.step(action)
        ep_return += reward
        done = termination or truncation
        obs = np.asarray(next_obs)
    print(f"Episode terminated in with {ep_return=}")
