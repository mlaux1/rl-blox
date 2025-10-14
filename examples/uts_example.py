from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.algorithm.sac import EntropyControl, create_sac_state, train_sac
from rl_blox.algorithm.uniform_task_sampling import TaskSet, train_uts
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList

env_name = "Pendulum-v1"
seed = 42
verbose = 1
backbone_algorithm = "SAC"

train_contexts = jnp.linspace(5, 15, 3)[:, jnp.newaxis]
train_envs = [
    RecordEpisodeStatistics(gym.make(env_name, g=5)),
    RecordEpisodeStatistics(gym.make(env_name, g=10)),
    RecordEpisodeStatistics(gym.make(env_name, g=15)),
]

train_set = TaskSet(train_contexts, train_envs)

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

match backbone_algorithm:
    case "SAC":
        sac_state = create_sac_state(
            train_set.get_task_env(0), **hparams_models
        )

        q_target = nnx.clone(sac_state.q)
        replay_buffer = ReplayBuffer(buffer_size=11_000)
        entropy_control = EntropyControl(
            train_set.get_task_env(0), 0.2, True, 1e-3
        )

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
        state = create_ddpg_state(train_set.get_task_env(0), seed=seed)
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

test_contexts = jnp.linspace(0, 20, 21)[:, jnp.newaxis]
test_envs = [
    RecordEpisodeStatistics(gym.make(env_name, g=0.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=1.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=2.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=3.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=4.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=5.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=6.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=7.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=8.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=9.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=10.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=11.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=12.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=13.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=14.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=15.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=16.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=17.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=18.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=19.0, render_mode="human")),
    RecordEpisodeStatistics(gym.make(env_name, g=20.0, render_mode="human")),
]
test_set = TaskSet(test_contexts, test_envs)

for i in range(len(test_envs)):
    env, context = test_set.get_task_and_context(i)
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
