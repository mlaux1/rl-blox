import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.multi_task.uniform_task_sampling import (
    TaskSet,
    train_uts,
)
from rl_blox.algorithm.sac import create_sac_state
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList

env_name = "Pendulum-v1"
seed = 42
verbose = 1

train_contexts = jnp.linspace(10, 11, 2)[:, jnp.newaxis]
train_envs = [
    RecordEpisodeStatistics(gym.make(env_name, g=10)),
    RecordEpisodeStatistics(gym.make(env_name, g=11)),
]

train_set = TaskSet(train_contexts, train_envs)

hparams_models = dict(
    q_hidden_nodes=[512, 512],
    q_learning_rate=3e-4,
    policy_learning_rate=1e-3,
    policy_hidden_nodes=[256, 256],
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=10_000,
    exploring_starts=1_000,
    episodes_per_task=1,
)

logger = LoggerList([AIMLogger(), OrbaxCheckpointer()])
logger.define_experiment(
    env_name=env_name,
    algorithm_name="UTS-SAC",
    hparams=hparams_models | hparams_algorithm,
)

sac_state = create_sac_state(train_envs[0], **hparams_models)
sac_result = train_uts(
    train_set,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    **hparams_algorithm,
    logger=logger,
)


policy, _, q, _, _, _, _ = sac_result


for i in range(2):
    env = train_set.get_task_env(i)
    ep_return = 0.0
    done = False
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs))[0])
        next_obs, reward, termination, truncation, info = env.step(action)
        ep_return += reward
        done = termination or truncation
        obs = np.asarray(next_obs)
    print(f"Episode terminated in with {ep_return=}")

for env in train_set:
    env.close()
