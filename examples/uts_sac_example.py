import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from rl_blox.algorithm.multi_task.uts_sac import TaskSet, train_uts_sac
from rl_blox.algorithm.sac import create_sac_state
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
seed = 1
verbose = 1

train_contexts = jnp.array([[10.0], [10.1], [9.9]])

train_envs = [
    RecordEpisodeStatistics(gym.make(env_name, g=10.0)),
    RecordEpisodeStatistics(gym.make(env_name, g=10.1)),
    RecordEpisodeStatistics(gym.make(env_name, g=9.9)),
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
    total_timesteps=100_000,
    exploring_starts=1_000,
    episodes_per_task=1,
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="UTS-SAC",
    hparams=hparams_models | hparams_algorithm,
)

sac_state = create_sac_state(train_envs[0], **hparams_models)
sac_result = train_uts_sac(
    train_set,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    **hparams_algorithm,
    logger=logger,
)


for env in train_envs:
    env.close()

policy, _, q, _, _, _, _ = sac_result

# Evaluation
test_contexts = jnp.array([[10.0], [9.9], [10.2]])
test_envs = [
    RecordEpisodeStatistics(gym.make(env_name, g=10.0)),
    RecordEpisodeStatistics(gym.make(env_name, g=9.9)),
    RecordEpisodeStatistics(gym.make(env_name, g=10.2)),
]
test_set = TaskSet(test_contexts, test_envs)

for i in range(3):
    env = test_set.get_task_env(i)
    done = False
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs))[0])
        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)

for env in test_envs:
    env.close()
