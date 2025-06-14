import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.td7 import create_td7_state, train_td7
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "Hopper-v5"
env = gym.make(env_name)

seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=1e-3,
    q_hidden_nodes=[256, 256],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    policy_delay=2,
    exploration_noise=0.1,
    target_policy_noise=0.2,
    noise_clip=0.5,
    total_timesteps=1_000_000,
    buffer_size=1_000_000,
    learning_starts=25_000,
    batch_size=256,
    use_checkpoints=False,
    seed=seed,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = LoggerList(
    [AIMLogger(),
     #StandardLogger(verbose=2)
     ]
)
logger.define_experiment(
    env_name=env_name,
    algorithm_name="TD7",
    hparams=hparams_models | hparams_algorithm,
)

td7_state = create_td7_state(env, **hparams_models)

td3_result = train_td7(
    env,
    embedding=td7_state.embedding,
    embedding_optimizer=td7_state.embedding_optimizer,
    actor=td7_state.actor,
    actor_optimizer=td7_state.actor_optimizer,
    critic=td7_state.critic,
    critic_optimizer=td7_state.critic_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()
# TODO
