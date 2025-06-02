import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.sac import create_sac_state, train_sac
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 1
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[128, 128],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[512, 512],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    total_timesteps=11_000,
    buffer_size=11_000,
    gamma=0.99,
    learning_starts=5_000,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = LoggerList([
    AIMLogger(),
    OrbaxCheckpointer("/tmp/rl-blox/sac_example/", verbose=verbose),
])
logger.define_experiment(
    env_name=env_name,
    algorithm_name="SAC",
    hparams=hparams_models | hparams_algorithm,
)
logger.define_checkpoint_frequency("policy", 5_000)

sac_state = create_sac_state(env, **hparams_models)
sac_result = train_sac(
    env,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q,
    sac_state.q_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()
policy, _, q, _, _, _ = sac_result


# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs))[0])
        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)

        if verbose >= 2:
            q_value = float(
                q(jnp.concatenate((obs, action), axis=-1)).squeeze()
            )
            print(f"{q_value=:.3f}")
