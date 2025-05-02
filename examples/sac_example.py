import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.sac import (
    create_sac_state,
    train_sac,
    NormalizeObservationStreamX,
)
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
verbose = 1
gamma = 0.99
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
    gamma=gamma,
    learning_starts=5_000,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="SAC",
    hparams=hparams_models | hparams_algorithm,
)

sac_state = create_sac_state(env, **hparams_models)
sac_result = train_sac(
    env,
    sac_state.policy,
    sac_state.policy_optimizer,
    sac_state.q1,
    sac_state.q1_optimizer,
    sac_state.q2,
    sac_state.q2_optimizer,
    observation_normalizer=NormalizeObservationStreamX(),
    logger=logger,
    **hparams_algorithm,
)
env.close()
policy, _, q1, _, _, q2, _, _, _, obs_norm, _ = sac_result


# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    obs, _ = env.reset()
    while not done:
        obs_in = jnp.asarray(obs).squeeze()
        if obs_norm is not None:
            obs_in = obs_norm.transform(obs_in)
        action = np.asarray(policy(obs_in))
        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)

        if verbose >= 2:
            obs_act = jnp.concatenate((obs_in, action), axis=-1)
            q1_value = float(q1(obs_act).squeeze())
            q2_value = float(q2(obs_act).squeeze())
            q_value = min(q1_value, q2_value)
            print(f"{q_value=:.3f} {q1_value=:.3f} {q2_value=:.3f}")
