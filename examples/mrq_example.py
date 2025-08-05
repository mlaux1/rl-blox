import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.mrq import create_mrq_state, train_mrq
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)

seed = 1
verbose = 2
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    seed=seed,
)
hparams_algorithm = dict(
    seed=seed,
    total_timesteps=12_000,
    buffer_size=12_000,
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
    algorithm_name="MR.Q",
    hparams=hparams_models | hparams_algorithm,
)

state = create_mrq_state(env, **hparams_models)

result = train_mrq(
    env,
    state.policy_with_encoder,
    state.encoder_optimizer,
    state.policy_optimizer,
    state.q,
    state.q_optimizer,
    state.the_bins,
    logger=logger,
    **hparams_algorithm,
)
env.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        obs = jnp.asarray(obs)
        action = result.policy_with_encoder(obs)
        next_obs, reward, termination, truncation, infos = env.step(
            np.asarray(action)
        )
        done = termination or truncation
        if verbose:
            zs = result.policy_with_encoder.encoder.encode_zs(obs)
            zsa = result.policy_with_encoder.encoder.encode_zsa(zs, action)
            q_value = result.q(zsa)
            print(f"{q_value=}")
        obs = np.asarray(next_obs)
