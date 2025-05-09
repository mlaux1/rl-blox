import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.td3 import create_td3_state, train_td3
from rl_blox.logging.logger import AIMLogger

env_name = "Hopper-v5"
env = gym.make(env_name)

seed = 1
verbose = 2
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=3e-4,
    q_hidden_nodes=[256, 256],
    q_learning_rate=3e-4,
    seed=seed,
)
hparams_algorithm = dict(
    policy_delay=2,
    exploration_noise=0.2,
    noise_clip=0.5,
    gradient_steps=1,
    total_timesteps=1_000_000,
    buffer_size=1_000_000,
    learning_starts=5_000,
    seed=seed,
)

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="TD3",
    hparams=hparams_models | hparams_algorithm,
)

ddpg_state = create_td3_state(env, **hparams_models)

td3_result = train_td3(
    env,
    ddpg_state.policy,
    ddpg_state.policy_optimizer,
    ddpg_state.q1,
    ddpg_state.q1_optimizer,
    ddpg_state.q2,
    ddpg_state.q2_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()
policy, _, _, q1, _, _, q2, _, _ = td3_result

# Evaluation
env = gym.make(env_name, render_mode="human")
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        if verbose >= 2:
            q1_value = float(
                q1(jnp.concatenate((obs, action), axis=-1)).squeeze()
            )
            q2_value = float(
                q2(jnp.concatenate((obs, action), axis=-1)).squeeze()
            )
            q_value = min(q1_value, q2_value)
            print(f"{q_value=:.3f} {q1_value=:.3f} {q2_value=:.3f}")
        obs = np.asarray(next_obs)
