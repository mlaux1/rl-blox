import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from rl_blox.algorithm.ddpg import create_ddpg_state, train_ddpg
from rl_blox.logging.logger import AIMLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)

seed = 1
verbose = 2
env = gym.wrappers.RecordEpisodeStatistics(env)

hparams_models = dict(
    policy_hidden_nodes=[256, 256],
    policy_learning_rate=1e-3,
    q_hidden_nodes=[256, 256],
    q_learning_rate=1e-3,
    seed=seed,
)
hparams_algorithm = dict(
    gradient_steps=1,
    seed=seed,
    total_timesteps=15_000,
    buffer_size=15_000,
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
    algorithm_name="DDPG",
    hparams=hparams_models | hparams_algorithm,
)

ddpg_state = create_ddpg_state(env, **hparams_models)

result = train_ddpg(
    env,
    ddpg_state.policy,
    ddpg_state.policy_optimizer,
    ddpg_state.q,
    ddpg_state.q_optimizer,
    logger=logger,
    **hparams_algorithm,
)
env.close()

policy = result.policy
q = result.q

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
        q_value = q(jnp.concatenate((obs, action)))
        if verbose:
            print(f"{q_value=}")
        obs = np.asarray(next_obs)
