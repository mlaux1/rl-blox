import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.cmaes import train_cmaes
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    DeterministicTanhPolicy,
)
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
hparams_model = dict(
    hidden_nodes=[64, 64],
    activation="relu",
)
hparams_algorithm = dict(
    n_samples_per_update=None,
    variance=0.3,
    active=False,
    total_episodes=10_000,
    seed=seed,
)
policy_net = MLP(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    **hparams_model,
    rngs=nnx.Rngs(seed),
)
policy = DeterministicTanhPolicy(policy_net, env.action_space)

logger = LoggerList([StandardLogger(verbose=1), AIMLogger()])
logger.define_experiment(
    env_name,
    "CMA-ES",
    hparams=hparams_model | hparams_algorithm,
)

policy = train_cmaes(env, policy, **hparams_algorithm, logger=logger)
env.close()

# Evaluation
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
while True:
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = np.asarray(policy(jnp.asarray(obs)))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
