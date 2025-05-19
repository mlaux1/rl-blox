import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import DeterministicTanhPolicy
from rl_blox.algorithm.cmaes import train_cmaes

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)
policy_net = MLP(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    [32, 32],
    "relu",
    nnx.Rngs(seed),
)
policy = DeterministicTanhPolicy(policy_net, env.action_space)
policy = train_cmaes(
    env,
    policy,
    10000,
    seed,
    n_samples_per_update=40,
    variance=1.0,
    active=True,
    verbose=2,
)
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
