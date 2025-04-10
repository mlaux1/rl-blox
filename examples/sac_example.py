import gymnasium as gym
import numpy as np
from flax import nnx
import jax.numpy as jnp

from rl_blox.algorithms.model_free.ddpg import MLP
from rl_blox.algorithms.model_free.sac import (
    GaussianMLP,
    GaussianPolicy,
    mean_action,
    train_sac,
)

env_name = "Pendulum-v1"
env = gym.make(env_name)
seed = 1
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

policy_net = GaussianMLP(
    False,
    env.observation_space.shape[0] + env.action_space.shape[0],
    env.action_space.shape[0],
    [256, 256],
    nnx.Rngs(seed),
)
policy = GaussianPolicy(policy_net, env.action_space)
q = MLP(
    env.observation_space.shape[0] + env.action_space.shape[0],
    1,
    [256, 256],
    nnx.Rngs(seed),
)
policy, q1, q2 = train_sac(
    env,
    policy,
    q,
    seed=seed,
    total_timesteps=8_000,
    buffer_size=1_000_000,
    gamma=0.99,
    learning_starts=5_000,
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
        action = np.asarray(mean_action(policy, obs)[0])
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        q1_value = q1(jnp.concatenate((obs, action), axis=-1))
        q2_value = q2(jnp.concatenate((obs, action), axis=-1))
        q_value = np.minimum(q1_value, q2_value)
        print(f"{q_value=}")
        obs = np.asarray(next_obs)
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"episodic_return={info['episode']['r']}")
            break
